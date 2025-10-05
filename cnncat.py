"""
Binary “cat / not-cat” on CIFAR-10 with:
• Inverse-frequency sampler + pos-weighted BCE (TRAIN OBJECTIVE)
• Optuna HPO (optional)
• One-Cycle LR (base LR = 5e-4, max LR = 3e-3, longer ramp-up)
• Strong regularization: dropout = 0.30, weight_decay = 1e-6
• Label smoothing (ε = 0.05) — TRAIN ONLY
• Simplified augmentation (flip, crop, rotate, mild color jitter, random erasing p=0.4)
• Larger batch (128) for smoother gradients
• No early stopping — fixed 200-epoch run
• Separate loss & accuracy plots
• BatchNorm on FC layers + gradient clipping (max_norm=1.0)
• Also compute PLAIN BCE on train (for fair plotting vs val).
• Fit temperature on val after training → reduces plain-BCE (train/val/test) with accuracy unchanged.
"""

import torch, random, optuna
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SMOOTH = 0.05  

class CatCNN(nn.Module):
    def __init__(self, c1, c2, c3, c4, h1, h2, d1, d2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  c1, 3, padding=1); self.bn1  = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1); self.bn2  = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1); self.bn3  = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1); self.bn4  = nn.BatchNorm2d(c4)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(c4*2*2, h1); self.bn5 = nn.BatchNorm1d(h1)
        self.fc2   = nn.Linear(h1,      h2); self.bn6 = nn.BatchNorm1d(h2)
        self.fc3   = nn.Linear(h2,      1)
        self.relu  = nn.ReLU()
        self.d1    = nn.Dropout(d1)
        self.d2    = nn.Dropout(d2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.d1(self.relu(self.bn5(self.fc1(x))))
        x = self.d2(self.relu(self.bn6(self.fc2(x))))
        return self.fc3(x)

#temperature scaling
class TempScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = torch.nn.Parameter(torch.zeros(()))  # T=1

    def forward(self, logits):
        return logits / self.logT.exp()

def collect_logits_targets(model, loader, device):
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            z = model(x).view(-1)
            all_z.append(z); all_y.append(y)
    return torch.cat(all_z), torch.cat(all_y)

def fit_temperature(model, val_loader, device):
    z, y = collect_logits_targets(model, val_loader, device)
    ts = TempScale().to(device)
    opt = torch.optim.LBFGS([ts.logT], lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(ts(z), y)
        loss.backward()
        return loss

    opt.step(closure)
    return ts

#Optuna objective
def objective(trial):
    hp = {
        'c1': trial.suggest_int('c1',  32, 256, 32),
        'c2': trial.suggest_int('c2',  64, 512, 64),
        'c3': trial.suggest_int('c3',  64, 512, 64),
        'c4': trial.suggest_int('c4',  64, 512, 64),
        'h1': trial.suggest_int('h1', 256,1024,256),
        'h2': trial.suggest_int('h2',  64, 512, 64),
        'd1': 0.30, 'd2': 0.30,
    }
    lr       = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    opt_type = trial.suggest_categorical('opt', ['Adam','SGD'])

    m   = CatCNN(**hp).to(device)
    optimizer = (optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
                 if opt_type=='Adam' else
                 optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6))
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    m.train()
    for _ in range(2):
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            y_s = y*(1-SMOOTH) + (1-y)*SMOOTH
            optimizer.zero_grad()
            loss = criterion_train(m(x).view(-1), y_s)
            loss.backward(); clip_grad_norm_(m.parameters(), max_norm=1.0)
            optimizer.step()

    #accuracy logic 
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device); y = y.to(device)
            preds = (torch.sigmoid(m(x).view(-1))>0.5).float().cpu()
            correct += (preds==y.cpu()).sum().item()
            total   += y.size(0)
    return 1 - correct/total

if __name__=='__main__':
    RUN_OPTUNA = True
    SEED       = 42
    torch.manual_seed(SEED); random.seed(SEED)
    default_hp = dict(conv1=32, conv2=192, conv3=128, conv4=256,
                      h1=1024, h2=64, d1=0.30, d2=0.30, lr=5e-4, opt='SGD')

    global device, train_loader, val_loader, pos_weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #augmentation
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.RandomRotation(15),
        T.ColorJitter(0.1,0.1,0.1),
        T.ToTensor(),
        T.Normalize((0.5,)*3,(0.5,)*3),
        T.RandomErasing(p=0.4)
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])

    #dataset/split
    raw = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=None)
    idx = list(range(len(raw)))
    tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    vl_idx, ts_idx = train_test_split(tmp, test_size=0.50, random_state=SEED)

    class CatWrap(torch.utils.data.Dataset):
        def __init__(self, idxs, tf):
            self.idxs, self.tf = idxs, tf
        def __len__(self): return len(self.idxs)
        def __getitem__(self,i):
            img,lbl = raw[self.idxs[i]]
            return self.tf(img), torch.tensor(lbl==3, dtype=torch.float32)

    train_ds = CatWrap(tr_idx, train_tf)
    val_ds   = CatWrap(vl_idx, test_tf)
    test_ds  = CatWrap(ts_idx, test_tf)

    #imbalance sampler for optimization; separate eval loader for training set
    labels      = torch.tensor([y.item() for _,y in train_ds])
    cnt         = torch.bincount(labels.long())
    wts         = 1.0/cnt.float()
    sampler     = WeightedRandomSampler(wts[labels.long()], len(labels), True)

    train_loader      = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
    train_eval_loader = DataLoader(train_ds, batch_size=128, shuffle=False,  num_workers=0) 
    val_loader        = DataLoader(val_ds,   batch_size=128, shuffle=False,  num_workers=0)
    test_loader       = DataLoader(test_ds,  batch_size=128, shuffle=False,  num_workers=0)

    pos_weight = torch.tensor([cnt[0]/cnt[1]], device=device)

    #criteria
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    criterion_eval  = nn.BCEWithLogitsLoss()                       

    #Optuna
    if RUN_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)
        hp = {**default_hp, **study.best_trial.params}
        print("Optuna best:", study.best_params)
    else:
        hp = default_hp

    #model/opt/sched
    model = CatCNN(hp['conv1'],hp['conv2'],hp['conv3'],hp['conv4'],
                   hp['h1'],hp['h2'],hp['d1'],hp['d2']).to(device)
    optimizer = (optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=1e-6)
                 if hp['opt']=='Adam' else
                 optim.SGD(model.parameters(), lr=hp['lr'], momentum=0.9, weight_decay=1e-6))
    epochs    = 300
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, pct_start=0.3,
                           steps_per_epoch=len(train_loader), epochs=epochs)

    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        obj_L, plain_L, corr, tot = 0.0, 0.0, 0, 0
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            y_s = y*(1-SMOOTH) + (1-y)*SMOOTH  
            if train: optimizer.zero_grad()
            logits = model(x).view(-1)

            #losses
            obj_loss   = (criterion_train(logits, y_s) if train else criterion_eval(logits, y))
            plain_loss = criterion_eval(logits, y) 

            if train:
                obj_loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            obj_L   += obj_loss.item()
            plain_L += plain_loss.item()

            preds = (torch.sigmoid(logits)>0.5).float()
            corr += (preds==y).sum().item()
            tot  += y.size(0)

        return (obj_L/len(dl), plain_L/len(dl), corr/tot) if train else (plain_L/len(dl), corr/tot)

    history = {"train_obj":[], "train_plain":[], "train_acc":[],
               "val_plain":[],   "val_acc":[]}
    best_val = float('inf')

    for ep in range(1, epochs+1):
        trObj,trPlain,trA = run_epoch(train_loader, True)
        vL,vA             = run_epoch(val_loader,   False)

        history["train_obj"].append(trObj)
        history["train_plain"].append(trPlain)
        history["train_acc"].append(trA)
        history["val_plain"].append(vL)
        history["val_acc"].append(vA)

        print(f"Epoch {ep:03d} | "
              f"Train obj {trObj:.3f} | Train plain {trPlain:.3f} | Acc {trA*100:.1f}% || "
              f"Val plain {vL:.3f} | Acc {vA*100:.1f}%")

        if vL < best_val:
            best_val = vL
            torch.save(model.state_dict(), 'best.pth')

    #test
    model.load_state_dict(torch.load('best.pth', map_location=device))
    #plain losses
    def eval_plain_loss(dl):
        model.eval(); L=0.0; corr=0; tot=0
        with torch.no_grad():
            for x,y in dl:
                x,y = x.to(device), y.to(device)
                z = model(x).view(-1)
                L += criterion_eval(z, y).item()
                preds = (torch.sigmoid(z)>0.5).float()  
                corr += (preds==y).sum().item(); tot += y.size(0)
        return L/len(dl), corr/tot

    tr_plain_final, tr_acc_final = eval_plain_loss(train_eval_loader)
    v_plain_final,  v_acc_final  = eval_plain_loss(val_loader)
    te_plain_final, te_acc_final = eval_plain_loss(test_loader)
    print(f"Final (plain BCE) — Train {tr_plain_final:.3f}/{tr_acc_final*100:.1f}% | "
          f"Val {v_plain_final:.3f}/{v_acc_final*100:.1f}% | "
          f"Test {te_plain_final:.3f}/{te_acc_final*100:.1f}%")

    #temperature scaling
    ts = fit_temperature(model, val_loader, device)
    def eval_plain_loss_with_T(dl, ts):
        model.eval(); L=0.0
        with torch.no_grad():
            for x,y in dl:
                x,y = x.to(device), y.to(device)
                z = model(x).view(-1)
                z = ts(z)  
                L += criterion_eval(z, y).item()
        return L/len(dl)

    tr_cal = eval_plain_loss_with_T(train_eval_loader, ts)
    va_cal = eval_plain_loss_with_T(val_loader,        ts)
    te_cal = eval_plain_loss_with_T(test_loader,       ts)
    print(f"Calibrated plain-BCE — Train {tr_cal:.3f} | Val {va_cal:.3f} | Test {te_cal:.3f}")

    #plots
#loss
plt.figure()
plt.plot(history["train_plain"], '--', label='Train loss (plain BCE)')
plt.plot(history["val_plain"],   '-',  label='Val   loss (plain BCE)')
plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("BCE Loss")
plt.legend(); plt.tight_layout(); plt.show()

#acc
epochs_done = len(history["train_acc"])
plt.figure()
plt.plot([a*100 for a in history["train_acc"]], '--', label='Train acc')
plt.plot([a*100 for a in history["val_acc"]],   '-',  label='Val   acc')
plt.scatter(epochs_done + 1, te_acc_final*100, marker='*', s=150, c='red', label='Test acc')
plt.title("Accuracy Curve"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.legend(); plt.tight_layout(); plt.show()