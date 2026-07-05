"""
HTDCNet Ablation Study on BNCI2014_004
Removes: GATE, Local Encoder, Global Encoder, SupCon loss
Runs LOSO for each variant and compares
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from htdcnet_loso import (
    load_subject, EEGDataset, HTDCNet, SupervisedContrastiveLoss, InfoNCELoss,
    GATEModule, LocalEncoder, GlobalEncoder,
    euclidean_alignment_per_subject,
    mixup_data, train_one_epoch, evaluate,
    NUM_SUBJECTS, NUM_CLASSES, DEVICE
)

OUTPUT_DIR = '/home/ubuntu/user_liu_20260403/python_projects/bibm2026/results/ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Ablation Model Variants
# ============================================================
class HTDCNet_NoGATE(nn.Module):
    """Ablation: Remove GATE module, feed raw EEG to both branches."""
    def __init__(self, C_in=6, d_model=32, n_heads=2, window=16, num_classes=2):
        super().__init__()
        # No GATE - use a simple projection instead
        self.proj = nn.Sequential(
            nn.Conv1d(C_in, d_model, 1), nn.BatchNorm1d(d_model), nn.ReLU())
        self.local_enc = LocalEncoder(d_model, n_heads, window)
        self.global_enc = GlobalEncoder(d_model, n_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes))
        self.supcon = SupervisedContrastiveLoss(0.5)

    def forward(self, x):
        f = self.proj(x)  # simple projection instead of GATE
        h_local = self.local_enc(f)
        h_global = self.global_enc(f)
        z_local = self.pool(h_local).squeeze(-1)
        z_global = self.pool(h_global).squeeze(-1)
        return self.classifier(z_global), z_local, z_global


class HTDCNet_NoLocal(nn.Module):
    """Ablation: Remove Local Encoder, only use Global Encoder."""
    def __init__(self, C_in=6, d_model=32, n_heads=2, window=16, num_classes=2):
        super().__init__()
        self.gate = GATEModule(C_in, d_model)
        self.global_enc = GlobalEncoder(d_model, n_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes))
        self.supcon = SupervisedContrastiveLoss(0.5)

    def forward(self, x):
        f_gate = self.gate(x)
        h_global = self.global_enc(f_gate)
        z_global = self.pool(h_global).squeeze(-1)
        return self.classifier(z_global), z_global, z_global  # no local branch


class HTDCNet_NoGlobal(nn.Module):
    """Ablation: Remove Global Encoder, only use Local Encoder."""
    def __init__(self, C_in=6, d_model=32, n_heads=2, window=16, num_classes=2):
        super().__init__()
        self.gate = GATEModule(C_in, d_model)
        self.local_enc = LocalEncoder(d_model, n_heads, window)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes))
        self.supcon = SupervisedContrastiveLoss(0.5)

    def forward(self, x):
        f_gate = self.gate(x)
        h_local = self.local_enc(f_gate)
        z_local = self.pool(h_local).squeeze(-1)
        return self.classifier(z_local), z_local, z_local  # no global branch


class HTDCNet_NoSupCon(nn.Module):
    """Ablation: Full architecture but only CE loss (no contrastive)."""
    def __init__(self, C_in=6, d_model=32, n_heads=2, window=16, num_classes=2):
        super().__init__()
        self.gate = GATEModule(C_in, d_model)
        self.local_enc = LocalEncoder(d_model, n_heads, window)
        self.global_enc = GlobalEncoder(d_model, n_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes))

    def forward(self, x):
        f_gate = self.gate(x)
        h_local = self.local_enc(f_gate)
        h_global = self.global_enc(f_gate)
        z_local = self.pool(h_local).squeeze(-1)
        z_global = self.pool(h_global).squeeze(-1)
        return self.classifier(z_global), z_local, z_global


def train_one_epoch_ablation(model, loader, optimizer, lam=0.3,
                              label_smoothing=0.1, mixup_alpha=0.2, use_supcon=True):
    """Training loop that handles both SupCon and no-SupCon variants."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        yb_orig = yb
        xb, yb_a, yb_b, mix_lam = mixup_data(xb, yb, alpha=mixup_alpha)
        optimizer.zero_grad()
        logits, z_local, z_global = model(xb)
        loss_ce = mix_lam * F.cross_entropy(logits, yb_a, label_smoothing=label_smoothing) + \
                  (1 - mix_lam) * F.cross_entropy(logits, yb_b, label_smoothing=label_smoothing)
        
        if use_supcon and hasattr(model, 'supcon'):
            loss_cl = model.supcon(z_local, z_global, yb_orig)
            loss = loss_ce + lam * loss_cl
        else:
            loss = loss_ce
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


def run_ablation_loso(model_class, model_kwargs, use_supcon=True, num_epochs=80):
    """Run LOSO for a given model variant."""
    # Load data
    subject_data = {}
    for sid in range(1, NUM_SUBJECTS + 1):
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
    
    fold_accs = []
    for test_sid in range(1, NUM_SUBJECTS + 1):
        X_test, y_test = subject_data[test_sid]
        X_train_list, y_train_list = [], []
        for sid in range(1, NUM_SUBJECTS + 1):
            if sid == test_sid:
                continue
            Xs, ys = subject_data[sid]
            X_train_list.append(Xs)
            y_train_list.append(ys)
        
        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)
        
        # EA
        X_train_all, X_test = euclidean_alignment_per_subject(
            subject_data, test_sid, X_train_all, X_test)
        
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(y_train_all))
        n_val = int(0.1 * len(idx))
        X_train, y_train = X_train_all[idx[n_val:]], y_train_all[idx[n_val:]]
        X_val, y_val = X_train_all[idx[:n_val]], y_train_all[idx[:n_val]]
        
        train_loader = DataLoader(EEGDataset(X_train, y_train, augment=True), batch_size=64, shuffle=True)
        val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=64)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=64)
        
        model = model_class(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_val_acc, best_state = 0, None
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch_ablation(
                model, train_loader, optimizer, lam=0.3,
                label_smoothing=0.1, mixup_alpha=0.2, use_supcon=use_supcon)
            val_acc = evaluate(model, val_loader)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        model.load_state_dict(best_state)
        model.to(DEVICE)
        test_acc = evaluate(model, test_loader)
        fold_accs.append(test_acc)
    
    return fold_accs


def run_all_ablations():
    print("=" * 60)
    print("HTDCNet Ablation Study - BNCI2014_004 LOSO")
    print("=" * 60)
    
    base_kwargs = dict(C_in=6, d_model=32, n_heads=2, window=16, num_classes=2)
    
    experiments = [
        ("Full Model (HTDCNet)", HTDCNet, base_kwargs, True),
        ("w/o GATE", HTDCNet_NoGATE, base_kwargs, True),
        ("w/o Local Encoder", HTDCNet_NoLocal, base_kwargs, True),
        ("w/o Global Encoder", HTDCNet_NoGlobal, base_kwargs, True),
        ("w/o SupCon Loss", HTDCNet_NoSupCon, base_kwargs, False),
    ]
    
    results = {}
    for name, model_class, kwargs, use_supcon in experiments:
        print(f"\n{'─'*50}")
        print(f"Ablation: {name}")
        print(f"{'─'*50}")
        t0 = time.time()
        accs = run_ablation_loso(model_class, kwargs, use_supcon=use_supcon, num_epochs=80)
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        results[name] = (mean_acc, std_acc, accs)
        print(f"  Mean: {mean_acc:.2f}% +/- {std_acc:.2f}%  ({time.time()-t0:.0f}s)")
        for i, a in enumerate(accs):
            print(f"    Fold {i+1}: {a*100:.2f}%")
    
    # Summary table
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'Mean Acc':>10} {'Std':>8}")
    print(f"{'─'*50}")
    for name, (mean, std, _) in results.items():
        delta = mean - results["Full Model (HTDCNet)"][0]
        print(f"{name:<30} {mean:>9.2f}% {std:>7.2f}%  ({delta:+.2f}%)")
    print(f"{'='*60}")
    
    # Save results
    import json
    out = {}
    for name, (mean, std, accs) in results.items():
        out[name] = {'mean': float(mean), 'std': float(std),
                     'folds': [float(a) for a in accs]}
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'ablation_results.json')}")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    t0 = time.time()
    run_all_ablations()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
