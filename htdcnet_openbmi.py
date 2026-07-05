"""
HTDCNet LOSO on OpenBMI (Lee2019-MI) Session 1
2-class MI (left hand vs right hand), 62 EEG channels, 1000Hz -> 250Hz
Best config from BNCI2014_004: d_model=32, 2 heads, EA + SupCon
"""
import os, sys, time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
DATA_ROOT = '/home/ubuntu/mne_data/MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542'
SESSION = 'session1'
FS = 1000
EPOCH_SEC = 4
EPOCH_LEN = EPOCH_SEC * FS  # 4000 samples
TARGET_FS = 250
TARGET_LEN = EPOCH_SEC * TARGET_FS  # 1000 samples after downsample
NUM_CLASSES = 2
NUM_CHANNELS = 62
BANDPASS = (4, 40)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Euclidean Alignment (same as htdcnet_loso.py)
# ============================================================
def _matrix_sqrt(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def _matrix_sqrt_inv(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

def euclidean_alignment_per_subject(subject_data, test_sid, X_train_all, X_test, reg=1e-5):
    C, T = X_train_all.shape[1], X_train_all.shape[2]
    subject_covs = {}
    for sid in subject_data:
        Xs = subject_data[sid][0]
        covs = []
        for i in range(Xs.shape[0]):
            x = Xs[i]
            cov = x @ x.T / T + reg * np.eye(C)
            covs.append(cov)
        subject_covs[sid] = np.mean(covs, axis=0)
    train_sids = [s for s in subject_covs if s != test_sid]
    R_ref = np.mean([subject_covs[s] for s in train_sids], axis=0)
    R_ref_sqrt = _matrix_sqrt(R_ref)
    X_train_aligned = np.zeros_like(X_train_all)
    offset = 0
    for sid in train_sids:
        N_s = subject_data[sid][0].shape[0]
        R_src_sqrt_inv = _matrix_sqrt_inv(subject_covs[sid])
        T_src = R_ref_sqrt @ R_src_sqrt_inv
        X_train_aligned[offset:offset+N_s] = np.einsum('ij,njk->nik', T_src, X_train_all[offset:offset+N_s])
        offset += N_s
    R_test_sqrt_inv = _matrix_sqrt_inv(subject_covs[test_sid])
    T_test = R_ref_sqrt @ R_test_sqrt_inv
    X_test_aligned = np.einsum('ij,njk->nik', T_test, X_test)
    return X_train_aligned, X_test_aligned

# ============================================================
# Data Loading
# ============================================================
def load_subject(subject_id, data_root=DATA_ROOT, session=SESSION):
    """Load one subject's session data. Returns (X, y) with shape (N, C, T)."""
    sess_num = int(session[-1])
    fpath = os.path.join(data_root, session, f's{subject_id}',
                         f'sess{sess_num:02d}_subj{subject_id:02d}_EEG_MI.mat')
    mat = sio.loadmat(fpath)

    all_X, all_y = [], []
    for key in ['EEG_MI_train', 'EEG_MI_test']:
        d = mat[key][0, 0]
        smt = d['smt']       # (4000, 100, 62) - pre-segmented epochs
        y_dec = d['y_dec'][0]  # (100,) - labels: 1=right, 2=left
        n_trials = smt.shape[1]

        for i in range(n_trials):
            epoch = smt[:, i, :]  # (4000, 62)
            label = y_dec[i] - 1  # 0-indexed: 0=right, 1=left
            all_X.append(epoch)
            all_y.append(label)

    all_X = np.array(all_X)  # (N, 4000, 62)
    all_y = np.array(all_y)

    # Bandpass filter continuous data (all trials at once for speed)
    nyq = 0.5 * FS
    b, a = butter(4, [BANDPASS[0]/nyq, BANDPASS[1]/nyq], btype='band')
    for i in range(all_X.shape[0]):
        all_X[i] = filtfilt(b, a, all_X[i], axis=0)

    # Downsample to TARGET_FS
    all_X_ds = np.zeros((all_X.shape[0], TARGET_LEN, all_X.shape[2]))
    for i in range(all_X.shape[0]):
        all_X_ds[i] = resample(all_X[i], TARGET_LEN, axis=0)
    all_X = all_X_ds

    # Z-score normalization per trial
    mean = all_X.mean(axis=1, keepdims=True)
    std = all_X.std(axis=1, keepdims=True) + 1e-8
    all_X = (all_X - mean) / std

    # Transpose to (N, C, T) for PyTorch
    all_X = all_X.transpose(0, 2, 1).astype(np.float32)
    return all_X, all_y


class EEGDataset(Dataset):
    def __init__(self, X, y, augment=False, channel_shuffle_p=0.3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.channel_shuffle_p = channel_shuffle_p

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment and torch.rand(1).item() < self.channel_shuffle_p:
            perm = torch.randperm(x.size(0))
            x = x[perm]
        return x, y


# ============================================================
# Model (same architecture as htdcnet_loso.py)
# ============================================================
class SEChannelAttention(nn.Module):
    def __init__(self, C_in, reduction=2):
        super().__init__()
        mid = max(C_in // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(C_in, mid), nn.ReLU(), nn.Linear(mid, C_in), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)

class GATEModule(nn.Module):
    def __init__(self, C_in, C_out=32):
        super().__init__()
        self.cheb_conv = nn.Sequential(
            nn.Conv1d(C_in, C_out, 1), nn.BatchNorm1d(C_out), nn.ReLU(),
            nn.Conv1d(C_out, C_out, 3, padding=1, groups=C_out), nn.BatchNorm1d(C_out), nn.ReLU())
        self.se = SEChannelAttention(C_out)
        self.ts_conv = nn.Sequential(
            nn.Conv1d(C_out, C_out, 25, padding=12), nn.BatchNorm1d(C_out), nn.ReLU(),
            nn.Conv1d(C_out, C_out, 1))
    def forward(self, x):
        x = self.cheb_conv(x)
        x = self.se(x) + x
        x = self.ts_conv(x) + x
        return x

class ResidualECA(nn.Module):
    def __init__(self, C, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(C, C, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(C)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(C, C, 1)
    def forward(self, x):
        out = self.bn(self.conv(x))
        w = torch.sigmoid(self.fc(self.pool(out)))
        return out * w + x

class LocalMaskedMHA(nn.Module):
    def __init__(self, d_model, n_heads=2, window=16):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self._mask_cache = {}
    def _get_mask(self, T, device):
        key = (T, str(device))
        if key not in self._mask_cache:
            idx = torch.arange(T, device=device)
            mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > self.window
            self._mask_cache[key] = mask
        return self._mask_cache[key]
    def forward(self, x):
        B, C, T = x.shape
        xt = x.permute(0, 2, 1)
        mask = self._get_mask(T, x.device)
        xt_ln = self.ln(xt)
        out, _ = self.attn(xt_ln, xt_ln, xt_ln, attn_mask=mask)
        out = self.proj(out) + xt
        return out.permute(0, 2, 1)

class GlobalMHA(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, C, T = x.shape
        xt = x.permute(0, 2, 1)
        xt_ln = self.ln(xt)
        out, _ = self.attn(xt_ln, xt_ln, xt_ln)
        out = self.proj(out) + xt
        return out.permute(0, 2, 1)

class TCNBlock(nn.Module):
    def __init__(self, C, n_layers=2, base_dilation=1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d = base_dilation * (2 ** i)
            layers.extend([nn.Conv1d(C, C, 3, dilation=d, padding=d), nn.BatchNorm1d(C), nn.ReLU(), nn.Dropout(0.3)])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x) + x

class LocalEncoder(nn.Module):
    def __init__(self, d_model, n_heads=2, window=16):
        super().__init__()
        self.eca = ResidualECA(d_model)
        self.local_mha = LocalMaskedMHA(d_model, n_heads, window)
        self.tcn = TCNBlock(d_model, n_layers=2, base_dilation=1)
    def forward(self, x):
        return self.tcn(self.local_mha(self.eca(x)))

class GlobalEncoder(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.eca = ResidualECA(d_model)
        self.global_mha = GlobalMHA(d_model, n_heads)
        self.tcn = TCNBlock(d_model, n_layers=4, base_dilation=1)
    def forward(self, x):
        return self.tcn(self.global_mha(self.eca(x)))

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature
    def forward(self, z1, z2, labels):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.size(0)
        logits = torch.mm(z1, z2.t()) / self.temp
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        n_pos = label_mask.sum(dim=1).clamp(min=1.0)
        lse = torch.logsumexp(logits, dim=1)
        pair_scores = logits - lse.unsqueeze(1)
        pos_scores = label_mask * pair_scores
        return -(pos_scores.sum(dim=1) / n_pos).mean()

class HTDCNet(nn.Module):
    def __init__(self, C_in=62, d_model=32, n_heads=2, window=16, num_classes=2):
        super().__init__()
        self.gate = GATEModule(C_in, d_model)
        self.local_enc = LocalEncoder(d_model, n_heads, window)
        self.global_enc = GlobalEncoder(d_model, n_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes))
        self.infonce = SupervisedContrastiveLoss(0.5)

    def forward(self, x):
        f_gate = self.gate(x)
        h_local = self.local_enc(f_gate)
        h_global = self.global_enc(f_gate)
        z_local = self.pool(h_local).squeeze(-1)
        z_global = self.pool(h_global).squeeze(-1)
        return self.classifier(z_global), z_local, z_global

# ============================================================
# Training
# ============================================================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def train_one_epoch(model, loader, optimizer, lam=0.5, label_smoothing=0.1, mixup_alpha=0.2):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        yb_orig = yb
        xb, yb_a, yb_b, mix_lam = mixup_data(xb, yb, alpha=mixup_alpha)
        optimizer.zero_grad()
        logits, zl, zg = model(xb)
        loss_ce = mix_lam * F.cross_entropy(logits, yb_a, label_smoothing=label_smoothing) + \
                  (1 - mix_lam) * F.cross_entropy(logits, yb_b, label_smoothing=label_smoothing)
        loss_cl = model.infonce(zl, zg, yb_orig)
        loss = loss_ce + lam * loss_cl
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _, _ = model(xb)
        correct += (logits.argmax(-1) == yb).sum().item()
        total += len(yb)
    return correct / total


def run_loso(num_subjects=54, num_epochs=80, batch_size=64, lr=1e-3, lam=0.5,
             d_model=32, n_heads=2, label_smoothing=0.1, mixup_alpha=0.2, use_ea=True):
    print(f"{'='*60}")
    print(f"HTDCNet LOSO on OpenBMI Session 1")
    print(f"Device: {DEVICE} | Subjects: {num_subjects} | Epochs: {num_epochs}")
    print(f"d_model: {d_model} | heads: {n_heads} | EA: {use_ea} | lam: {lam}")
    print(f"Channels: {NUM_CHANNELS} | Seq length: {TARGET_LEN} ({TARGET_FS}Hz)")
    print(f"{'='*60}\n")

    print("Loading data...")
    t0 = time.time()
    subject_data = {}
    for sid in range(1, num_subjects + 1):
        ts = time.time()
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
        print(f"  Subject {sid:02d}: {X.shape[0]} trials, C={X.shape[1]}, T={X.shape[2]} ({time.time()-ts:.1f}s)")
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    fold_accs = []
    for test_sid in range(1, num_subjects + 1):
        print(f"\n{'─'*50}")
        print(f"Fold {test_sid}/{num_subjects}: Test=Subject {test_sid:02d}")

        X_test, y_test = subject_data[test_sid]
        X_train_list, y_train_list = [], []
        for sid in range(1, num_subjects + 1):
            if sid == test_sid:
                continue
            Xs, ys = subject_data[sid]
            X_train_list.append(Xs)
            y_train_list.append(ys)

        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)

        if use_ea:
            print("  Applying Euclidean Alignment...")
            X_train_all, X_test = euclidean_alignment_per_subject(
                subject_data, test_sid, X_train_all, X_test)
        else:
            print("  Skipping EA...")

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(y_train_all))
        n_val = int(0.1 * len(idx))
        X_train, y_train = X_train_all[idx[n_val:]], y_train_all[idx[n_val:]]
        X_val, y_val = X_train_all[idx[:n_val]], y_train_all[idx[:n_val]]

        train_loader = DataLoader(EEGDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=batch_size, num_workers=0)

        C_in = X_train.shape[1]
        model = HTDCNet(C_in=C_in, d_model=d_model, n_heads=n_heads, window=16, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_val_acc, best_state = 0, None
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, lam,
                                                     label_smoothing=label_smoothing, mixup_alpha=mixup_alpha)
            val_acc = evaluate(model, val_loader)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d} | Loss:{train_loss:.4f} TrAcc:{train_acc:.3f} ValAcc:{val_acc:.3f}")

        model.load_state_dict(best_state)
        model.to(DEVICE)
        test_acc = evaluate(model, test_loader)
        fold_accs.append(test_acc)
        print(f"  >> Test Acc (S{test_sid:02d}): {test_acc:.4f} ({test_acc*100:.2f}%) | best val: {best_val_acc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("LOSO Results Summary - OpenBMI Session 1")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1} (Subj {i+1:02d}): {acc*100:.2f}%")
    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    print(f"\n  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"\n  Reference baselines (OpenBMI LOSO):")
    print(f"    SFE-GAT: ~65-67% | ATCNet: ~63% | EEGNet: ~55%")
    print(f"{'='*60}")
    return fold_accs


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    t_start = time.time()
    run_loso(num_subjects=54, num_epochs=80, batch_size=64, lr=1e-3, lam=0.5,
             d_model=32, n_heads=2, label_smoothing=0.1, mixup_alpha=0.2, use_ea=True)
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
