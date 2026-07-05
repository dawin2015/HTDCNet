"""
HTDCNet: Hierarchical Temporal Disentangled Contrastive Network
Quick LOSO validation on BNCI2014_004 dataset

Architecture:
  Raw EEG -> [Shared GATE] -> F_GATE
    -> [Branch A: Local Encoder] Local MHA + Shallow TCN -> H_local
    -> [Branch B: Global Encoder] Global MHA + Deep TCN  -> H_global
  Loss = CE(Pool(H_global), y) + lambda * InfoNCE(Pool(H_local), Pool(H_global))
"""

import os, sys, time
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample
from scipy.linalg import sqrtm, inv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Data Loading & Preprocessing
# ============================================================

DATA_DIR = '/home/ubuntu/mne_data/MNE-bnci-data/database/data-sets/004-2014/'
FS = 250
EPOCH_SEC = 4
EPOCH_LEN = EPOCH_SEC * FS  # 1000 samples
TARGET_FS = 125             # downsample to 125Hz for speed
TARGET_LEN = EPOCH_SEC * TARGET_FS  # 500 samples after downsample
NUM_SUBJECTS = 9
NUM_CLASSES = 2
BANDPASS = (4, 40)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_subject(subject_id, data_dir=DATA_DIR):
    """Load one subject's train + eval data, extract epochs with optimized filtering."""
    sid = f'{subject_id:02d}'
    all_X, all_y = [], []

    # Precompute filter coefficients
    nyq = 0.5 * FS
    b, a = butter(4, [BANDPASS[0]/nyq, BANDPASS[1]/nyq], btype='band')

    for suffix in ['T', 'E']:
        fpath = os.path.join(data_dir, f'B{sid}{suffix}.mat')
        mat = sio.loadmat(fpath)
        data = mat['data']
        n_runs = data.shape[1]
        for r in range(n_runs):
            run = data[0, r]
            X = run['X'][0, 0]                    # (samples, channels)
            trials = run['trial'][0, 0].flatten()
            y = run['y'][0, 0].flatten()
            artifacts = run['artifacts'][0, 0].flatten()

            # Filter continuous data once (much faster than per-epoch)
            X_filt = filtfilt(b, a, X, axis=0)

            for t_start, label, art in zip(trials, y, artifacts):
                if art:
                    continue
                t_end = t_start + EPOCH_LEN
                if t_end > X_filt.shape[0]:
                    continue
                epoch = X_filt[t_start:t_end, :]  # (T, C)
                # Downsample to TARGET_FS
                epoch_ds = resample(epoch, TARGET_LEN, axis=0)
                all_X.append(epoch_ds)
                all_y.append(label - 1)

    all_X = np.array(all_X)   # (N, T, C)
    all_y = np.array(all_y)

    # Z-score normalization per trial
    mean = all_X.mean(axis=1, keepdims=True)
    std = all_X.std(axis=1, keepdims=True) + 1e-8
    all_X = (all_X - mean) / std

    # Transpose to (N, C, T)
    all_X = all_X.transpose(0, 2, 1).astype(np.float32)
    return all_X, all_y


# ============================================================
# Euclidean Alignment (EA) for cross-subject domain adaptation
# ============================================================

def _matrix_sqrt(M):
    """Compute symmetric matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical stability
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt_inv(M):
    """Compute inverse of symmetric matrix square root."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def euclidean_alignment(X_train, X_test, reg=1e-5):
    """
    Euclidean Alignment for LOSO cross-subject adaptation.
    
    Aligns all trials to a common reference covariance space:
      X_aligned = R_ref^(1/2) @ R_subject^(-1/2) @ X
    
    Args:
        X_train: (N_train, C, T) training trials from multiple subjects
        X_test:  (N_test, C, T)  test trials from one held-out subject
        reg:     regularization for covariance matrices
    
    Returns:
        X_train_aligned, X_test_aligned
    """
    C, T = X_train.shape[1], X_train.shape[2]
    
    # 1. Compute reference covariance from ALL training trials
    # R_ref = mean covariance across training subjects
    covs_train = []
    for i in range(X_train.shape[0]):
        x = X_train[i]  # (C, T)
        cov = x @ x.T / T + reg * np.eye(C)
        covs_train.append(cov)
    R_ref = np.mean(covs_train, axis=0)
    R_ref_sqrt = _matrix_sqrt(R_ref)
    
    # 2. Compute subject-specific mean covariance from test subject's own data
    covs_test = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        cov = x @ x.T / T + reg * np.eye(C)
        covs_test.append(cov)
    R_test_mean = np.mean(covs_test, axis=0)
    R_test_sqrt_inv = _matrix_sqrt_inv(R_test_mean)
    
    # Transform for test: R_ref^(1/2) @ R_test_subject^(-1/2) @ X
    T_test = R_ref_sqrt @ R_test_sqrt_inv
    
    # 3. Apply EA to test data
    X_test_aligned = np.zeros_like(X_test)
    for i in range(X_test.shape[0]):
        X_test_aligned[i] = T_test @ X_test[i]
    
    # 4. For training data, each source subject needs its own alignment
    # We need per-subject mean covariances. Group training trials by subject.
    # Since X_train_all concatenates all source subjects, we track offsets.
    # Simpler: use R_ref as both reference and source mean -> identity transform
    # This means training data is already in the reference space.
    # For better alignment, we compute per-subject transforms.
    
    return X_train, X_test_aligned


def euclidean_alignment_per_subject(subject_data, test_sid, X_train_all, X_test, reg=1e-5):
    """
    Full EA with per-subject alignment for training data.
    
    For each source subject: X_src_aligned = R_ref^(1/2) @ R_src^(-1/2) @ X_src
    For test subject:        X_test_aligned = R_ref^(1/2) @ R_test^(-1/2) @ X_test
    """
    C, T = X_train_all.shape[1], X_train_all.shape[2]
    
    # 1. Compute per-subject mean covariance from raw subject data
    subject_covs = {}
    for sid in subject_data:
        Xs = subject_data[sid][0]  # (N, C, T)
        covs = []
        for i in range(Xs.shape[0]):
            x = Xs[i]
            cov = x @ x.T / T + reg * np.eye(C)
            covs.append(cov)
        subject_covs[sid] = np.mean(covs, axis=0)
    
    # 2. Reference covariance = mean of all training subject covariances
    train_sids = [s for s in subject_covs if s != test_sid]
    R_ref = np.mean([subject_covs[s] for s in train_sids], axis=0)
    R_ref_sqrt = _matrix_sqrt(R_ref)
    
    # 3. Align training data per source subject
    X_train_aligned = np.zeros_like(X_train_all)
    offset = 0
    for sid in train_sids:
        N_s = subject_data[sid][0].shape[0]
        R_src_sqrt_inv = _matrix_sqrt_inv(subject_covs[sid])
        T_src = R_ref_sqrt @ R_src_sqrt_inv
        X_train_aligned[offset:offset+N_s] = np.einsum('ij,njk->nik', T_src, X_train_all[offset:offset+N_s])
        offset += N_s
    
    # 4. Align test data
    R_test_sqrt_inv = _matrix_sqrt_inv(subject_covs[test_sid])
    T_test = R_ref_sqrt @ R_test_sqrt_inv
    X_test_aligned = np.einsum('ij,njk->nik', T_test, X_test)
    
    return X_train_aligned, X_test_aligned


class EEGDataset(Dataset):
    def __init__(self, X, y, augment=False, channel_shuffle_p=0.3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.channel_shuffle_p = channel_shuffle_p

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        y = self.y[idx]
        if self.augment:
            # Channel shuffle augmentation
            if torch.rand(1).item() < self.channel_shuffle_p:
                perm = torch.randperm(x.size(0))
                x = x[perm]
        return x, y


# ============================================================
# 2. Model Components
# ============================================================

class SEChannelAttention(nn.Module):
    def __init__(self, C_in, reduction=2):
        super().__init__()
        mid = max(C_in // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(C_in, mid), nn.ReLU(),
            nn.Linear(mid, C_in), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * w


class GATEModule(nn.Module):
    def __init__(self, C_in, C_out=32):
        super().__init__()
        self.cheb_conv = nn.Sequential(
            nn.Conv1d(C_in, C_out, 1), nn.BatchNorm1d(C_out), nn.ReLU(),
            nn.Conv1d(C_out, C_out, 3, padding=1, groups=C_out),
            nn.BatchNorm1d(C_out), nn.ReLU(),
        )
        self.se = SEChannelAttention(C_out)
        self.ts_conv = nn.Sequential(
            nn.Conv1d(C_out, C_out, 25, padding=12), nn.BatchNorm1d(C_out), nn.ReLU(),
            nn.Conv1d(C_out, C_out, 1),
        )
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
    """Local windowed attention (vectorized mask)."""
    def __init__(self, d_model, n_heads=4, window=16):
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
            # Vectorized: True where |i-j| > window (i.e., NOT allowed)
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
    def __init__(self, d_model, n_heads=4):
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
            pad = d  # kernel_size=3, pad = dilation*(3-1)/2 = dilation
            layers.extend([
                nn.Conv1d(C, C, 3, dilation=d, padding=pad),
                nn.BatchNorm1d(C), nn.ReLU(), nn.Dropout(0.3),
            ])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x) + x


class LocalEncoder(nn.Module):
    def __init__(self, d_model, n_heads=4, window=16):
        super().__init__()
        self.eca = ResidualECA(d_model)
        self.local_mha = LocalMaskedMHA(d_model, n_heads, window)
        self.tcn = TCNBlock(d_model, n_layers=2, base_dilation=1)
    def forward(self, x):
        return self.tcn(self.local_mha(self.eca(x)))


class GlobalEncoder(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.eca = ResidualECA(d_model)
        self.global_mha = GlobalMHA(d_model, n_heads)
        self.tcn = TCNBlock(d_model, n_layers=4, base_dilation=1)
    def forward(self, x):
        return self.tcn(self.global_mha(self.eca(x)))


class InfoNCELoss(nn.Module):
    """Instance-level contrastive loss (kept for reference/comparison)."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature
    def forward(self, z1, z2, labels=None):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        logits = torch.mm(z1, z2.t()) / self.temp
        return F.cross_entropy(logits, torch.arange(z1.size(0), device=z1.device))


class SupervisedContrastiveLoss(nn.Module):
    """
    Class-aware supervised contrastive loss.
    
    Unlike instance-level InfoNCE, this treats same-class pairs as positives:
    - z1[i] vs z2[j] is a POSITIVE pair if labels[i] == labels[j]
    - z1[i] vs z2[j] is a NEGATIVE pair if labels[i] != labels[j]
    
    This is critical for MI-EEG where different trials of the same class
    should be pulled together in representation space.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2, labels):
        """
        Args:
            z1: (B, D) representations from branch 1
            z2: (B, D) representations from branch 2
            labels: (B,) class labels
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.size(0)

        # Similarity matrix between z1 and z2 (cosine sim / temperature)
        # Values are in [-1/temp, 1/temp] = [-2, 2] for temp=0.5 — bounded, no overflow
        logits = torch.mm(z1, z2.t()) / self.temp  # (B, B)

        # Positive mask: same class pairs (including self, since z1 and z2 are different views)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B, B)

        # Number of positive pairs per sample
        n_pos = label_mask.sum(dim=1).clamp(min=1.0)  # (B,)

        # Numerically stable loss using logsumexp
        # loss_i = -mean_{j: same class} [logits_ij - log(sum_k exp(logits_ik))]
        lse = torch.logsumexp(logits, dim=1)  # (B,) log-sum-exp over all candidates

        # For each positive pair, compute: logits_ij - lse_i
        # Then average over positive pairs per sample
        pair_scores = logits - lse.unsqueeze(1)  # (B, B) log-probabilities
        pos_scores = label_mask * pair_scores    # zero out negative pairs
        loss_per_sample = -pos_scores.sum(dim=1) / n_pos  # (B,)

        return loss_per_sample.mean()


class HTDCNet(nn.Module):
    def __init__(self, C_in=6, d_model=32, n_heads=4, window=16, num_classes=2):
        super().__init__()
        self.gate = GATEModule(C_in, d_model)
        self.local_enc = LocalEncoder(d_model, n_heads, window)
        self.global_enc = GlobalEncoder(d_model, n_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes),
        )
        self.infonce = SupervisedContrastiveLoss(0.5)

    def forward(self, x):
        f_gate = self.gate(x)
        h_local = self.local_enc(f_gate)
        h_global = self.global_enc(f_gate)
        z_local = self.pool(h_local).squeeze(-1)
        z_global = self.pool(h_global).squeeze(-1)
        return self.classifier(z_global), z_local, z_global

    def compute_loss(self, x, y, lam=0.5, label_smoothing=0.1):
        logits, z_local, z_global = self(x)
        loss_ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss_cl = self.infonce(z_local, z_global, y)
        return loss_ce + lam * loss_cl, loss_ce, loss_cl


# ============================================================
# 3. Training & Evaluation
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blend pairs of samples and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, loader, optimizer, lam, label_smoothing=0.1, mixup_alpha=0.2):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # Mixup augmentation
        xb, yb_a, yb_b, mix_lam = mixup_data(xb, yb, alpha=mixup_alpha)

        optimizer.zero_grad()
        logits, z_local, z_global = model(xb)
        # Mixup loss: weighted combination of CE on both label sets
        loss_ce = mix_lam * F.cross_entropy(logits, yb_a, label_smoothing=label_smoothing) + \
                  (1 - mix_lam) * F.cross_entropy(logits, yb_b, label_smoothing=label_smoothing)
        # For mixup: use original labels for contrastive loss (not mixed)
        # Use yb (original labels before mixup) for supervised contrastive
        yb_orig = yb  # keep original labels before mixup changes them
        loss_cl = model.infonce(z_local, z_global, yb_orig)
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


def run_loso(num_epochs=80, batch_size=64, lr=1e-3, lam=0.3, d_model=32,
              n_heads=2, label_smoothing=0.1, mixup_alpha=0.2, dropout=0.3):
    print(f"{'='*60}")
    print(f"HTDCNet LOSO Validation on BNCI2014_004")
    print(f"Device: {DEVICE} | Epochs: {num_epochs} | d_model: {d_model} | heads: {n_heads}")
    print(f"label_smooth: {label_smoothing} | mixup_alpha: {mixup_alpha} | dropout: {dropout} | lam: {lam}")
    print(f"Seq length: {TARGET_LEN} (downsampled to {TARGET_FS}Hz)")
    print(f"{'='*60}\n")

    # Preload all subjects
    print("Loading data...")
    t0 = time.time()
    subject_data = {}
    for sid in range(1, NUM_SUBJECTS + 1):
        ts = time.time()
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
        print(f"  Subject {sid:02d}: {X.shape[0]} trials, C={X.shape[1]}, T={X.shape[2]} ({time.time()-ts:.1f}s)")
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    fold_accs = []
    for test_sid in range(1, NUM_SUBJECTS + 1):
        print(f"\n{'─'*50}")
        print(f"Fold {test_sid}/9: Test=Subject {test_sid:02d}")

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

        # Euclidean Alignment (per-subject)
        print("  Applying Euclidean Alignment...")
        X_train_all, X_test = euclidean_alignment_per_subject(
            subject_data, test_sid, X_train_all, X_test)

        # Train/val split (90/10)
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

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d} | Loss:{train_loss:.4f} TrAcc:{train_acc:.3f} ValAcc:{val_acc:.3f}")

        model.load_state_dict(best_state)
        model.to(DEVICE)
        test_acc = evaluate(model, test_loader)
        fold_accs.append(test_acc)
        print(f"  >> Test Acc (S{test_sid:02d}): {test_acc:.4f} ({test_acc*100:.2f}%) | best val: {best_val_acc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("LOSO Results Summary")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1} (Subj {i+1:02d}): {acc*100:.2f}%")
    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    print(f"\n  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"\n  Reference baselines (BCI IV 2a LOSO):")
    print(f"    GAH-TNet: 69.39% | EEG-TransNet: 64.53% | EEGNet: 56.80%")
    print(f"{'='*60}")
    return fold_accs


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    t_start = time.time()
    run_loso(num_epochs=80, batch_size=64, lr=1e-3, lam=0.3, d_model=32,
             n_heads=2, label_smoothing=0.1, mixup_alpha=0.2)
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
