"""
t-SNE Visualization for HTDCNet
Compares: EA vs no-EA, SupCon vs InfoNCE, z_local vs z_global
Run on BNCI2014_004 (already trained model available)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from htdcnet_loso import (
    load_subject, EEGDataset, HTDCNet, SupervisedContrastiveLoss, InfoNCELoss,
    euclidean_alignment_per_subject, DATA_DIR, NUM_SUBJECTS, NUM_CLASSES, DEVICE
)

OUTPUT_DIR = '/home/ubuntu/user_liu_20260403/python_projects/bibm2026/results/tsne'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_features(model, loader, device='cuda'):
    """Extract z_local and z_global features from all samples."""
    model.eval()
    z_locals, z_globals, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, zl, zg = model(xb)
            z_locals.append(zl.cpu().numpy())
            z_globals.append(zg.cpu().numpy())
            labels.append(yb.numpy())
    return (np.concatenate(z_locals), np.concatenate(z_globals), np.concatenate(labels))


def plot_tsne(features, labels, title, filename, perplexity=30):
    """Create t-SNE scatter plot colored by class."""
    n_samples = min(len(features), 5000)
    idx = np.random.choice(len(features), n_samples, replace=False)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(features[idx])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    unique_labels = np.unique(labels[idx])
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    class_names = ['Class 0', 'Class 1']
    
    for i, label in enumerate(unique_labels):
        mask = labels[idx] == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[colors[i]], label=class_names[i], alpha=0.6, s=10)
    
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_tsne_by_subject(features, subject_ids, title, filename, perplexity=30):
    """Create t-SNE colored by subject to show EA alignment effect."""
    n_samples = min(len(features), 5000)
    idx = np.random.choice(len(features), n_samples, replace=False)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(features[idx])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    unique_sids = np.unique(subject_ids[idx])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sids)))
    
    for i, sid in enumerate(unique_sids):
        mask = subject_ids[idx] == sid
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[colors[i]], label=f'S{sid:02d}', alpha=0.6, s=10)
    
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def run_tsne_experiments():
    print("=" * 60)
    print("HTDCNet t-SNE Visualization Experiments")
    print("=" * 60)
    
    # Load all subject data
    print("\nLoading data...")
    subject_data = {}
    for sid in range(1, NUM_SUBJECTS + 1):
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
        print(f"  Subject {sid:02d}: {X.shape[0]} trials")

    # Use fold 1 (test=Subject 01) for visualization
    test_sid = 1
    X_test, y_test = subject_data[test_sid]
    X_train_list, y_train_list = [], []
    subject_id_list = []
    for sid in range(1, NUM_SUBJECTS + 1):
        if sid == test_sid:
            continue
        Xs, ys = subject_data[sid]
        X_train_list.append(Xs)
        y_train_list.append(ys)
        subject_id_list.extend([sid] * Xs.shape[0])
    
    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    subject_ids = np.array(subject_id_list)
    
    # ============================================================
    # Experiment 1: EA vs no-EA (both with SupCon)
    # ============================================================
    print("\n--- Experiment 1: EA vs no-EA (SupCon) ---")
    
    # Train model WITHOUT EA
    print("  Training without EA...")
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_train_all))
    n_val = int(0.1 * len(idx))
    X_train_noEA, y_train_noEA = X_train_all[idx[n_val:]], y_train_all[idx[n_val:]]
    
    train_loader = DataLoader(EEGDataset(X_train_noEA, y_train_noEA, augment=True), batch_size=64, shuffle=True)
    model_noEA = HTDCNet(C_in=6, d_model=32, n_heads=2, window=16, num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model_noEA.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    for epoch in range(1, 81):
        model_noEA.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yb_orig = yb
            optimizer.zero_grad()
            logits, zl, zg = model_noEA(xb)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_cl = model_noEA.infonce(zl, zg, yb_orig)
            loss = loss_ce + 0.3 * loss_cl
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    z_l_noEA, z_g_noEA, _ = get_features(model_noEA, train_loader)
    
    # Train model WITH EA
    print("  Training with EA...")
    X_train_ea, X_test_ea = euclidean_alignment_per_subject(
        subject_data, test_sid, X_train_all, X_test)
    X_train_ea_split, y_train_ea_split = X_train_ea[idx[n_val:]], y_train_all[idx[n_val:]]
    
    train_loader_ea = DataLoader(EEGDataset(X_train_ea_split, y_train_ea_split, augment=True), batch_size=64, shuffle=True)
    model_ea = HTDCNet(C_in=6, d_model=32, n_heads=2, window=16, num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model_ea.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    for epoch in range(1, 81):
        model_ea.train()
        for xb, yb in train_loader_ea:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yb_orig = yb
            optimizer.zero_grad()
            logits, zl, zg = model_ea(xb)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_cl = model_ea.infonce(zl, zg, yb_orig)
            loss = loss_ce + 0.3 * loss_cl
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    z_l_ea, z_g_ea, _ = get_features(model_ea, train_loader_ea)
    
    # Plot t-SNE: EA vs no-EA (class-colored)
    plot_tsne(z_g_noEA, y_train_all[idx[n_val:]], 'z_global WITHOUT EA (SupCon)', 'tsne_noEA_supcon.png')
    plot_tsne(z_g_ea, y_train_all[idx[n_val:]], 'z_global WITH EA (SupCon)', 'tsne_withEA_supcon.png')
    
    # Plot t-SNE by subject: EA vs no-EA
    plot_tsne_by_subject(z_g_noEA, subject_ids[idx[n_val:]], 'z_global WITHOUT EA (by Subject)', 'tsne_noEA_by_subject.png')
    plot_tsne_by_subject(z_g_ea, subject_ids[idx[n_val:]], 'z_global WITH EA (by Subject)', 'tsne_withEA_by_subject.png')
    
    # ============================================================
    # Experiment 2: SupCon vs InfoNCE (both with EA)
    # ============================================================
    print("\n--- Experiment 2: SupCon vs InfoNCE (both with EA) ---")
    
    # Already have SupCon+EA model (model_ea)
    z_l_supcon, z_g_supcon, _ = get_features(model_ea, train_loader_ea)
    
    # Train model with InfoNCE + EA
    print("  Training with InfoNCE + EA...")
    model_infonce = HTDCNet(C_in=6, d_model=32, n_heads=2, window=16, num_classes=2).to(DEVICE)
    model_infonce.infonce = InfoNCELoss(0.5)  # replace with InfoNCE
    optimizer = torch.optim.Adam(model_infonce.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    for epoch in range(1, 81):
        model_infonce.train()
        for xb, yb in train_loader_ea:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, zl, zg = model_infonce(xb)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_cl = model_infonce.infonce(zl, zg)  # InfoNCE doesn't use labels
            loss = loss_ce + 0.3 * loss_cl
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    z_l_infonce, z_g_infonce, _ = get_features(model_infonce, train_loader_ea)
    
    # Plot t-SNE: SupCon vs InfoNCE
    plot_tsne(z_g_supcon, y_train_all[idx[n_val:]], 'z_global with SupCon (EA)', 'tsne_supcon_ea.png')
    plot_tsne(z_g_infonce, y_train_all[idx[n_val:]], 'z_global with InfoNCE (EA)', 'tsne_infonce_ea.png')
    
    # ============================================================
    # Experiment 3: z_local vs z_global (same model)
    # ============================================================
    print("\n--- Experiment 3: z_local vs z_global ---")
    plot_tsne(z_l_supcon, y_train_all[idx[n_val:]], 'z_local (Local Encoder)', 'tsne_zlocal.png')
    plot_tsne(z_g_supcon, y_train_all[idx[n_val:]], 'z_global (Global Encoder)', 'tsne_zglobal.png')
    
    # Combined 2x2 figure
    print("\n  Creating combined figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, features, labels_arr, title in [
        (axes[0,0], z_g_noEA, y_train_all[idx[n_val:]], 'No EA + SupCon'),
        (axes[0,1], z_g_ea, y_train_all[idx[n_val:]], 'EA + SupCon'),
        (axes[1,0], z_g_infonce, y_train_all[idx[n_val:]], 'EA + InfoNCE'),
        (axes[1,1], z_g_supcon, y_train_all[idx[n_val:]], 'EA + SupCon (best)')
    ]:
        n = min(len(features), 3000)
        sidx = np.random.choice(len(features), n, replace=False)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
        emb = tsne.fit_transform(features[sidx])
        colors = plt.cm.Set2(np.linspace(0, 1, 2))
        for c in [0, 1]:
            mask = labels_arr[sidx] == c
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]], label=f'Class {c}', alpha=0.5, s=8)
        ax.set_title(title)
        ax.legend(fontsize=9)
    plt.suptitle('HTDCNet t-SNE Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_combined.png'), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'tsne_combined.png')}")
    
    print("\n" + "=" * 60)
    print("All t-SNE visualizations saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    t0 = time.time()
    run_tsne_experiments()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
