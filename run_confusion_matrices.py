"""
Confusion Matrix Visualization for OpenBMI LOSO
Per-subject confusion matrices + aggregate confusion matrix
"""
import os, sys, time, re
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from htdcnet_openbmi import (
    load_subject, EEGDataset, HTDCNet, euclidean_alignment_per_subject,
    DATA_ROOT, SESSION, NUM_CHANNELS, NUM_CLASSES, DEVICE, TARGET_LEN
)

OUTPUT_DIR = '/home/ubuntu/user_liu_20260403/python_projects/bibm2026/results/confusion_matrices'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Right Hand', 'Left Hand']


def train_model_for_fold(X_train, y_train, X_val, y_val, num_epochs=80, d_model=32, n_heads=2):
    """Train a model and return best state dict."""
    train_loader = DataLoader(EEGDataset(X_train, y_train, augment=True), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=64, num_workers=0)
    
    C_in = X_train.shape[1]
    model = HTDCNet(C_in=C_in, d_model=d_model, n_heads=n_heads, window=16, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc, best_state = 0, None
    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yb_orig = yb
            optimizer.zero_grad()
            logits, zl, zg = model(xb)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_cl = model.infonce(zl, zg, yb_orig)
            loss = loss_ce + 0.5 * loss_cl
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _, _ = model(xb)
                correct += (logits.argmax(-1) == yb).sum().item()
                total += len(yb)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model


def get_predictions(model, loader):
    """Get predictions and labels from a model."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits, _, _ = model(xb)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(y_true, y_pred, title, filename, class_names=None):
    """Plot and save a confusion matrix."""
    if class_names is None:
        class_names = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def run_confusion_matrices(num_subjects=54, num_epochs=80, d_model=32, n_heads=2):
    print("=" * 60)
    print("OpenBMI Confusion Matrix Analysis")
    print(f"Subjects: {num_subjects} | Epochs: {num_epochs}")
    print("=" * 60)
    
    # Load all subject data
    print("\nLoading data...")
    t0 = time.time()
    subject_data = {}
    for sid in range(1, num_subjects + 1):
        ts = time.time()
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
        if sid % 10 == 0:
            print(f"  Loaded {sid}/{num_subjects} subjects ({time.time()-ts:.1f}s each)")
    print(f"All data loaded in {time.time()-t0:.1f}s\n")
    
    # Collect per-subject predictions
    all_y_true, all_y_pred = [], []
    per_subject_accs = []
    
    for test_sid in range(1, num_subjects + 1):
        print(f"Fold {test_sid}/{num_subjects}: Subject {test_sid:02d}", end=" ", flush=True)
        
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
        
        # EA
        X_train_all, X_test = euclidean_alignment_per_subject(
            subject_data, test_sid, X_train_all, X_test)
        
        # Split
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(y_train_all))
        n_val = int(0.1 * len(idx))
        X_train, y_train = X_train_all[idx[n_val:]], y_train_all[idx[n_val:]]
        X_val, y_val = X_train_all[idx[:n_val]], y_train_all[idx[:n_val]]
        
        # Train
        model = train_model_for_fold(X_train, y_train, X_val, y_val, 
                                      num_epochs=num_epochs, d_model=d_model, n_heads=n_heads)
        
        # Predict
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=64, num_workers=0)
        preds, labels = get_predictions(model, test_loader)
        acc = (preds == labels).mean() * 100
        per_subject_accs.append(acc)
        
        all_y_true.append(labels)
        all_y_pred.append(preds)
        
        # Per-subject confusion matrix
        plot_confusion_matrix(labels, preds, 
                             f'S{test_sid:02d} (Acc: {acc:.1f}%)', 
                             f'cm_s{test_sid:02d}.png')
        print(f"Acc={acc:.1f}%")
    
    # ============================================================
    # Aggregate confusion matrix
    # ============================================================
    all_true = np.concatenate(all_y_true)
    all_pred = np.concatenate(all_y_pred)
    
    # Overall aggregate
    plot_confusion_matrix(all_true, all_pred, 
                         f'Aggregate (N={len(all_true)}, Acc={np.mean(per_subject_accs):.1f}%)',
                         'cm_aggregate.png')
    
    # Per-class recall
    cm = confusion_matrix(all_true, all_pred)
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # ============================================================
    # Grid of all subjects
    # ============================================================
    print("\n  Creating grid visualization...")
    n_cols = 9
    n_rows = (num_subjects + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3*n_rows))
    axes = axes.flatten()
    
    for i in range(num_subjects):
        ax = axes[i]
        cm_i = confusion_matrix(all_y_true[i], all_y_pred[i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_i, display_labels=['R', 'L'])
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False,
                 xticks_rotation=0)
        ax.set_title(f'S{i+1:02d}\n{per_subject_accs[i]:.0f}%', fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    for i in range(num_subjects, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'OpenBMI LOSO Per-Subject Confusion Matrices\nMean: {np.mean(per_subject_accs):.1f}% ± {np.std(per_subject_accs):.1f}%',
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cm_grid_all_subjects.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'cm_grid_all_subjects.png')}")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("Confusion Matrix Analysis Complete")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {np.mean(per_subject_accs):.2f}% ± {np.std(per_subject_accs):.2f}%")
    print(f"Per-class Recall: Right={recall_per_class[0]*100:.1f}%, Left={recall_per_class[1]*100:.1f}%")
    print(f"Aggregate Confusion Matrix:\n{cm}")
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    run_confusion_matrices(num_subjects=54, num_epochs=80, d_model=32, n_heads=2)
