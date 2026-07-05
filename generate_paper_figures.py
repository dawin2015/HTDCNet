"""
HTDCNet Paper Figures Generator
1. Per-subject accuracy bar charts (BNCI2014_004 + OpenBMI)
2. Training curve plots (loss + accuracy over epochs)
3. Statistical significance tests (Wilcoxon signed-rank)
"""
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJDIR = '/home/ubuntu/user_liu_20260403/python_projects/bibm2026'
RESULTS_DIR = os.path.join(PROJDIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_loso_log(log_path):
    """Parse LOSO log file to extract per-subject accuracies and training curves."""
    with open(log_path) as f:
        lines = f.readlines()

    # Per-subject test accuracies
    test_accs = {}
    test_acc_pattern = re.compile(r'Test Acc \(S(\d+)\):\s+([\d.]+)')
    for line in lines:
        m = test_acc_pattern.search(line)
        if m:
            sid = int(m.group(1))
            acc = float(m.group(2))
            test_accs[sid] = acc

    # Training curves: extract epoch, train_acc, val_acc per fold
    fold_curves = {}
    current_fold = None
    fold_pattern = re.compile(r'Fold\s+(\d+)/\d+:')
    epoch_pattern = re.compile(r'Ep\s+(\d+)\s*\|\s*Loss:([\d.]+)\s*TrAcc:([\d.]+)\s*ValAcc:([\d.]+)')
    test_pattern = re.compile(r'Test Acc \(S\d+\)')

    for line in lines:
        fm = fold_pattern.search(line)
        if fm:
            current_fold = int(fm.group(1))
            fold_curves[current_fold] = {'epochs': [], 'loss': [], 'train_acc': [], 'val_acc': []}

        em = epoch_pattern.search(line)
        if em and current_fold is not None and current_fold in fold_curves:
            fold_curves[current_fold]['epochs'].append(int(em.group(1)))
            fold_curves[current_fold]['loss'].append(float(em.group(2)))
            fold_curves[current_fold]['train_acc'].append(float(em.group(3)))
            fold_curves[current_fold]['val_acc'].append(float(em.group(4)))

        if test_pattern.search(line):
            current_fold = None  # next fold

    return test_accs, fold_curves


# ============================================================
# 1. Per-subject bar charts
# ============================================================
def plot_per_subject_bars():
    print("=" * 60)
    print("1. Per-subject accuracy bar charts")
    print("=" * 60)

    # --- BNCI2014_004 ---
    _, bnci_curves = parse_loso_log(os.path.join(RESULTS_DIR, 'logs', 'bci2a_v2_output.log'))
    # For BNCI2014_004, use the ablation results (full model)
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        ablation = json.load(f)
    bnci_accs = np.array(ablation['Full Model (HTDCNet)']['folds']) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # BNCI2014_004 bar chart
    ax = axes[0]
    sids = list(range(1, len(bnci_accs) + 1))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sids)))
    bars = ax.bar(sids, bnci_accs, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(y=np.mean(bnci_accs), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(bnci_accs):.1f}%')
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('BNCI2014_004 LOSO (9 subjects, 2-class MI, 6 channels)', fontsize=11)
    ax.set_xticks(sids)
    ax.set_xticklabels([f'S{i:02d}' for i in sids], rotation=45, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    # Add value labels
    for bar, acc in zip(bars, bnci_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=7)

    # OpenBMI bar chart
    openbmi_accs, _ = parse_loso_log(os.path.join(RESULTS_DIR, 'logs', 'openbmi_loso_output.log'))
    ax = axes[1]
    openbmi_sids = sorted(openbmi_accs.keys())
    openbmi_vals = [openbmi_accs[s] * 100 for s in openbmi_sids]
    mean_val = np.mean(openbmi_vals)

    # Color bars: green if >= mean, orange if below
    bar_colors = ['#2ecc71' if v >= mean_val else '#e67e22' for v in openbmi_vals]
    bars = ax.bar(openbmi_sids, openbmi_vals, color=bar_colors, edgecolor='gray', linewidth=0.3)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.1f}%')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Chance (50%)')
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('OpenBMI Session 1 LOSO (54 subjects, 2-class MI, 62 channels)', fontsize=11)
    ax.set_xticks(openbmi_sids[::3])  # show every 3rd tick
    ax.set_xticklabels([f'S{i:02d}' for i in openbmi_sids[::3]], rotation=45, fontsize=7)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'per_subject_accuracy.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return bnci_accs, np.array(openbmi_vals)


# ============================================================
# 2. Training curve plots
# ============================================================
def plot_training_curves():
    print("\n" + "=" * 60)
    print("2. Training curve plots")
    print("=" * 60)

    _, openbmi_curves = parse_loso_log(os.path.join(RESULTS_DIR, 'logs', 'openbmi_loso_output.log'))

    # Select representative folds: best, worst, median
    fold_ids = sorted(openbmi_curves.keys())
    final_val_accs = {}
    for fid in fold_ids:
        curves = openbmi_curves[fid]
        if curves['val_acc']:
            final_val_accs[fid] = max(curves['val_acc'])

    sorted_folds = sorted(final_val_accs, key=final_val_accs.get)
    n = len(sorted_folds)
    best_fold = sorted_folds[-1]
    worst_fold = sorted_folds[0]
    median_fold = sorted_folds[n // 2]

    representative = {
        'Best': best_fold,
        'Median': median_fold,
        'Worst': worst_fold,
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col, (label, fid) in enumerate(representative.items()):
        curves = openbmi_curves[fid]
        epochs = curves['epochs']
        loss = curves['loss']
        train_acc = [a * 100 for a in curves['train_acc']]
        val_acc = [a * 100 for a in curves['val_acc']]

        # Loss plot
        ax = axes[0, col]
        ax.plot(epochs, loss, 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'{label} (Fold {fid})', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Accuracy plot
        ax = axes[1, col]
        ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc')
        ax.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Acc')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{label} (Fold {fid})', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    plt.suptitle('HTDCNet Training Curves on OpenBMI (representative folds)', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'training_curves.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Aggregate training curve (average over all folds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    all_train = []
    all_val = []
    all_loss = []
    common_epochs = [1, 20, 40, 60, 80]
    for fid in fold_ids:
        curves = openbmi_curves[fid]
        if curves['epochs'] == common_epochs:
            all_train.append([a * 100 for a in curves['train_acc']])
            all_val.append([a * 100 for a in curves['val_acc']])
            all_loss.append(curves['loss'])

    if all_train:
        mean_train = np.mean(all_train, axis=0)
        std_train = np.std(all_train, axis=0)
        mean_val = np.mean(all_val, axis=0)
        std_val = np.std(all_val, axis=0)
        mean_loss = np.mean(all_loss, axis=0)
        std_loss = np.std(all_loss, axis=0)

        ax = axes[0]
        ax.plot(common_epochs, mean_loss, 'b-', linewidth=2)
        ax.fill_between(common_epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Average Loss (N={len(all_train)} folds)')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(common_epochs, mean_train, 'b-', linewidth=2, label='Train Acc')
        ax.fill_between(common_epochs, mean_train - std_train, mean_train + std_train, alpha=0.2, color='blue')
        ax.plot(common_epochs, mean_val, 'r-', linewidth=2, label='Val Acc')
        ax.fill_between(common_epochs, mean_val - std_val, mean_val + std_val, alpha=0.2, color='red')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Average Accuracy (N={len(all_train)} folds)')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'training_curves_aggregate.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# 3. Statistical tests (Wilcoxon signed-rank)
# ============================================================
def run_statistical_tests():
    print("\n" + "=" * 60)
    print("3. Statistical significance tests")
    print("=" * 60)

    # Load ablation data
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        ablation = json.load(f)

    full_accs = np.array(ablation['Full Model (HTDCNet)']['folds'])
    n_folds = len(full_accs)

    print(f"\n--- BNCI2014_004 Ablation Wilcoxon Tests (N={n_folds} folds) ---")
    print(f"{'Variant':<25} {'Mean Acc':>10} {'W-stat':>10} {'p-value':>12} {'Significant':>12}")
    print("-" * 70)

    results = {}
    for name, data in ablation.items():
        if name == 'Full Model (HTDCNet)':
            continue
        variant_accs = np.array(data['folds'])
        # Wilcoxon signed-rank test: paired test per fold
        stat, pval = wilcoxon(full_accs, variant_accs)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        results[name] = (stat, pval, sig)
        print(f"{name:<25} {np.mean(variant_accs)*100:>9.2f}% {stat:>10.1f} {pval:>12.4f} {sig:>12}")

    # Also test: Full model vs chance (50% for binary)
    chance = np.full(n_folds, 0.5)
    stat, pval = wilcoxon(full_accs, chance)
    print(f"\n{'Full vs Chance (50%)':<25} {'':>10} {stat:>10.1f} {pval:>12.4f} {'***' if pval < 0.001 else 'n.s.':>12}")

    # OpenBMI vs chance
    openbmi_accs, _ = parse_loso_log(os.path.join(RESULTS_DIR, 'logs', 'openbmi_loso_output.log'))
    openbmi_vals = np.array([openbmi_accs[s] for s in sorted(openbmi_accs.keys())])
    chance_obmi = np.full(len(openbmi_vals), 0.5)
    stat_obmi, pval_obmi = wilcoxon(openbmi_vals, chance_obmi)
    print(f"\n--- OpenBMI Session 1 (N={len(openbmi_vals)} subjects) ---")
    print(f"{'Full vs Chance (50%)':<25} {np.mean(openbmi_vals)*100:>9.2f}% {stat_obmi:>10.1f} {pval_obmi:>12.4f} {'***' if pval_obmi < 0.001 else 'n.s.':>12}")

    # Create statistical test visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ablation paired comparison
    ax = axes[0]
    variant_names = list(ablation.keys())
    variant_means = [np.array(ablation[n]['folds']).mean() * 100 for n in variant_names]
    variant_stds = [np.array(ablation[n]['folds']).std() * 100 for n in variant_names]

    x_pos = np.arange(len(variant_names))
    bars = ax.bar(x_pos, variant_means, yerr=variant_stds,
                  color=['#3498db'] + ['#e74c3c'] * (len(variant_names) - 1),
                  capsize=5, edgecolor='gray', linewidth=0.5)

    # Add significance stars
    for i, name in enumerate(variant_names):
        if name in results:
            sig = results[name][2]
            ax.text(x_pos[i], variant_means[i] + variant_stds[i] + 1, sig,
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('BNCI2014_004 Ablation (Wilcoxon signed-rank)', fontsize=11)
    ax.set_xticks(x_pos)
    short_names = ['Full', 'w/o GATE', 'w/o Local', 'w/o Global', 'w/o SupCon']
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(50, 75)
    ax.grid(True, alpha=0.3, axis='y')

    # OpenBMI: HTDCNet vs chance
    ax = axes[1]
    openbmi_sorted = sorted(openbmi_accs.items())
    sids = [s for s, _ in openbmi_sorted]
    vals = [v * 100 for _, v in openbmi_sorted]
    mean_val = np.mean(vals)

    colors = ['#2ecc71' if v >= mean_val else '#e67e22' for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor='gray', linewidth=0.3)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.1f}%')
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Chance')

    # Add significance annotation
    ax.text(0.95, 0.95, f'Wilcoxon p = {pval_obmi:.2e}\n(*** p < 0.001)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Subject (sorted)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('OpenBMI Session 1 vs Chance Level', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'statistical_tests.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("HTDCNet Paper Figures Generator")
    print("=" * 60)

    bnci_accs, openbmi_accs = plot_per_subject_bars()
    plot_training_curves()
    stat_results = run_statistical_tests()

    print("\n" + "=" * 60)
    print("All paper figures generated:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f"  {os.path.join(FIGURES_DIR, f)}")
    print("=" * 60)
