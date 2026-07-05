# HTDCNet: Hierarchical Temporal Disentangled Contrastive Network for Cross-Subject Motor Imagery Decoding

Official implementation for IEEE BIBM 2026 submission.

## Overview

HTDCNet is a novel deep learning architecture for cross-subject motor imagery (MI) EEG decoding. The key innovations are:

1. **Parallel dual-branch architecture**: Decouples local and global temporal processing into independent encoder pathways
2. **Class-aware supervised contrastive loss (SupCon)**: Aligns representations across branches according to task semantics
3. **Euclidean Alignment (EA)**: Harmonizes cross-subject covariance distributions as a label-free preprocessing step

## Project Structure

```
HTDCNet/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── htdcnet_loso.py               # Core model + BNCI2014_004 LOSO training
├── htdcnet_openbmi.py            # OpenBMI Session 1 LOSO training
├── htdcnet_bci2a.py              # BCI Competition IV-2a LOSO training
├── run_ablation.py               # Ablation study experiments
├── run_tsne.py                   # t-SNE visualization experiments
├── generate_paper_figures.py     # Paper figure generation (bar charts, curves, stats)
├── run_confusion_matrices.py     # Confusion matrix generation
├── run_eegnet_baseline.py        # EEGNet baseline validation
├── htdcnet_abstract.tex          # LaTeX manuscript source
├── results/
│   ├── ablation/                 # Ablation study results (JSON)
│   ├── confusion_matrices/       # Per-subject confusion matrices (PNG)
│   ├── paper_figures/            # Publication-ready figures (PNG)
│   ├── tsne/                     # t-SNE visualization plots (PNG)
│   └── logs/                     # Experiment training logs
└── datasets/                     # (Not included) See Dataset section below
```

## Datasets

The experiments use two public MI datasets:

| Dataset | Subjects | Channels | Classes | Sampling Rate | Source |
|---------|----------|----------|---------|---------------|--------|
| **BNCI2014_004** (BCI Competition IV-2b) | 9 | 6 | 2 (L/R hand) | 250 Hz | [MOABB](https://moabb.neurotechx.com) |
| **OpenBMI Session 1** | 54 | 62 | 2 (L/R hand) | 1000 Hz | [GigaScience](https://doi.org/10.5524/100542) |

### Data Download

Datasets are NOT included in this repository. To reproduce results:

1. **BNCI2014_004** (BCI Competition IV-2b):
   ```python
   from moabb.datasets import BNCI2014_004
   dataset = BNCI2014_004()
   ```

2. **OpenBMI** (Lee2019-MI):
   Download from: https://doi.org/10.5524/100542
   Or use MOABB:
   ```python
   from moabb.datasets import Lee2019_MI
   dataset = Lee2019_MI()
   ```

3. Update the `DATA_DIR` paths in each training script to point to your local data directory.

## Environment Setup

```bash
# Create conda environment
conda create -n HTDCNet python=3.10 -y
conda activate HTDCNet

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### Hardware Requirements

- GPU: NVIDIA RTX 3090 (or equivalent with ≥24GB VRAM) recommended
- RAM: ≥32GB (for OpenBMI 54-subject data loading)
- Disk: ~5GB for datasets, ~2GB for results

## Reproduction Instructions

### 1. BNCI2014_004 LOSO (Main Experiment)

```bash
# Update DATA_DIR in htdcnet_loso.py first
python htdcnet_loso.py
```

Expected output: Mean LOSO accuracy of ~64.88% ± 9.02% across 9 subjects.

### 2. OpenBMI Session 1 LOSO

```bash
# Update DATA_DIR in htdcnet_openbmi.py first
python htdcnet_openbmi.py
```

Expected output: Mean LOSO accuracy of ~75.58% ± 11.56% across 54 subjects.

### 3. Ablation Study

```bash
python run_ablation.py
```

Generates per-variant LOSO results and saves to `results/ablation/ablation_results.json`.

### 4. t-SNE Visualization

```bash
python run_tsne.py
```

Generates t-SNE plots comparing EA/no-EA and SupCon/InfoNCE configurations in `results/tsne/`.

### 5. Paper Figures

```bash
python generate_paper_figures.py
```

Generates per-subject bar charts, training curves, and statistical test figures in `results/paper_figures/`.

### 6. Confusion Matrices

```bash
python run_confusion_matrices.py
```

Generates per-subject and aggregate confusion matrices in `results/confusion_matrices/`.

### 7. Run All Experiments Sequentially

```bash
bash run_all_experiments.sh
```

This script runs the full experiment pipeline (BCI IV-2a, t-SNE, ablation) and saves logs to `results/logs/`.

## Model Architecture

```
Raw EEG (B × C × T)
  │
  ├── [Shared] GATE Module
  │     ├── Chebyshev-style spatial conv (1×1 + depthwise 3×3)
  │     ├── SE channel attention
  │     └── Temporal-spatial conv (k=25 + 1×1)
  │
  ├── Branch A: Local Encoder
  │     ├── Residual-ECA
  │     ├── Local Masked MHA (w=16, h=2)
  │     └── Shallow TCN (2 layers, dilation [1,2])
  │
  ├── Branch B: Global Encoder
  │     ├── Residual-ECA
  │     ├── Global MHA (h=2, full sequence)
  │     └── Deep TCN (4 layers, dilation [1,2,4,8])
  │
  └── Output
        ├── Pooling: z_local = AvgPool(H_local)  → (B × 32)
        ├── Pooling: z_global = AvgPool(H_global) → (B × 32)
        ├── Classifier: z_global → Linear(32→16) → ReLU → Dropout → Linear(16→K)
        └── Loss: L = L_CE(label_smooth=0.1) + λ·L_SupCon(τ=0.5)
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model dimension (d_model) | 32 |
| Attention heads | 2 |
| Local window size | 16 |
| Contrastive loss weight (λ) | 0.5 |
| SupCon temperature (τ) | 0.5 |
| Label smoothing (ε) | 0.1 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 80 |
| Optimizer | Adam (weight_decay=5e-4) |
| Scheduler | CosineAnnealingLR |

## Experimental Results

### BNCI2014_004 (BCI Competition IV-2b) — 9 Subjects

| Method | Accuracy (%) | Std |
|--------|-------------|-----|
| Shallow ConvNet | 74.96 | 7.21 |
| EEGNet | 76.26 | 5.74 |
| EEG-TCNet | 77.08 | — |
| TCNet-Fusion | 77.53 | — |
| ATCNet | 75.95 | 5.65 |
| EEG-TransNet | 79.68 | — |
| GAH-TNet | **81.00** | — |
| **HTDCNet (ours)** | 64.88 | 9.02 |

### OpenBMI Session 1 — 54 Subjects

| Method | Accuracy (%) |
|--------|-------------|
| EEGNet | ~55 |
| ATCNet | ~63 |
| SFE-GAT | 65–67 |
| **HTDCNet (ours)** | **75.58 ± 11.56** |

### Ablation Study (BNCI2014_004)

| Variant | Accuracy (%) | Δ | p-value |
|---------|-------------|---|---------|
| Full HTDCNet | 64.88 ± 9.02 | — | — |
| w/o GATE | 62.83 ± 10.03 | -2.05 | 0.098 |
| w/o Local Encoder | 62.86 ± 8.67 | -2.02 | **0.020** |
| w/o Global Encoder | 62.51 ± 8.18 | -2.37 | 0.129 |
| w/o SupCon Loss | 63.88 ± 7.22 | -1.00 | 0.734 |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{htdcnet2026,
  title={HTDCNet: Parallel Hierarchical Temporal Decoupling with Class-Aware Contrastive Learning for Cross-Subject Motor Imagery Decoding},
  author={Author Name(s)},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration, please contact: `{author}@institution.edu`
