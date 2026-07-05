"""
EEGNet baseline on BNCI2014_004 (BCI Competition IV-2b)
Implements Lawhern et al. (2018) EEGNet architecture
Uses same data loading and LOSO protocol as HTDCNet
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
import scipy.io as sio

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
sys.path.insert(0, '/home/ubuntu/user_liu_20260403/python_projects/bibm2026')

# Import data loading from HTDCNet
from htdcnet_loso import load_subject, DATA_DIR, NUM_SUBJECTS

# EEGNet model (Lawhern et al., 2018)
class EEGNet(nn.Module):
    def __init__(self, num_channels=3, num_samples=1000, num_classes=2,
                 F1=8, F2=16, D=2, kernel_length=64, dropout=0.5):
        super(EEGNet, self).__init__()
        
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_length = kernel_length
        self.num_channels = num_channels
        self.num_samples = num_samples
        
        # Block 1: Temporal Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise Conv
        self.depthwise = nn.Conv2d(F1, F1*D, (num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 2: Separable Conv
        self.separable = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Classification
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (num_samples // 32), num_classes, bias=False)
        
    def forward(self, x):
        # x shape: (batch, channels, samples) -> (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


def train_eegnet(model, train_loader, val_loader, num_epochs=200, lr=0.001, device='cuda'):
    """Train EEGNet model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            sys.stdout.flush()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


def evaluate_eegnet(model, test_loader, device='cuda'):
    """Evaluate EEGNet model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels


class EEGDataset(Dataset):
    """Simple dataset wrapper"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_eegnet_loso():
    """Run EEGNet on BNCI2014_004 with LOSO protocol"""
    print("=" * 60)
    print("EEGNet Baseline on BNCI2014_004 (BCI Competition IV-2b)")
    print("=" * 60)
    
    # Load data for all subjects
    print("\nLoading BNCI2014_004 data...")
    subject_data = {}
    for sid in range(1, NUM_SUBJECTS + 1):
        X, y = load_subject(sid)
        subject_data[sid] = (X, y)
        print(f"  Subject {sid:2d}: {X.shape[0]} trials, shape {X.shape[1:]}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # LOSO cross-validation
    subject_results = []
    
    for test_sid in range(1, NUM_SUBJECTS + 1):
        print(f"\n{'='*60}")
        print(f"Fold {test_sid}/9: Test Subject {test_sid}")
        print(f"{'='*60}")
        
        # Get test data
        X_test, y_test = subject_data[test_sid]
        
        # Get train data (all other subjects)
        X_train_list = []
        y_train_list = []
        for sid in range(1, NUM_SUBJECTS + 1):
            if sid != test_sid:
                X_s, y_s = subject_data[sid]
                X_train_list.append(X_s)
                y_train_list.append(y_s)
        
        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)
        
        # Split train/val (90/10)
        n_train = int(0.9 * len(X_train_all))
        indices = np.random.permutation(len(X_train_all))
        X_train = X_train_all[indices[:n_train]]
        y_train = y_train_all[indices[:n_train]]
        X_val = X_train_all[indices[n_train:]]
        y_val = y_train_all[indices[n_train:]]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create data loaders
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize EEGNet
        # For 125Hz downsampled data, use kernel_length=32 (original was 64 for 250Hz)
        model = EEGNet(
            num_channels=X_train.shape[1],
            num_samples=X_train.shape[2],
            num_classes=2,
            F1=8,
            F2=16,
            D=2,
            kernel_length=32,
            dropout=0.5
        )
        
        # Train
        model, best_val_acc = train_eegnet(
            model, train_loader, val_loader,
            num_epochs=200, lr=0.001, device=device
        )
        
        # Evaluate
        test_acc, test_preds, test_labels = evaluate_eegnet(model, test_loader, device)
        
        print(f"\nResults for Subject {test_sid}:")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        subject_results.append({
            'subject': int(test_sid),
            'val_accuracy': float(best_val_acc),
            'test_accuracy': float(test_acc)
        })
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("EEGNet Baseline Results Summary")
    print("=" * 60)
    
    test_accuracies = [r['test_accuracy'] for r in subject_results]
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    
    print(f"\nPer-subject test accuracies:")
    for r in subject_results:
        print(f"  Subject {r['subject']:2d}: {r['test_accuracy']:.4f}")
    
    print(f"\nMean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
    
    # Save results
    output_file = '/home/ubuntu/user_liu_20260403/python_projects/bibm2026/results/eegnet_bnci2014_004_results.json'
    results = {
        'model': 'EEGNet',
        'dataset': 'BNCI2014_004',
        'evaluation': 'LOSO',
        'num_subjects': 9,
        'subject_results': subject_results,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'comparison_with_gah_tnet': {
            'GAH-TNet reported': '76.33%',
            'Our EEGNet': f'{mean_acc*100:.2f}% ± {std_acc*100:.2f}%'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Comparison with GAH-TNet Table 3
    print("\n" + "=" * 60)
    print("Comparison with GAH-TNet Table 3 (BCI IV-2b LOSO)")
    print("=" * 60)
    print(f"GAH-TNet reported EEGNet: 76.33%")
    print(f"Our EEGNet implementation:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Difference: {(mean_acc - 0.7633)*100:+.2f} percentage points")
    
    if abs(mean_acc - 0.7633) < 0.05:  # Within 5%
        print("\n✓ Our EEGNet result is close to GAH-TNet's reported value")
        print("  This validates the GAH-TNet Table 3 baselines")
    else:
        print("\n⚠ Our EEGNet result differs significantly from GAH-TNet's reported value")
        print("  This suggests GAH-TNet Table 3 baselines may not be directly comparable")


if __name__ == '__main__':
    run_eegnet_loso()
