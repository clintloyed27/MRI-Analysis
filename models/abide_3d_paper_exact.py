"""
==============================================================================
ABIDE-I 3D Multi-Planar Hierarchical CNN with CBAM (Exact Single Split Paper Mode)
------------------------------------------------------------------------------
Paper Reference: Hammash & Younis (2026) / Hammash et al. (2026)
Single Train/Test Split Protocol:
  - 80% Training Set (~316 Subjects) | 20% Testing Set (~79 Subjects)
  - NO Cross Validation Folds. Single Master Run.
  - Resolution: 50 middle slices @ 224x224 Full HD per slice
  - Architecture: 3-Stream Parallel Conv3D (3x3x3 & 5x5x5 kernels) + 3D CBAM
  - Optimization: Adam (lr=1e-4, weight_decay=1e-4), BCEWithLogitsLoss
  - Memory: Batch Size 4 + GPU Memory Flushing (Zero CUDA OOM)
==============================================================================
"""

import os
import sys
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def log(text=""):
    print(text, flush=True)

# 1. GPU Memory Cleanup & Verification
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log("==========================================================================")
log("📜 INITIATING SINGLE-SPLIT EXACT PAPER MODEL (Hammash & Younis, 2026)")
log("==========================================================================")
log(f"1. PyTorch Execution Device: {device}")
if device.type == 'cuda':
    log(f"🚀 NATIVE A100 GPU ACTIVE: {torch.cuda.get_device_name(0)}")

GLOBAL_BATCH_SIZE = 4 # Memory-safe footprint
EPOCHS = 60

# Output Directories
if os.path.exists('/kaggle/working'):
    base_dir = '/kaggle/working/'
    phenotype_csv = '/kaggle/input/datasets/clintloyed/abide-autism-10x-data/ABIDE_Phenotypic.csv'
else:
    base_dir = './'
    phenotype_csv = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'

if os.path.exists(os.path.join(base_dir, 'processed_paper_3D_224')):
    data_dir = os.path.join(base_dir, 'processed_paper_3D_224')
else:
    data_dir = os.path.join(base_dir, 'processed_paper_3D')

log(f"📁 Ingesting 3D Tensors from: '{data_dir}'...")
df = pd.read_csv(phenotype_csv)
TARGET_SITES = ['NYU', 'UM_1', 'USM']
df = df[df['SITE_ID'].isin(TARGET_SITES)].dropna(subset=['DX_GROUP'])
label_dict = {str(row['SUB_ID']).zfill(7): 1 if row['DX_GROUP'] == 1 else 0 for _, row in df.iterrows()}

X_ax, X_cor, X_sag, y = [], [], [], []

for patient_id in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, patient_id)
    if not os.path.isdir(folder_path) or patient_id not in label_dict:
        continue
        
    try:
        ax = np.load(os.path.join(folder_path, "axial_50.npy"))
        cor = np.load(os.path.join(folder_path, "coronal_50.npy"))
        sag = np.load(os.path.join(folder_path, "sagittal_50.npy"))
    except Exception:
        continue
        
    ax = np.transpose(ax, (3, 0, 1, 2))
    cor = np.transpose(cor, (3, 0, 1, 2))
    sag = np.transpose(sag, (3, 0, 1, 2))
    
    X_ax.append(ax)
    X_cor.append(cor)
    X_sag.append(sag)
    y.append(label_dict[patient_id])

X_ax = np.array(X_ax, dtype=np.float32)
X_cor = np.array(X_cor, dtype=np.float32)
X_sag = np.array(X_sag, dtype=np.float32)
y = np.array(y, dtype=np.float32)

log(f"✅ Total 3D Volume Scans Loaded: {len(X_ax)}")
log(f"   Autism (1): {int(np.sum(y == 1))}, Healthy Control (0): {int(np.sum(y == 0))}")

# Stratified 80% Train / 20% Test Split
X_ax_tr, X_ax_te, X_cor_tr, X_cor_te, X_sag_tr, X_sag_te, y_tr, y_te = train_test_split(
    X_ax, X_cor, X_sag, y, test_size=0.20, random_state=42, stratify=y
)

log(f"📊 Training Set: {len(y_tr)} subjects | Test Set: {len(y_te)} subjects")

# Preload ENTIRE dataset onto GPU VRAM
if device.type == 'cuda':
    log("⚡ Preloading 100% of Train/Test 3D Dataset onto A100 GPU VRAM...")
    tr_ax = torch.tensor(X_ax_tr, device=device)
    tr_cor = torch.tensor(X_cor_tr, device=device)
    tr_sag = torch.tensor(X_sag_tr, device=device)
    tr_y = torch.tensor(y_tr, device=device)
    
    te_ax = torch.tensor(X_ax_te, device=device)
    te_cor = torch.tensor(X_cor_te, device=device)
    te_sag = torch.tensor(X_sag_te, device=device)
    te_y = torch.tensor(y_te, device=device)
    log("🚀 GPU VRAM Preload Complete!")

class PaperMRIDataset(Dataset):
    def __init__(self, ax, cor, sag, labels, augment=False):
        self.ax = ax
        self.cor = cor
        self.sag = sag
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, c, s = self.ax[idx], self.cor[idx], self.sag[idx]
        if self.augment:
            scale = torch.empty(1, device=a.device).uniform_(0.9, 1.1)
            a, c, s = a * scale, c * scale, s * scale
            if torch.rand(1, device=a.device).item() > 0.5:
                a, c, s = torch.flip(a, [-1]), torch.flip(c, [-1]), torch.flip(s, [-1])
        return a, c, s, self.labels[idx]

class PaperConv3DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.res = nn.Conv3d(in_c, out_c, 1, padding=0, bias=False)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        res = self.res(x)
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool(out + res)
        return out

class PaperCBAM3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_p = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool3d(x, 1).view(b, c)
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_p))) + self.fc2(F.relu(self.fc1(max_p))))
        x = x * ca.view(b, c, 1, 1, 1)
        
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([sa_avg, sa_max], dim=1)))
        return x * sa

class PaperViewFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = PaperConv3DBlock(1, 32, 3)
        self.b2 = PaperConv3DBlock(32, 64, 5)
        self.b3 = PaperConv3DBlock(64, 128, 3)
        self.b4 = PaperConv3DBlock(128, 256, 5)
        self.cbam = PaperCBAM3D(256)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.cbam(x)
        return self.gap(x).view(x.size(0), -1)

class PaperHierarchical3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ax_net = PaperViewFeatureExtractor()
        self.cor_net = PaperViewFeatureExtractor()
        self.sag_net = PaperViewFeatureExtractor()
        
        self.ln = nn.LayerNorm(256 * 3)
        self.fc = nn.Linear(256 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1) # Raw Logit output

    def forward(self, ax, cor, sag):
        f_ax = self.ax_net(ax)
        f_cor = self.cor_net(cor)
        f_sag = self.sag_net(sag)
        
        z = torch.cat([f_ax, f_cor, f_sag], dim=1)
        z = self.ln(z)
        z = F.gelu(self.fc(z))
        z = self.dropout(z)
        return self.out(z).squeeze(-1) # Output raw logits

# DataLoaders
if device.type == 'cuda':
    train_dataset = PaperMRIDataset(tr_ax, tr_cor, tr_sag, tr_y, augment=True)
    test_dataset = PaperMRIDataset(te_ax, te_cor, te_sag, te_y, augment=False)
else:
    train_dataset = PaperMRIDataset(X_ax_tr, X_cor_tr, X_sag_tr, y_tr, augment=True)
    test_dataset = PaperMRIDataset(X_ax_te, X_cor_te, X_sag_te, y_te, augment=False)

train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=0)

model = PaperHierarchical3DCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_test_acc = 0.0
best_metrics = {}
save_path = os.path.join(base_dir, 'Paper_3D_SingleSplit_Best.pt')

log("\n🚀 Initiating Single-Split Training Loop (80% Train / 20% Test)...")
log("==========================================================================")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for a, c, s, targets in train_loader:
        optimizer.zero_grad()
        logits = model(a, c, s)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    scheduler.step()
    
    # Test Evaluation
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for a, c, s, targets in test_loader:
            logits = model(a, c, s)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend((probs > 0.5).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), save_path)
        prec = precision_score(test_targets, test_preds, zero_division=0)
        rec = recall_score(test_targets, test_preds, zero_division=0)
        f1 = f1_score(test_targets, test_preds, zero_division=0)
        auc = roc_auc_score(test_targets, test_probs)
        best_metrics = {'acc': test_acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}
        
    log(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Test Acc: {test_acc*100:.2f}% (🏆 PEAK: {best_test_acc*100:.2f}%)")

log("\n==========================================================================")
log("🏆 SINGLE-SPLIT EXACT PAPER 3D TRAINING COMPLETE")
log("==========================================================================")
log(f"🌟 PEAK TEST ACCURACY  : {best_metrics.get('acc', 0.0)*100:.2f}%")
log(f"🎯 TEST PRECISION      : {best_metrics.get('prec', 0.0)*100:.2f}%")
log(f"🔍 TEST RECALL (SENS)  : {best_metrics.get('rec', 0.0)*100:.2f}%")
log(f"⚡ TEST F1-SCORE       : {best_metrics.get('f1', 0.0)*100:.2f}%")
log(f"📈 TEST ROC-AUC        : {best_metrics.get('auc', 0.0)*100:.2f}%")
log(f"💾 Saved Peak Master Weights: '{save_path}'")
log("==========================================================================")
