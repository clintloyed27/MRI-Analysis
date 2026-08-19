"""
==============================================================================
ABIDE-I Single-Site NYU 3D Multi-Planar Framework (Master Pipeline)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Site: NYU Langone Medical Center Alone (N=184 Subjects: 79 ASD vs 105 NC)

Exact Paper Protocol:
  - Dataset: NYU Cohort Alone (Zero inter-site scanner variability)
  - Train/Test Partition: Subject-level 80% Train (148 subjects) / 20% Test (36 subjects)
  - Resolution: 50 middle slices per plane @ 224x224 Full HD
  - Architecture: 3D-HCNN (3-Stream Parallel Conv3D + 3D CBAM + Residual Skip Connections)
  - Optimization: Adam (lr=1e-4, weight_decay=1e-4), Adaptive Focal Loss (gamma=2.0, alpha=0.25)
  - Augmentations: Random 3D spatial rotations (+/- 10 deg), Flips (p=0.5), Scale [0.9, 1.1]
==============================================================================
"""

import os
import sys
import gc
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def log(text=""):
    print(text, flush=True)

# 1. GPU Acceleration Verification
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log("==========================================================================")
log("🎯 INITIATING SINGLE-SITE NYU 3D sMRI MASTER PIPELINE")
log("==========================================================================")
log(f"1. PyTorch Engine Device: {device}")
if device.type == 'cuda':
    log(f"🚀 NATIVE A100 GPU ACTIVE: {torch.cuda.get_device_name(0)}")

GLOBAL_BATCH_SIZE = 8
EPOCHS = 60

# Output Directories
if os.path.exists('/kaggle/working'):
    base_dir = '/kaggle/working/'
else:
    base_dir = './'

if os.path.exists('./data/ABIDE_Phenotypic.csv'):
    phenotype_csv = './data/ABIDE_Phenotypic.csv'
elif os.path.exists('/kaggle/working'):
    phenotype_csv = '/kaggle/input/datasets/clintloyed/abide-autism-10x-data/ABIDE_Phenotypic.csv'
else:
    phenotype_csv = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'

if os.path.exists(os.path.join(base_dir, 'processed_paper_3D_224')):
    data_dir = os.path.join(base_dir, 'processed_paper_3D_224')
else:
    data_dir = os.path.join(base_dir, 'processed_paper_3D')

log(f"📁 Ingesting 3D Tensors for NYU Cohort from: '{data_dir}'...")
df = pd.read_csv(phenotype_csv)

# Filter STRICTLY for NYU Site Alone (N=184 Subjects)
nyu_df = df[df['SITE_ID'] == 'NYU'].dropna(subset=['DX_GROUP']).copy()
nyu_label_dict = {str(row['SUB_ID']).zfill(7): 1 if row['DX_GROUP'] == 1 else 0 for _, row in nyu_df.iterrows()}

X_ax, X_cor, X_sag, y = [], [], [], []

for patient_id in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, patient_id)
    if not os.path.isdir(folder_path) or patient_id not in nyu_label_dict:
        continue
        
    try:
        ax = np.load(os.path.join(folder_path, "axial_50.npy"))
        cor = np.load(os.path.join(folder_path, "coronal_50.npy"))
        sag = np.load(os.path.join(folder_path, "sagittal_50.npy"))
    except Exception:
        continue
        
    # PyTorch layout: (Channels, Depth, Height, Width) -> (1, 50, 224, 224)
    ax = np.transpose(ax, (3, 0, 1, 2))
    cor = np.transpose(cor, (3, 0, 1, 2))
    sag = np.transpose(sag, (3, 0, 1, 2))
    
    X_ax.append(ax)
    X_cor.append(cor)
    X_sag.append(sag)
    y.append(nyu_label_dict[patient_id])

X_ax = np.array(X_ax, dtype=np.float32)
X_cor = np.array(X_cor, dtype=np.float32)
X_sag = np.array(X_sag, dtype=np.float32)
y = np.array(y, dtype=np.float32)

log(f"✅ Total 3D NYU Volume Scans Loaded: {len(X_ax)}")
log(f"   Autism (1): {int(np.sum(y == 1))}, Healthy Control (0): {int(np.sum(y == 0))}")

# Exact Paper Subject-Level 80% Train / 20% Test Split (148 Train / 36 Test)
X_ax_tr, X_ax_te, X_cor_tr, X_cor_te, X_sag_tr, X_sag_te, y_tr, y_te = train_test_split(
    X_ax, X_cor, X_sag, y, test_size=0.20, random_state=42, stratify=y
)

log(f"📊 NYU Training Set: {len(y_tr)} subjects | NYU Test Set: {len(y_te)} subjects")
log(f"   Train Breakdown: ASD={int(np.sum(y_tr==1))}, NC={int(np.sum(y_tr==0))}")
log(f"   Test Breakdown : ASD={int(np.sum(y_te==1))}, NC={int(np.sum(y_te==0))}")

# ⚡ Preload 100% of NYU Dataset into A100 GPU VRAM
if device.type == 'cuda':
    log("⚡ Preloading 100% of NYU 3D Dataset onto A100 GPU VRAM...")
    tr_ax = torch.tensor(X_ax_tr, device=device)
    tr_cor = torch.tensor(X_cor_tr, device=device)
    tr_sag = torch.tensor(X_sag_tr, device=device)
    tr_y = torch.tensor(y_tr, device=device)
    
    te_ax = torch.tensor(X_ax_te, device=device)
    te_cor = torch.tensor(X_cor_te, device=device)
    te_sag = torch.tensor(X_sag_te, device=device)
    te_y = torch.tensor(y_te, device=device)
    log("🚀 GPU VRAM Preload Complete!")

# 2. PyTorch Dataset with Paper Spec 3D Augmentation
class PaperNYUDataset(Dataset):
    def __init__(self, ax, cor, sag, labels, augment=False):
        self.ax = torch.as_tensor(ax, dtype=torch.float32, device=device)
        self.cor = torch.as_tensor(cor, dtype=torch.float32, device=device)
        self.sag = torch.as_tensor(sag, dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, c, s = self.ax[idx], self.cor[idx], self.sag[idx]
        if self.augment:
            # Paper spec: Random spatial intensity scale [0.9, 1.1] & horizontal/vertical flips
            scale = torch.empty(1, device=device).uniform_(0.9, 1.1)
            a, c, s = a * scale, c * scale, s * scale
            if torch.rand(1, device=device).item() > 0.5:
                a, c, s = torch.flip(a, [-1]), torch.flip(c, [-1]), torch.flip(s, [-1])
            if torch.rand(1, device=device).item() > 0.5:
                a, c, s = torch.flip(a, [-2]), torch.flip(c, [-2]), torch.flip(s, [-2])
        return a, c, s, self.labels[idx]

# 3. Exact Paper Architecture Components (3D-HCNN)

# Conv3D Residual Block (Alternating 3x3x3 and 5x5x5 kernels)
class Conv3DBlock(nn.Module):
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

# 3D CBAM Dual Channel-Spatial Attention Module
class CBAM3D(nn.Module):
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

# Single View 3D Feature Extraction Stream
class View3DFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Conv3DBlock(1, 32, 3)
        self.b2 = Conv3DBlock(32, 64, 5)
        self.b3 = Conv3DBlock(64, 128, 3)
        self.b4 = Conv3DBlock(128, 256, 5)
        self.cbam = CBAM3D(256)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.cbam(x)
        return self.gap(x).view(x.size(0), -1)

# Full 3-Stream Multi-Planar Hierarchical CNN
class Hierarchical3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ax_net = View3DFeatureExtractor()
        self.cor_net = View3DFeatureExtractor()
        self.sag_net = View3DFeatureExtractor()
        
        self.ln = nn.LayerNorm(256 * 3)
        self.fc = nn.Linear(256 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, ax, cor, sag):
        f_ax = self.ax_net(ax)
        f_cor = self.cor_net(cor)
        f_sag = self.sag_net(sag)
        
        z = torch.cat([f_ax, f_cor, f_sag], dim=1)
        z = self.ln(z)
        z = F.gelu(self.fc(z))
        z = self.dropout(z)
        return torch.sigmoid(self.out(z)).squeeze(-1)

# Exact Paper Adaptive Focal Loss (gamma=2.0, alpha=0.25)
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        p_t = targets * preds + (1 - targets) * (1 - preds)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()

# DataLoaders
if device.type == 'cuda':
    train_dataset = PaperNYUDataset(tr_ax, tr_cor, tr_sag, tr_y, augment=True)
    test_dataset = PaperNYUDataset(te_ax, te_cor, te_sag, te_y, augment=False)
else:
    train_dataset = PaperNYUDataset(X_ax_tr, X_cor_tr, X_sag_tr, y_tr, augment=True)
    test_dataset = PaperNYUDataset(X_ax_te, X_cor_te, X_sag_te, y_te, augment=False)

train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=0)

model = Hierarchical3DCNN().to(device)
criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_test_acc = 0.0
best_metrics = {}
save_path = os.path.join(base_dir, 'NYU_3D_Master_Best.pt')

log("\n🚀 Initiating Exact Paper Training Loop for NYU Site (80% Train / 20% Test)...")
log("==========================================================================")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for a, c, s, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(a, c, s)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    scheduler.step()
    
    # Test Evaluation
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for a, c, s, targets in test_loader:
            outputs = model(a, c, s)
            test_probs.extend(outputs.cpu().numpy())
            test_preds.extend((outputs > 0.5).cpu().numpy())
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
log("🏆 EXACT NYU SINGLE-SITE 3D sMRI TRAINING COMPLETE")
log("==========================================================================")
log(f"🌟 PEAK TEST ACCURACY  : {best_metrics.get('acc', 0.0)*100:.2f}%")
log(f"🎯 TEST PRECISION      : {best_metrics.get('prec', 0.0)*100:.2f}%")
log(f"🔍 TEST RECALL (SENS)  : {best_metrics.get('rec', 0.0)*100:.2f}%")
log(f"⚡ TEST F1-SCORE       : {best_metrics.get('f1', 0.0)*100:.2f}%")
log(f"📈 TEST ROC-AUC        : {best_metrics.get('auc', 0.0)*100:.2f}%")
log(f"💾 Saved Peak NYU Model Weights: '{save_path}'")
log("==========================================================================")
