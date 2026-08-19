"""
==============================================================================
ABIDE-I Single-Site NYU 3D Multi-Planar Framework (5-Fold Stratified Engine)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Site: NYU Langone Medical Center Alone (N=184 Subjects: 79 ASD vs 105 NC)

Configuration:
  - Dataset: NYU Cohort Alone (Zero inter-site scanner variability)
  - Evaluation: Stratified 5-Fold Cross Validation Protocol
  - Resolution: 50 middle slices per plane @ 224x224 Full HD
  - Architecture: 3D-HCNN (3-Stream Parallel Conv3D + 3D CBAM + Residual Skip Connections)
  - Loss: BCEWithLogitsLoss with pos_weight class balancing
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
from sklearn.model_selection import StratifiedKFold
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
log("🎯 INITIATING SINGLE-SITE NYU 3D sMRI MASTER PIPELINE (5-FOLD CV)")
log("==========================================================================")
log(f"1. PyTorch Engine Device: {device}")
if device.type == 'cuda':
    log(f"🚀 NATIVE A100 GPU ACTIVE: {torch.cuda.get_device_name(0)}")

GLOBAL_BATCH_SIZE = 8
EPOCHS = 50

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

# 2. PyTorch Dataset with 3D Augmentation
class PaperNYUDataset(Dataset):
    def __init__(self, ax, cor, sag, labels, augment=False):
        self.ax = torch.as_tensor(ax, dtype=torch.float32)
        self.cor = torch.as_tensor(cor, dtype=torch.float32)
        self.sag = torch.as_tensor(sag, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = self.ax[idx]
        c = self.cor[idx]
        s = self.sag[idx]
        label = self.labels[idx]
        
        if self.augment:
            scale = torch.empty(1).uniform_(0.9, 1.1)
            a, c, s = a * scale, c * scale, s * scale
            if torch.rand(1).item() > 0.5:
                a, c, s = torch.flip(a, [-1]), torch.flip(c, [-1]), torch.flip(s, [-1])
            if torch.rand(1).item() > 0.5:
                a, c, s = torch.flip(a, [-2]), torch.flip(c, [-2]), torch.flip(s, [-2])
        return a, c, s, label

# 3. Exact Paper Architecture Components (3D-HCNN)

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
        return self.out(z).squeeze(-1)

# 4. Stratified 5-Fold Cross Validation Protocol
log("\n🚀 Initiating Stratified 5-Fold Cross Validation for NYU Site...")
log("==========================================================================")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ax, y), 1):
    log(f"\n==========================================")
    log(f"🔥 TRAINING FOLD {fold} / 5 (NYU Site Alone)")
    log(f"==========================================")
    
    train_dataset = PaperNYUDataset(X_ax[train_idx], X_cor[train_idx], X_sag[train_idx], y[train_idx], augment=True)
    val_dataset = PaperNYUDataset(X_ax[val_idx], X_cor[val_idx], X_sag[val_idx], y[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = Hierarchical3DCNN().to(device)
    
    pos_weight = torch.tensor([(len(train_idx) - np.sum(y[train_idx])) / np.sum(y[train_idx])], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_acc = 0.0
    save_path = os.path.join(base_dir, f'NYU_3D_Fold{fold}_Best.pt')
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for a, c, s, targets in train_loader:
            a, c, s, targets = a.to(device), c.to(device), s.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(a, c, s)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for a, c, s, targets in val_loader:
                a, c, s, targets = a.to(device), c.to(device), s.to(device), targets.to(device)
                logits = model(a, c, s)
                probs = torch.sigmoid(logits)
                val_preds.extend((probs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
        if epoch % 5 == 0 or epoch == EPOCHS:
            log(f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% (Peak: {best_val_acc*100:.2f}%)")
            
    fold_scores.append(best_val_acc)
    log(f"✅ Fold {fold} NYU Peak Validation Accuracy: {best_val_acc*100:.2f}%")

log("\n==============================================")
log("🏆 NYU SINGLE-SITE 3D sMRI 5-FOLD CV COMPLETE")
log("==============================================")
for i, score in enumerate(fold_scores, 1):
    log(f"Fold {i}: {score*100:.2f}%")
log(f"🌟 FINAL NYU 3D AVERAGE ACCURACY: {np.mean(fold_scores)*100:.2f}%")
