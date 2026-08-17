"""
==============================================================================
ABIDE-I 3D Hierarchical Multi-Planar PyTorch Neural Network (224x224 Full HD)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Sites: NYU, UM_1, USM (~400 Subjects Total)
Native PyTorch CUDA GPU Accelerator (Guaranteed A100 / L4 Native Speed)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. GPU Acceleration Verification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"1. Initializing PyTorch 3D Engine on Device: {device}")
if device.type == 'cuda':
    print(f"🚀 NATIVE GPU ACCELERATOR ACTIVE: {torch.cuda.get_device_name(0)}")

GLOBAL_BATCH_SIZE = 8
EPOCHS = 50

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

print(f"📁 Ingesting 3D Tensors from: '{data_dir}'...")
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
        
    # Reshape (50, H, W, 1) -> (1, 50, H, W) for PyTorch Conv3D
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

print(f"✅ Total 3D Volume Tensors Loaded: {len(X_ax)}")
print(f"   Autism (1): {int(np.sum(y == 1))}, Healthy Control (0): {int(np.sum(y == 0))}")

class sMRIDataset(Dataset):
    def __init__(self, ax, cor, sag, labels, augment=False):
        self.ax = torch.tensor(ax)
        self.cor = torch.tensor(cor)
        self.sag = torch.tensor(sag)
        self.labels = torch.tensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, c, s = self.ax[idx], self.cor[idx], self.sag[idx]
        if self.augment:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            a, c, s = a * scale, c * scale, s * scale
            if torch.rand(1).item() > 0.5:
                a, c, s = torch.flip(a, [-1]), torch.flip(c, [-1]), torch.flip(s, [-1])
        return a, c, s, self.labels[idx]

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
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)
        self.spatial_conv = nn.Conv3d(2, 1, 3, padding=1)

    def forward(self, x):
        # Channel Attention
        b, c, _, _, _ = x.size()
        avg_p = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool3d(x, 1).view(b, c)
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_p))) + self.fc2(F.relu(self.fc1(max_p))))
        x = x * ca.view(b, c, 1, 1, 1)
        
        # Spatial Attention
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
        return torch.sigmoid(self.out(z)).squeeze(-1)

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

print("\n🚀 4. Initiating PyTorch 5-Fold Stratified Cross Validation Protocol...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ax, y), 1):
    print(f"\n==========================================")
    print(f"🔥 TRAINING FOLD {fold} / 5 (PyTorch GPU Mode)")
    print(f"==========================================")
    
    train_dataset = sMRIDataset(X_ax[train_idx], X_cor[train_idx], X_sag[train_idx], y[train_idx], augment=True)
    val_dataset = sMRIDataset(X_ax[val_idx], X_cor[val_idx], X_sag[val_idx], y[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)
    
    model = Hierarchical3DCNN().to(device)
    criterion = BinaryFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for a, c, s, targets in train_loader:
            a, c, s, targets = a.to(device), c.to(device), s.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(a, c, s)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for a, c, s, targets in val_loader:
                a, c, s, targets = a.to(device), c.to(device), s.to(device), targets.to(device)
                outputs = model(a, c, s)
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(base_dir, f'PyTorch_3D_Fold{fold}.pt'))
            
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% (Peak: {best_acc*100:.2f}%)")
            
    fold_scores.append(best_acc)
    print(f"✅ Fold {fold} PyTorch Peak Validation Accuracy: {best_acc*100:.2f}%")

print("\n==============================================")
print("🏆 PyTorch 3D MULTI-PLANAR 5-FOLD CV COMPLETE")
print("==============================================")
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: {score*100:.2f}%")
print(f"🌟 FINAL PyTorch 3D AVERAGE ACCURACY: {np.mean(fold_scores)*100:.2f}%")
