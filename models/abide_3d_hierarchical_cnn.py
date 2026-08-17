"""
==============================================================================
ABIDE-I 3D Hierarchical Multi-Planar Convolutional Neural Network (Multi-Site)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Sites: NYU, UM_1, USM (~400 Subjects Total)
Based on the sMRI 3D Architecture by Hammash & Younis (2026)

Key Features:
  1. True 3D Convolutions (Conv3D) over 50-slice spatial volumes
  2. Alternating kernel sizes (3x3x3 for fine features, 5x5x5 for coarse structures)
  3. Hierarchical Channel Scaling (32 -> 64 -> 128 -> 256)
  4. ResNet-Style Skip Connections
  5. 3D CBAM Channel-Spatial Attention Mechanism
  6. Adaptive Binary Focal Loss (gamma=2.0, alpha=0.25)
  7. LayerNorm + Dense(256) + GELU + Dropout(0.5) Classifier Head
  8. Cross-Platform Compatibility (Lightning AI, Kaggle, Colab, Local)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.ops as K
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, Add, 
                                     Conv3D, MaxPooling3D, GlobalAveragePooling3D, 
                                     GlobalMaxPooling3D, Activation, LayerNormalization, BatchNormalization, Reshape)
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("1. Initializing 3D Tensor Ecosystem...")
# Optimized batch size for Lightning AI / Enterprise GPUs (A100, L4, L40S)
GLOBAL_BATCH_SIZE = 8 

# Cross-Platform Output Directory Configuration
if os.path.exists('/kaggle/working'):
    base_dir = '/kaggle/working/'
    phenotype_csv = '/kaggle/input/datasets/clintloyed/abide-autism-10x-data/ABIDE_Phenotypic.csv'
else:
    base_dir = './'
    phenotype_csv = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'

data_dir = os.path.join(base_dir, 'processed_paper_3D')

print("2. Loading Clinical Metadata (Multi-Site: NYU, UM_1, USM)...")
df = pd.read_csv(phenotype_csv)
TARGET_SITES = ['NYU', 'UM_1', 'USM']
df = df[df['SITE_ID'].isin(TARGET_SITES)].dropna(subset=['DX_GROUP'])
label_dict = {str(row['SUB_ID']).zfill(7): 1 if row['DX_GROUP'] == 1 else 0 for _, row in df.iterrows()}

print("3. Ingesting 3D NumPy Volume Tensors (Axial, Coronal, Sagittal)...")
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
        
    X_ax.append(ax)
    X_cor.append(cor)
    X_sag.append(sag)
    y.append(label_dict[patient_id])

X_ax, X_cor, X_sag = np.array(X_ax), np.array(X_cor), np.array(X_sag)
y = np.array(y)

print(f"✅ Total Multi-Site 3D Volume Tensors Loaded: {len(X_ax)}")
print(f"   Autism (1): {np.sum(y == 1)}, Healthy Control (0): {np.sum(y == 0)}")

def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Adaptive Focal Loss targeting difficult borderline subjects"""
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return focal_loss_fn

def build_paper_3d_cnn():
    ax_input = Input(shape=(50, 128, 128, 1), name='axial')
    cor_input = Input(shape=(50, 128, 128, 1), name='coronal')
    sag_input = Input(shape=(50, 128, 128, 1), name='sagittal')
    
    def hierarchical_block(x, filters, kernel_size):
        res = Conv3D(filters, 1, padding='same', use_bias=False)(x)
        x = Conv3D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, res])
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def build_3d_feature_extractor(view_input):
        x = hierarchical_block(view_input, 32, (3, 3, 3))
        x = hierarchical_block(x, 64, (5, 5, 5))
        x = hierarchical_block(x, 128, (3, 3, 3))
        x = hierarchical_block(x, 256, (5, 5, 5))
        
        # 3D CBAM Attention
        channel_avg = GlobalAveragePooling3D()(x)
        channel_max = GlobalMaxPooling3D()(x)
        dense_1 = Dense(256 // 8, activation='relu')
        dense_2 = Dense(256)
        
        avg_out = dense_2(dense_1(channel_avg))
        max_out = dense_2(dense_1(channel_max))
        
        channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
        channel_attention = Reshape((1, 1, 1, 256))(channel_attention)
        x = x * channel_attention
        
        spatial_avg = K.mean(x, axis=-1, keepdims=True)
        spatial_max = K.max(x, axis=-1, keepdims=True)
        spatial_concat = Concatenate(axis=-1)([spatial_avg, spatial_max])
        
        spatial_attention = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(spatial_concat)
        x = x * spatial_attention
        
        return GlobalAveragePooling3D()(x)

    ax_feat = build_3d_feature_extractor(ax_input)
    cor_feat = build_3d_feature_extractor(cor_input)
    sag_feat = build_3d_feature_extractor(sag_input)
    
    z = Concatenate()([ax_feat, cor_feat, sag_feat])
    z = LayerNormalization()(z)
    z = Dense(256, activation='gelu')(z)
    z = Dropout(0.5)(z)
    predictions = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[ax_input, cor_input, sag_input], outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss=binary_focal_loss(), 
                  metrics=['accuracy'])
    return model

print("\n🚀 4. Initiating Stratified 5-Fold Cross Validation Protocol...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

def augment_3d_tensors(inputs, label):
    scale = tf.random.uniform([], 0.9, 1.1)
    aug_inputs = {
        'axial': inputs['axial'] * scale,
        'coronal': inputs['coronal'] * scale,
        'sagittal': inputs['sagittal'] * scale
    }
    return aug_inputs, label

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ax, y), 1):
    print(f"\n==========================================")
    print(f"🔥 TRAINING FOLD {fold} / 5 (MULTI-SITE 3D MODE)")
    print(f"==========================================")
    
    tf.keras.backend.clear_session()
    
    X_ax_t, X_ax_v = X_ax[train_idx], X_ax[val_idx]
    X_cor_t, X_cor_v = X_cor[train_idx], X_cor[val_idx]
    X_sag_t, X_sag_v = X_sag[train_idx], X_sag[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        ({'axial': X_ax_t, 'coronal': X_cor_t, 'sagittal': X_sag_t}, y_train)
    ).shuffle(len(y_train)).map(augment_3d_tensors).batch(GLOBAL_BATCH_SIZE)
    
    val_ds = tf.data.Dataset.from_tensor_slices(
        ({'axial': X_ax_v, 'coronal': X_cor_v, 'sagittal': X_sag_v}, y_val)
    ).batch(GLOBAL_BATCH_SIZE)
    
    model = build_paper_3d_cnn()
    
    checkpoint_filepath = os.path.join(base_dir, f'Paper_3D_GodMode_Fold{fold}.keras')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30, 
        callbacks=[checkpoint_callback]
    )
    
    model.load_weights(checkpoint_filepath)
    y_pred_prob = model.predict(val_ds)
    y_pred = (y_pred_prob > 0.5).astype(int)
    fold_acc = accuracy_score(y_val, y_pred)
    fold_scores.append(fold_acc)
    print(f"\n✅ Fold {fold} Multi-Site 3D Validation Accuracy: {fold_acc:.4f}")

print("\n==============================================")
print("🏆 MULTI-SITE 3D MULTI-PLANAR 5-FOLD CV COMPLETE")
print("==============================================")
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: {score*100:.2f}%")
print(f"🌟 FINAL MULTI-SITE 3D AVERAGE ACCURACY: {np.mean(fold_scores)*100:.2f}%")
