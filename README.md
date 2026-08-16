# 🧠 Structural MRI Analysis & Neuroimaging System
### Deep Learning Pipeline for Brain Tumor, Alzheimer's, and Autism Spectrum Disorder (ASD) Classification

This repository contains a comprehensive, end-to-end Artificial Intelligence system designed to analyze Magnetic Resonance Imaging (MRI) scans and raw hospital DICOM/NIfTI files to detect structural brain anomalies and neurodevelopmental conditions.

---

## 🔬 Neurological Diseases Identified

This framework features three highly specialized deep learning pipelines:

### 1. Autism Spectrum Disorder (ASD) — Volumetric Neurodevelopmental Analysis
* **Modality:** Structural MRI (sMRI, T1-weighted)
* **Dataset:** ABIDE-I (NYU Langone Medical Center Cohort — 184 subjects: 79 ASD / 105 Healthy Control).
* **Architecture:** True 3D Hierarchical Multi-Planar Convolutional Neural Network (`Conv3D`) based on Hammash & Younis (2026).
* **Key Innovations:**
  * **3D Multi-Planar Extraction:** Extracts 50 consecutive middle slices across Axial, Coronal, and Sagittal planes into raw 3D NumPy arrays `(50, 128, 128, 1)`.
  * **Alternating Kernel Strategy:** Alternates `3x3x3` kernels (fine anatomical details like gyri/sulci) with `5x5x5` kernels (coarse structures like lateral ventricles).
  * **3D CBAM Attention:** Channel and 3D Spatial attention modules dynamically pinpoint ASD-correlated anatomical regions.
  * **Adaptive Focal Loss:** Handles borderline, hard-to-classify diagnostic cases ($\gamma=2.0, \alpha=0.25$).
* **Experimental Performance:**
  * **2D Single-View Baseline:** `56.25%`
  * **2.5D Global Run:** `63.62%`
  * **True 3D Hierarchical 5-Fold CV Average:** **`69.59%`**
  * **Peak Fold Result (Fold 5):** **`75.00%`** *(+18.75 pp gain over baseline)*

---

### 2. Brain Tumors — Macroscopic Tissue Growth
Detects large-scale abnormal tissue growth and classifies 2D scans into four distinct categories:
* **Glioma Tumor:** Tumors occurring in the brain and spinal cord tissue.
* **Meningioma Tumor:** Tumors arising from the meninges surrounding the brain.
* **Pituitary Tumor:** Tumors developing in the pituitary gland at the cranial base.
* **No Tumor:** Healthy brain structure.

---

### 3. Alzheimer's Disease — Microscopic Tissue Atrophy
Detects microscopic tissue shrinkage (atrophy), primarily focusing on the Hippocampus and medial temporal lobes. Classifies scans into four clinical stages:
* **Very Mild Impairment:** Earliest perceptible tissue loss.
* **Mild Impairment:** Early-stage structural cognitive decline.
* **Moderate Impairment:** Obvious, significant tissue atrophy.
* **No Impairment:** Healthy brain structure.

---

## 💻 Technology Stack & Architectural Overview

| Technology | Purpose | Technical Justification |
|------------|---------|-------------------------|
| **TensorFlow & Keras 3** | Core Deep Learning | Standard framework for building 2D and 3D convolutional networks. |
| **Conv3D (3D Convolution)** | Volumetric Feature Extraction | Processes true 3D spatial tensors `(Depth, Height, Width, Channels)` rather than isolated 2D slice images. |
| **EfficientNetB0 & DenseNet121** | 2D Feature Extraction | Compound scaling and dense feature reuse for 2D classification pipelines. |
| **3D CBAM Attention** | Explainable Feature Gating | Dynamically weights feature channels and 3D spatial regions linked to ASD biomarkers. |
| **Adaptive Focal Loss** | Objective Function | Down-weights easy samples and focuses gradient updates on hard borderline subjects. |
| **Pydicom & NiBabel** | Neuroimaging I/O | Parses raw 16-bit hospital `.dcm` files and 3D NIfTI `.nii.gz` volumetric MRI data. |
| **Streamlit** | Interactive Web Dashboard | Local web application deploying models for real-time inference and Grad-CAM visualization. |

---

## ⚙️ Preprocessing & Evaluation Pipeline

### 3D sMRI Preprocessing Pipeline (`preprocessing/abide_3d_preprocessing.py`)
1. **Bounding Box Skull Stripping:** Detects non-zero voxel intensity coordinates to crop away empty background space, reducing computational load.
2. **Z-Score Intensity Normalization:** Standardizes pixel intensity distributions across scans: $I_{\text{norm}} = (I_{\text{brain}} - \mu) / \sigma$, with outlier truncation at $[-3\sigma, +3\sigma]$.
3. **Multi-Planar Slice Sampling:** Samples 50 middle slices along Axial (Z-axis), Coronal (Y-axis), and Sagittal (X-axis) physical planes into raw `.npy` tensors.

### Evaluation Protocol
* **Stratified 5-Fold Cross Validation:** Evaluates models under an 80% Train / 20% Test split repeated across 5 folds, maintaining exact class balance and preventing data leakage.

---

## 📄 References & Literature Citations

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
