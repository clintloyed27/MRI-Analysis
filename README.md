# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Preserved Benchmark Models & Results

| Model File | Dataset / Sites | Specs & Features | Peak Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **Multi-Site (NYU + UM_1 + USM)** <br> $N=395$ Subjects | PyTorch 3D Conv3D CBAM + GPU VRAM Preload + Full HD 224x224 (50 Slices) | **`75.95%` (Fold 1 Peak)** 🔥 |
| **[`models/abide_3d_hierarchical_cnn.py`](models/abide_3d_hierarchical_cnn.py)** | **Single-Site (NYU Alone)** <br> $N=184$ Subjects | Keras/TensorFlow 3D Conv3D CBAM + 128x128 / 224x224 | **`75.00%` (Fold 5 Peak)** 🌟 |

---

## 🔬 Dataset Overview (ABIDE-I Multi-Site & NYU Cohorts)

* **Multi-Site Cohort (`NYU`, `UM_1`, `USM`):** $N = 395$ Total Subjects
  * **Autism Spectrum Disorder (ASD):** 192 Subjects (48.6%)
  * **Healthy Control (HC):** 203 Subjects (51.4%)
* **Single-Site Cohort (`NYU` Alone):** $N = 184$ Total Subjects
  * **Autism Spectrum Disorder (ASD):** 79 Subjects (42.9%)
  * **Healthy Control (HC):** 105 Subjects (57.1%)

---

## ⚙️ Preprocessing & Model Architecture

### 1. Preprocessing Pipeline (`preprocessing/abide_3d_preprocessing.py`)
* **Otsu Adaptive Brain Masking:** Erases non-brain tissue, skull, and scalp fat.
* **N4 Bias Field Correction:** Removes spatial magnetic field shading gradients.
* **Z-Score Intensity Normalization:** Rescales voxel brightness per scan: $I_{\text{norm}} = (I_{\text{brain}} - \mu)/\sigma$.
* **Multi-Planar 50-Slice Extraction:** Extracts 50 middle slices for Axial (Z-axis), Coronal (Y-axis), and Sagittal (X-axis) physical planes into 3D tensors `(50, 224, 224, 1)`.

### 2. Neural Network Architecture (`models/abide_3d_hierarchical_cnn_pytorch.py`)
* **Volumetric Engine:** Genuine `Conv3D` layers operating on 50-slice spatial volumes.
* **Alternating Kernel Strategy:** Alternates $3 \times 3 \times 3$ kernels (fine gyri/sulci) with $5 \times 5 \times 5$ kernels (coarse lateral ventricles/cerebellum).
* **Hierarchical Scaling:** $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ channel progression.
* **ResNet Skip Connections:** Residual addition blocks preserve gradient flow across deep volumetric stages.
* **3D CBAM Attention:** Channel and 3D Spatial attention modules dynamically isolate ASD-correlated neuroanatomical biomarkers.
* **Adaptive Focal Loss:** Binary Focal Cross-Entropy ($\gamma=2.0, \alpha=0.25$) forces the network to focus on hard borderline cases.

---

## 🚀 How to Run

### Step 1: Execute 3D Preprocessing
```bash
python preprocessing/abide_3d_preprocessing.py
```

### Step 2: Run Multi-Site PyTorch 3D Training (Peak 75.95%)
```bash
python models/abide_3d_hierarchical_cnn_pytorch.py
```

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── preprocessing/
│   └── abide_3d_preprocessing.py               # Multi-site AWS S3 downloader & Full HD 224x224 3D tensor extractor
├── models/
│   ├── abide_3d_hierarchical_cnn_pytorch.py    # Multi-site (NYU+UM_1+USM) PyTorch 3D Conv3D CBAM Model (Peak: 75.95%)
│   └── abide_3d_hierarchical_cnn.py            # NYU Single-site 3D Conv3D CBAM Model (Peak: 75.00%)
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md   # Research report & methodology
└── README.md                                   # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
