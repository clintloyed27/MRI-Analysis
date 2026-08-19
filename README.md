# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Preserved Benchmark Models & Preprocessing Pipelines

| Model Script | Preprocessing Script | Target Cohort | Input Dimensions | Output Location | Benchmark Peak |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[`models/abide_3d_hierarchical_cnn.py`](models/abide_3d_hierarchical_cnn.py)** | **[`preprocessing/abide_3d_preprocessing.py`](preprocessing/abide_3d_preprocessing.py)** | **NYU Site Alone** ($N=184$) | 50 Slices @ **`128x128`** | `./processed_paper_3D/` | **`75.00%` Peak** 🌟 |
| **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **[`preprocessing/abide_3d_preprocessing.py`](preprocessing/abide_3d_preprocessing.py)** | **Multi-Site** ($N=395$) | 50 Slices @ **`224x224`** | `./processed_paper_3D_224/` | **`75.95%` Peak** 🔥 |

---

## 🔬 Dataset Overview (ABIDE-I NYU Cohort)

* **Single-Site Cohort (`NYU` Alone):** $N = 184$ Total Subjects
  * **Autism Spectrum Disorder (ASD):** 79 Subjects (42.9%)
  * **Healthy Control (HC):** 105 Subjects (57.1%)

---

## ⚙️ Preprocessing & Model Architecture

### 1. NYU 128x128 Preprocessing (`preprocessing/abide_3d_preprocessing.py`)
* **AWS S3 Automated Downloader:** Pulls raw NIfTI scans for the NYU cohort.
* **Bounding Box Skull Stripping:** Detects non-zero voxel intensity coordinates to crop empty background space around brain tissue.
* **Z-Score Normalization:** Rescales voxel brightness to $[-3\sigma, +3\sigma]$.
* **Multi-Planar 50-Slice Tensor Extractor:** Extracts 50 middle slices for Axial, Coronal, and Sagittal physical planes into 3D tensors `(50, 128, 128, 1)`.
* **Output Destination:** Saves 3D arrays to `./processed_paper_3D/`.

### 2. Keras 3D Conv3D Model (`models/abide_3d_hierarchical_cnn.py`)
* **3-Stream Parallel Backbone:** Axial, Coronal, and Sagittal 3D feature extractors.
* **Alternating Kernels:** $3\times3\times3$ and $5\times5\times5$ 3D Conv blocks with residual skip connections.
* **3D CBAM Attention:** Channel & 3D Spatial Attention modules.
* **Adaptive Focal Loss:** $\gamma=2.0, \alpha=0.25$.

---

## 🚀 How to Run

### Step 1: Run Preprocessing
```bash
python preprocessing/abide_3d_preprocessing.py
```

### Step 2: Run 3D Model Training
```bash
python models/abide_3d_hierarchical_cnn.py
```

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── data/
│   └── ABIDE_Phenotypic.csv                # Official ABIDE-I phenotypic metadata table
├── preprocessing/
│   └── abide_3d_preprocessing.py           # 128x128 3D tensor extraction pipeline (outputs to ./processed_paper_3D/)
├── models/
│   ├── abide_3d_hierarchical_cnn.py        # Keras 3D Conv3D CBAM Model (NYU 128x128)
│   ├── abide_3d_hierarchical_cnn_pytorch.py# PyTorch 3D Conv3D CBAM Model (Multi-Site)
│   └── abide_3d_nyu_master.py              # Single-site NYU PyTorch Master Model
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md# Complete research report & methodology
└── README.md                               # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
