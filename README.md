# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Preserved Benchmark Models & Preprocessing Pipelines

| Model Script | Preprocessing Script | Dataset / Sites | Resolution | Peak Validation Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **[`preprocessing/abide_3d_preprocessing.py`](preprocessing/abide_3d_preprocessing.py)** | **Multi-Site (NYU + UM_1 + USM)** <br> $N=395$ Subjects | Full HD **`224x224`** | **`75.95%` (Fold 1 Peak)** 🔥 |
| **[`models/abide_3d_hierarchical_cnn.py`](models/abide_3d_hierarchical_cnn.py)** | **[`preprocessing/abide_3d_preprocessing_nyu_128.py`](preprocessing/abide_3d_preprocessing_nyu_128.py)** | **Single-Site (NYU Alone)** <br> $N=184$ Subjects | Standard **`128x128`** | **`75.00%` (Fold 5 Peak)** 🌟 |

---

## 🔬 Dataset Overview (ABIDE-I Multi-Site & NYU Cohorts)

* **Multi-Site Cohort (`NYU`, `UM_1`, `USM`):** $N = 395$ Total Subjects
  * **Autism Spectrum Disorder (ASD):** 192 Subjects (48.6%)
  * **Healthy Control (HC):** 203 Subjects (51.4%)
* **Single-Site Cohort (`NYU` Alone):** $N = 184$ Total Subjects
  * **Autism Spectrum Disorder (ASD):** 79 Subjects (42.9%)
  * **Healthy Control (HC):** 105 Subjects (57.1%)

---

## ⚙️ Preprocessing & Model Architectures

### 1. Multi-Site Full HD Preprocessing (`preprocessing/abide_3d_preprocessing.py`)
* **Otsu Adaptive Brain Masking:** Erases non-brain tissue, skull, and scalp fat.
* **N4 Bias Field Correction:** Removes spatial magnetic field shading gradients.
* **Site-Harmonized Z-Score Normalization:** Rescales voxel brightness per scan while aligning GE/Siemens contrast histograms.
* **Multi-Planar 50-Slice Extraction:** Extracts 50 middle slices for Axial, Coronal, and Sagittal physical planes into 3D tensors `(50, 224, 224, 1)`.

### 2. NYU Single-Site Preprocessing (`preprocessing/abide_3d_preprocessing_nyu_128.py`)
* **Bounding Box Skull Stripping:** Crops empty background space around brain tissue.
* **Z-Score Normalization:** Rescales voxel brightness to $[-3\sigma, +3\sigma]$.
* **Multi-Planar 50-Slice Extraction:** Extracts 50 middle slices into `(50, 128, 128, 1)` tensors.

---

## 🚀 How to Run

### Pipeline A: Multi-Site PyTorch 3D Model (Peak 75.95%)
```bash
python preprocessing/abide_3d_preprocessing.py
python models/abide_3d_hierarchical_cnn_pytorch.py
```

### Pipeline B: NYU Single-Site Keras 3D Model (Peak 75.00%)
```bash
python preprocessing/abide_3d_preprocessing_nyu_128.py
python models/abide_3d_hierarchical_cnn.py
```

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── preprocessing/
│   ├── abide_3d_preprocessing.py            # Multi-site Full HD 224x224 3D tensor extractor (NYU+UM_1+USM)
│   └── abide_3d_preprocessing_nyu_128.py    # NYU Single-site 128x128 3D tensor extractor (NYU)
├── models/
│   ├── abide_3d_hierarchical_cnn_pytorch.py # Multi-site PyTorch 3D Conv3D CBAM Model (Peak: 75.95%)
│   └── abide_3d_hierarchical_cnn.py         # NYU Single-site 3D Conv3D CBAM Model (Peak: 75.00%)
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md# Complete research report & methodology
└── README.md                                # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
