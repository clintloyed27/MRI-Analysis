# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Preserved Preprocessing Pipelines & Benchmark Models

| Preprocessing Script | Target Dataset / Sites | Input Dimensions | Output Location | Paired Model Script | Peak Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[`preprocessing/abide_3d_preprocessing_224_multisite.py`](preprocessing/abide_3d_preprocessing_224_multisite.py)** <br> *(or `abide_3d_preprocessing.py`)* | **Multi-Site (3 Universities: NYU + UM_1 + USM)** <br> $N=395$ Subjects | 50 Slices @ **`224x224` Full HD** | `./processed_paper_3D_224/` | **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **`75.95%` Peak** 🔥 |
| **[`preprocessing/abide_3d_preprocessing_128_nyu.py`](preprocessing/abide_3d_preprocessing_128_nyu.py)** | **Single-Site (NYU Alone)** <br> $N=184$ Subjects | 50 Slices @ **`128x128` Standard** | `./processed_paper_3D/` | **[`models/abide_3d_hierarchical_cnn.py`](models/abide_3d_hierarchical_cnn.py)** <br> **[`models/abide_3d_nyu_master.py`](models/abide_3d_nyu_master.py)** | **`75.68%` Peak** 🌟 |

---

## ⚙️ How to Run

### Option 1: Multi-Site 224x224 Full HD Preprocessing (3 Universities)
```bash
python preprocessing/abide_3d_preprocessing_224_multisite.py
python models/abide_3d_hierarchical_cnn_pytorch.py
```

### Option 2: NYU Single-Site 128x128 Preprocessing (NYU Alone)
```bash
python preprocessing/abide_3d_preprocessing_128_nyu.py
python models/abide_3d_hierarchical_cnn.py
```

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── data/
│   └── ABIDE_Phenotypic.csv                      # Official ABIDE-I phenotypic metadata table
├── preprocessing/
│   ├── abide_3d_preprocessing.py                 # Default launcher (runs 224x224 multi-site pipeline)
│   ├── abide_3d_preprocessing_224_multisite.py   # 224x224 Full HD 3-university pipeline (NYU+UM_1+USM)
│   └── abide_3d_preprocessing_128_nyu.py         # 128x128 single-site NYU pipeline
├── models/
│   ├── abide_3d_hierarchical_cnn_pytorch.py      # Multi-site PyTorch 3D model (Peak: 75.95%)
│   ├── abide_3d_hierarchical_cnn.py               # NYU single-site Keras 3D model
│   └── abide_3d_nyu_master.py                    # Single-site NYU PyTorch Master model
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md     # Complete research report & methodology
└── README.md                                     # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
