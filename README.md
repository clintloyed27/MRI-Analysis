# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I Dataset**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Benchmark Models & Preprocessing Pairings

| Model Script | Resolution & Target | Input Tensor Folder | Paired Preprocessing Script | Benchmark Peak |
| :--- | :--- | :--- | :--- | :--- |
| **[`models/abide_3d_hierarchical_cnn_128_nyu.py`](models/abide_3d_hierarchical_cnn_128_nyu.py)** | **`128x128` (NYU Alone)** | `./processed_paper_3D/` | **[`preprocessing/abide_3d_preprocessing_128_nyu.py`](preprocessing/abide_3d_preprocessing_128_nyu.py)** | **`75.00%` Peak** 🌟 |
| **[`models/abide_3d_hierarchical_cnn_pytorch.py`](models/abide_3d_hierarchical_cnn_pytorch.py)** | **`224x224` (Multi-Site: NYU+UM_1+USM)** | `./processed_paper_3D_224/` | **[`preprocessing/abide_3d_preprocessing_224_multisite.py`](preprocessing/abide_3d_preprocessing_224_multisite.py)** | **`75.95%` Peak** 🔥 |
| **[`models/abide_3d_nyu_master.py`](models/abide_3d_nyu_master.py)** | **`224x224` PyTorch (NYU Master)** | `./processed_paper_3D_224/` | **[`preprocessing/abide_3d_preprocessing_224_multisite.py`](preprocessing/abide_3d_preprocessing_224_multisite.py)** | **`75.68%` Peak** 🎯 |

---

## 🚀 How to Run

### Option 1: 128x128 Single-Site NYU Pipeline
```bash
python preprocessing/abide_3d_preprocessing_128_nyu.py
python models/abide_3d_hierarchical_cnn_128_nyu.py
```

### Option 2: 224x224 Full HD Multi-Site PyTorch Pipeline (Peak 75.95%)
```bash
python preprocessing/abide_3d_preprocessing_224_multisite.py
python models/abide_3d_hierarchical_cnn_pytorch.py
```

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── data/
│   └── ABIDE_Phenotypic.csv                      # Official ABIDE-I phenotypic metadata table
├── preprocessing/
│   ├── abide_3d_preprocessing_224_multisite.py   # 224x224 Full HD 3-university pipeline (NYU+UM_1+USM)
│   └── abide_3d_preprocessing_128_nyu.py         # 128x128 single-site NYU pipeline
├── models/
│   ├── abide_3d_hierarchical_cnn_128_nyu.py      # Dedicated 128x128 NYU 3D Keras Model (inputs from ./processed_paper_3D/)
│   ├── abide_3d_hierarchical_cnn_pytorch.py      # Multi-site 224x224 PyTorch 3D Model (Peak: 75.95%)
│   └── abide_3d_nyu_master.py                    # NYU PyTorch Master Model
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md     # Complete research report & methodology
└── README.md                                     # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
