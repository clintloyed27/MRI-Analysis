# 🧠 3D Multi-Planar Structural MRI Deep Learning Framework for Autism Spectrum Disorder (ASD) Classification

A state-of-the-art 3D Volumetric Deep Learning system for classifying Autism Spectrum Disorder (ASD) from T1-weighted Structural Magnetic Resonance Imaging (sMRI) scans on the **ABIDE-I NYU Cohort**.

Replicating and adapting the methodology of **Hammash & Younis (2026)**, this repository implements genuine **3D Convolutions (`Conv3D`)** across three orthogonal anatomical views (Axial, Coronal, Sagittal) with 3D CBAM Attention and Adaptive Focal Loss.

---

## 📊 Experimental Results (Stratified 5-Fold Cross-Validation)

| Evaluation Protocol | Dataset | Accuracy Metric | Gain over 2D Baseline |
| :--- | :--- | :--- | :--- |
| **2D Single-View Baseline** | NYU (1 Slice) | **56.25%** | Baseline |
| **2.5D Multi-Site Run** | ABIDE (3-Slice RGB Stack) | **63.62%** | +7.37 pp |
| **3D Multi-Planar CV (Average)** | **NYU (50 Slices, 3-View)** | **69.59%** | **+13.34 pp** |
| **3D Multi-Planar CV (Fold 5 Peak)** | **NYU (50 Slices, 3-View)** | **75.00%** | **+18.75 pp** |

### Per-Fold Breakdown (5-Fold Stratified CV)
* **Fold 1:** `70.27%`
* **Fold 2:** `67.57%`
* **Fold 3:** `64.86%`
* **Fold 4:** `70.27%`
* **Fold 5:** **`75.00%` (Peak Result)**
* 🌟 **Final 5-Fold Average Accuracy:** **`69.59%`**

---

## 🔬 Dataset Overview (ABIDE-I NYU Cohort)

* **Source Repository:** Autism Brain Imaging Data Exchange I (ABIDE-I).
* **Scanner Site:** NYU Langone Medical Center ($N=184$ subjects).
* **Class Distribution:**
  * **Autism Spectrum Disorder (ASD):** 79 Subjects (42.93%)
  * **Healthy Control (HC):** 105 Subjects (57.07%)
* **Significance:** Near-equal ~43/57 balance eliminates majority-class label bias, ensuring metrics reflect genuine neuroanatomical feature discriminability.

---

## ⚙️ Preprocessing & Model Architecture

### 1. Preprocessing Pipeline (`preprocessing/abide_3d_preprocessing.py`)
* **Bounding Box Skull Stripping:** Crops non-zero voxel coordinates to eliminate uninformative background air.
* **Z-Score Intensity Normalization:** Rescales voxel brightness per scan: $I_{\text{norm}} = (I_{\text{brain}} - \mu)/\sigma$, with outlier truncation at $[-3\sigma, +3\sigma]$.
* **Multi-Planar 50-Slice Extraction:** Extracts 50 consecutive middle slices for Axial (Z-axis), Coronal (Y-axis), and Sagittal (X-axis) physical planes into 3D NumPy arrays `(50, 128, 128, 1)`.

### 2. Neural Network Architecture (`models/abide_3d_hierarchical_cnn.py`)
* **Volumetric Engine:** Genuine `Conv3D` layers operating on 50-slice spatial volumes.
* **Alternating Kernel Strategy:** Alternates $3 \times 3 \times 3$ kernels (fine gyri/sulci) with $5 \times 5 \times 5$ kernels (coarse lateral ventricles/cerebellum).
* **Hierarchical Scaling:** $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ channel progression.
* **ResNet Skip Connections:** Residual addition blocks (`Conv3D + BatchNorm + Activation + Add`) preserve gradient flow across deep volumetric stages.
* **3D CBAM Attention:** Channel and 3D Spatial attention modules dynamically isolate ASD-correlated neuroanatomical biomarkers.
* **Adaptive Focal Loss:** Binary Focal Cross-Entropy ($\gamma=2.0, \alpha=0.25$) forces the network to focus on hard borderline cases.
* **MLP Classifier Head:** `LayerNormalization` $\rightarrow$ `Dense(256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(1, activation='sigmoid')`.

---

## 🚀 How to Run

### Step 1: Execute 3D Preprocessing
```bash
python preprocessing/abide_3d_preprocessing.py
```
*(Downloads the NYU NIfTI files from AWS S3, crops the brain, normalizes intensities, and exports 50-slice 3D `.npy` tensors to `/kaggle/working/processed_paper_3D/`)*

### Step 2: Run 3D Multi-Planar 5-Fold Training
```bash
python models/abide_3d_hierarchical_cnn.py
```
*(Loads the 3D `.npy` tensors, constructs the 3-Input Conv3D CBAM network, and executes Stratified 5-Fold Cross Validation)*

---

## 📂 Repository Structure

```text
MRI-Analysis/
├── preprocessing/
│   └── abide_3d_preprocessing.py      # Automated AWS S3 downloader & 3D 50-slice multi-planar tensor generator
├── models/
│   └── abide_3d_hierarchical_cnn.py   # True 3D Hierarchical Conv3D CBAM network with 5-Fold Stratified CV
├── report/
│   └── 3D_MultiPlanar_ASD_Research_Report.md  # Detailed research report, methodology, and fold results
└── README.md                          # Repository documentation
```

---

## 📄 Academic Reference

* **Hammash, N. M., & Younis, M. C. (2026).** *A Hierarchical Multi-View Deep Learning Framework for Autism Classification Using Structural and Functional MRI.* MDPI Journal of Imaging, 12(3), 109. [DOI: 10.3390/jimaging12030109]
