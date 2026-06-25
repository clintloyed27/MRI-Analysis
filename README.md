# 🧠 Structural MRI Analysis System
### Deep Learning Pipeline for Brain Tumor and Alzheimer's Detection

This repository contains an end-to-end Artificial Intelligence software system designed to analyze Magnetic Resonance Imaging (MRI) scans and raw hospital DICOM files to detect structural brain anomalies. 

---

## 🔬 Diseases Identified
This system features two highly specialized neural networks trained to identify distinct neurological conditions:

### 1. Brain Tumors (Macroscopic Anomalies)
The first model focuses on detecting large-scale abnormal tissue growth and classifies scans into four categories:
* **Glioma Tumor:** Tumors occurring in the brain and spinal cord.
* **Meningioma Tumor:** Tumors arising from the meninges (the membranes surrounding the brain).
* **Pituitary Tumor:** Tumors developing in the pituitary gland at the base of the brain.
* **No Tumor:** A perfectly healthy brain structure.

### 2. Alzheimer's Disease (Microscopic Anomalies)
The second model detects microscopic tissue shrinkage (atrophy), primarily focusing on the Hippocampus and medial temporal lobes. It classifies scans into four stages:
* **Very Mild Impairment:** The absolute earliest, almost imperceptible signs of tissue loss.
* **Mild Impairment:** Noticeable early-stage cognitive decline structure.
* **Moderate Impairment:** Significant, obvious tissue shrinkage.
* **No Impairment:** A perfectly healthy brain structure.

---

## 💻 Technology Stack & Justification

| Technology | Purpose | Technical Justification |
|------------|---------|-------------------------|
| **TensorFlow & Keras** | Core Deep Learning | The industry standard framework for building and training complex convolutional neural networks. |
| **EfficientNetB0** | Neural Architecture | We utilized a fully unfrozen EfficientNet architecture. Its advanced compound scaling method extracts incredibly fine, microscopic features without requiring massive computational overhead. This "Micro-Vision" was critical for detecting the highly subtle tissue changes in Very Mild Alzheimer's. |
| **Grad-CAM** | Explainable AI | Doctors cannot trust a "black box" AI. Gradient-weighted Class Activation Mapping mathematically traces the AI's logic backward to generate a glowing heatmap over the specific anatomical structures (e.g., the hippocampus) that triggered the diagnosis. |
| **Streamlit** | Web Dashboard | Allowed for the rapid deployment of a fully interactive, locally hosted web application to demonstrate the models in real-time. |
| **Pydicom** | Clinical Integration | MRI machines output raw 16-bit `.dcm` files, not standard JPEGs. Pydicom intercepts these raw hospital files, extracts hidden patient metadata, and normalizes the pixel data so the AI can ingest real-world clinical scans. |
| **Google Colab** | Cloud Computing | Bypassed local Mac hardware limitations by utilizing cloud GPUs for the millions of matrix multiplications required to train the neural networks. |

---

## ⚙️ Methodology & Pipeline
1. **Data Augmentation:** Used `ImageDataGenerator` to mathematically multiply the training data. We heavily relied on `horizontal_flip=True` to prevent "Sidedness Bias" (ensuring the AI learns the actual texture of the disease, rather than falsely memorizing that tumors only appear on the left or right side of the brain).
2. **Transfer Learning Optimization:** We leveraged pre-trained ImageNet weights, but explicitly unfroze the entire base model (`trainable = True`). Paired with a micro-learning rate (`1e-4`), this forced the network to rewrite its internal logic specifically for human neuroanatomy.
3. **Inference Engine:** The local Streamlit dashboard acts as a seamless inference engine. It intercepts the user's file, applies the exact mathematical preprocessing required for the selected disease, runs the neural network, and renders the predictions alongside the Grad-CAM overlay in milliseconds.
