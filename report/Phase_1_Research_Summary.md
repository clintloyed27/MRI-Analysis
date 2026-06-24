# Phase 1 Research Summary: Structural MRI & Autism Spectrum Disorder (ASD)

## 1. What is MRI and Structural MRI (sMRI)?
**Magnetic Resonance Imaging (MRI)** is a non-invasive medical imaging technique that uses strong magnetic fields and radio waves to generate detailed images of the organs and tissues within the body. Unlike X-rays or CT scans, MRI does not use ionizing radiation, making it extremely safe for neurological studies.

**Structural MRI (sMRI)** is a specific type of MRI used to examine the static 3D anatomy of the brain. It provides high-contrast, high-resolution images of brain tissues, allowing researchers to measure the size, shape, and integrity of brain structures. (This is distinct from *functional* MRI (fMRI), which measures brain activity over time by detecting changes in blood flow).

## 2. Brain Anatomy Basics
To analyze sMRI images, it is crucial to understand the three primary tissue types visible in the brain:
* **Gray Matter (GM):** Consists mostly of neuronal cell bodies. This is where processing is done. It forms the outer layer of the brain (the cerebral cortex) and deep structures.
* **White Matter (WM):** Consists of myelinated axons, acting as the communication network connecting different parts of the gray matter to each other.
* **Cerebrospinal Fluid (CSF):** The clear fluid surrounding the brain and spinal cord, providing mechanical cushioning and clearing waste.

## 3. Autism Spectrum Disorder (ASD) Basics
**ASD** is a complex neurodevelopmental condition characterized by challenges in social interaction, verbal and nonverbal communication, and restricted/repetitive behaviors. The exact etiology is unknown, but it is widely accepted to be a combination of genetic and environmental factors that affect early brain development.

### What changes occur in ASD brains?
Research using sMRI has found several structural differences in the brains of individuals with ASD compared to neurotypical individuals:
* **Early Brain Overgrowth:** Children with ASD often show an accelerated rate of brain growth early in life (ages 1-3), followed by an arrested growth phase. This leads to increased overall brain volume and abnormal cortical thickness in early childhood.
* **Altered Gray/White Matter Ratios:** Differences in the volumes of specific gray matter and white matter tracts are common.
* **Atypical Connectivity:** While sMRI primarily shows structure, the structural differences in white matter tracts suggest abnormal connectivity (often local over-connectivity and long-range under-connectivity).

### Which brain regions are important?
When developing a Deep Learning model for ASD classification, the model will likely look for subtle structural changes in these key regions:
* **Amygdala:** Involved in emotion regulation and social behavior. It is often found to be enlarged in young children with ASD.
* **Corpus Callosum:** The main bundle of white matter connecting the two hemispheres. Reduced volume here is frequently observed, impacting interhemispheric communication.
* **Frontal & Temporal Lobes:** Areas critical for language, social cognition, and executive function. Cortical thickness anomalies are often found here.
* **Cerebellum:** Traditionally associated with motor control, but now known to play a role in cognition; abnormalities in Purkinje cells and cerebellar volume are common in ASD.

## 4. Why Use MRI for ASD?
* **Objective Biomarkers:** ASD diagnosis is currently behavioral and clinical. MRI provides objective, quantifiable biomarkers that can aid in earlier and more accurate diagnosis.
* **Subtle Pattern Detection:** Deep Learning can detect complex, non-linear structural patterns across multiple brain regions that are invisible to the naked eye of a clinician.
* **Non-invasive:** Safe for infants and children, allowing longitudinal studies of brain development.

## 5. MRI File Formats
When working with medical imaging datasets, you will encounter specific file formats rather than standard image files (like JPG or PNG):
* **DICOM (.dcm):** The standard format used in hospitals. It contains the image data plus extensive metadata (patient info, scanner settings). DICOMs are often bulky and complex to process directly in ML pipelines.
* **NIfTI (.nii or .nii.gz):** The standard format for neuroimaging research and Deep Learning. It combines the slices of a 3D scan into a single file and strips away sensitive patient metadata. Most Python libraries (like `NiBabel`) are designed to read NIfTI files easily, representing them as 3D Numpy arrays.

---
**Next Steps for Deep Learning:**
Our model will take 3D NIfTI files as input, extract these subtle volumetric and structural features using Convolutional Neural Networks (CNNs), and output a probability of the scan belonging to the ASD or Neurotypical class.
