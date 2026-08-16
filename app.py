import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
import os
import tempfile

st.set_page_config(page_title="Autism Detection MRI", layout="wide")
st.title("🧠 ABIDE Autism Detection & Explainability")
st.markdown("Upload a raw 3D `.nii.gz` structural MRI scan. The engine will extract the central slices, run them through the pre-trained DenseNet121 model, and generate a Grad-CAM heatmap explaining its clinical decision.")

@st.cache_resource
def load_medical_model():
    # Load the DenseNet model (compile=False because we only need it for inference)
    return tf.keras.models.load_model("DenseNet121_Autism.h5", compile=False)

try:
    model = load_medical_model()
    st.sidebar.success("✅ DenseNet121 Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error("❌ Model not found! Please ensure 'DenseNet121_Autism.h5' is downloaded from Google Drive and placed in this folder.")
    st.stop()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="relu"):
    """Generates the Grad-CAM Heatmap to explain the AI's decision."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_array, heatmap, alpha=0.4):
    """Overlays the heatmap onto the original MRI slice."""
    img = np.uint8(255 * img_array[0])
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img, 1-alpha, jet, alpha, 0)
    return superimposed_img

uploaded_file = st.file_uploader("Upload 3D NIfTI Scan (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    st.info("Extracting 2D Slices from 3D Volume...")
    
    # Save uploaded file temporarily so nibabel can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
        
    try:
        # Load 3D Volume
        scan = nib.load(tmp_path).get_fdata()
        cx, cy, cz = scan.shape[0]//2, scan.shape[1]//2, scan.shape[2]//2
        
        # Extract slices
        slices = {
            'Axial (Top-Down)': scan[:, :, cz],
            'Coronal (Front-Back)': scan[:, cy, :],
            'Sagittal (Side)': scan[cx, :, :]
        }
        
        st.subheader("1. Extracted Slices")
        cols = st.columns(3)
        processed_slices = []
        
        for i, (name, img_data) in enumerate(slices.items()):
            # Intensity Normalization
            if np.max(img_data) != 0:
                img_data = (img_data / np.max(img_data)) * 255.0
            img_data = img_data.astype(np.uint8)
            
            # Resize to DenseNet's exact 224x224 requirement and convert to RGB
            img_rgb = np.stack((img_data,)*3, axis=-1) 
            img_rgb = cv2.resize(img_rgb, (224, 224))
            processed_slices.append(img_rgb)
            
            cols[i].image(img_rgb, caption=name, use_container_width=True)
            
        st.subheader("2. AI Diagnosis & Explainability (Grad-CAM)")
        
        if st.button("Run DenseNet121 Analysis"):
            with st.spinner("Analyzing brain structures..."):
                # We analyze the Axial slice for the Grad-CAM demo
                input_tensor = np.expand_dims(processed_slices[0] / 255.0, axis=0)
                
                # Run DenseNet Inference
                prediction = model.predict(input_tensor)[0][0]
                confidence = prediction if prediction > 0.5 else 1 - prediction
                diagnosis = "Autism Spectrum (ASD)" if prediction > 0.5 else "Neurotypical (Control)"
                color = "red" if prediction > 0.5 else "green"
                
                st.markdown(f"### Diagnosis: <span style='color:{color}'>{diagnosis}</span> ({confidence*100:.1f}% Confidence)", unsafe_allow_html=True)
                
                try:
                    # Generate Grad-CAM Heatmap
                    # The final conv layer in DenseNet121 before pooling is usually 'relu'
                    heatmap = make_gradcam_heatmap(input_tensor, model, "relu")
                    overlay = display_gradcam(input_tensor, heatmap)
                    
                    gc_cols = st.columns(2)
                    gc_cols[0].image(processed_slices[0], caption="Original Axial Slice")
                    gc_cols[1].image(overlay, caption="Grad-CAM (Where the AI looked)")
                except Exception as e:
                    st.warning(f"Grad-CAM Error: {e}")
                    
    finally:
        os.remove(tmp_path)
