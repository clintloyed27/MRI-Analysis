import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Configure the Webpage
st.set_page_config(page_title="MRI Analysis System", layout="wide", page_icon="🧠")

st.title("🧠 Structural MRI Analysis System")
st.markdown("Upload an MRI scan to instantly generate a diagnosis and an AI visual explanation.")

# Sidebar Configuration
st.sidebar.header("Analysis Configuration")
task_selection = st.sidebar.radio("Select Diagnostic Task:", ["Brain Tumor Detection", "Alzheimer's Screening"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- DenseNet121 (Tumor Model)")
st.sidebar.markdown("- EfficientNetB0 (Alzheimer's Model)")

# Define Class Labels
TUMOR_CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
ALZHEIMER_CLASSES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Smart Model Caching (So it doesn't reload the massive model on every click)
@st.cache_resource
def load_tumor_model():
    return tf.keras.models.load_model('models/densenet_mri_model_ROBUST.h5')

@st.cache_resource
def load_alzheimer_model():
    return tf.keras.models.load_model('models/efficientnet_alzheimers_model_ULTIMATE.h5')

model = None
classes = []

# Load the requested model dynamically
if task_selection == "Brain Tumor Detection":
    classes = TUMOR_CLASSES
    if os.path.exists('models/densenet_mri_model_ROBUST.h5'):
        model = load_tumor_model()
    else:
        st.error("⚠️ Tumor Model missing! Please download `densenet_mri_model_ROBUST.h5` into your `models/` folder.")
else:
    classes = ALZHEIMER_CLASSES
    if os.path.exists('models/efficientnet_alzheimers_model_ULTIMATE.h5'):
        model = load_alzheimer_model()
    else:
        st.error("⚠️ Alzheimer's Model missing! Please download `efficientnet_alzheimers_model_ULTIMATE.h5` into your `models/` folder.")

# --- GRAD-CAM FUNCTIONS ---
def generate_tumor_gradcam(img_array, model):
    # Tumor model is Functional (DenseNet121)
    last_conv_layer_name = "relu" 
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]
        
    heatmap = np.mean(last_conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
    return heatmap, preds[0]

def generate_alzheimer_gradcam(img_array, model):
    # Alzheimer's model is Nested Sequential (EfficientNetB0)
    base_model = model.layers[0]
    last_conv_layer_name = base_model.layers[-1].name
    last_conv_model = tf.keras.Model(base_model.inputs, base_model.get_layer(last_conv_layer_name).output)
    
    classifier_input = tf.keras.Input(shape=last_conv_model.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]: x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_output = last_conv_model(img_array)
        tape.watch(last_conv_output)
        preds = classifier_model(last_conv_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]
        
    heatmap = np.mean(last_conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap)
    return heatmap, preds[0]

# --- MAIN UI ---
uploaded_file = st.file_uploader("Select an MRI Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    # Process Image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    
    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)
        
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner(f"Analyzing for {task_selection}..."):
            
            if task_selection == "Brain Tumor Detection":
                # Tumors: Gaussian Blur + Rescale
                processed_img = cv2.GaussianBlur(img_resized, (5, 5), 0)
                input_tensor = np.expand_dims(processed_img / 255.0, axis=0)
                heatmap, preds = generate_tumor_gradcam(input_tensor, model)
            else:
                # Alzheimer's: No Blur, No Rescale
                input_tensor = np.expand_dims(img_resized, axis=0)
                heatmap, preds = generate_alzheimer_gradcam(input_tensor, model)
            
            # Extract Results
            top_pred_idx = np.argmax(preds)
            diagnosis = classes[top_pred_idx]
            confidence = preds[top_pred_idx] * 100
            
            # Generate Overlay
            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend original with heatmap
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Grad-CAM Explanation")
                st.image(overlay_rgb, use_container_width=True)
                
            st.success("Analysis Complete!")
            
            # Display Final Metrics
            st.markdown("---")
            st.markdown(f"### 🩺 **Diagnosis:** `{diagnosis}`")
            st.markdown(f"### 📊 **Confidence:** `{confidence:.2f}%`")
