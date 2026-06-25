import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def generate_gradcam_heatmap(img_array, model):
    """
    Generates a Grad-CAM heatmap for a nested EfficientNet model.
    """
    # 1. Setup the Models for Grad-CAM
    base_model = model.layers[0]
    last_conv_layer_name = base_model.layers[-1].name

    # Model 1: Input to last conv layer
    last_conv_model = tf.keras.Model(base_model.inputs, base_model.get_layer(last_conv_layer_name).output)

    # Model 2: Last conv layer to Final Prediction
    classifier_input = tf.keras.Input(shape=last_conv_model.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # 2. Compute the Gradient
    with tf.GradientTape() as tape:
        last_conv_output = last_conv_model(img_array)
        tape.watch(last_conv_output)
        
        preds = classifier_model(last_conv_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_output)

    # 3. Process Gradients into Heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) # ReLU
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap) # Normalize

    return heatmap, top_pred_index

def plot_gradcam(img_array, heatmap, true_label, predicted_label):
    """
    Superimposes the heatmap onto the original image and plots them side-by-side.
    """
    img = img_array[0]
    img = np.uint8(255 * img) 

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + img

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original MRI\nTrue Diagnosis: {true_label}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tf.keras.utils.array_to_img(superimposed_img))
    plt.title(f"Grad-CAM Explanation\nAI Predicted: {predicted_label}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
