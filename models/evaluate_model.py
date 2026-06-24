import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2
import os

def apply_noise_removal(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def evaluate_mri_model(model_path, test_data_path):
    # Load Model
    print("Loading Trained Model...")
    model = load_model(model_path)

    # Prepare Test Data (NO Augmentation, NO Shuffle)
    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=apply_noise_removal
    )

    print("Loading Testing Dataset...")
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False # CRITICAL for evaluation
    )

    # Get Predictions
    print("\nRunning AI Predictions on Test Data...")
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Generate Metrics Report
    print("\n=== PERFORMANCE REPORT ===")
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Generate Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Tumor Classification Confusion Matrix')
    plt.ylabel('True MRI Diagnosis')
    plt.xlabel('AI Predicted Diagnosis')
    plt.show()

# If running locally:
# evaluate_mri_model("models/densenet_mri_model.h5", "datasets/Testing")
