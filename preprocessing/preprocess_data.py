import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_noise_removal(image):
    """
    Custom preprocessing function for noise removal using OpenCV.
    Applies a Gaussian Blur to smooth the MRI image and remove high-frequency noise.
    """
    # The image comes in as a float32 array from Keras, OpenCV handles it perfectly.
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def create_data_generators(dataset_dir, batch_size=32, target_size=(224, 224)):
    """
    Creates training and validation data generators with required preprocessing.
    - Normalization (0-1)
    - Resize (224x224)
    - Noise Removal (GaussianBlur)
    - Data Augmentation (Rotation, Flip, Brightness)
    """
    
    # 1. Initialize the ImageDataGenerator with all augmentations and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,              # Normalize 0-255 -> 0-1
        preprocessing_function=apply_noise_removal, # Custom OpenCV Noise removal
        rotation_range=15,              # Rotation (+/- 15 degrees)
        horizontal_flip=True,           # Horizontal Flip
        brightness_range=[0.8, 1.2],    # Brightness adjustment
        validation_split=0.2            # Split 20% of data for validation testing
    )
    
    print("Loading Training Data...")
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,        # Automatically Resize to 224x224
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    print("\nLoading Validation Data...")
    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator

if __name__ == "__main__":
    # If running in Colab, the path will be:
    # DATASET_PATH = "/content/drive/MyDrive/MRI-Analysis/datasets/Training" (or wherever the tumor folders are)
    
    # For local testing, we'll use the local path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dataset_path = os.path.join(base_dir, 'datasets')
    
    # This will test the pipeline on your local folders (which are currently empty)
    train_gen, val_gen = create_data_generators(local_dataset_path)
    print("\nPreprocessing pipeline is ready!")
