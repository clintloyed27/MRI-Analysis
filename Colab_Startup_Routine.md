# Google Colab Startup Routine

Every time you open your Google Colab notebook after closing it or disconnecting, Google gives you a fresh, blank computer. 

You MUST run these three steps before you can do any training or analysis.

### 1. Connect your Google Drive
Run this in a code cell to allow Colab to see your `MRI-Analysis` folder.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Required Software
Run this in a code cell to install the medical imaging and computer vision libraries.
```python
!pip install nibabel opencv-python monai
```

### 3. Load the MRI Images into Memory
Run your preprocessing pipeline to load the images from your Google Drive into the temporary Colab RAM.
```python
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_noise_removal(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    preprocessing_function=apply_noise_removal,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Remember to ensure this path points to your actual dataset!
DATASET_PATH = "/content/drive/MyDrive/MRI-Analysis/datasets/Training" 

print("Loading Training Dataset...")
train_data = datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training'
)

print("\nLoading Validation Dataset...")
val_data = datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)
```

Once you run these three blocks, your Colab environment is fully restored and ready to do more AI training!
