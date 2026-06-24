import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_efficientnet_model(train_data, val_data, save_path="models/efficientnet_mri_model.h5"):
    print("1. Loading Google's EfficientNetB0 Architecture...")
    # Swapped DenseNet for EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Keep completely frozen!

    print("2. Building Robust Top Classifier...")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        
        # Deep Learning Funnel
        Dense(512, activation='relu'),
        Dropout(0.5), 
        
        Dense(256, activation='relu'),
        Dropout(0.3), 
        
        Dense(train_data.num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )

    print("\n3. Starting EfficientNet Training (Up to 30 Epochs)...")
    history = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=30, 
        callbacks=[early_stop]
    )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nEfficientNet Model Saved Successfully to: {save_path}")

    return model, history
