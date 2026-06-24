import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_robust_model(train_data, val_data, save_path="models/densenet_robust_model.h5"):
    print("1. Loading DenseNet121 Architecture...")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Keep completely frozen!

    print("2. Building Robust Top Classifier...")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        
        # New Deep Learning Funnel
        Dense(512, activation='relu'),
        Dropout(0.5), # 50% Dropout
        
        Dense(256, activation='relu'),
        Dropout(0.3), # 30% Dropout
        
        Dense(train_data.num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early Stopping safeguard
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )

    print("\n3. Starting Robust Training (Up to 30 Epochs)...")
    history = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=30, 
        callbacks=[early_stop]
    )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nRobust Model Saved Successfully to: {save_path}")

    return model, history
