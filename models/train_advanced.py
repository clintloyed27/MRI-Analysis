import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_and_finetune_model(train_data, val_data, save_path="models/densenet_advanced_model.h5"):
    print("1. Loading DenseNet121 Architecture...")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze for warmup

    print("2. Building Advanced Classifier with Dropout...")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(train_data.num_classes, activation='softmax') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print("\n3. Phase 1: Warming up top layers (10 Epochs)...")
    history1 = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[early_stop])

    print("\n4. Phase 2: Unfreezing Base Model for Fine-Tuning...")
    base_model.trainable = True # Unfreeze everything!
    
    # Recompile with tiny learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\nStarting Phase 2 Training (15 Epochs)...")
    history2 = model.fit(train_data, validation_data=val_data, epochs=15, callbacks=[early_stop])

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nAdvanced Model Saved Successfully to: {save_path}")

    return model, history1, history2
