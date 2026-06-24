import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

def run_silver_bullet_finetuning(train_data, val_data, model_path="models/densenet_mri_model_ROBUST.h5", save_path="models/densenet_mri_model_ULTIMATE.h5"):
    print("1. Loading Robust Model...")
    model = load_model(model_path)

    print("2. Unlocking the Top 30 Layers of DenseNet...")
    # The first layer in our Sequential model is the massive DenseNet121 base
    base_model = model.layers[0] 
    base_model.trainable = True

    # Freeze everything EXCEPT the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Check how many layers are trainable now
    trainable_count = sum([1 for w in model.trainable_weights])
    print(f"Total Trainable Weight Tensors: {trainable_count}")

    print("\n3. Recompiling with Microscopic Learning Rate...")
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )

    print("\n4. Starting Partial Fine-Tuning (Up to 20 Epochs)...")
    history = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=20, 
        callbacks=[early_stop]
    )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nUltimate Model Saved Successfully to: {save_path}")

    return model, history
