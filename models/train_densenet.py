import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

def build_and_train_densenet(train_data, val_data, epochs=20, save_path="models/densenet_model.h5"):
    print("1. Loading DenseNet121 Architecture...")
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    print("2. Building Custom MRI Classifier...")
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(train_data.num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print(f"\n3. Starting Training for {epochs} Epochs...")
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel Saved Successfully to: {save_path}")

    return model, history

# Note: In a full local run, this script would be called by a main script that first generates train_data and val_data.
