# model.py
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, Flatten,
    MaxPooling2D, Input, RandomFlip,
    RandomTranslation, RandomRotation, RandomZoom,
    GlobalAveragePooling2D)
from config import INPUT_SHAPE, NUM_CLASSES


def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        # Data Augmentation
        RandomFlip("horizontal"),
        RandomTranslation(0.1, 0.1),
        RandomRotation(0.1),
        RandomZoom(0.1),

        Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
        Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
