# model.py
from keras.models import Sequential
from keras.layers import (
     Input, Conv2D, MaxPooling2D, Dense,
    RandomFlip, RandomTranslation, RandomRotation, RandomZoom,
    BatchNormalization, Dropout, GlobalAveragePooling2D)

from config import INPUT_SHAPE, NUM_CLASSES

#The model was improved by adding Batch Normalization and Dropout layers 
# and increasing convolutional depth, which stabilizes training, reduces 
# overfitting, and allows the network to learn more robust features for 
# better generalization on CIFAR-10.

def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),
# Data Augmentation
        RandomFlip("horizontal"),
        RandomTranslation(0.1, 0.1),
        RandomRotation(0.1),
        RandomZoom(0.1),
        
        
        Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.15),

        Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.15),

        Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.30),

        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
