# model.py
from keras.models import Sequential
from keras.layers import ( Dense, Conv2D, Flatten, MaxPooling2D, Input,
 RandomFlip, RandomTranslation, RandomRotation, RandomZoom )

from config import INPUT_SHAPE, NUM_CLASSES


def build_model():
    model = Sequential([
        Input(shape=INPUT_SHAPE),
# Data Augmentation
        RandomFlip("horizontal"),
        RandomTranslation(0.1, 0.1),
        RandomRotation(0.1),
        RandomZoom(0.1),
        
        
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
