from keras.datasets import cifar10
import numpy as np


def load_cifar10():
    """
    Load CIFAR-10 dataset from Keras.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x_* shape: (N, 32, 32, 3), dtype uint8
        - y_* shape: (N, 1)
    """
    return cifar10.load_data()


def preprocess_cifar10(x, y):
    """
    Preprocess CIFAR-10 data:
    - Normalize images to [0, 1]
    - Flatten labels to shape (N,)

    Args:
        x: numpy array of images
        y: numpy array of labels

    Returns:
        x_processed, y_processed
    """
    x = x.astype("float32") / 255.0
    y = y.squeeze()

    return x, y


def get_cifar10_data():
    """
    Full CIFAR-10 pipeline:
    load → preprocess

    Returns:
        x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    x_train, y_train = preprocess_cifar10(x_train, y_train)
    x_test, y_test = preprocess_cifar10(x_test, y_test)

    return x_train, y_train, x_test, y_test
