"""
Data Loading and Preprocessing
Assignment-compliant: No TensorFlow, No Keras backend
"""

import numpy as np
import os
import urllib.request


def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)


def load_mnist():
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def load_fashion_mnist():
    from keras.datasets import fashion_mnist

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


def preprocess(X, y):
    X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0

    y_onehot = np.zeros((y.size, 10))
    y_onehot[np.arange(y.size), y] = 1

    return X, y_onehot


def load_data(dataset="mnist"):

    if dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist()
    elif dataset in {"fashion", "fashion_mnist"}:
        X_train, y_train, X_test, y_test = load_fashion_mnist()
    else:
        raise ValueError("Invalid dataset")

    X_train, y_train = preprocess(X_train, y_train)
    X_test, y_test = preprocess(X_test, y_test)

    return X_train, y_train, X_test, y_test