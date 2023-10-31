#!/usr/bin/env python3
"""Script to init data
from MNIST dataset
and saves in numpy array"""


import tensorflow as tf

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = download_mnist()


def download_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    return x_train, y_train, x_test, y_test
