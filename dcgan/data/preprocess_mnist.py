#!/usr/bin/env python3
"""DCGANs:
Preprocessing the
MNIST dataset
using tensorflow"""


import tensorflow as tf

if __name__ == "__main__":
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()


def preprocess_mnist(x_train, x_test):
    # Preprocess the dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test

    # Preprocess and save the dataset
    x_train, x_test = preprocess_mnist(x_train, x_test)
