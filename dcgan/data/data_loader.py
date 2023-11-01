#!/usr/bin/env python3
"""DCGANs:
Data Loader -
downloads MINST dataset
Preprocesses data
loads to bacth."""


import numpy as np
import tensorflow as tf
from tensorflow import keras

def download_mnist():

    print("Downloading MNIST dataset...")
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  
    print("Download complete!")

    return train_images, train_labels, test_images, test_labels

def preprocess_mnist(train_images, test_images):

    print("Preprocessing MNIST dataset...")
  
    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
  
    print("Preprocessing complete!")
  
    return train_images, test_images

def load_mnist(validation_split=0.1):

    print("Loading MNIST dataset...")
  
    train_images, train_labels, test_images, test_labels = download_mnist()
    train_images, test_images = preprocess_mnist(train_images, test_images)
    num_train = int((1-validation_split) * len(train_images))
    x_train, y_train = train_images[:num_train], train_labels[:num_train]
    x_val, y_val = train_images[num_train:], train_labels[num_train:]
  
    print(f"Training data: {len(x_train)} samples")
    print(f"Validation data: {len(x_val)} samples")
    print(f"Test data: {len(test_images)} samples")
  
    return (x_train, y_train), (x_val, y_val), (test_images, test_labels)
    
(x_train, y_train), (x_val, y_val), (test_images, test_labels) = load_mnist()
