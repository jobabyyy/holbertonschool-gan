#!/usr/bin/env python3
#baseline_discriminator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def discriminator():
    model = keras.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    print("Added Conv2D layer with 64 filters")
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Second convolutional layer
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    print("Added Conv2D layer with 128 filters")
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    print("Added Flatten layer")
    model.add(layers.Dense(1))
    print("Added Dense layer with 1 unit")

    return model

# Calling the function
discriminator = discriminator()
