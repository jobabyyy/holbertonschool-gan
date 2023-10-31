#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model

def build_discriminator(input_shape):
    # Define the discriminator model
    discriminator_input = Input(shape=input_shape)
    
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(discriminator_input)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs=discriminator_input, outputs=x)

    return discriminator

if __name__ == "__main__":
    input_shape = (28, 28, 1)  # Input shape for discriminator (assuming MNIST image shape)

    # Build the discriminator
    discriminator = build_discriminator(input_shape)
    
    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    # Print the discriminator summary
    discriminator.summary()

    # Print information about the discriminator
    print("Discriminator Architecture:")
    discriminator.summary(print_fn=lambda x: print(x))

    # Verify the discriminator's input shape
    print(f"Discriminator Input Shape: {discriminator.input_shape}")


