#!/usr/bin/env python3
"""DCGANs:
Builds and
compiles generator model."""


"""import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape"""


import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose

def build_generator(input_shape, latent_dim=100):
    model = tf.keras.Sequential()

    # Start with a fully connected layer to transform noise into a suitable tensor shape
    model.add(Dense(7 * 7 * 128, input_shape=(latent_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Reshape into a 7x7x128 tensor
    model.add(Reshape((7, 7, 128)))

    # Upsample to 14x14x64 tensor
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Upsample to 28x28x1 tensor
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

    return model


"""def build_generator(input_shape, latent_dim=100):
    # Calculate the number of units needed for the initial Dense layer
    num_units = 7 * 7 * 128

    # Define the generator model
    generator = tf.keras.Sequential()

    # Add a fully connected layer with appropriate input shape
    generator.add(Dense(num_units, input_shape=input_shape))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))

    # Reshape to match the shape expected by convolutional layers
    generator.add(Reshape((7, 7, 128)))

    # Add convolutional layers with upsampling
    generator.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))

    # Output layer with tanh activation
    generator.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), activation='tanh', padding='same'))

    return generator"""

"""if __name__ == "__main__":
    input_shape = (100,)  # Input shape for generator

    # Build and compile the generator
    generator = build_generator(input_shape)
    generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    # Print the generator summary
    generator.summary()

    # Print information about the generator
    print("Generator Architecture:")
    generator.summary(print_fn=lambda x: print(x))

    # Verify the generator's input shape
    print(f"Generator Input Shape: {generator.input_shape}")
"""