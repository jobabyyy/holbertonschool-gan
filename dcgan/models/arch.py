#!/usr/bin/env python3
"""Architecture modification(s)
for DCGAN baseline."""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
import time
import matplotlib.pyplot as plt
import os 

wandb.init(project='gans', entity='joeannaavila123', dir='/logs/')

# Config Variables
noise = 100
noise_dim = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 128 # increased from 100 to 128
EPOCHS = 50 # updated from 30 to 50
save_interval = 10
learning_rate = 1e-4
num_examples_to_generate = 16


# Load And Size Data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Gen Model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Discriminate
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# Create Objects Of Our Functions
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)


discriminator = make_discriminator_model()
decision = discriminator(generated_image)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# Loss Tracking
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Optimizer Functions
generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



# Modify the save path in the generate_and_log_images function
def generate_and_log_images(model, test_input):
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig("generated_image.png")

    # Log images to WandB
    wandb.log({"generated_images": [wandb.Image("generated_image.png")]})


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        gen_losses, disc_losses = [], []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        avg_gen_loss = sum(gen_losses) / len(gen_losses)
        avg_disc_loss = sum(disc_losses) / len(disc_losses)

        print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}, Time: {}'.format(epoch + 1, avg_gen_loss, avg_disc_loss, time.time() - start))

        # Produce images for WandB
        generate_and_log_images(generator, seed)

        # Log generator and discriminator loss to WandB
        wandb.log({"generator_loss": avg_gen_loss, "discriminator_loss": avg_disc_loss, "Epoch ": epoch + 1, "Time ": time.time() - start})


# Trains Model
train(train_dataset, EPOCHS)

