#!/usr/bin/env python3
"""Baseline for DCGAN"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
import time
import matplotlib.pyplot as plt

wandb.init(project='gans', entity='joeannaavila123', dir='/logs/')
latent_dim = 50
batch_size = 128
epochs = 30
save_interval = 10
seed = tf.random.normal([16, latent_dim])


def generator():
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim)),
            layers.Dense(7 * 7 * 256),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", activation="tanh"),
        ],
        name="generator",
    )
    print("Generator created with layers:")
    for layer in generator.layers:
        print(f"- {layer.name}")
    return generator

def discriminator():
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    print("Discriminator created with layers:")
    for layer in discriminator.layers:
        print(f"- {layer.name}")

    return discriminator

def generate(model, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0  # Rescale to [0, 1]

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig("generated_image.png")

    # Log images to WandB
    wandb.log({"generated_images": [wandb.Image("generated_image.png")]})

generator = generator()
discriminator = discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

discriminator.compile(
    loss=cross_entropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan.compile(
    loss=cross_entropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)

(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = (x_train.reshape(-1, 28, 28, 1).astype("float32") - 127.5) / 127.5


real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
  noise = tf.random.normal([batch_size, latent_dim])

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

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
# train function
for epoch in range(epochs):
    start = time.time()
    gen_losses, disc_losses = [], []

    for image_batch in dataset:
        gen_loss, disc_loss = train_step(image_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)

    avg_gen_loss = sum(gen_losses) / len(gen_losses)
    avg_disc_loss = sum(disc_losses) / len(disc_losses)



    # Logging the losses, epoch, and time to wandb
    wandb.log({"generator_loss": avg_gen_loss, "discriminator_loss": avg_disc_loss, "Epoch": epoch + 1, "Time": time.time() - start})

    generate(generator, seed)

    print(f"Epoch {epoch + 1}: Discriminator loss: {avg_disc_loss}, Generator loss: {avg_gen_loss}")
