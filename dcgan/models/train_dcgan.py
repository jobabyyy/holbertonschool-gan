#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
import time

wandb.init(project='gans', entity='joeannaavila123', dir='/logs/')

def generator():
    generator = keras.Sequential(
        [   keras.Input(shape=(latent_dim)),
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
        [   keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    print("Discriminator created with layers:")
    for layer in discriminator.layers:
        print(f"- {layer.name}")
    return discriminator

generator = generator()
discriminator()

discriminator.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False), 
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5) 
)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)

(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0


batch_size = 128
epochs = 30
save_interval = 10

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    start = time.time() # Start time for this epoch

    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real)

    # Logging the losses, epoch, and time to wandb
    wandb.log({"generator_loss": g_loss, "discriminator_loss": d_loss, "Epoch": epoch + 1, "Time": time.time() - start})

    if epoch % save_interval == 0:
        print(f"Epoch {epoch}: Discriminator loss: {d_loss}, Generator loss: {g_loss}")

