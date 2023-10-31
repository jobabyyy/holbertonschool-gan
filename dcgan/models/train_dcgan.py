#!/usr/bin/env python3
import tensorflow as tf
import time
import wandb
from ..models.discriminator import build_discriminator
from ..models.generator import build_generator
#import sys
import numpy as np
from ..data.data_loader import create_data_loaders


wandb.init(project='gans', entity='joeannaavila123', dir='/logs/')

input_shape = (28, 28, 1)

# Create instances of the generator and discriminator
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)

# Define loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

latent_dim = 100 

# Load images
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

batch_size = 64
train_data_gen = data_generator.flow_from_directory(
    'data/data_loader.py',
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True
)

# Convert the image dataset to a NumPy array
x_train = np.concatenate([train_data_gen.next() for _ in range(len(train_data_gen))])


num_epochs = 100

for epoch in range(num_epochs):
    for batch in create_data_loaders():
        start = time.time()

        # Generate random noise vectors
        noise = tf.random.normal([batch_size, latent_dim])

        # Generate fake images using the generator
        fake_images = generator(noise, training=True)

        # Discriminator loss on real data
        real_labels = tf.ones((batch_size, 1))
        real_loss = loss_fn(real_labels, discriminator(batch, training=True))

        # Discriminator loss on fake data
        fake_labels = tf.zeros((batch_size, 1))
        fake_loss = loss_fn(fake_labels, discriminator(fake_images, training=True))

        # Total discriminator loss
        discriminator_loss = real_loss + fake_loss

        # Generator loss
        generator_loss = loss_fn(real_labels, discriminator(fake_images, training=True))

        # wnb
        wandb.log({"generator_loss": generator_loss, "discriminator_loss": discriminator_loss, "Epoch ": epoch + 1, "Time ": time.time() - start})
        
        # Gradient tape for backpropagation
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator loss gradient
            gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)

            # Discriminator loss gradient
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        # Update generator and discriminator weights
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Save the trained generator model for generating new images
generator.save('dcgan_generator.h5')
#print(sys.path)
