#!/usr/bin/env python3
"""DCGANs:
Data Loader -
Creates a dataloader
for batch training."""


import tensorflow as tf

if __name__ == "__main__":
    batch_size = 64
    train_dataset = create_data_loaders(x_train, batch_size)

def create_data_loaders(x_train, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size)

    return train_dataset
