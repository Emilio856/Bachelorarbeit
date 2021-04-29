# This program trains an AlexNet architecture to measure the length
# of the pitch in the center of an image. Early Stopping, Dropout
# and L1L2 Regularization are applied.
# 
# The results are stored in
# a .txt file after testing the trained network.
#
# author: Emilio Rivera

# for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import random as rn
rn.seed(42)

import matplotlib.pyplot as plt
import os
import gc

from datetime import datetime


device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], enable=True)
rnd_seed = 42


def decode_image(image):
    """
    Decodes, casts, normalizes and reshapes an encoded image into its original size.

    Args:
      image: An image stored as a sequence of binary records.
    
    Returns:
      A .png image.
    """
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [450, 450, 3])

    return image

def read_tfrecord(example):
    """
    Reads a .TFRecord file and extracts its information.

    Args:
      example: Name of the file.
    
    Returns:
      The images and labels from the file.
    """
    tfrecord_format = (
        {
            "img": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.float32)
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["img"])
    label = tf.cast(example["label"], tf.float32)
    return image, label

def load_dataset(filenames):
    """
    Instantiates a tf.Dataset object using a .TFRecord file.
    
    Args:
      filename: The name of the file.
    
    Returns:
      A dataset containing the images and labels from a file.
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=20)
    return dataset

def get_dataset(filenames):
    """
    Loads and prepares a dataset.

    Args:
      filename: The name of the file containing the data.

    Returns: A batched dataset.
    """
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(len(list(dataset)), seed=rnd_seed)
    dataset = dataset.batch(8, drop_remainder=True)
    return dataset

os.chdir(".")
print(os.getcwd())
gc.collect()
train_dataset = get_dataset("train.tfrecords")
val_dataset = get_dataset("val.tfrecords")
gc.collect()


# Unique id for the folder where the results are stored
run_id = datetime.now().strftime("AlexNet %Y_%m_%d T %H-%M-%S")
logdir = f"logs/{run_id}"

# Early Stopping
callback1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=200,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

# Uses Tensorboard to monitor training
callback2 = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    write_graph=True,
    write_images=True
)


# Model architecture from:
# https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
def make_model():
    """
    Creates an AlexNet model and compiles it with the Adam algorithm
    and the Mean Squared Error.

    Returns:
      An AlexNet model.
    """
    model = tf.keras.models.Sequential([
        # First Convolutional layer
        tf.keras.layers.Conv2D(filters=96,kernel_size=(11, 11), strides=(4, 4), activation="relu",
                               input_shape=(450, 450, 3), padding="same",
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Second Convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                               padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Third Convolutional layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.BatchNormalization(),

        # Fourth Convolutional layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.BatchNormalization(),

        # Fifth Convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Pass to Fully Connected layer
        tf.keras.layers.Flatten(),

        # First Fully Connected layer
        tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.Dropout(0.5, seed=rnd_seed),

        # Second Fully Connected layer
        tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed)),
        tf.keras.layers.Dropout(0.5, seed=rnd_seed),

        # Third Fully Connected
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed))
    ])
    model.compile(
        tf.keras.optimizers.Adam(
            lr=0.000001
        ),
        tf.keras.losses.MeanSquaredError(),
        ["accuracy"] 
    )
    return model


model = make_model()

# Add regularizer with l1 and l2 norm
model.trainable = True
regularizer = tf.keras.regularizers.l1_l2()
for layer in model.layers:
    for attr in ["kernel_regularizer"]:
        if hasattr(layer, attr):
            setattr(layer, attr, regularizer)

history = model.fit(
    train_dataset,
    epochs=450,
    validation_data=val_dataset,
    shuffle=False,
    callbacks=[callback1, callback2]
)


# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(os.path.join(logdir, "accuracy.png"), dpi=500)
plt.close()
# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(os.path.join(logdir, "loss.png"), dpi=500)


# Test the trained network
test_dataset = get_dataset("test.tfrecords")
score = model.evaluate(test_dataset)

# File containing detailed information of every step in the training process
with open(os.path.join(logdir, "evaluation.txt"), "a") as f:
    f.write(f"Final training results after {history.epoch[-1] + 1} epochs:\n")
    values = list(history.history.values())

    loss = values[0][-1]
    acc = values[1][-1]
    val_loss = values[2][-1]
    val_acc = values[3][-1]
    f.write(f"accuracy: {acc}\n")
    f.write(f"loss: {loss}\n")
    f.write(f"val_accuracy: {val_acc}\n")
    f.write(f"val_loss: {val_loss}\n\n")
    f.write(f"Test loss: {score[0]} / Test accuracy: {score[1]}\n\n\n")
    
    for i in range(len(values[0])):
        loss = values[0][i]
        acc = values[1][i]
        val_loss = values[2][i]
        val_acc = values[3][i]
        f.write(f"Epoch {i + 1}:\n")
        f.write(f"accuracy: {acc}\n")
        f.write(f"loss: {loss}\n")
        f.write(f"val_accuracy: {val_acc}\n")
        f.write(f"val_loss: {val_loss}\n\n")

model.save(os.path.join(logdir, "AlexNet"))

