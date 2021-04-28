# for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import random as rn
rn.seed(42)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os
import gc

from datetime import datetime


device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], enable=True)


def decode_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [450, 450, 3])

    return image

def read_tfrecord(example):
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
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=20)
    return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(len(list(dataset)), seed=42)
    dataset = dataset.batch(8, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(1)
    return dataset

os.chdir(".")
print(os.getcwd())
gc.collect()
train_dataset = get_dataset("train.tfrecords")
val_dataset = get_dataset("val.tfrecords")
gc.collect()


run_id = datetime.now().strftime("VGG16 %Y_%m_%d T %H-%M-%S")
logdir = f"logs/{run_id}"

# Early Stopping
callback1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=25,
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


def make_model():

    base_vgg = tf.keras.applications.VGG16(input_shape=(450, 450, 3), include_top=False, weights="imagenet")

    x = base_vgg.output
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(2048, activation="elu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(x)
    x = tf.keras.layers.Dropout(0.5, seed=42)(x)

    x = tf.keras.layers.Dense(2048, activation="elu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(x)
    x = tf.keras.layers.Dropout(0.5, seed=42)(x)
    
    output_layer = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(x)
    vgg16_model = tf.keras.Model(inputs=base_vgg.input, outputs=output_layer)
    vgg16_model.compile(
        tf.keras.optimizers.Adam(
            lr=0.000001
        ),
        tf.keras.losses.MeanSquaredError(),
        ["accuracy"]    
    )

    return vgg16_model


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
    epochs=500,
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

model.save(os.path.join(logdir, "VGG16"))

test_dataset = get_dataset("test.tfrecords")
score = model.evaluate(test_dataset)

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

