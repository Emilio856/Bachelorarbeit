# This program uses a pretrained neural network to inspect the pitch lengths
# of images in a dataset. The measured values are compared with the setpoint
# using a tolerance. The program outputs a .txt file containing the paths to
# all images where an error was detected and the size of the deviation.
#
# The paths to the neural network and the tolerance have to be specified below.
#
# author: Emilio Rivera

# for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import random as rn
rn.seed(42)

import csv
import joblib

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

def load_dataset(filename):
    """
    Instantiates a tf.Dataset object using a .TFRecord file.
    
    Args:
      filename: The name of the file.
    
    Returns:
      A dataset containing the images and labels from a file.
    """
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=20)
    return dataset

def get_dataset(filename):
    """
    Loads and prepares a dataset.

    Args:
      filename: The name of the file containing the data.

    Returns: A batched dataset.
    """
    dataset = load_dataset(filename)
    dataset = dataset.shuffle(len(list(dataset)), seed=rnd_seed)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(1)
    return dataset

# Set values for tolerance, path and path2.
tolerance = 10
normalizer = joblib.load("normalizer.save")
# path to the saved model
path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master\\Bachelorarbeit-master\\logs\\EffNetB1 2021_04_14 T 19-33-44\\EffNetB1"
# path to the folder of the saved model
path2 = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master\\Bachelorarbeit-master\\logs\\EffNetB1 2021_04_14 T 19-33-44"

model_name = path.split("\\")[-1]
model = tf.keras.models.load_model(path)

run_id = datetime.now().strftime(f"{model_name} %Y_%m_%d T %H-%M-%S")
test_dataset = get_dataset("test.tfrecords")
test_predictions = model.predict(test_dataset)
denormalized_preds = normalizer.inverse_transform(test_predictions)

y = np.concatenate([y for x, y in test_dataset], axis=0)
y = y.reshape(-1, 1)
y = normalizer.inverse_transform(y)

with open("img_names.csv", newline="") as f:
    reader = csv.reader(f)
    data = list(reader)

diff = list()
error_indices = list()
for i in range(len(y)):
    diff.append(y[i][0] - denormalized_preds[i][0])

for i in range(len(diff)):
    diff[i] = abs(diff[i])
    if diff[i] > tolerance:
        error_indices.append(i)

with open(f"errors located {run_id}.txt", "a") as f:
    f.write("Testevaluation mit insgesamt 1303 Bildern.\n")
    f.write(f"Es wurde eine Toleranz von {tolerance} Pixel betrachtet.\n")
    f.write("Alle Größen in Pixeln angegeben! \n\n\n")
    
    f.write(f"Im Folgenden sind {len(error_indices)} gefundene Abweichungen außerhalb des Toleranzbereichs aufgelistet:\n\n")
    for i in range(len(error_indices)):
        index = error_indices[i]
        f.write(f"Abweichung: {diff[index]}\n")
        f.write(f"Bild: {data[0][index]}\n\n")
