import tensorflow as tf
import json
import os
import cv2
import numpy as np
from glob import glob

def load_data(path):
    with open(os.path.join(path, "labels_augmented_imgs.json")) as f:
        labels = list()
        json_file = json.load(f)

    for folder in json_file:
        for img in json_file[folder]:
            if json_file[folder][img] != None:
                labels.append(json_file[folder][img][-1])

    # Only load images that have a label
    images = list()
    for subdir, dirs, files in os.walk(path):
        for file in files:
            folder = subdir.split("\\")[-1]
            if ".png" in file and file in json_file[folder] and json_file[folder][file] != None:
                images.append(os.path.join(subdir, file))
            else:
                pass

    # TODO: Se puede quitar sorted?
    images = sorted(images)
    return images, labels

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_label(path):

    index = images.index(path)
    label = labels[index]

    # label = labels[folder][img][-1]
    return label

def preprocess(x, y):
    def f(x, y):
        # x = x.decode()
        # y = y.decode()
        # TODO tf.image.decode_png
    
        x = read_image(x)
        y = read_label(x)

        return x, y
    
    images, labels = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([450, 450, 3])
    labels.set_shape(1,)

    return images, labels

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess(x, y))
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":
    path = "K:\\Data\\2020-12-03\\debayered_images_cropped_augmented_180degree"
    images, labels = load_data(path)
    print(f"Images: {len(images)} - Labels: {len(labels)}")

    dataset = tf_dataset(images, labels)
    for x, y in dataset:
        x = x * 255   # x = x[0] * 255
        y = y

        x = x.numpy()
        y = y.numpy()
