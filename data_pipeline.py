# This script creates a test, validation and training dataset as TFRecord files.
# Thus, a continuos connection to the location of the images and labels is not
# necessary. The data is scaled between 0 and 1 fitting a normalizer on the
# training data and then using it on the testing and validation data.
#
# The variable "path" has to be specified before using this script.
#
# author: Emilio Rivera


import os
import json
import sys
import random
import joblib

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# Replace with the path to the folder where the folders with the cropped and augmented
# images are stored
path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master"

img_type = "CLAHE"
rnd_seed = 42
labels = list()
images = list()
my_dict = dict()

"""
Data from augmented imgs
"""
# Open labels
with open(os.path.join(path,"Augmented180", "labels_augmented_imgs.json")) as f:
        json_file = json.load(f)

for folder in json_file:
        for img in json_file[folder]:
            if json_file[folder][img] != None:
                labels.append(json_file[folder][img][-1])

# Only load images that have a label
for subdir, dirs, files in os.walk(os.path.join(path, "Augmented180", img_type)):
    for file in files:
        folder = subdir.split("\\")[-1]
        if ".png" in file and file in json_file[folder] and json_file[folder][file] != None:
            images.append(os.path.join(subdir, file))
        else:
            pass

"""
Data from normal imgs
"""
# Open labels
with open(os.path.join(path, "Cropped", "combined_labels.json")) as f:
    json_file = json.load(f)

for folder in json_file:
    if "gerissen" in folder or "licht" in folder:
        pass
    else:
        for img in json_file[folder]:
            if json_file[folder][img] != None:
                labels.append(json_file[folder][img][-1])

# Only loads images that have a label
for subdir, dirs, files in os.walk(os.path.join(path, "Cropped", img_type)):
    if "gerissen" in subdir or "licht" in subdir or "weiß" in subdir or "papier" in subdir:
        pass
    else:
        for file in files:
            folder = subdir.split("\\")[-1]
            if ".png" in file and file in json_file[folder] and json_file[folder][file] != None:
                images.append(os.path.join(subdir, file))
            else:
                pass
                

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def create_data_record(out_filename, paths, pick_lengths):
    """
    Creates and stores a TFRecord file.

    Args:
      out_filename: The specified name for the file.
      paths: Paths to the images that will be stored in the file.
      pick_lengths: Labels belonging to the images that will be stored in the file.
    
    Returns:
      A TFRecord file containing several images and labels stored as a sequence of
      binary records.
    """
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(paths)):
        # print nr of saved images every 500 images
        if not i % 500:
            print(f"Train data: {i} / {len(paths)}")
            sys.stdout.flush()
        image_string = open(paths[i], "rb").read()
        label = pick_lengths[i]

        feature = {
            "img": _bytes_feature(image_string),
            "label": _float_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()

# Zip and shuffle the data
all_data = list(zip(images, labels))
np.random.seed(rnd_seed)
np.random.shuffle(all_data)
images, labels = zip(*all_data)

normalizer = MinMaxScaler()

# Training dataset containing 70% of the data
train_imgs = images[0:int(0.7 * len(images))]
train_labels = labels[0:int(0.7 * len(labels))]
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(-1, 1)
train_labels = normalizer.fit_transform(train_labels)
train_labels = train_labels.reshape(1, -1)
train_labels = train_labels[0].tolist()
print("train", len(train_imgs), len(train_labels))

# Validation dataset containing 15% of the data
val_imgs = images[int(0.7 * len(images)):int(0.85 * len(images))]
val_labels = labels[int(0.7 * len(labels)):int(0.85 * len(labels))]
val_labels = np.array(val_labels)
val_labels = val_labels.reshape(-1, 1)
val_labels = normalizer.transform(val_labels)
val_labels = val_labels.reshape(1, -1)
val_labels = val_labels[0].tolist()
print("val", len(val_imgs), len(val_labels))

# Testing dataset containing 15% of the data
test_imgs = images[int(0.85 * len(images)):]
test_labels = labels[int(0.85 * len(labels)):]
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1, 1)
test_labels = normalizer.transform(test_labels)
test_labels = test_labels.reshape(1, -1)
test_labels = test_labels[0].tolist()
print("test", len(test_imgs), len(test_labels))


create_data_record("train.tfrecords", train_imgs, train_labels)
create_data_record("val.tfrecords", val_imgs, val_labels)
create_data_record("test.tfrecords", test_imgs, test_labels)

# Save normalizer
joblib.dump(normalizer, "normalizer.save")