import csv
import os
import json
import time
import sys
import cv2
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from retry import retry


# path = "K:\\Data\\2020-12-03"
path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master"
img_type = "HarrisCorner"   # Replace!! -> CLAHE or HarrisCorner
seed = 42

"""
Data from augmented imgs
"""
# Open labels
with open(os.path.join(path,"Augmented180", "labels_augmented_imgs.json")) as f:
        labels = list()
        json_file = json.load(f)

for folder in json_file:
        for img in json_file[folder]:
            if json_file[folder][img] != None:
                labels.append(json_file[folder][img][-1])

# Only load images that have a label
images = list()
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









df = pd.DataFrame({"img_path": images, "label": labels})

# Write sample data
df.to_csv("./sample.csv", index=False)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def encoder(row0, row1):
    return tf.train.Example(
        features=tf.train.Features(feature={
            "img_path": _bytes_feature(row0),
            "label": _float_feature(row1)
        })
    ).SerializeToString()

# Iterate over data and generate serializable data
def read_csv(file_path="./sample.csv", skip_rows=1):
    with open("./sample.csv", "r") as csv_file:
        data = csv.reader(csv_file, delimiter=",")
        for index, row in enumerate(data):
            if index < skip_rows:
                continue
            yield encoder(
                row0=row[0].encode("utf-8"),
                row1=float(row[1])
            )

def load_img(address):
    img = cv2.imread(address)
    return img

# TFRecord writer for encoded data
def tf_record_writer(in_file="./sample.csv", out_file="./sample.tfrecords"):
    with tf.io.TFRecordWriter(out_file) as writer:
        for record in read_csv(file_path=in_file):
            writer.write(record)

def create_data_record(out_filename, paths, pick_lengths):
    # open TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(paths)):
        # print nr of saved images every 500 images
        if not i % 500:
            print(f"Train data: {i} / {len(paths)}")
            sys.stdout.flush()
        img = load_img(paths[i])
        label = pick_lengths[i]

        feature = {
            "img_path": _bytes_feature(img.tostring()),
            "label": _float_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()


# tf_record_writer()

@retry(Exception, delay=5, tries=100)
def img_decoder(img_path, label):
    read_img = tf.io.read_file(img_path)
    img = tf.image.decode_png(read_img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    return img, label

"""dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Shuffle and repeat with buffer size equal to length of dataset -> ensures good shuffling
dataset = dataset.shuffle(len(images), seed=seed)

dataset = dataset.map(img_decoder, num_parallel_calls=4)
#dataset = dataset.batch(16, drop_remainder=True)
#dataset = dataset.prefetch(1)

# TODO .cache() ??

#train_size = round(0.7 * len(dataset))
#train = dataset.take(train_size)
#test = dataset.skip(train_size)




ddd = get_dataset()
train_size = round(0.7 * len(ddd))
train = ddd.take(train_size)
test = ddd.skip(train_size)
print("Done")"""

print(len(images))
print(len(labels))

# labels = np.random.random((8683, 1))

all_data = list(zip(images, labels))
np.random.seed(42)
np.random.shuffle(all_data)
images, labels = zip(*all_data)

# 70% training
train_imgs = images[0:int(0.7 * len(images))]
train_labels = labels[0:int(0.7 * len(labels))]
print("train", len(train_imgs), len(train_labels))

# 15% testing
val_imgs = images[int(0.7 * len(images)):int(0.85 * len(images))]
val_labels = labels[int(0.7 * len(labels)):int(0.85 * len(labels))]
print("val", len(val_imgs), len(val_labels))

# 15% validation
test_imgs = images[int(0.85 * len(images)):]
test_labels = labels[int(0.85 * len(labels)):]
print("test", len(test_imgs), len(test_labels))

create_data_record("train.tfrecords", train_imgs, train_labels)
create_data_record("val.tfrecords", val_imgs, val_labels)
create_data_record("test.tfrecords", test_imgs, test_labels)
