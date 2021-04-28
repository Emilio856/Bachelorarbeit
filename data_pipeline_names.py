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

from sklearn.preprocessing import MinMaxScaler
from matplotlib.image import imread


# path = "K:\\Data\\2020-12-03"
path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master"
img_type = "CLAHE"   # Replace!! -> CLAHE or HarrisCorner or normal or clahe_gradient
seed = 42
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

def encoder(row0, row1):
    return tf.train.Example(
        features=tf.train.Features(feature={
            "img": _bytes_feature(row0),
            "label": _float_feature(row1)
        })
    ).SerializeToString()


def load_img(address):
    img = cv2.imread(address)
    #img = img / 255.0
    #img = img.astype(np.float32)
    return img


def create_data_record(out_filename, paths, pick_lengths):
    # open TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(paths)):
        # print nr of saved images every 500 images
        if not i % 500:
            print(f"Train data: {i} / {len(paths)}")
            sys.stdout.flush()
        # img = imread(paths[i])
        image_string = open(paths[i], "rb").read()
        # img = img / 255.0
        label = pick_lengths[i]

        feature = {
            "img": _bytes_feature(image_string),   # img.encode("utf-8")   # img.tostring()
            "label": _float_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()



print(len(images))
print(len(labels))


all_data = list(zip(images, labels))
np.random.seed(42)
np.random.shuffle(all_data)
images, labels = zip(*all_data)

# 15% validation
test_imgs = images[int(0.85 * len(images)):]
test_labels = labels[int(0.85 * len(labels)):]
test_imgs = list(test_imgs)
print(len(test_imgs))
print(type(test_imgs))


with open("img_names.csv", 'w', newline='') as f:
     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
     wr.writerow(test_imgs)
