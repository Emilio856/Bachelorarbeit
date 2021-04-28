# Creates a csv file containing all the paths to the images in the test dataset
# in the same order as the .TFRecord file. The variable "path" has to be specified
# before using this script.
#
# author: Emilio Rivera


import csv
import os
import json
import random

import numpy as np


# Replace with the path to the folder where the folders with the cropped and augmented
# images are stored
path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master"

img_type = "CLAHE"
rnd_seed = 42
labels = list()
images = list()

"""
Data from augmented images
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
Data from normal images
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

# Only load images that have a label
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


all_data = list(zip(images, labels))
np.random.seed(rnd_seed)
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
