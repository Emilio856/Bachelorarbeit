# Augments images from given folders by turning them 180 degrees. 
#
# author: Emilio Rivera

import json
import os
import sys
from PIL import Image

path = "K:\\Data\\2020-12-03\\Debayered_images_cropped"   # Replace with path to Drive
labels_path = "C:\\Users\\emili\\Desktop\\Python\\data\\combined_labels.json"   # Replace with path to labels
new_folder_path = "..\\AugmentedImgs180Degrees"   # Can be replaced
folders_to_augment = [
    "iteration_0_30grad",
    "iteration_0_40grad",
    "iteration_0_50grad",
    "iteration_0_60grad",
    "iteration_0_70grad",
    "iteration_30grad",
    "iteration_30grad_2",
    "iteration_30grad_3",
    "iteration_30grad_4",
    "iteration_40grad",
    "iteration_40grad_2",
    "iteration_40grad_3",
    "iteration_40grad_4",
    "iteration_50grad",
    "iteration_50grad_1",
    "iteration_50grad_2",
    "iteration_60grad",
    "iteration_60grad_2",
    "iteration_60grad_4",
    "iteration_70grad",
    "iteration_70grad_1",
    "iteration_70grad_2",
    "iteration_70grad_3"
]

os.mkdir(new_folder_path)
for folder in folders_to_augment:
    folder_path = os.path.join(new_folder_path, folder)
    os.mkdir(folder_path)

with open(labels_path) as f:
    labels = json.load(f)

session = []

for i, session in enumerate(folders_to_augment):
    imgs = sorted([
        img for img in os.listdir(
            os.path.join(path, session)
        )
        if ".png" in img
    ])

    for j, image in enumerate(imgs):
        if image not in labels[session]:
            continue
        elif labels[session][image] is None:
            continue
        else:
            print(labels[session][image])
            image_to_rotate = Image.open(os.path.join(path, session, image))
            rotated = image_to_rotate.rotate(180)
            rotated = rotated.save(os.path.join(new_folder_path, session, image))


