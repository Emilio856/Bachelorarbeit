# This script creates various folders with the images from the dataset and applies
# different tecniques to preprocess them.
#
# author: Emilio Rivera

import json
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# Replace with path to the cropped images
path_cropped = "K:\\Data\\2020-12-03\\Debayered_images_cropped"

# Replace with path to the augmented images
path_augmented = "K:\\Data\\2020-12-03\\debayered_images_cropped_augmented_180degree"

# Replace with the path to the labels from the cropped images
labels_path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master\\Bachelorarbeit-master\\Bilder_Labeln\\combined_labels.json"   # Replace with path to labels

# Replace with the path to the labels from the augmeted images
aug_labels_path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master\\Bachelorarbeit-master\\Bilder_Labeln\\labels_augmented_imgs.json"   # Replace with path to aug labels

new_path_cropped = "..\\Cropped"
new_path_aug = "..\\Augmented180"

harris_path_cropped = "..\\Cropped\\HarrisCorner"
harris_path_aug = "..\\Augmented180\\HarrisCorner"

clahe_path_cropped = "..\\Cropped\\CLAHE"
clahe_path_aug = "..\\Augmented180\\CLAHE"

clahe_grad_path_cropped = "..\\Cropped\\clahe_gradient"
clahe_grad_path_aug = "..\\Augmented180\\clahe_gradient"

clahe_eq_path_cropped = "..\\Cropped\\clahe_equalized"
clahe_eq_path_aug = "..\\Augmented180\\clahe_equalized"

normal_path_cropped = "..\\Cropped\\normal"
normal_path_aug = "..\\Augmented180\\normal"

folders = [
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

# Find and mark all corners found by the harris corner detector on the image
def harris_corner(path_to_img):

    img = cv2.imread(path_to_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # find corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    centroids = np.float32(centroids)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 0.01)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    i = 0
    while i < len(corners):
        if corners[i][0] <= 0.0 or corners[i][0] >= 450.0 or centroids[i][0] <= 0.0 or centroids[i][0] >= 450.0 or \
            corners[i][1] <= 0.0 or corners[i][1] >= 450.0 or centroids[i][1] <= 0.0 or centroids[i][1] >= 450.0:
            centroids = np.delete(centroids, i, 0)
            corners = np.delete(corners, i, 0)
        else:
            i += 1

    # draw corners
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:,3], res[:,2]] = [255, 0, 0]

    return img

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the images
def clahe(path_to_img):
    img = cv2.imread(path_to_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    cl1 = np.float32(cl1)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

    return cl1

# CLAHE with equalized light
def clahe_equalized(path_to_img):
    img = cv2.imread(path_to_img)
    img = equalize_light(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    cl1 = np.float32(cl1)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

    return cl1

# Normal image (no preprocessing)
def normal(path_to_img):
    img = cv2.imread(path_to_img)
    return img

# Equalize light on the image
def equalize_light(image, limit=3, grid=(7,7), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

# Apply a CLAHE gradient on the image
def clahe_gradient(path_to_img):
    img = cv2.imread(path_to_img)
    img = equalize_light(img, gray=True)
    
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    return gradient


print("Creating main directories...")
os.mkdir(new_path_cropped)
os.mkdir(new_path_aug)
os.mkdir(harris_path_cropped)
os.mkdir(harris_path_aug)
os.mkdir(clahe_path_cropped)
os.mkdir(clahe_path_aug)
os.mkdir(clahe_grad_path_cropped)
os.mkdir(clahe_grad_path_aug)
os.mkdir(clahe_eq_path_cropped)
os.mkdir(clahe_eq_path_aug)
os.mkdir(normal_path_cropped)
os.mkdir(normal_path_aug)


# For normal cropped images
print("Editing normal cropped images...")
for folder in folders:
    folder_path = os.path.join(harris_path_cropped, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_path_cropped, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_grad_path_cropped, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_eq_path_cropped, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(normal_path_cropped, folder)
    os.mkdir(folder_path)

with open(labels_path) as f:
    labels_json = json.load(f)

session = []
for i, session in enumerate(folders):
    print(session)
    imgs = sorted([
        img for img in os.listdir(
            os.path.join(path_cropped, session)
        )
        if ".png" in img
    ])
    
    for j, image in enumerate(imgs):
        if image not in labels_json[session]:
            continue
        elif labels_json[session][image] is None:
            continue
        else:
            harris_img = harris_corner(os.path.join(path_cropped, session, image))
            clahe_img = clahe(os.path.join(path_cropped, session, image))
            clahe_grad_img = clahe_gradient(os.path.join(path_cropped, session, image))
            clahe_eq_img = clahe_equalized(os.path.join(path_cropped, session, image))
            normal_img = normal(os.path.join(path_cropped, session, image))

            cv2.imwrite(os.path.join(harris_path_cropped, session, image), harris_img)
            cv2.imwrite(os.path.join(clahe_path_cropped, session, image), clahe_img)
            cv2.imwrite(os.path.join(clahe_grad_path_cropped, session, image), clahe_grad_img)
            cv2.imwrite(os.path.join(clahe_eq_path_cropped, session, image), clahe_eq_img)
            cv2.imwrite(os.path.join(normal_path_cropped, session, image), normal_img)


# For 180° augmented images
print("Editing 180° augmented images...")
for folder in folders:
    folder_path = os.path.join(harris_path_aug, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_path_aug, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_grad_path_aug, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(clahe_eq_path_aug, folder)
    os.mkdir(folder_path)
    folder_path = os.path.join(normal_path_aug, folder)
    os.mkdir(folder_path)

with open(aug_labels_path) as f:
    labels_json = json.load(f)

session = []
for i, session in enumerate(folders):
    print(session)
    imgs = sorted([
        img for img in os.listdir(
            os.path.join(path_augmented, session)
        )
        if ".png" in img
    ])

    for j, image in enumerate(imgs):
        if image not in labels_json[session]:
            continue
        elif labels_json[session][image] is None:
            continue
        else:
            harris_img = harris_corner(os.path.join(path_augmented, session, image))
            clahe_img = clahe(os.path.join(path_augmented, session, image))
            clahe_grad_img = clahe_gradient(os.path.join(path_augmented, session, image))
            clahe_eq_img = clahe_equalized(os.path.join(path_augmented, session, image))
            normal_img = normal(os.path.join(path_augmented, session, image))

            cv2.imwrite(os.path.join(harris_path_aug, session, image), harris_img)
            cv2.imwrite(os.path.join(clahe_path_aug, session, image), clahe_img)
            cv2.imwrite(os.path.join(clahe_grad_path_aug, session, image), clahe_grad_img)
            cv2.imwrite(os.path.join(clahe_eq_path_aug, session, image), clahe_eq_img)
            cv2.imwrite(os.path.join(normal_path_aug, session, image), normal_img)

print("Done")
