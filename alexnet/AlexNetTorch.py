import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
import pickle
import csv
import time
import torchvision.models as models
from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize, plots, space
from skopt.utils import use_named_args
from NDStandardScaler import NDStandardScaler


# Input data
batch_size = 32
img_height = 600
img_width = 200
channels = 3


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Model from:
# https://arxiv.org/pdf/1404.5997.pdf
def init_alexnet(feature_extract=True):   # lr, hidden, drop
    # lr = 1 / np.power(10, lr)

    model_ft = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 1)
    input_size = (600, 200)
    return model_ft, input_size




NAME = f"Just a Testfile{int(time.time())}"
# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

run_id = datetime.now().strftime("Xception %Y_%m_%d T %H-%M-%S")
# logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//AlexNet//' + run_id   # TODO dir
logdir = os.getcwd() + "//" + run_id
# directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
os.chdir("..")
directory = os.getcwd()
os.mkdir(logdir)
logdir = logdir + "//"

x_data = list()
y_data = list()


labels = csv.reader(open("Aufnahmen Test//Labels2.csv"), "excel", delimiter=";")

next(labels, None)
y_data2 = list()
for row in labels:
    y_data2.append(float(row[1]))
numImg = len(y_data2)
counter = 0

for img in glob.glob("C://Users//emili//Desktop//Python//Bachelorarbeit Code//Aufnahmen Test//*.png"):   # TODO relative Pfad
# print(os.getcwd())
# print(os.path.abspath(os.curdir))
# os.chdir("../Aufnahmen Test")
# for img in os.getcwd() + "//*.png":
    x_data.append(image.img_to_array(cv2.imread(img)))
    y_data.append(y_data2[counter])
    counter += 1

# Convert images to numpy array and normalize them
x_data = np.asarray(x_data)
y_data = np.array(y_data)


scaler = NDStandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

imp = IterativeImputer()
y_data = np.array(y_data).reshape(-1, 1)

# MinMaxScaler
mnscaler = MinMaxScaler()
y_data = mnscaler.fit_transform(imp.fit_transform(y_data))


# Label log file
h = open(logdir + 'out.txt', 'a')
h.write('lr,drop,drop2,loss1,loss2,batch size,min loss\n')
h.close()




alexnet, input_size = init_alexnet()
print(alexnet)

print(torch.__version__)
print("YVBHNJ")

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)




"""

USE BoTorch for Bayesian Opt. Already installed on virt. env.


"""