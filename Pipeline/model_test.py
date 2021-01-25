import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pathlib
import pickle
import csv
import time
import tensorflow as tf
import tensorflow.keras
from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from skopt import gp_minimize, plots, space
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental import preprocessing
sys.path.insert(1, "C:\\Users\\emili\\Desktop\\Python\\Bachelorarbeit Code\\CNNs")
from vgg16 import VGG16



model = VGG16(input_shape=(450, 450, 3), include_top=False, weights="imagenet")
print(model.summary())
print("Saving...")
model.save("C:\\Users\\emili\\Desktop\\Python\\Bachelorarbeit Code\\vgg16_model")
print("DONE")