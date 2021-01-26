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
import tensorflow as tf
import tensorflow.keras
from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from skopt import gp_minimize, plots, space
from skopt.utils import use_named_args
from NDStandardScaler import NDStandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental import preprocessing


# Model from:
# https://arxiv.org/pdf/1512.00567v3.pdf
def init_inception_v3(hidden):
    lr = 1 / np.power(10, lr)

    base_inception = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet",
                                                       input_shape=(450, 450, 3))
    
    x = base_inception.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    inception_v3_model = tf.keras.Model(inputs=base_inception, outputs=output_layer)

    return inception_v3_model