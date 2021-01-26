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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize, plots, space
from skopt.utils import use_named_args
from NDStandardScaler import NDStandardScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image

# Models from:
# https://arxiv.org/pdf/1603.05027.pdf
def init_resnet50v2(hidden):
    base_resnet50 = tf.keras.applications.ResNet50V2(input_shape=(450, 450, 3), include_top=False)
    x = base_resnet50.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet50v2 = tf.keras.Model(inputs=base_resnet50.input, outputs=output_layer)

    return resnet50v2


def init_resnet101v2(hidden):
    base_resnet101 = tf.keras.applications.ResNet101V2(input_shape=(450, 450, 3), include_top=False)
    x = base_resnet101.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet101v2 = tf.keras.Model(inputs=base_resnet101.input, outputs=output_layer)

    return resnet101v2


def init_resnet152v2(hidden):
    base_resnet152 = tf.keras.applications.ResNet152V2(input_shape=(600, 200, 3), include_top=False)
    x = base_resnet152.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet152v2 = tf.keras.Model(inputs=base_resnet152.input, outputs=output_layer)

    return resnet152v2