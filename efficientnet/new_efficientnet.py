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

img_height = 450
img_width = 450
channels = 3


# EfficientNet Models from:
# https://arxiv.org/pdf/1905.11946.pdf
def init_efficientnet_b0(lr, hidden, drop):

    base_b0 = tf.keras.applications.EfficientNetB0(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b0.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b0_model = tf.keras.Model(inputs=base_b0.input, outputs=output_layer)

    return b0_model


def init_efficientnet_b1(lr, hidden, drop):

    base_b1 = tf.keras.applications.EfficientNetB1(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b1.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b1_model = tf.keras.Model(inputs=base_b1.input, outputs=output_layer)

    return b1_model


def init_efficientnet_b2(lr, hidden, drop):

    base_b2 = tf.keras.applications.EfficientNetB2(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b2.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b2_model = tf.keras.Model(inputs=base_b2.input, outputs=output_layer)

    return b2_model


def init_efficientnet_b3(lr, hidden, drop):

    base_b3 = tf.keras.applications.EfficientNetB3(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b3.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b3_model = tf.keras.Model(inputs=base_b3.input, outputs=output_layer)

    return b3_model


def init_efficientnet_b4(lr, hidden, drop):

    base_b4 = tf.keras.applications.EfficientNetB4(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b4.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b4_model = tf.keras.Model(inputs=base_b4.input, outputs=output_layer)

    return b4_model


def init_efficientnet_b5(lr, hidden, drop):

    base_b5 = tf.keras.applications.EfficientNetB5(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b5.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b5_model = tf.keras.Model(inputs=base_b5.input, outputs=output_layer)

    return b5_model


def init_efficientnet_b6(lr, hidden, drop):

    base_b6 = tf.keras.applications.EfficientNetB6(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b6.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b6_model = tf.keras.Model(inputs=base_b6.input, outputs=output_layer)

    return b6_model


def init_efficientnet_b7(lr, hidden, drop):

    base_b7 = tf.keras.applications.EfficientNetB7(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b7.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b7_model = tf.keras.Model(inputs=base_b7.input, outputs=output_layer)

    return b7_model