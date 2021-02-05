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
drop = 0.5
hidden = 512

# EfficientNet Models from:
# https://arxiv.org/pdf/1905.11946.pdf
def init_efficientnet_b0():

    base_b0 = tf.keras.applications.EfficientNetB0(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b0.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b0_model = tf.keras.Model(inputs=base_b0.input, outputs=output_layer)

    return b0_model


def init_efficientnet_b1():

    base_b1 = tf.keras.applications.EfficientNetB1(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b1.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b1_model = tf.keras.Model(inputs=base_b1.input, outputs=output_layer)

    return b1_model


def init_efficientnet_b2():

    base_b2 = tf.keras.applications.EfficientNetB2(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b2.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b2_model = tf.keras.Model(inputs=base_b2.input, outputs=output_layer)

    return b2_model


def init_efficientnet_b3():

    base_b3 = tf.keras.applications.EfficientNetB3(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b3.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b3_model = tf.keras.Model(inputs=base_b3.input, outputs=output_layer)

    return b3_model


def init_efficientnet_b4():

    base_b4 = tf.keras.applications.EfficientNetB4(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b4.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b4_model = tf.keras.Model(inputs=base_b4.input, outputs=output_layer)

    return b4_model


def init_efficientnet_b5():

    base_b5 = tf.keras.applications.EfficientNetB5(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b5.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b5_model = tf.keras.Model(inputs=base_b5.input, outputs=output_layer)

    return b5_model


def init_efficientnet_b6():

    base_b6 = tf.keras.applications.EfficientNetB6(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b6.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b6_model = tf.keras.Model(inputs=base_b6.input, outputs=output_layer)

    return b6_model


def init_efficientnet_b7():

    base_b7 = tf.keras.applications.EfficientNetB7(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b7.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b7_model = tf.keras.Model(inputs=base_b7.input, outputs=output_layer)

    return b7_model

"""
efficientnet_b0 = init_efficientnet_b0()
efficientnet_b1 = init_efficientnet_b1()
efficientnet_b2 = init_efficientnet_b2()
efficientnet_b3 = init_efficientnet_b3()"""
efficientnet_b4 = init_efficientnet_b4()
efficientnet_b5 = init_efficientnet_b5()
efficientnet_b6 = init_efficientnet_b6()
efficientnet_b7 = init_efficientnet_b7()


"""efficientnet_b0.save("efficientnet\\efficientnet_b0_model")
efficientnet_b1.save("efficientnet\\efficientnet_b1_model")
efficientnet_b2.save("efficientnet\\efficientnet_b2_model")
efficientnet_b3.save("efficientnet\\efficientnet_b3_model")"""
efficientnet_b4.save("efficientnet\\efficientnet_b4_model")
efficientnet_b5.save("efficientnet\\efficientnet_b5_model")
efficientnet_b6.save("efficientnet\\efficientnet_b6_model")
efficientnet_b7.save("efficientnet\\efficientnet_b7_model")