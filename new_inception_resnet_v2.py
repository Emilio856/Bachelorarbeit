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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize, plots, space
from skopt.utils import use_named_args
from NDStandardScaler import NDStandardScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image

# Model from:
# https://arxiv.org/pdf/1602.07261.pdf
def init_inception_resnetv2(hidden):
    
    base_inception = tf.keras.applications.InceptionResNetV2(input_shape=(450, 450, 3),
                                                             include_top=False, weights="imagenet")

    x = base_inception.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    inceptionresnet_model = tf.keras.Model(inputs=base_inception.input, outputs=output_layer)
    return inceptionresnet_model