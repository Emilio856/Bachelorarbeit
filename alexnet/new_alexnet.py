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

def alexnet(lr, loss1, loss2):

    model = tf.keras.models.Sequential([
        # First Convolutional layer
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu",
                               input_shape=(450, 450, 3), padding="same",
                               kernel_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               bias_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               activity_regularizer=tf.keras.regularizers.L1L2(loss1, loss2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Second Convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu",
                               padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               bias_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               activity_regularizer=tf.keras.regularizers.L1L2(loss1, loss2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Third Convolutional layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               bias_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               activity_regularizer=tf.keras.regularizers.L1L2(loss1, loss2)),
        tf.keras.layers.BatchNormalization(),

        # Fourth Convolutional layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               bias_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               activity_regularizer=tf.keras.regularizers.L1L2(loss1, loss2)),
        tf.keras.layers.BatchNormalization(),

        # Fifth Convolutional layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               bias_regularizer=tf.keras.regularizers.L1L2(loss1, loss2),
                               activity_regularizer=tf.keras.regularizers.L1L2(loss1, loss2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"),

        # Pass to Fully Connected layer
        tf.keras.layers.Flatten(),

        # First Fully Connected layer
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Second Fully Connected layer
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Third Fully Connected
        tf.keras.layers.Dense(1, activation="softmax")
    ])
    return model