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

import alexnet.new_alexnet as new_alexnet
import vgg.new_vgg as new_vgg
"""import efficientnet.new_efficientnet
import inception_resnet_v2.new_inception_resnet_v2
import inception.new_inception
import mobilenet_v3.new_mobilenet_v3
import resnet.new_resnet_v2
import xception.new_xception"""


available_models = [
    "alexnet",
    "vgg16",
    "vgg19"
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "inception_resnet_v2",
    "inception",
    "mobilenet_v3_small",
    "mobilenet_v3_large"
    "resnet50_v2",
    "resnet101_v2",
    "resnet152_v2",
    "xception"
]

# TODO Switch case
def get_model(model_name):
    if model_name in available_models:
        if model_name == "alexnet":
            model = new_alexnet.alexnet()
        elif model_name == "vgg16":
            model = new_vgg.create_vvg16()
        elif model_name == "vgg19":
            # model = new_vgg.create_vgg19()
            pass
        elif model_name == "efficientnet_b0":
            # model = new_efficientnet.init_efficientnet_b0()
            pass
        elif model_name == "efficientnet_b1":
            # model = new_efficientnet.init_efficientnet_b1()
            pass
        elif model_name == "efficientnet_b2":
            # model = new_efficientnet.init_efficientnet_b2()
            pass
        elif model_name == "efficientnet_b3":
            # model = new_efficientnet.init_efficientnet_b3()
            pass
        elif model_name == "efficientnet_b4":
            # model = new_efficientnet.init_efficientnet_b4()
            pass
        elif model_name == "efficientnet_b5":
            # model = new_efficientnet.init_efficientnet_b5()
            pass
        elif model_name == "efficientnet_b6":
            # model = new_efficientnet.init_efficientnet_b6()
            pass
        elif model_name == "efficientnet_b7":
            # model = new_efficientnet.init_efficientnet_b7()
            pass
        elif model_name == "inception_resnet_v2":
            # model = new_inception_resnet_v2.init_inception_resnetv2()
            pass
        elif model_name == "inception":
            # model = new_inception.init_inception_v3()
            pass
        elif model_name == "mobilenet_v3_small":
            # model = new_mobilenet_v3.init_mn_v3_small()
            pass
        elif model_name == "mobilenet_v3_large":
            # model = new_mobilenet_v3.init_mn_v3_large()
            pass
        elif model_name == "resnet50_v2":
            # model = new_resnet_v2.init_resnet50v2()
            pass
        elif model_name == "resnet101_v2":
            # model = new_resnet_v2.init_resnet101v2()
            pass
        elif model_name == "resnet152_v2":
            # model = new_resnet_v2.init_resnet152v2()
            pass
        elif model_name == "xception":
            # model = new_xception.init_xception()
            pass
        else:
            print("Error, spelling mistake in model name")
    else:
        print("Error, model is not implemented")

    return model