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


# Input data
batch_size = 32
img_height = 600
img_width = 200
channels = 3


# EfficientNet Models from:
# https://arxiv.org/pdf/1905.11946.pdf
def init_efficientnet_b0(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b0 = tf.keras.applications.EfficientNetB0(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b0.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b0_model = tf.keras.Model(inputs=base_b0.input, outputs=output_layer)

    b0_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b0_model


def init_efficientnet_b1(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b1 = tf.keras.applications.EfficientNetB1(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b1.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b1_model = tf.keras.Model(inputs=base_b1.input, outputs=output_layer)

    b1_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b1_model


def init_efficientnet_b2(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b2 = tf.keras.applications.EfficientNetB2(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b2.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b2_model = tf.keras.Model(inputs=base_b2.input, outputs=output_layer)

    b2_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b2_model


def init_efficientnet_b3(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b3 = tf.keras.applications.EfficientNetB3(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b3.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b3_model = tf.keras.Model(inputs=base_b3.input, outputs=output_layer)

    b3_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b3_model


def init_efficientnet_b4(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b4 = tf.keras.applications.EfficientNetB4(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b4.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b4_model = tf.keras.Model(inputs=base_b4.input, outputs=output_layer)

    b4_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b4_model


def init_efficientnet_b5(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b5 = tf.keras.applications.EfficientNetB5(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b5.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b5_model = tf.keras.Model(inputs=base_b5.input, outputs=output_layer)

    b5_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b5_model


def init_efficientnet_b6(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b6 = tf.keras.applications.EfficientNetB6(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b6.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b6_model = tf.keras.Model(inputs=base_b6.input, outputs=output_layer)

    b6_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b6_model


def init_efficientnet_b7(lr, hidden, drop):
    lr = 1 / np.power(10, lr)

    base_b7 = tf.keras.applications.EfficientNetB7(input_shape=(img_height, img_width, channels),
                                                   include_top=False, weights="imagenet")

    x = base_b7.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    b7_model = tf.keras.Model(inputs=base_b7.input, outputs=output_layer)

    b7_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                     ["mae", "accuracy"])
    return b7_model
    

search_space = [space.Real(2.5, 5.5, name='lr'),
                space.Integer(200, 1000, name='hidden'),
                space.Real(0, 0.7, name='drop'),
                space.Integer(1, 32, name='batch_size')]


#Funktion, die von gp_minimize aufgerufen wird. Enthält die fit Funktion
@use_named_args(search_space)
def evaluate_func(**kwargs):

    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=init_efficientnet_b0,epochs=2)
    model.set_params(**kwargs)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    y_train = np.random.random((10,1))
    y_test = np.random.random((5,1))


    #Fit model
    history = model.fit(x_train, y_train, validation_split=0.2,callbacks=[callb,callb2],shuffle=True)
    minloss = np.min(history.history['val_mae'][-11:-1])
	#logging des Ergebnis 
    f = open(logdir+'out.csv','a')
    f.write(str(model.sk_params['batch_size'])+','+str(minloss)+'\n')
    f.close()
    return minloss

#Plots und aktuellen Stand der Optmierung nach jedem Parametersatz speichern
def plotting(results):
    fig1 = plt.figure(1)
    ax1 = plt.axes()
    plots.plot_convergence(results)
    plt.savefig(logdir+'convergence.png',dpi=500)
    plt.close()
    fig2 = plt.figure(1)
    ax2 = plt.axes()
    plots.plot_evaluations(results)
    plt.savefig(logdir+'eval.png',dpi=500)
    plt.close()
	#plot_objective kann nur aufgerufen werden, sobald die Parameter berechnet und nicht mehr zufällig gewählt sind
    if len(results.models) > 1:
        fig3 = plt.figure(3)
        ax3 = plt.axes()
        plots.plot_objective(results)
        plt.savefig(logdir+'objective.png',dpi=500)
        plt.close()
    filename = logdir+'gp_min.sav'
    pickle.dump(results, open(filename, 'wb'))


run_id = datetime.now().strftime("EfficientNet B0 - B7 %Y_%m_%d T %H-%M-%S")
# logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//AlexNet//' + run_id   # TODO dir
os.chdir("..")
logdir = os.getcwd() + "//" + run_id
# directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
# os.chdir("..")
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

# Early Stopping
callb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='min',
                                         restore_best_weights=True)

# Uses Tensorboard to monitor training
callb2 = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=10, write_graph=False)

# Starts optimization
# n_calls are number of diferent parameter sets
result = gp_minimize(evaluate_func, search_space, n_calls=10, n_jobs=1, verbose=True, acq_func='gp_hedge',
                     acq_optimizer='auto', callback=[plotting])  # , callback=[tensorboard]

print("Best Accuracy: %  3f" % (1.0 - result.fun))