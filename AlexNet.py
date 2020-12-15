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
from tensorflow import keras
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
from keras.callbacks import TensorBoard
from keras.preprocessing import image


# Input data
batch_size = 32
img_height = 600
img_width = 200
channels = 3


# Model architecture from:
# https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
def alexnet(lr, drop1, drop2, loss1, loss2):
    # Learning rate as exponent
    lr = 1 / np.power(10, lr)

    model = tf.keras.models.Sequential([
        # First Convolutional layer
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu",
                               input_shape=(img_height, img_width, channels), padding="same",
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

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.MeanSquaredError(),
                  metrics=["accuracy"])

    return model

    """model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", input_shape=(600, 200, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="softmax")
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    return model"""


# Search space for the hyperparameter optimization
search_space = [space.Real(2, 6, name='lr'),   # 1e-7, 1e-2   # Consider lr between 10^-6 and 1
                space.Real(0, 0.7, name='drop1'),
                space.Real(0, 0.7, name='drop2'),
                space.Real(0, 0.15, name='loss1'),
                space.Real(0, 0.15, name='loss2'),
                space.Integer(1, 32, name='batch_size')]


# Splits the data to create the test and validation sets and trains the model. Evaluates a given Configuration and
# creates a .txt file that stores the current values of the parameters after every iteration. Returns the minimal
# reached loss across all iterations.
@use_named_args(search_space)
def evaluate_func(**kwargs):
    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=alexnet, epochs=2)
    model.set_params(**kwargs)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    # x_train = np.random.random((10, 600, 200, 3))
    # x_test = np.random.random((5, 600, 200, 3))
    #y_train = np.random.random((10,))
    #y_test = np.random.random((5,))

    """y_test = tf.convert_to_tensor(y_test, np.float32)
    y_train = tf.convert_to_tensor(y_train, np.float32)
    x_test = tf.convert_to_tensor(x_test, np.float32)
    x_train = tf.convert_to_tensor(x_train, np.float32)"""

    fit_model = model.fit(x_train, y_train, validation_split=0.2, callbacks=[callb, callb2], shuffle=True)

    score = mean_squared_error(y_test, model.predict(x_test))
    print("score", score)

    # eval_model = model.evaluate(x_test, y_test)

    min_loss = np.min(fit_model.history['val_loss'][-21:-1])  # [-21:-1]         # history.history
    f = open(logdir + 'out.txt', 'a')
    f.write(str(model.sk_params['batch_size']) + ' ,' + str(min_loss) + '\n')
    f.close()
    return min_loss


# Plot and save current data status. Gets called after each completed parameter set
def plotting(results):
    fig1 = plt.figure(1)
    ax1 = plt.axes()
    plots.plot_convergence(results)
    plt.savefig(logdir + 'convergence.png', dpi=500)
    plt.close()
    fig2 = plt.figure(1)
    ax2 = plt.axes()
    plots.plot_evaluations(results)
    plt.savefig(logdir + 'eval.png', dpi=500)
    plt.close()
    # plot_objective can only be called if the parameters are being calculated instead of selected randomly
    if len(results.models) > 1:
        fig3 = plt.figure(3)
        ax3 = plt.axes()
        plots.plot_objective(results)
        plt.savefig(logdir + 'objective.png', dpi=500)
        plt.close()
    filename = logdir + 'gp_min.sav'
    pickle.dump(results, open(filename, 'wb'))


NAME = f"Just a Testfile{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

run_id = datetime.now().strftime("AlexNet %Y_%m_%d T %H-%M-%S")
# logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//AlexNet//' + run_id   # TODO dir
os.chdir("..")
logdir = os.getcwd() + "//" + run_id
# directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
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
