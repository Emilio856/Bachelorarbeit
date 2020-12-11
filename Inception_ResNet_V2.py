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


# Model from:
# https://arxiv.org/pdf/1602.07261.pdf
def init_inception_resnetv2(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    
    base_inception = tf.keras.applications.InceptionResNetV2(input_shape=(img_height, img_width, channels),
                                                             include_top=False, weights="imagenet")

    x = base_inception.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    inceptionresnet_model = tf.keras.Model(inputs=base_inception.input, outputs=output_layer)

    inceptionresnet_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(),
                                  ["mae", "accuracy"])
    return inceptionresnet_model


search_space = [space.Real(2.5, 5.5, name='lr'),
                space.Integer(200, 1000, name='hidden'),
                space.Real(0, 0.7, name='drop'),
                space.Integer(1, 32, name='batch_size')]


#Funktion, die von gp_minimize aufgerufen wird. Enthält die fit Funktion
@use_named_args(search_space)
def evaluate_func(**kwargs):

    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=init_inception_resnetv2,epochs=2)
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


NAME = f"Just a Testfile{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

run_id = datetime.now().strftime("Inception ResNet V2 %Y_%m_%d T %H-%M-%S")
# logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//AlexNet//' + run_id   # TODO dir
logdir = os.getcwd() + "//" + run_id
# directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
os.chdir("..")
directory = os.getcwd()
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