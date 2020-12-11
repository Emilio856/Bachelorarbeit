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



NAME = f"Just a Testfile{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")


x_data = list()
y_data = list()

labels = csv.reader(open("C://Users//emili//Desktop//Python//Bachelorarbeit Code//Aufnahmen Test//Labels2.csv"),"excel", delimiter=";")
next(labels, None)
y_data2 = list()
for row in labels:
    y_data2.append(float(row[1]))
numImg = len(y_data2)
counter = 0
for img in glob.glob("C://Users//emili//Desktop//Python//Bachelorarbeit Code//Aufnahmen Test//*.png"):
    x_data.append(image.img_to_array(cv2.imread(img)))
    y_data.append(y_data2[counter])
    counter += 1

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
x_data = np.asarray(x_data)
scaler = NDStandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)


# Models from:
# https://arxiv.org/pdf/1409.1556.pdf
def init_vgg16(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    base_vgg16 = tf.keras.applications.VGG16(input_shape=(600, 200, 3), include_top=False, weights="imagenet")
    x = base_vgg16.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    vgg16_model = tf.keras.Model(inputs=base_vgg16.input, outputs=output_layer)

    vgg16_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])
    return vgg16_model

def init_vgg19(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    base_vgg19 = tf.keras.applications.VGG19(input_shape=(600, 200, 3), include_top=False, weights="imagenet")
    x = base_vgg19.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    vgg19_model = tf.keras.Model(inputs=base_vgg19, outputs=output_layer)

    vgg19_model.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])
    return vgg19_model


search_space = [space.Real(2,6, name='lr'),
         space.Integer(300,2000, name='hidden'),
         space.Real(0.01,0.7,name='drop'),
         space.Integer(1,4,name='batch_size')]


#Funktion, die von gp_minimize aufgerufen wird. Enthält die fit Funktion
@use_named_args(search_space)
def objectiv(**kwargs):

    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=init_vgg16,epochs=2)
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
	

directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
run_id = datetime.now().strftime("VGG %Y_%m_%d T %H-%M-%S")
logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//VGG//' + run_id
os.mkdir(logdir)
logdir = logdir + '//'

# Label log file
h = open(logdir+'out.txt', 'a')
h.write('lr,drop,drop2,loss1,loss2,batch size,min loss\n')
h.close()

#EarlyStopping
callb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.01,patience=10,mode='min',restore_best_weights=True)
#TensorBoard zum Loggen der Trainings
callb2 = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=0,write_graph=False)   # histogram_greq=10

#Starten des Optimierungsvorgangs, n_calls sind die Anzahl verschiedener Parametersätze, n_random_starts sind die Anzahl zufälliger zu Beginn
res = gp_minimize(objectiv,search_space,n_calls=30,n_jobs=1,n_random_starts=30,verbose=True,acq_func='gp_hedge',callback=[plotting])