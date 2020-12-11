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
def init_resnet50v2(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    base_resnet50 = tf.keras.applications.ResNet50V2(input_shape=(600, 200, 3), include_top=False)
    x = base_resnet50.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(drop)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet50v2 = tf.keras.Model(inputs=base_resnet50.input, outputs=output_layer)

    resnet50v2.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])
    return resnet50v2


def init_resnet101v2(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    base_resnet101 = tf.keras.applications.ResNet101V2(input_shape=(600, 200, 3), include_top=False)
    x = base_resnet101.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(drop)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet101v2 = tf.keras.Model(inputs=base_resnet101.input, outputs=output_layer)

    resnet101v2.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])
    return resnet101v2


def init_resnet152v2(lr, hidden, drop):
    lr = 1 / np.power(10, lr)
    base_resnet152 = tf.keras.applications.ResNet152V2(input_shape=(600, 200, 3), include_top=False)
    x = base_resnet152.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(drop)(x)

    x = tf.keras.layers.Dense(hidden, activation="elu")(x)
    x = tf.keras.layers.Dropout(drop)(x)

    output_layer = tf.keras.layers.Dense(1)(x)
    resnet152v2 = tf.keras.Model(inputs=base_resnet152.input, outputs=output_layer)

    resnet152v2.compile(tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])


search_space = [space.Real(2,6, name='lr'),
         space.Integer(300,2000, name='hidden'),
         space.Real(0.01,0.7,name='drop'),
         space.Integer(1,4,name='batch_size')]

@use_named_args(search_space)
def evaluate_func(**kwargs):
    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=init_resnet50v2, epochs=1)
    model.set_params(**kwargs)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    fit_model = model.fit(x_train, y_train, validation_split=0.2, callbacks=[callb, callb2], shuffle=True)

    score = mean_squared_error(y_test, model.predict(x_test))
    print("score", score)

    min_loss = np.min(fit_model.history['val_mae'][-21:-1])  # [-21:-1]         # history.history
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

    """
    try:
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
    except:
        f = open(logdir+'error.txt','w')
        f.write('failed to plot convergence or evaluations')
    try:
        if len(results.models) > 1:
            fig3 = plt.figure(3)
            ax3 = plt.axes()
            plots.plot_objective(results)
            plt.savefig(logdir+'objective.png',dpi=500)
            plt.close()
    except:
        f = open(logdir+'error.txt','w')
        f.write('failed to plot objective')
    try:
        filename = logdir+'gp_min.sav'
        pickle.dump(results, open(filename, 'wb')) 
    except:
        f = open('error.txt','w')
        f.write('failed to save model')
        """



NAME = f"Just a Testfile{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

directory = "C://Users//emili//Desktop//Python//Bachelorarbeit Code"
run_id = datetime.now().strftime("ResNet %Y_%m_%d T %H-%M-%S")
logdir = 'C://Users//emili//Desktop//Python//Bachelorarbeit Code//ResNet//' + run_id
os.mkdir(logdir)
logdir = logdir + '//'

data_dir = pathlib.Path(directory)
print(data_dir)

image_count = len(list(data_dir.glob("*/*.png")))
print(image_count)
print(data_dir)

batch_size = 32
img_height = 600
img_width = 200

x_data = list()
y_data = list()

labels = csv.reader(open("C://Users//emili//Desktop//Python//Bachelorarbeit Code//Aufnahmen Test//Labels2.csv"),
                    "excel", delimiter=";")

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

# Convert images to numpy array and normalize them
x_data = np.asarray(x_data)
y_data = np.array(y_data)


"""y_data = np.expand_dims(y_data, axis=1)
y_data = np.array(y_data).reshape(-1, 1)"""

scaler = NDStandardScaler()
scaler.fit(x_data)
# x_data = scaler.transform(x_data)
x_data = tf.keras.utils.normalize(x_data, axis=-1, order=2)

"""# Label encoder
le = LabelEncoder()
y_data = le.fit_transform(y_data)"""

imp = IterativeImputer()
y_data = np.array(y_data).reshape(-1, 1)

# MinMaxScaler
mnscaler = MinMaxScaler()
# y_data = mnscaler.fit_transform(y_data)
y_data = mnscaler.fit_transform(imp.fit_transform(y_data))

# Standard Scaler
stscaler = StandardScaler()
# y_data = stscaler.fit_transform(y_data)
# y_data = stscaler.fit_transform(imp.fit_transform(y_data))


# Label log file
h = open(logdir + 'out.txt', 'a')
h.write('lr,drop,drop2,loss1,loss2,batch size,min loss\n')
h.close()

# Early Stopping
callb = tf.keras.callbacks.EarlyStopping(monitor='mae', min_delta=0.01, patience=20, mode='min',
                                         restore_best_weights=True)

# Uses Tensorboard to monitor training
callb2 = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=10, write_graph=False)

# Starts optimization
# n_calls are number of diferent parameter sets
result = gp_minimize(evaluate_func, search_space, n_calls=10, n_jobs=1, verbose=True, acq_func='gp_hedge',
                     acq_optimizer='auto', callback=[plotting])  # , callback=[tensorboard]

print("Best Accuracy: %  3f" % (1.0 - result.fun))