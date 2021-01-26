import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Pipeline import DataPipeline
from Pipeline import model_test


models_to_train = [
    "alexnet",
    "vgg16",
    "vgg19",
    "efficientnet",
    "efficientdet",
    "inceptionresnetv2",
    "inception",
    "mobilenetv3",
    "resnetv2",
    "gcnn",
    "xception"
]

def get_model():
    return model_test.create_vvg16()

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.plot(hist["epoch"], hist["mae"], label="Training ERror")
    plt.plot(hist["epoch"], hist["val_mae"], label="Validation error")
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.plot(hist["epoch"], hist["mse"], label="Training error")
    plt.plot(hist["epoch"], hist["val_mse"], label="Validation error")
    plt.ylim(0,20)
    plt.legend()
    plt.show()



dataset = DataPipeline.get_dataset()
callback1, callback2 = model_test.get_callbacks()
train_size = round(0.7 * len(dataset))
train = dataset.take(train_size)
test = dataset.skip(train_size)

model = get_model()
model.compile(tf.keras.optimizers.Adam(lr=0.001, amsgrad=True,), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])
history = model.fit(train, validation_split=0.2, callbacks=[callback1, callback2], epochs=200, verbose=2)