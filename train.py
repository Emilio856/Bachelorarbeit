import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import data_pipeline as data_pipeline
import model_manager as model_manager

from numba import cuda
from numpy import mean
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


cuda_device = cuda.get_current_device()
cuda_device.reset()

device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], enable=True)

exceptions_num = 0
#for attempt in range(10):
    #try:

models_to_train = [
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
    "efficientdet_d0",
    "inception_resnet_v2",
    "inception",
    "mobilenet_v3_small",
    "mobilenet_v3_large"
    "resnet50_v2",
    "resnet101_v2",
    "resnet152_v2",
    "gcnn",
    "xception"
]

def get_model(model_name):
    return model_manager.get_model(model_name)

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

def parser(record):
    keys_to_feature = {
        "img_path": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(record, keys_to_feature)
    img = tf.io.decode_raw(parsed["img_path"], tf.uint8)
    img = tf.cast(img, tf.float32)
    label = tf.cast(parsed["label"], tf.float32)

    # return {"img": img}, label
    return img, label

def get_tfrecord(filenames):
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=20)
    dataset = dataset.shuffle(8683, seed=42)
    dataset = dataset.map(parser, num_parallel_calls=20)
    dataset = dataset.batch(16, drop_remainder=True)
    dataset = dataset.prefetch(2)
    return dataset

def get_train_data():
    return get_tfrecord(filenames=["train.tfrecords"])

def get_val_data():
    return get_tfrecord(filenames=["val.tfrecords"])

def get_test_data():
    return get_tfrecord(filenames=["test.tfrecords"])



"""dataset = data_pipeline.get_dataset()
train_size = round(0.7 * len(dataset))
train = dataset.take(train_size)
test = dataset.skip(train_size)"""


training_model = "vgg16"
model = get_model(training_model)

run_id = datetime.now().strftime("VGG %Y_%m_%d T %H-%M-%S")
os.chdir("..")
# logdir = os.getcwd() + "//" + run_id
logdir = os.path.join(os.getcwd(), run_id)
os.mkdir(logdir)
# logdir = logdir + '//'

# Label log file
# h = open(logdir+'out.txt', 'a')
h = open(os.path.join(os.getcwd(), run_id, "out.txt"), "a")
h.write('lr,loss1,loss2,min loss\n')
h.close()

# Early Stopping
callback1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=20,
    mode="min",
    restore_best_weights=True
)

# Uses Tensorboard to monitor training
callback2 = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Checkpoint
# checkpoint_path ="training\\cp.ckpt"
checkpoint_path = os.path.join("training, cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

callback3 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    mode="max"
)

model.compile(tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=True,), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])


"""train = get_train_data()
val = get_val_data()
test = get_test_data()"""

x_train = np.random.random((20, 450, 450, 3))
y_train = np.random.random((20,))

# history = model.fit(train, validation_data=val, callbacks=[callback1, callback2, callback3], epochs=200, verbose=2)
history = model.fit(x_train, y_train, batch_size=8, validation_split=0.2, epochs=10)

# Evaluate model on test set
print("Evaluate")
result = model.evaluate(test)
result_dict = dict(zip(model.metrics, result))

with open("testing_result.txt", "w") as f:
    for key, value in result_dict.items():
        f.write(f"{key} = {value}\n")

model.save(training_model + "_" + run_id + ".h5")

"""except Exception:
    exceptions_num += 1
    print(f"Caught exception number {exceptions_num}!")
    time.sleep(5)
else:
    break"""
    