import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import DataPipeline
import model_manager
import pandas as pd
import os
import time
from numba import cuda

from numpy import mean
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


cuda_device = cuda.get_current_device
cuda_device.reset()

device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], enable=True)

exceptions_num = 0
for attempt in range(10):
    try:

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
            return model_manager.create_vvg16(model_name)

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
        train_size = round(0.7 * len(dataset))
        train = dataset.take(train_size)
        test = dataset.skip(train_size)


        training_model = ""
        model = get_model(training_model)

        run_id = datetime.now().strftime("VGG %Y_%m_%d T %H-%M-%S")
        os.chdir("...")
        logdir = os.getcwd() + "//" + run_id
        os.mkdir(logdir)
        logdir = logdir + '//'

        # Label log file
        h = open(logdir+'out.txt', 'a')
        h.write('lr,drop,drop2,loss1,loss2,batch size,min loss\n')
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
        checkpoint_path ="training/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        callback3 = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode="max"
        )

        model.compile(tf.keras.optimizers.Adam(lr=0.001, amsgrad=True,), tf.keras.losses.MeanSquaredError(), ["mae", "accuracy"])

        history = model.fit(train, validation_split=0.2, callbacks=[callback1, callback2], epochs=200, verbose=2)

        # Evaluate model on test set
        print("Evaluate")
        result = model.evaluate(test)
        result_dict = dict(zip(model.metrics, result))

        with open("testing_result.txt", "w") as f:
            for key, value in result_dict.items():
                f.write(f"{key} = {value}\n")

    except Exception:
        exceptions_num += 1
        print(f"Caught exception number {exceptions_num}!")
        time.sleep(10)
    else:
        break