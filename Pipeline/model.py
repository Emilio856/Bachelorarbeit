import sys
import tensorflow as tf

from tensorflow import Tensor
# from .CNNs.VGG import init_vgg16
sys.path.insert(1, "C:\\Users\\emili\\Desktop\\Python\\Bachelorarbeit Code\\CNNs")
from VGG import init_vgg16

class Inputs(object):
    def __init__(self, img: Tensor, label: Tensor):
        self.img = img
        self.label = label

class Model(object):
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        # self.predictions = self.predictions(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, inputs: Inputs):
        with tf.name_scope("conv_relu_maxpool"):
            for conv_layer_i in range(5):
                x = tf.layers

print("GUBBHJ")