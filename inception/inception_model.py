# This program builds the inception v4 architecture from the
# paper https://arxiv.org/pdf/1602.07261.pdf
#
# author: Emilio Rivera

# for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import random as rn
rn.seed(42)

import numpy as np

rnd_seed = 42


def conv_bn(x, filter_nr, row_nr, col_nr, padding="same", strides=(1, 1)):
    """
    Convolution followed by Batch Normalization

    Args:
      x: Input.
      filter_nr: Number of filters.
      row_nr: Heigth of a filter.
      col_nr: Width of a filter.
      padding: Type of padding to use.
      strides: Size of strides to use.

    Returns:
      A block of a Convolutional and a Batch Normalization Layer.
    """
    x = tf.keras.layers.Convolution2D(filter_nr,
                                     (row_nr, col_nr),
                                      strides=strides,
                                      padding=padding,
                                      use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def inception_a(input):
    """
    Builds an Inception-A-Block.

    Args:
      input: Input of the Block.
    
    Returns:
      An Inception-A-Block.
    """
    limb1 = conv_bn(input, 96, 1, 1)

    limb2 = conv_bn(input, 64, 1, 1)
    limb2 = conv_bn(limb2, 96, 3, 3)

    limb3 = conv_bn(input, 64, 1, 1)
    limb3 = conv_bn(limb3, 96, 3, 3)
    limb3 = conv_bn(limb3, 96, 3, 3)

    limb4 = tf.keras.layers.AveragePooling2D((3, 3),
                                             strides=(1, 1),
                                             padding="same")(input)
    limb4 = conv_bn(limb4, 96, 1, 1)

    x = tf.keras.layers.concatenate([limb1, limb2, limb3, limb4])
    return x


def inception_b(input):
    """
    Builds an Inception-B-Block.

    Args:
      input: Input of the Block.
    
    Returns:
      An Inception-B-Block.
    """
    limb1 = conv_bn(input, 384, 1, 1)

    limb2 = conv_bn(input, 192, 1, 1)
    limb2 = conv_bn(limb2, 224, 1, 7)
    limb2 = conv_bn(limb2, 256, 7, 1)

    limb3 = conv_bn(input, 192, 1, 1)
    limb3 = conv_bn(limb3, 192, 7, 1)
    limb3 = conv_bn(limb3, 224, 1, 7)
    limb3 = conv_bn(limb3, 224, 7, 1)
    limb3 = conv_bn(limb3, 256, 1, 7)

    limb4 = tf.keras.layers.AveragePooling2D((3, 3),
                                             strides=(1, 1),
                                             padding="same")(input)
    limb4 = conv_bn(limb4, 128, 1, 1)

    x = tf.keras.layers.concatenate([limb1, limb2, limb3, limb4])
    return x


def inception_c(input):
    """
    Builds an Inception-C-Block.

    Args:
      input: Input of the Block.
    
    Returns:
      An Inception-C-Block.
    """
    limb1 = conv_bn(input, 256, 1, 1)

    limb2 = conv_bn(input, 384, 1, 1)
    limb20 = conv_bn(limb2, 256, 1, 3)
    limb21 = conv_bn(limb2, 256, 3, 1)
    limb2 = tf.keras.layers.concatenate([limb20, limb21])

    limb3 = conv_bn(input, 384, 1, 1)
    limb3 = conv_bn(limb3, 448, 3, 1)
    limb3 = conv_bn(limb3, 512, 1, 3)
    limb30 = conv_bn(limb3, 256, 1, 3)
    limb31 = conv_bn(limb3, 256, 3, 1)
    limb3 = tf.keras.layers.concatenate([limb30, limb31])

    limb4 = tf.keras.layers.AveragePooling2D((3, 3),
                                             strides=(1, 1),
                                             padding="same")(input)
    
    x = tf.keras.layers.concatenate([limb1, limb2, limb3, limb4])
    return x


def reduction_a(input):
    """
    Builds a Reduction-A-Block for the Inception architecture.

    Args:
      input: Input of the Block.
    
    Returns:
      A Reduction-A-Block.
    """
    limb1 = conv_bn(input, 384, 3, 3, strides=(2, 2), padding="valid")

    limb2 = conv_bn(input, 192, 1, 1)
    limb2 = conv_bn(limb2, 224, 3, 3)
    limb2 = conv_bn(limb2, 256, 3, 3, strides=(2, 2), padding="valid")

    limb3 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(input)

    x = tf.keras.layers.concatenate([limb1, limb2, limb3])
    return x


def reduction_b(input):
    """
    Builds a Reduction-B-Block for the Inception architecture.

    Args:
      input: Input of the Block.
    
    Returns:
      A Reduction-B-Block.
    """
    limb1 = conv_bn(input, 192, 1, 1)
    limb1 = conv_bn(limb1, 192, 3, 3, strides=(2, 2), padding="valid")

    limb2 = conv_bn(input, 256, 1, 1)
    limb2 = conv_bn(limb2, 256, 1, 7)
    limb2 = conv_bn(limb2, 320, 7, 1)
    limb2 = conv_bn(limb2, 320, 3, 3, strides=(2, 2), padding="valid")

    limb3 = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding="valid")(input)

    x = tf.keras.layers.concatenate([limb1, limb2, limb3])
    return x


def inception(input):
    """
    Builds the basic structure of the Inception v4 architecture.

    Args:
      input: Input of the model.
    
    Returns:
      The base of Inception v4.
    """
    stem = conv_bn(input, 32, 3, 3, strides=(2, 2), padding="valid")
    stem = conv_bn(stem, 32, 3, 3, padding="valid")
    stem = conv_bn(stem, 64, 3, 3)

    limb1 = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding="valid")(stem)
    
    limb2 = conv_bn(stem, 96, 3, 3, strides=(2, 2), padding="valid")

    stem = tf.keras.layers.concatenate([limb1, limb2])

    limb1 = conv_bn(stem, 64, 1, 1)
    limb1 = conv_bn(limb1, 96, 3, 3, padding="valid")

    limb2 = conv_bn(stem, 64, 1, 1)
    limb2 = conv_bn(limb2, 64, 1, 7)
    limb2 = conv_bn(limb2, 64, 7, 1)
    limb2 = conv_bn(limb2, 96, 3, 3, padding="valid")

    stem = tf.keras.layers.concatenate([limb1, limb2])

    limb1 = conv_bn(stem, 192, 3, 3, strides=(2, 2), padding="valid")
    limb2 = tf.keras.layers.MaxPooling2D((3, 3),
                                         strides=(2, 2),
                                         padding="valid")(stem)
    
    stem = tf.keras.layers.concatenate([limb1, limb2])

    # 4 Inception A
    for _ in range(4):
        stem = inception_a(stem)

    # Reduction A
    stem = reduction_a(stem)

    # 7 Inception B
    for _ in range(7):
        stem = inception_b(stem)
    
    # 3 Inception C
    for _ in range(3):
        stem = inception_c(stem)
    
    return stem
    

def create_inception():
    """
    Builds a complete architecture of Inception v4 with a 450 x 450 input size
    """
    inputs = tf.keras.layers.Input((450, 450, 3))

    x = inception(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D((8, 8),
                                               padding="valid")(x)
    x = tf.keras.layers.Dropout(0.2, seed=rnd_seed)
    x = tf.keras.layers.Dense(1,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=rnd_seed))(x)

    model = tf.keras.models.Model(inputs, x)
    return model