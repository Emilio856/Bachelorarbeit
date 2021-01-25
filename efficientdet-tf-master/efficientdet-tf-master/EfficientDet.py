import tensorflow as tf
import hub
import tensorflow_hub as hub
import tarfile
from efficientdet import visualizer
from efficientdet import EfficientDet
from efficientdet.data import preprocess
from efficientdet.utils.io import load_image
from efficientdet.data.voc import IDX_2_LABEL
# import efficientdet

print(hub.__version__)
print(tf.__version__)

model_url = "https://tfhub.dev/tensorflow/efficientdet/d3/1"
# base_model = hub.KerasLayer(model_url, input_shape=(299, 299, 3))
# base_model = hub.load(model_url)
# print(base_model.summary())

"""my_model = tf.keras.applications.VGG16()
print(my_model.summary())
"""

"""hub_layer = hub.KerasLayer(model_url, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation="relu", input_shape=[20]))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
print(model.summary())"""

"""bla = tarfile.open("efficientdet-d0.tar.gz")
bla.extractall()
print(bla)
print("done")"""


model = EfficientDet.from_pretrained("D0-VOC", score_threshold=0.3)
image_size = model.config.input_size
print(image_size)
