import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import IPython.display as display
import gc

file_name = "train.tfrecords"
raw_dataset = tf.data.TFRecordDataset(file_name)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

"""
raw_img_dataset = tf.data.TFRecordDataset("train.tfrecords")

img_feature_description = {
    "img_path": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.float32)
}

def parse(example_proto):
    return tf.io.parse_single_example(example_proto, img_feature_description)

parsed_img_dataset = raw_img_dataset.map(parse)
print("1.   ", parsed_img_dataset)

for bla in parsed_img_dataset.take(1):
    image_raw = bla["img_path"].numpy()
    display.display(display.Image(data=image_raw))"""
