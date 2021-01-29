import csv
import os
import json
import time

import pandas as pd
import tensorflow as tf

from retry import retry

def get_dataset():
    with tf.device("/cpu:0"):
        # path = "K:\\Data\\2020-12-03"
        path = "C:\\Users\\uffie\\bwSyncAndShare\\Bachelorarbeit-master"
        img_type = "CLAHE"   # Replace!! -> CLAHE or HarrisCorner
        seed = 42

        """
        Data from augmented imgs
        """
        # Open labels
        with open(os.path.join(path,"Augmented180", "labels_augmented_imgs.json")) as f:
                labels = list()
                json_file = json.load(f)

        for folder in json_file:
                for img in json_file[folder]:
                    if json_file[folder][img] != None:
                        labels.append(json_file[folder][img][-1])

        # Only load images that have a label
        images = list()
        for subdir, dirs, files in os.walk(os.path.join(path, "Augmented180", img_type)):
            for file in files:
                folder = subdir.split("\\")[-1]
                if ".png" in file and file in json_file[folder] and json_file[folder][file] != None:
                    images.append(os.path.join(subdir, file))
                else:
                    pass
        
        """
        Data from normal imgs
        """
        # Open labels
        with open(os.path.join(path, "Cropped", "combined_labels.json")) as f:
            json_file = json.load(f)
        
        for folder in json_file:
            if "gerissen" in folder or "licht" in folder:
                pass
            else:
                for img in json_file[folder]:
                    if json_file[folder][img] != None:
                        labels.append(json_file[folder][img][-1])
        
        # Only loads images that have a label
        for subdir, dirs, files in os.walk(os.path.join(path, "Cropped", img_type)):
            if "gerissen" in subdir or "licht" in subdir or "weiß" in subdir or "papier" in subdir:
                pass
            else:
                for file in files:
                    folder = subdir.split("\\")[-1]
                    if ".png" in file and file in json_file[folder] and json_file[folder][file] != None:
                        images.append(os.path.join(subdir, file))
                    else:
                        pass

        

        df = pd.DataFrame({"img_path": images, "label": labels})

        # Write sample data
        df.to_csv("./sample.csv", index=False)

        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def encoder(row0, row1):
            return tf.train.Example(
                features=tf.train.Features(feature={
                    "img_path": _bytes_feature(row0),
                    "label": _float_feature(row1)
                })
            ).SerializeToString()

        # Iterate over data and generate serializable data
        def read_csv(file_path="./sample.csv", skip_rows=1):
            with open("./sample.csv", "r") as csv_file:
                data = csv.reader(csv_file, delimiter=",")
                for index, row in enumerate(data):
                    if index < skip_rows:
                        continue
                    yield encoder(
                        row0=row[0].encode("utf-8"),
                        row1=float(row[1])
                    )

        # TFRecord writer for encoded data
        def tf_record_writer(in_file="./sample.csv", out_file="./sample.tfrecords"):
            with tf.io.TFRecordWriter(out_file) as writer:
                for record in read_csv(file_path=in_file):
                    writer.write(record)


        tf_record_writer()
        
        @retry(Exception, delay=5, tries=100)
        def img_decoder(img_path, label):
            read_img = tf.io.read_file(img_path)
            img = tf.image.decode_png(read_img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            return img, label

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Shuffle and repeat with buffer size equal to length of dataset -> ensures good shuffling
        dataset = dataset.shuffle(len(images), seed=seed)

        dataset = dataset.map(img_decoder, num_parallel_calls=4)
        dataset = dataset.batch(16, drop_remainder=True)
        dataset = dataset.prefetch(1)

        # TODO .cache() ??

        train_size = round(0.7 * len(dataset))
        train = dataset.take(train_size)
        test = dataset.skip(train_size)

        return train, test

tr, te = get_dataset()
print("Done")