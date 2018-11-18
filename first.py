from tensorflow.contrib.data.python.ops import sliding
import numpy as np
import tensorflow as tf
#import gc
import cv2
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import listdir
from os.path import isfile, join
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")    #for tensorboard
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
root_logdir_m = "tf_models"
logdir_m = "{}/run-{}/".format(root_logdir_m, now)

def _parse_function(example_proto):
    features = {
                "image_y": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,1])
    image_y = tf.cast(image_y,dtype=tf.float32)

    return image_y
