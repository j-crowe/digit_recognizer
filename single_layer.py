import tensorflow as tf
import numpy as np


# Load in mnist dataset using one_hot
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Get basic information about dataset
print("Training Size:\t\t{}".format(len(data.train.labels)))
print("Test Size:\t\t{}".format(len(data.test.labels)))


# Each image size is 28px by 28px
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

data.test.cls = np.array([label.argmax() for label in data.test.labels])

X = tf.placeholder(tf.float32, [None, img_size_flat])
y_one_hot = tf.placeholder(tf.float32, [None, num_classes])
y_true = tf.placeholder(tf.int64, [None])
