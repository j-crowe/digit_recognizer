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

# Weight matrix that is img_size_flat X num_classes
# To be modified in TF backprop
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# Bias 1D Array of size num_classes
biases = tf.Variable(tf.zeros([num_classes]))

# Generate logits
logits = tf.matmul(X, weights) + biases
predicted = tf.nn.softmax(logits)
predicted_cls = tf.argmax(predicted), dimension=1)

# Calculate cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true_one_hot)

# Generate cost function by calculating mean of cross entropy
cost_fn = tf.reduce_mean(cross_entropy)

# Using gradient decent to optimize our single layer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost_fn)
correct_prediction = tf.equal(predicted_cls, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
