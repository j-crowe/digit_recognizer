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
batch_size = 100

data.test.cls = np.array([label.argmax() for label in data.test.labels])

X = tf.placeholder(tf.float32, [None, img_size_flat])
y_one_hot = tf.placeholder(tf.float32, [None, num_classes])
y_true = tf.placeholder(tf.int64, [None])

feed_dict_test = {X: data.test.images,
                  y_one_hot: data.test.labels,
                  y_true: data.test.cls}

# Weight matrix that is img_size_flat X num_classes
# To be modified in TF backprop
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# Bias 1D Array of size num_classes
biases = tf.Variable(tf.zeros([num_classes]))

# Generate logits
logits = tf.matmul(X, weights) + biases
predicted = tf.nn.softmax(logits)
predicted_cls = tf.argmax(predicted, dimension=1)

# Calculate cross entropy + cost function
cost_fn = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=y_one_hot))

# Using gradient decent to optimize our single layer
optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.5).minimize(cost_fn)

correct_prediction = tf.equal(predicted_cls, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create new TF session and initialize
session = tf.Session()
session.run(tf.global_variables_initializer())


def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_one_hot_batch = data.train.next_batch(batch_size)
        feed_dict_train = {X: x_batch, y_one_hot: y_one_hot_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


# Set up the test feed dict with the separate test data
feed_dict_test = {X: data.test.images,
                  y_one_hot: data.test.labels,
                  y_true: data.test.cls}


def optimization_run(num_iters=1):
    optimize(num_iterations=num_iters)
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print('Accuracy for {0} iteration: {1:.1%}'.format(num_iters, acc))


# Attempt running a varible number of times to locate where we begin to see
# diminishing returns
optimization_run(1)
optimization_run(5)
optimization_run(20)
optimization_run(100)
optimization_run(1000)
# We can see diminishing returns after 10000 iterations
optimization_run(10000)
