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

# Define number of nodes in each hidden layer
h1_nodes = 500
h2_nodes = 500
h3_nodes = 500

data.test.cls = np.array([label.argmax() for label in data.test.labels])

X = tf.placeholder(tf.float32, [None, img_size_flat])
y = tf.placeholder(tf.float32, [None, num_classes])
y_true = tf.placeholder(tf.int64, [None])

feed_dict_test = {X: data.test.images,
                  y: data.test.labels,
                  y_true: data.test.cls}


# Define the neural network model including the weights and biases
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([img_size_flat,
                      h1_nodes])),
                      'biases': tf.Variable(tf.random_normal([h1_nodes]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([h1_nodes,
                      h2_nodes])),
                      'biases': tf.Variable(tf.random_normal([h2_nodes]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([h2_nodes,
                      h3_nodes])),
                      'biases': tf.Variable(tf.random_normal([h3_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([h3_nodes,
                    num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}

    l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    logits = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(data.train.num_examples/batch_size)):
                epoch_x, epoch_y = data.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost],
                                feed_dict={X: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',
                  hm_epochs, 'loss', epoch_loss)

        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy', accuracy.eval(
                        {x: data.test.images, y: data.test.labels}))


train_neural_network(X)
