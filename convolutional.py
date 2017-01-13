import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Interactive session. Fix at end after testing
sess = tf.InteractiveSession()

# Training data and true labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize TF global variables
sess.run(tf.global_variables_initializer())

# cost fn
# optimization minimizing cost

# Calculate logit
y = tf.matmul(x, W) + b


# Cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Batch training/optimizing
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Built correct_prediction vector
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Calculate numerical accuracy of predictor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Evaluate accuracy on test data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
