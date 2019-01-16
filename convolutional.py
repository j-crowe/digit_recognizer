import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("data", one_hot=True)

# Interactive session. Fix at end after testing
sess = tf.InteractiveSession()


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10
# Training data and true labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])


# Helper weight generation function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Helper bias generation function
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Helper convolutional 2d nn generation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def create_conv_layer(input,
                      num_input_channels,
                      filter_size,
                      num_filters,
                      pool=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Generate weights and biases from provided arguments
    W_conv = weight_variable(shape)
    b_conv = bias_variable([num_filters])

    # Create conv2d
    conv = conv2d(input, W_conv)
    conv = tf.nn.relu(conv + b_conv)

    if pool:
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    return conv


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    relu=True):

    weights = weight_variable(shape=[num_inputs, num_outputs])
    biases = bias_variable([num_outputs])

    layer = tf.matmul(input, weights) + biases

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    # General way to figure out how many features we have
    num_features = layer.get_shape()[1:4].num_elements()
    # Flatten the tensor into [num_images, img_height*img_width*num_channels]
    flattened = tf.reshape(layer, [-1, num_features])
    return flattened, num_features


# Transform image
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First Convolutional layer
layer_1 = create_conv_layer(input=x_image,
                            num_input_channels=num_channels,
                            filter_size=filter_size1,
                            num_filters=num_filters1,
                            pool=True)

# Second Convolutional layer
layer_2 = create_conv_layer(input=layer_1,
                            num_input_channels=num_filters1,
                            filter_size=filter_size2,
                            num_filters=num_filters2,
                            pool=True)

# Flatten image to original dimention
layer_flat, num_features = flatten_layer(layer_2)

# Create first connected layer
connected_layer = create_fc_layer(input=layer_flat,
                                  num_inputs=num_features,
                                  num_outputs=fc_size,
                                  relu=True)

# Create final connected layer (readout_layer)
readout_layer = create_fc_layer(input=connected_layer,
                                num_inputs=fc_size,
                                num_outputs=num_classes,
                                relu=False)

# Train and test accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=readout_layer, labels=y_true))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(readout_layer, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                       y_true: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0],
                              y_true: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_true: mnist.test.labels}))

f = open("/tmp/accuracy.txt", "a")
f.write(accuracy.eval( str(accuracy.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels})))