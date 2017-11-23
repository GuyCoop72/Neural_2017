from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

# parameter definitions
h_dim = 32 #set the horizontal dimension object images
v_dim = 32 #set the vertical dimension of the object images
num_classes = 10 # how many different input classes there are

batch_size = 100
learning_rate = 0.005
max_steps = 1000

# prepare the data set
# ....
# images_train = 50000 x 3072 (=32x32x3) array of training images
# labels_train = 50000 labels, 1 per image
# images_test = 10000 x 3072 (=32x32x3) array of test images
# labels_test = 10000 labels, 1 per image
# classes = 10 text labels to translate numerical class into words (car, plane, etc)

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, (h_dim * v_dim * 3)])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables, these are the variables to be optimized
weights = tf.Variable(tf.zeros([h_dim * v_dim * 3, num_classes]))
biases = tf.Variable(tf.zeros([10]))

# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases

# Define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_placeholder))

# Define the training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ----------------------------------------------------------------------------------------------------------------------
# Run the TensorFlow graph
# ----------------------------------------------------------------------------------------------------------------------
beginTime = time.time()

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.initialize_all_variables())

    # Repeat max_steps times
    for i in range(max_steps):

        # Generate input data batch
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        # Periodically print out the models current accuracy
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

        # Perform a single training step
        sess.run(train_step, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})

    # After finishing the training, evaluate on the test set
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: data_sets['images_test'],
        labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

