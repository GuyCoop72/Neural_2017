from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time


# ----------------------------------------------------------------------------------------------------------------------
# Data imports
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Global parameter set up
# ----------------------------------------------------------------------------------------------------------------------
h_dim = 32 #set the horizontal dimension object images
v_dim = 32 #set the vertical dimension of the object images
num_classes = 10 # how many different input classes there are

# ----------------------------------------------------------------------------------------------------------------------
# Set up image recognizer network
# ----------------------------------------------------------------------------------------------------------------------
batch_size = 100
learning_rate = 0.005
max_steps = 1000

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



