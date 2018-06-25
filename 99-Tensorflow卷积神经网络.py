from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''
Tensorflow卷积神经网络案例
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
'''
# 超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# 神经网络参数设定
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf 计算图输入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)
加参数名称即可。
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = pred, logits= y))

'''
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 报错如下：
ValueError: No gradients provided for any variable, check your graph
for ops that do not support gradients, between variables
["<tf.Variable 'Variable:0' shape=(5, 5, 1, 32) dtype=float32_ref>",
"<tf.Variable 'Variable_1:0' shape=(5, 5, 32, 64) dtype=float32_ref>",
"<tf.Variable 'Variable_2:0' shape=(3136, 1024) dtype=float32_ref>",
"<tf.Variable 'Variable_3:0' shape=(1024, 10) dtype=float32_ref>",
"<tf.Variable 'Variable_4:0' shape=(32,) dtype=float32_ref>",
"<tf.Variable 'Variable_5:0' shape=(64,) dtype=float32_ref>",
"<tf.Variable 'Variable_6:0' shape=(1024,) dtype=float32_ref>",
"<tf.Variable 'Variable_7:0' shape=(10,) dtype=float32_ref>"]
and loss Tensor("Mean:0", shape=(), dtype=float32).

https://stackoverflow.com/questions/46506646/valueerror-no-gradients-provided-for-any-variable-check-your-graph-for-ops-tha

使用TensorFlow训练神经网络时，出现以下报错信息：
ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients,
between variables ["", "", "", "", "", ""] and loss Tensor("Mean_2:0", shape=(), dtype=float32).
报错信息的意思是，提供给minimize函数的var_list参数中的变量没有梯度，需要检查你的图的操作是否在这些变量中支持梯度。
经过检查，发现优化器指定的loss参数中，与var_list参数中的变量无任何关系导致了上述错误，因此，需要检查loss和var_list两个参数的关联性，确保loss由指定的var_list中的变量来计算

'''
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:",
           sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))

