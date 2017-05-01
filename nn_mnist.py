from __future__ import division

import gzip
import cPickle

import tensorflow as tf
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y,10)
valid_y = one_hot(valid_y,10)
test_y = one_hot(test_y,10)

"""
plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]
"""

x = tf.placeholder(tf.float32, [None, 28*28])  # samples
y_ = tf.placeholder(tf.float32, [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(28*28, 10*10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10*10)) * 0.1)
W2 = tf.Variable(np.float32(np.random.rand(10*10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 100

error = 1;
epoch = 0;
lastError = 1000
bound = 0.001
errorHistory = []

while 1:
    for jj in xrange((int)(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    epoch += 1
    error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    errorHistory.append(error)

    print ("Epoch #:", epoch, "Error: ", error)

    if abs(error - lastError) < bound:
        break

    lastError = error

print ("------Test-------")
result = sess.run(y, feed_dict={x: test_x})
esBien = 0
for b, r in zip(test_y, result):
    if b.argmax() == r.argmax():
        esBien += 1

error = sess.run(loss, feed_dict={x: test_x, y_: test_y})
print 'Error = ', error
porcentaje = esBien/len(test_y)*100
print 'Acierto = ', porcentaje, ' %'

plt.plot(errorHistory)
plt.ylabel("Error")
plt.xlabel("Epoca")
plt.show()