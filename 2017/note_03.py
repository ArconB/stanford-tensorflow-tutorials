#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 2018/1/11 23:16
# Author  : Shi Bo
# File    : note_03.py

# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import xlrd
#
# DATA_FILE = './data/fire_theft.xls'
#
# # df = pd.read_excel(DATA_FILE)
# # data = np.asarray(df.values)
# # n_samples, _ = data.shape
# # print('n_samples=%d' % n_samples)
# # print(data)
# # for x, y in data:
# #     print('herw', x.dtype, y.dtype)
#
# book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
# sheet = book.sheet_by_index(0)
# data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
# n_sample = data.shape[0]
# x_input = np.linspace(-1, 1, 100)
# y_input = x_input * 3 + np.random.randn(x_input.shape[0]) * 0.5
# print(x_input.shape, y_input.shape)
#
# # define network
# x = tf.placeholder(tf.float32, name='x')
# y = tf.placeholder(tf.float32, name='y')
# w1 = tf.Variable(0.0, name='w1')
# w2 = tf.Variable(0.0, name='w2')
# b = tf.Variable(0.0, name='b')
# y_pred = x * x * w1 + x * w2 + b
# loss = tf.square(y - y_pred, name='loss')
# optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(100):
#         total_loss = 0.0
#         _, loss_v, w1_v, w2_v, b_v = sess.run([optim, loss, w1, w2, b], feed_dict={x: data[:,0], y: data[:,1]})
#         print('w1_v=%f, w2_v=%f, b_v=%f' % (w1_v, w2_v, b_v))
#         print('after epoch=%d: mean loss=%f' % (i, np.mean(loss_v)) )
#         # total_loss += loss_v
#         # for x_v, y_v in data:
#         #     _, loss_v, w1_v, w2_v, b_v = sess.run([optim, loss, w1, w2, b], feed_dict={x: x_v, y: y_v})
#         #     print('w1_v=%f, w2_v=%f, b_v=%f' % (w1_v, w2_v, b_v))
#         #     total_loss += loss_v
#         # print('after epoch=%d: total_loss=%f' % (i, total_loss / n_sample))

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
learning_rate = 0.01
batch_size = 128
n_epochs = 25

x = tf.placeholder(tf.float32, [None, 784], name='x_image')
y = tf.placeholder(tf.float32, [None, 10], name='y_label')
w = tf.Variable(tf.random_uniform([784, 10]), name='w')
b = tf.Variable(tf.zeros([1, 10]), name='b')
logits = tf.matmul(x, w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy)
optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):
        total_loss = 0.0
        for _ in range(n_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optim, loss], feed_dict={x: x_batch, y: y_batch})
            total_loss += loss_batch
        print('after epoch=%d, average loss on training set is %f' % (i, total_loss / float(n_batches)))

        total_correct_preds = 0
        n_batches = int(mnist.test.num_examples / batch_size)
        for _ in range(n_batches):
            x_batch, y_batch = mnist.test.next_batch(batch_size)
            _, logits_batch = sess.run([optim, logits], feed_dict={x: x_batch, y: y_batch})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
        print(
            'after epoch=%d, accuracy on testing set is %f' % (i, total_correct_preds / float(n_batches * batch_size)))
