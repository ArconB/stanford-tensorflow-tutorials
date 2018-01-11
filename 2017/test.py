#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 2018/1/11 22:06
# Author  : Shi Bo
# File    : test.py.py

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))

writer.close()