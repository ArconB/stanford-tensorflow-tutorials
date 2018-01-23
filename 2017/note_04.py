#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 18/1/22 下午4:01
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : note_04.py

import tensorflow as tf

BATCH_SIZE = 32
VOCAB_SIZE = 1000
EMBED_SIZE = 100
NUM_SAMPLED = 10
LEARNING_RATE = 0.001

center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], - 1.0, 1.0))

embed = tf.nn.embedding_lookup(embed_matrix, center_words)

nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / EMBED_SIZE ** 0.5))
nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed,
                   num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
