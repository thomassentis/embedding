#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:04:14 2017

@author: thomas
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

import os.path

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# EMBEDDING

# Here the shape of x is [m, X*Y*D] (X=Y=28, D=1, m=BATCH_SIZE)


# dimensions of the image
X, Y = 28, 28
# depth (number of images for the same object)
D = 1
# dimension of the embedding space
N = 128

ORIGINAL_SIZE = 28*28
SIZE = 7*7*32


BATCH_SIZE = 200

ALPHA = 1e-2

LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")

lx = np.linspace(0,1, 2001)
ly = []


def first_layer(h1):

    x_image = tf.reshape(h1, [-1, 28, 28, 1])
    W_conv1 = weight_variable([2, 2, 1, 32], "W_CONV1")
    b_conv1 = bias_variable([32], "B_CONV1")
    h2 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h3 = max_pool_2x2(h2)
    #print(x.shape)
    x = tf.reshape(h3, [-1, SIZE])
    return x



    

def lifted_embedding(x):
    
    # SIZE rows, N columns
    W_embedding = weight_variable([SIZE, N], "W_EMBEDDING")
    
    # BATCH_SIZE rows, N columns
    y = tf.matmul(x, W_embedding)
    return y

    
def cost_function(y):
    
    # N rows, BATCH_SIZE columns. Each column is f(x_i)
    y_vertical = tf.transpose(y)
    
    # N rows, BATCH_SIZE columns
    y_vertical_square = tf.square(y_vertical)
    
    
    # 1 row, BATCH_SIZE columns. Each element is |f(x_i)\^2
    measures_square = tf.matmul(tf.ones([1, N]), y_vertical_square)
    
    # BATCH_SIZE square matrix
    
    part1 = tf.matmul(measures_square, tf.ones([1, BATCH_SIZE]), transpose_a = True)
    
    part2 = tf.matmul(tf.ones([1, BATCH_SIZE]), measures_square, transpose_a = True)
    
    part3 = 2*tf.matmul(y_vertical, y_vertical, transpose_a=True)
    
    distances_square = part1 + part2 - part3
    
    SHIFT = 1e-4
    
    distances_square = tf.nn.relu(distances_square - SHIFT * matrix_ones) + SHIFT * matrix_ones
    
    with tf.control_dependencies([tf.assert_positive(distances_square)]):
        distances = tf.sqrt(distances_square)
        #distances = tf.check_numerics(distances, "NOOOOOOOOO")
        #distances = distances_square
    
    
    # Same_label matrix : S_ij = 1 if x_i and x_j have same label, 0 otherwise
    
    pos_pairs = tf.matmul(y_,y_,transpose_b=True)
    
    # total number of positive pairs, counting twice each {i,j}={j,i}
    # equal to 2|P|
    number_pos_pairs = tf.reduce_sum(pos_pairs)
    
    neg_pairs = matrix_ones - pos_pairs
    
    # negative pairs
    
    terms_neg_pairs = tf.exp(ALPHA * matrix_ones - distances) * neg_pairs
    
    sums_rowwise = tf.matmul(terms_neg_pairs, vect_ones)
    
    terms_in_log = tf.matmul(sums_rowwise, vect_ones, transpose_b = True) + tf.matmul(vect_ones, sums_rowwise, transpose_b = True)
    
    J_matrix = tf.log(terms_in_log) + distances
    
    
    J = tf.reduce_sum(tf.square(tf.nn.relu(J_matrix * pos_pairs))) / number_pos_pairs
    
    return J


# Create randomly initialized embedding weights which will be trained
LOG_DIR = "/Users/thomas/Documents/Pzartech/TensorFlow/"
    
vect_ones = tf.ones([BATCH_SIZE, 1])
matrix_ones = tf.ones([BATCH_SIZE, BATCH_SIZE])

h1 = tf.placeholder(tf.float32, [None, ORIGINAL_SIZE])
    
# LABELS
# BATCH_SIZE rows, N columns
y_ = tf.placeholder(tf.float32, [None, 10])

def next_graph(embedding_size):
    
    x = first_layer(h1)
    
    y = lifted_embedding(x)
    
    J = cost_function(y)
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(J)
    
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(y)
    saver = tf.train.Saver()
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR + "BANANA")
    writer.add_graph(sess.graph)
    
    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()
    
    # adding one embedding
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.metadata_path = LABELS
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'labels_1024.tsv')
    
    
    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(writer, config)
    
    for i in range(2001):
        batch = mnist.train.next_batch(BATCH_SIZE)
        
        if(i==2000):
            sess.run(assignment, feed_dict={h1:mnist.test.images[:1024], y_:mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)
        if(i%100==0):
            print(i)
        t_val, J_val = sess.run([train_step, J], feed_dict={h1:batch[0], y_:batch[1]})
        ly.append(J_val)
    
    sess.close()

next_graph(N)

plt.plot(lx, ly)