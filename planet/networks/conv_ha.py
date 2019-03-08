# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools
from tflearn.layers.conv import global_avg_pool
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

# Hyperparameter
growth_k = 18  # growth rate, how many feature map we generate each layer
nb_block = 5        # how many (dense block + Transition Layer)

# 96*96*3

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers) :
    return tf.concat(layers, axis=3)


def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        """1*1 conv to reduce depth og feature map then use 3*3 conv"""
        with tf.name_scope(scope):

            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope+'_conv1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope+'_conv2')

            return x

    def transition_layer(self, x, scope):
        """connect different dense net block"""
        with tf.name_scope(scope):
            x = Relu(x)
            x = conv_layer(x, filter=4, kernel=[3, 3], layer_name=scope+'_conv1')
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        """nb_layers: how many layers(BN-Relu-1*1conv-dropout-BN-Relu-3*3conv-dropout ) in block"""
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            # it is like global state
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)   # all feature map in one dense block

            return x

    def Dense_net(self, input_x):
        """Dense net composed with many block and the transition layers between those block"""
        x = conv_layer(input_x, filter=12, kernel=[5, 5], stride=3, layer_name='conv0')
        x = self.dense_block(input_x=x, nb_layers=8, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = flatten(x)

        return x

# def encoder(obs):
#   """Extract deterministic features from an observation."""
#   kwargs = dict(strides=2, activation=tf.nn.relu)
#   hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())   # (50,50,64,64,3) reshape to (2500,64,64,3)
#   hidden = tf.layers.conv2d(hidden, 24, 8, **kwargs)
#   # hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
#   hidden = tf.layers.conv2d(hidden, 48, 5, **kwargs)
#   hidden = tf.layers.conv2d(hidden, 64, 5, **kwargs)
#   hidden = tf.layers.conv2d(hidden, 128, 5, **kwargs)
# #   print(hidden)
#   hidden = tf.layers.conv2d(hidden, 1024, 3, strides=1)
#   # print(hidden)
#   hidden = tf.layers.flatten(hidden)
#
#   assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
#   hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
#       np.prod(hidden.shape[1:].as_list())])
#   return hidden                                                                # shape(50,50,1024)

def encoder(obs):
  print("*****************************************input shape is******************************************************", obs['image'].shape)
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())
 #  hidden = tf.reshape(obs['image'], [-1, 96, 96, 3])
  hidden = DenseNet(x=hidden, nb_blocks=nb_block, filters=growth_k).model
  assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [np.prod(hidden.shape[1:].as_list())])
  return hidden

def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.layers.dense(state, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
  hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs) 
  hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 3, 9, strides=3)
  mean = hidden
  assert mean.shape[1:].as_list() == [96, 96, 3], mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tools.MSEDistribution(mean)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
