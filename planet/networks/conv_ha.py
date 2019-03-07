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

def encoder(obs):
  """Extract deterministic features from an observation."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.reshape(obs['image'], [-1] + obs['image'].shape[2:].as_list())   # (50,50,64,64,3) reshape to (2500,64,64,3)
  hidden = tf.layers.conv2d(hidden, 24, 8, **kwargs) #####
  # hidden = tf.layers.conv2d(hidden, 32, 4, **kwargs)
  hidden = tf.layers.conv2d(hidden, 48, 5, **kwargs)
  hidden = tf.layers.conv2d(hidden, 64, 5, **kwargs)
  hidden = tf.layers.conv2d(hidden, 128, 5, **kwargs)
#   print(hidden)
  hidden = tf.layers.conv2d(hidden, 1024, 3, strides=1)
  # print(hidden)
  hidden = tf.layers.flatten(hidden)
  
  assert hidden.shape[1:].as_list() == [1024], hidden.shape.as_list()
  hidden = tf.reshape(hidden, tools.shape(obs['image'])[:2] + [
      np.prod(hidden.shape[1:].as_list())])
  return hidden                                                                # shape(50,50,1024)


def decoder(state, data_shape):
  """Compute the data distribution of an observation from its state."""
  kwargs = dict(strides=2, activation=tf.nn.relu)
  hidden = tf.layers.dense(state, 1024, None)
  hidden = tf.reshape(hidden, [-1, 1, 1, hidden.shape[-1].value])
  hidden = tf.layers.conv2d_transpose(hidden, 128, 5, **kwargs) 
  hidden = tf.layers.conv2d_transpose(hidden, 64, 5, **kwargs)
  hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
  # hidden = tf.layers.conv2d_transpose(hidden, 32, 6, **kwargs)
  # print(hidden)
  hidden = tf.layers.conv2d_transpose(hidden, 3, 9, strides=3)
  mean = hidden
  # print(mean)
  # assert mean.shape[1:].as_list() == [64, 64, 3], mean.shape
  assert mean.shape[1:].as_list() == [96, 96, 3], mean.shape
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tools.MSEDistribution(mean)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
