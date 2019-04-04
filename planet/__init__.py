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
import datetime
import os
import os.path
from os import path


ENV_CONFIG1 = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "None",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 2,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG2 = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "soft",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 4,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG3 = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "hard",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 4,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG4 = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "soft",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 5,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG5 = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "hard",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 5,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG_test = {
    "x_res": 96,
    "y_res": 96,
    "image_mode": "encode",
    "host": "localhost",
    "early_stop": True,        # if we use planet this has to be False
    "attention_mode": "None",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 2,           # 4 for one point attention, 5 for control view field
}


ENV_CONFIG = ENV_CONFIG1

LARGE_NET = False

PID_FILE_NAME = '/tmp/pid_mode_{}_channel_{}_dim_{}.txt'.format(ENV_CONFIG['attention_mode'], ENV_CONFIG['attention_channel'], ENV_CONFIG['action_dim'])

# ENV_CONFIG = ENV_CONFIG3
LOG_NAME = './log_mode_{}_channel_{}_dim_{}'.format(ENV_CONFIG['attention_mode'], ENV_CONFIG['attention_channel'], ENV_CONFIG['action_dim'])

