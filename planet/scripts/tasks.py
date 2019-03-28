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

import collections
import functools
import os

import numpy as np

import gym

from planet import control
from planet import networks
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat                     # max_length = 1000 // 8 = 125.
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance')
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup')
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin')
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run')
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 6)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch')
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk')
  return Task('walker_walk', env_ctor, max_length, state_components)


def humanoid_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = [
      'reward', 'com_velocity', 'extremities', 'head_height', 'joint_angles',
      'torso_vertical', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'humanoid', 'walk')
  return Task('humanoid_walk', env_ctor, max_length, state_components)


def _dm_control_env(action_repeat, max_length, domain, task):
  from dm_control import suite
  def env_ctor():
    env = control.wrappers.DeepMindWrapper(suite.load(domain, task), (64, 64))
    env = control.wrappers.ActionRepeat(env, action_repeat)
    env = control.wrappers.LimitDuration(env, max_length)
    env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = control.wrappers.ConvertTo32Bit(env)
    return env
  env = control.wrappers.ExternalProcess(env_ctor)
  return env


# gym classic_control
#=============================================================

def pendulum(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = [
      'reward', 'state']
  env_ctor = functools.partial(
      _dm_control_env_gym, action_repeat, max_length, 'Pendulum-v0')
  return Task('pendulum', env_ctor, max_length, state_components)


class DeepMindWrapper_gym(object):
  """Wraps a Gym environment into an interface for downstream process"""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id
    self.observation_space = gym.spaces.Dict({'state':self.observation_space})

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {'state':obs}
    return obs, reward, done, info

  def reset(self):
    return {'state':self._env.reset()}

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.render(mode='rgb_array',render_size=(100,100))[18:82,18:82]  # pendulum.py is modified.



def _dm_control_env_gym(action_repeat, max_length, env_name):
  import gym
  def env_ctor():
    env = gym.make(env_name)     # 'Pendulum-v0'
    env = env.env                # 'remove TimeLimit wrapper
    env = DeepMindWrapper_gym(env, (96, 96))
    env = control.wrappers.ActionRepeat(env, action_repeat)
    env = control.wrappers.LimitDuration(env, max_length)
    env = control.wrappers.PixelObservations(env, (96, 96), np.uint8, 'image')
    env = control.wrappers.ConvertTo32Bit(env)
    return env
  env = control.wrappers.ExternalProcess(env_ctor)
  return env


# carla
#=============================================================


def carla(config, params):
  action_repeat = params.get('action_repeat', 1)
  # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
  max_length = 100 // action_repeat
  state_components = [
      'reward', 'state']
  env_ctor = functools.partial(
    _dm_control_env_carla, action_repeat, max_length, 'carla')
  return Task('carla', env_ctor, max_length, state_components)


class DeepMindWrapper_carla(object):
  """Wraps a Gym environment into an interface for downstream process"""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(96, 96), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id
    self.observation_space = gym.spaces.Dict({'state':gym.spaces.Box(low=-1,high=1,shape=(1,))})

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    self.img, reward, done, info = self._env.step(action)
    obs = {'state':np.array([0.0])}
    return obs, reward, done, {}

  def reset(self):
    self.img = self._env.reset()
    return {'state':np.array([0.0])}

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self.img



def _dm_control_env_carla(action_repeat, max_length, env_name):
  assert env_name == 'carla'
  from planet.envs.carla.env import CarlaEnv
  def env_ctor():
    env = CarlaEnv()
    env = DeepMindWrapper_carla(env, (96, 96))
    env = control.wrappers.ActionRepeat(env, action_repeat)
    env = control.wrappers.LimitDuration(env, max_length)
    env = control.wrappers.PixelObservations(env, (96, 96), np.float32, 'image')
    env = control.wrappers.ConvertTo32Bit(env)
    return env
  env = control.wrappers.ExternalProcess(env_ctor)
  return env


