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

from planet.control import wrappers


def random_episodes(env_ctor, num_episodes, output_dir=None):
  env = env_ctor()
  env = wrappers.CollectGymDataset(env, output_dir)
  episodes = []
  for _ in range(num_episodes):
    policy = lambda env, obs: env.action_space.sample()
    done = False
    obs = env.reset()
    # cnt = 0
    while not done:
      action = policy(env, obs)
      obs, _, done, info = env.step(action)  # env.step
    #   cnt += 1
    # print(cnt)
    episodes.append(info['episode'])  # if done is True, info stores the 'episode' information and 'episode' is written in a file(e.g. "~/planet/log_debug/00001/test_episodes").
  return episodes
