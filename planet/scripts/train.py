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

r"""Train a Deep Latent Planning agent.

Full training run:

python3 -m planet.scripts.train \
    --logdir /path/to/logdir \
    --config default \
    --params '{tasks: [cheetah_run]}'

For debugging:

python3 -m planet.scripts.train \
    --logdir ./log \
    --resume_runs False \
    --num_runs 1000 \
    --config debug \
    --params '{tasks: [cheetah_run]}'

To run multiple experiments using a smaller number of workers, pass
`--ping_every 30` to enable coordination between the workers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys

# Need offline backend to render summaries from within tf.py_func.
import matplotlib
matplotlib.use('Agg')

import ruamel.yaml as yaml
import tensorflow as tf


import sys
sys.path
sys.path.append('/home/zdx/github/Planet/planet')
sys.path.append('/home/zdx/github/Planet')
print(sys.path)


from planet import models
from planet import tools
from planet import training
from planet.scripts import configs


def start(logdir, args):
  with args.params.unlocked:
    args.params.logdir = logdir
  config = tools.AttrDict()
  with config.unlocked:
    config = getattr(configs, args.config)(config, args.params)  # function args.config('default'/'debug') is defined in configs(configs.py). task is defined.
  training.utility.collect_initial_episodes(config)
  return config


def resume(logdir, args):
  with args.params.unlocked:
    args.params.logdir = logdir
  config = tools.AttrDict()
  with config.unlocked:
    config = getattr(configs, args.config)(config, args.params)
  return config


def process(logdir, config, args):
  tf.reset_default_graph()
  dataset = tools.numpy_episodes(
      config.train_dir, config.test_dir, config.batch_shape,
      loader=config.data_loader,
      preprocess_fn=config.preprocess_fn,
      scan_every=config.scan_episodes_every,
      num_chunks=config.num_chunks,
      resize=config.resize,
      sub_sample=config.sub_sample,
      max_length=config.max_length,
      max_episodes=config.max_episodes,
      action_noise=config.fixed_action_noise)
  for score in training.utility.train(
      training.define_model, dataset, logdir, config):
    yield score


def main(args):
  training.utility.set_up_logging()
  experiment = training.Experiment(
      args.logdir,
      process_fn=functools.partial(process, args=args),
      start_fn=functools.partial(start, args=args),
      resume_fn=functools.partial(resume, args=args),
      num_runs=args.num_runs,
      ping_every=args.ping_every,
      resume_runs=args.resume_runs)
  for run in experiment:     # the method __iter__(self) defined in experiment.
    for unused_score in run: # the method __iter__(self) defined in run.
      pass


if __name__ == '__main__':
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--logdir', default='./log1')
  parser.add_argument(
      '--num_runs', type=int, default=1)
  parser.add_argument(
      '--config', default='default',
      help='Select a configuration function from scripts/configs.py.')
  parser.add_argument(
      '--params', default="{tasks: [carla]}", type=str,   # pendulum carla
      help='YAML formatted dictionary to be used by the config.')
  parser.add_argument(
      '--ping_every', type=int, default=0,
      help='Used to prevent conflicts between multiple workers; 0 to disable.')
  parser.add_argument(
      '--resume_runs', type=boolean, default=True,
      help='Whether to resume unfinished runs in the log directory.')
  args_, remaining = parser.parse_known_args()  # args_ = Namespace(config='default', logdir='./log_debug', num_runs=1, params={'tasks': ['cheetah_run']}, ping_every=0, resume_runs=True)
  args_.params = tools.AttrDict(yaml.safe_load(args_.params.replace('#', ',')))  # class AttrDict: """Wrap a dictionary to access keys as attributes."""
  args_.logdir = args_.logdir and os.path.expanduser(args_.logdir)
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)   # tf.app.run(main, argv): Runs the program with an optional 'main' function and 'argv' list.
