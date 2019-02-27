import gym
import numpy as np

env0 = gym.make("Breakout-v4")
env0.reset()
for t in range(1000):
  print(env0.env.ale.getScreenRGB2()[10, 10, 1])
  s, _, _, _ = env0.step(env0.action_space.sample())



class DeepMindWrapper(object):
  """Wraps a DM Control environment into a Gym interface."""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(
        action_spec.minimum, action_spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)



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
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)



# envgym = gym.make("Breakout-v4")
# envgym = DeepMindWrapper_gym(envgym)
# envgym.reset()
# for t in range(1000):
#   img = envgym.render()
#   s, _, _, _ = envgym.step(envgym.action_space.sample())


from dm_control import suite
# from planet.control.wrappers import DeepMindWrapper
#env = suite.load('cheetah', 'run')
env = suite.load('walker','walk')
env = DeepMindWrapper(env)

env.reset()
for t in range(1000):
  img = env.render()
  s, _, _, _ = env.step(env.action_space.sample())

print(env.action_space)
print(env.observation_space)

# env = DeepMindWrapper(env)
