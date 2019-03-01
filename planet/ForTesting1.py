import gym
import time

import matplotlib.pyplot as plt
def displayImage(image):
    plt.imshow(image)
    plt.show()

env0 = gym.make("Breakout-v4")
env0.reset()
for t in range(1000):
    displayImage(env0.env.ale.getScreenRGB2())
    s, _, _, _ = env0.step(env0.action_space.sample())


class Wrapperx(object):
    def __init__(self,env):
        self._env = env

envgym = gym.make("Pendulum-v0")

#envgym = Wrapperx(envgym)

t0 = time.time()

envgym.reset()
for t in range(200):
  img = envgym.render(mode='rgb_array',render_size=(100,100))
  #img = envgym.render(mode='rgb_array',close=True)
  #print(img[50,40,2])
  #displayImage(img[20:80,20:80,...])
  s, _, _, _ = envgym.step(envgym.action_space.sample())

t1 = time.time()
print(t1-t0)