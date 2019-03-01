import gym
import time

import matplotlib.pyplot as plt
def displayImage(image):
    plt.imshow(image)
    plt.show()



class Wrapperx(object):
    def __init__(self,env):
        self._env = env

envgym = gym.make("Pendulum-v0")
envgym = envgym.env

#envgym = Wrapperx(envgym)

t0 = time.time()

envgym.reset()
for t in range(1000):
    img = envgym.render()
    # img = envgym.render(mode='rgb_array');displayImage(img)
    #img = envgym.render(mode='rgb_array',render_size=(100,100));displayImage(img[18:82,18:82])
    #img = envgym.render(mode='rgb_array',close=True)
    #print(img[50,40,2])
    s, _, done, _ = envgym.step(envgym.action_space.sample())
    if done:
        print(t,done)

t1 = time.time()
print(t1-t0)