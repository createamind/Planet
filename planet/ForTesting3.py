
from planet.envs.carla.env import CarlaEnv

env = CarlaEnv()

env.reset()
for t in range(1000):
    #img = env.render()
    s, _, done, _ = env.step(env.action_space.sample()+0.1)
    if done:
        print(t,done)