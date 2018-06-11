import gym
import numpy as np

env = gym.envs.make("MountainCar-v0")

env.reset()
env.render()

for x in range(1000):
    # env.step(env.action_space.sample())
    # env.step(0)
    env.step(1)
    # env.step(2)
    env.render()

env.close()
