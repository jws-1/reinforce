import gym

from reinforce import Core
from reinforce.agents import EpsilonGreedy
import numpy as np

env = gym.make("CliffWalking-v0")
eps = 1.0
eps_min = 0.01
num_episodes = 1000
agent = EpsilonGreedy(
    env,
    epsilon=np.float32(eps),
    decay_fn=lambda e: e * ((eps_min / eps) ** (1 / num_episodes)),
)
runner = Core(agent, env)
rewards = runner.learn(num_episodes)
print(rewards)
