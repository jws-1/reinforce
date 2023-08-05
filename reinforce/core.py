import gym
import numpy as np
from reinforce.agents import BaseAgent
from typing import Optional


class Core:
    def __init__(self, agent: BaseAgent, env: gym.Env):
        self.agent = agent
        self.env = env

    def learn(self, n_episodes: Optional[int] = 1000) -> np.ndarray:
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            s, _ = self.env.reset()
            done = False
            print(i, self.agent.epsilon)
            while not done:
                a = self.agent.select_action(s)
                s_, r, done, _, _ = self.env.step(a)
                self.agent.update(s, a, s_, r, done)
                s = s_
                rewards[i] += r
        return rewards
