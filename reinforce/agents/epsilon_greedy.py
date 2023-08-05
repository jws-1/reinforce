from reinforce.agents.q_learning import QLearning

import gym
import numpy as np
import math
from typing import Optional, Callable


class EpsilonGreedy(QLearning):
    def __init__(
        self,
        env: gym.Env,
        q_0: Optional[np.float32] = 0.0,
        learning_rate: Optional[np.float32] = 0.1,
        epsilon: Optional[np.float32] = 0.5,
        decay_fn: Optional[Callable[[np.float32], np.float32]] = lambda e: e,
    ):
        self.epsilon = self.epsilon_0 = epsilon
        self.decay_fn = decay_fn
        super().__init__(env, q_0, learning_rate)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, next_state, reward, done):
        if done:
            self.epsilon = self.decay_fn(self.epsilon)
        return super().update(state, action, next_state, reward, done)
