from reinforce.agents.base import BaseAgent

import gym
import numpy as np
from typing import Optional


class QLearning(BaseAgent):
    def __init__(
        self,
        env: gym.Env,
        q_0: Optional[np.float32] = 0.0,
        learning_rate: Optional[np.float32] = 0.1,
        model: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.alpha = learning_rate
        self.Q = np.full((env.nS, env.nA), q_0)
        super().__init__(env)

    def update(self, state, action, next_state, reward, done):
        self.Q[state, action] = self.alpha * (
            reward + np.max(self.Q[next_state]) - self.Q[state, action]
        )
