from reinforce.agents.base import BaseAgent

import numpy as np


class EpsilonGreedyAgent(BaseAgent):
    def train(
        self, num_episodes=1000, q_0=0.0, epsilon=0.1, alpha=0.1, gamma=1.0
    ) -> np.array:
        rewards = np.zeros(num_episodes)
        N_s = self.env.nS
        N_a = self.env.nA
        Q = np.full(
            (N_s, N_a),
            fill_value=q_0,
        )

        for i in range(num_episodes):
            s, _ = self.env.reset()
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(Q[s])
                s_1, r, done, _, _ = self.env.step(a)
                Q[s, a] = alpha * (r + gamma * np.max(Q[s_1]) - Q[s, a])
                s = s_1
                rewards[i] += r
        return rewards
