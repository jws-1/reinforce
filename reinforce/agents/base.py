import gym


class BaseAgent:
    def __init__(self, env: gym.Env):
        self.env = env  # gym.wrappers.FlattenObservation(env)

    def train(self, num_episodes=1000):
        raise NotImplementedError(
            "train method must be implemented in derived classes."
        )
