import gym


class BaseAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def select_action(self, state):
        raise NotImplementedError(
            "`select_action` method must be implemented in derived classes."
        )

    def update(self, state, action, next_state, reward, done):
        raise NotImplementedError(
            "`update` method must be implemented in derived classes."
        )
