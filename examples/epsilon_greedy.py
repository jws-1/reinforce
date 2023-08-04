from reinforce.agents import EpsilonGreedyAgent
import gym

env = gym.make("CliffWalking-v0")
agent = EpsilonGreedyAgent(env)
rewards = agent.train()
print(rewards)
