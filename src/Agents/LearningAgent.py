from Agents.AbstractAgent import Agent
import random
import numpy as np

class LearningAgent(Agent):
    def __init__(self):
        pass

    def policy(self, state):
        """ Epsilon greedy policy """
        if random.uniform(0,1) < self.config.epsilon:
              return self.env.action_space.sample()
        else:
            return self.model.getArgMaxQPolicy(state, self.env.action_space.n, self.device)
