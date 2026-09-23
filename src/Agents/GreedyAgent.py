from Agents.AbstractAgent import Agent
import torch
import numpy as np

class GreedyAgent(Agent):
    def __init__(self):
        pass

    def policy(self, state):
        """ greedy policy """
        return self.model.getArgMaxQPolicy(state, self.env.action_space.n, self.device)
