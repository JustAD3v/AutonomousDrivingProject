from AbstractAgent import Agent
import random
import torch
import numpy as np

class LearningAgent(Agent):
    def __init__(self, device):
        super().__init__(device)

    @property
    def env(self):
        return self.env

    @property
    def config(self):
        return self.config 

    @property
    def model(self):
        return self.model

    def policy(self, state):
        """ Epsilon greedy policy """
        if random.uniform(0,1) < self.config.epsilon:
              return self.env.action_space.sample()
        else:
            return self.getArgMaxQPolicy(state)

    def getArgMaxQPolicy(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(np.ravel(state)) # (25,)
            # unsqueeze and expand to have => (5, 25)
            state_t_exp = torch.unsqueeze(state_t, 0).expand(5, 25)
            # append actions => (5,5) + (5, 25) => (5, 30)
            actions = torch.tensor([self.one_hot(x) for x in range(self.env.action_space.n)])
            input_tensor = torch.cat((state_t_exp,actions), dim=1)
            # get qvalues from model
            # calling model(...) instead of model.forward(..) works and is better because we named it forward
            qvalues = self.model(input_tensor.to(self.device)) # out => (5,)
            return torch.argmax(qvalues).item()

    def process_batch(self, batch_tuple):
        states, actions, _, _, _ = batch_tuple
        one_hot_actions = torch.tensor([self.one_hot(x) for x in actions]) # (B,5)
        batch_input = torch.cat((states, one_hot_actions), dim=1) # turn states = (B, 25) to => (B, 30)
        # calling model(...) instead of model.forward(..) works and is better because we named it forward
        return self.model(batch_input.to(self.device))

    def one_hot(x):
        l = [0.0,0.0,0.0,0.0,0.0]
        l[x] = 1.0
        return l