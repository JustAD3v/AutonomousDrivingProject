from Agents.AbstractAgent import Agent
import torch
import numpy as np

class OptimalAgent(Agent):
    def __init__(self):
        pass

    def policy(self, state):
        """ greedy policy """
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
        pass

    def one_hot(self, x):
        l = [0.0,0.0,0.0,0.0,0.0]
        l[x] = 1.0
        return l