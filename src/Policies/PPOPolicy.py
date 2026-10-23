import torch
import random
import numpy as np

class PPOPolicy:
    def __init__(self, network=None, action_space=None):
        self.network = network
        self.action_space = action_space

    def predict(self, state, device="cpu", deterministic=True):
        with torch.no_grad():
            state_flat = np.asarray(state, dtype=np.float32).flatten()
            state_tensor = torch.tensor(state_flat, dtype=torch.float32).to(device)
            output = self.network(state_tensor)
            # Categorical applies softmax on network output
            # to get a probability distribution from the output vector
            # it also wraps it into an interface that allows for
            # sampling that distribution
            # and finding the log_prob for a given action
            distribution = torch.distributions.Categorical(logits=output)
            action = distribution.sample()
        return action.item(), distribution.log_prob(action)