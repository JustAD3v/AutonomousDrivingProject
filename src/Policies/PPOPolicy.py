import torch
import random
import numpy as np

class PPOPolicy:
    def __init__(self, network=None, action_space=None):
        self.network = network
        self.action_space = action_space

    def predict(self, state, epsilon=0.0, device="cpu", deterministic=False):
        # investigate this weird np conversion for no reason ???
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        output = self.network(state_tensor)
        # Categorical applies softmax on network output
        # to get a probability distribution from the output vector
        # it also wraps it into an interface that allows for
        # sampling that distribution
        # and finding the log_prob for a given action
        distribution = torch.distributions.Categorical(output)
        action = distribution.sample().item()
        return action, distribution.log_prob(action)