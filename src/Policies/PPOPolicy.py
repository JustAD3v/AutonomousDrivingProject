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
        action_probs = self.network(state_tensor)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample().item()
        return action, distribution[action]