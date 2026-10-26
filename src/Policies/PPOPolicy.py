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
            distribution = self.network.distribution(state_tensor)
            action = distribution.sample()
        if self.network.continuous:
            # the environment clips the action to its bounds, the unclipped sample is kept for the log_prob
            return action.cpu().numpy(), distribution.log_prob(action)
        return action.item(), distribution.log_prob(action)