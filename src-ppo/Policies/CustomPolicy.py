import torch
import random
import numpy as np

class CustomPolicy:
    def __init__(self, network=None, action_space=None):
        self.network = network
        self.action_space = action_space

    def predict(self, state, epsilon=0.0, device="cpu", deterministic=False):
        if deterministic:
            epsilon = 0.0
        if random.random() < epsilon and self.action_space is not None:
            return self.action_space.sample(), None

        if self.network is not None:
            if hasattr(self.network, "getArgMaxQPolicy"):
                action = self.network.getArgMaxQPolicy(state, self.action_space.n if self.action_space else 5, device)
            else:
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = self.network(state_tensor)
                    action = torch.argmax(q_values, dim=1).item()
            return action, None

        if self.action_space is not None:
            return self.action_space.sample(), None

        return 0, None