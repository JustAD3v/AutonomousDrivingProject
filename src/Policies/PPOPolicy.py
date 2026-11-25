import torch
import random
import numpy as np

class PPOPolicy:
    def __init__(self, network=None, action_space=None):
        self.network = network
        self.action_space = action_space

    def predict(self, state, device="cpu", deterministic=False):
        with torch.no_grad():
            state_flat = np.asarray(state, dtype=np.float32).flatten()
            state_tensor = torch.tensor(state_flat, dtype=torch.float32).to(device)
            distribution = self.network.distribution(state_tensor)
            if deterministic:
                # greedy action : the gaussian mean
                action = distribution.mean if self.network.continuous else distribution.probs.argmax(-1)
            else:
                action = distribution.sample()
        if self.network.continuous:
            log_prob = distribution.log_prob(action)
            # raw sampled action, consistent with log_prob (use to_env_action before env.step)
            return action.cpu().numpy(), log_prob
        return action.item(), distribution.log_prob(action)

    def to_env_action(self, action):
        if not self.network.continuous:
            return action
        # map the raw action to the environment action space and clip it to its bounds
        low, high = self.action_space.low, self.action_space.high
        action = ((high - low) / 2) * action + ((high + low) / 2)
        return np.clip(action, low, high)