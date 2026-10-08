import torch
import random
import numpy as np

class DQNWrapper:
    def __init__(self, model, use_custom=True, config=None, seed=random.randint(0, 2**32 - 1)):
        self.model = model
        self.use_custom = use_custom
        self.config = config
        self.set_seed(seed)

    def learn(self):
        if self.use_custom:
            return self.model.learn()
        else:
            self.model.learn(total_timesteps=self.config.iterations)
            return None

    def save(self, path):
        if self.use_custom:
            torch.save(self.model.target_net.state_dict(), path)
        else:
            self.model.save(path)
        print(f"[+] Target network saved to file : {path}")


    def predict(self, state, deterministic=True):
        if self.use_custom:
            return self.model.policy.predict(state, epsilon=0.0)
        else:
            return self.model.predict(state, deterministic=deterministic)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)