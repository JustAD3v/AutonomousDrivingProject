import torch
import random
import numpy as np

class TrainerWrapper:
    def __init__(self, trainer, trainer_type="dqn", config=None, seed=random.randint(0, 2**32 - 1)):
        self.trainer = trainer
        self.trainer_type = trainer_type
        self.config = config
        self.set_seed(seed)

    def learn(self):
        match self.trainer_type:
            case "sb3_dqn":
                self.trainer.learn(total_timesteps=self.config.iterations)
                return None
            case _:
                return self.trainer.learn()

    def save(self, path):
        match self.trainer_type:
            case "dqn":
                torch.save(self.trainer.target_net.state_dict(), path)
            case "wocar":
                torch.save(self.trainer.robust_net.state_dict(), path)
            case "ppo":
                torch.save(self.trainer.actor_network.state_dict(), path)                
            case _:
                self.trainer.save(path)
        print(f"[+] Target network saved to file : {path}")


    def predict(self, state, deterministic=True):
        match self.trainer_type:
            case "sb3_dqn":
                return self.trainer.predict(state, deterministic=deterministic)
            case "ppo":
                action, _ = self.trainer.policy.predict(state)
                return self.trainer.policy.to_env_action(action)
            case _:
                return self.trainer.policy.predict(state, epsilon=0.0)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)