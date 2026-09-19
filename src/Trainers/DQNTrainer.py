import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
from HyperParams import HyperParams

class DQNTrainer:
    def __init__(self, 
                 hyperparams: HyperParams, 
                 target_network: nn.Module, 
                 main_network: nn.Module, 
                 r_buffer,
                 env_name: str,  
                 env_config=None):
        self.env = gym.make(env_name, config=env_config)
        self.hyperparams = hyperparams
        self.r_buffer = r_buffer
        self.target_net = target_network
        self.main_net = main_network
        self.target_network.load_state_dict(self.main_net.state_dict())

        