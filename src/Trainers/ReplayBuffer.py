from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, warmup_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.device = device

    def push(self, state_tuple):
        self.buffer.append(state_tuple)

    def warmed_up(self):
        return len(self.buffer) >= self.warmup_size

    def get_batch(self):
        """ Draws a uniform batch of size <self.batch_size> from buffer """
        shuffled_idx = np.random.choice(np.arange(len(self.buffer)), size=self.batch_size, replace=False)
        batch = [self.buffer[i] for i in shuffled_idx]
        # /!\ This part is too inefficient and lowers the performance a lot
        states = torch.tensor(np.array([r[0] for r in batch])).flatten(start_dim=1) # flatten from (5, 5) to (25,)
        actions = torch.tensor(np.array([r[1] for r in batch]))
        rewards = torch.tensor(np.array([r[2] for r in batch]), device=self.device, dtype=torch.float32)
        next_states = torch.tensor(np.array([r[3] for r in batch])).flatten(start_dim=1) # flatten from (5, 5) to (25,)
        dones = torch.tensor(np.array([r[4] for r in batch]), device=self.device, dtype=torch.float32) # convert to float to make operation (1-dones) possible
        return (states, actions, rewards, next_states, dones)