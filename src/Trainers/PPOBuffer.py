from collections import deque
import numpy as np
import torch

class PPOBuffer:
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
        indices = np.random.choice(
            len(self.buffer),
            size=self.batch_size,
            replace=True
        )
        batch = [self.buffer[i] for i in indices]
        
        states = np.stack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.stack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        
        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(next_states).to(self.device),
            torch.from_numpy(dones).to(self.device)
        )