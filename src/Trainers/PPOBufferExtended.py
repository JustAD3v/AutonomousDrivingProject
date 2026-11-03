import numpy as np
import torch

from Trainers.PPOBuffer import PPOBuffer

class PPOBufferExtended(PPOBuffer):
    """PPOBuffer that also keeps the next state and the done flag of every
    transition, so a Q-value target can be computed as in the DQN version."""

    def push(self, state_tuple):
        # state_tuple = (state, action, reward, log_p, critic_value, next_state, done)
        if len(state_tuple) != 7:
            raise ValueError("PPOBufferExtended expects (state, action, reward, log_p, critic_value, next_state, done)")
        super().push(state_tuple)

    def batch_tensors(self):
        rewards = np.array([x[2] for x in self.buffer], dtype=np.float32)
        next_states = np.stack([x[5] for x in self.buffer])
        dones = np.array([x[6] for x in self.buffer], dtype=np.float32)

        return super().batch_tensors() + (
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(next_states).to(self.device),
            torch.from_numpy(dones).to(self.device)
        )
