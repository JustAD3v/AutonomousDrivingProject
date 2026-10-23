import numpy as np
import torch

class PPOBuffer:
    def __init__(self, buffer_size, device, gamma, lm):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lm = lm
        self.device = device
        self.reset()

    def reset(self):
        self.buffer = []
        self.advantage_buffer = np.zeros(self.buffer_size)
        self.return_buffer = np.zeros(self.buffer_size)
        self.ptr = 0
        self.episode_start = 0
        self.full = False
        
    def push(self, state_tuple):
        # state_tuple = (state, action, reward, log_p, critic_value)
        if self.ptr >= self.buffer_size:
            raise IndexError("PPO buffer is full")
        self.buffer.append(state_tuple)
        self.ptr += 1
        self.full = self.ptr >= self.buffer_size

    def finish_episode(self, extra_critic_value):
        ep_slice = slice(self.episode_start, self.ptr)
        # calculate the advantages using GAE method
        advantages = []
        returns = []
        for start in range(self.episode_start, self.ptr):
            episode_slice = slice(start, self.ptr)
            episode_steps = self.buffer[episode_slice]
            rewards = np.array([e[2] for e in episode_steps])
            critic_values = np.array([e[-1] for e in episode_steps] + [extra_critic_value])
            delta = rewards + self.gamma * critic_values[1:] - critic_values[:-1]
            lm_factors = np.array([pow(self.lm, i) for i in range(self.ptr - start)])
            gamma_factors = np.array([pow(self.gamma, i) for i in range(self.ptr - start)])
            advantage = np.sum(delta * gamma_factors * lm_factors)
            return_val = np.sum(gamma_factors * rewards)
            advantages.append(advantage)
            returns.append(return_val)
        self.return_buffer[ep_slice] = returns
        self.advantage_buffer[ep_slice] = advantages
        # calculate return
        gamma_factors = np.array([pow(self.gamma, i) for i in range(self.ptr - self.episode_start)])
        episode_steps = self.buffer[ep_slice]
        rewards = np.array([e[2] for e in episode_steps])
         
        
    def get_batch(self):
        assert self.ptr == self.buffer_size
        states = np.stack([x[0] for x in self.buffer])
        actions = np.array([x[1] for x in self.buffer]) 
        log_action_probs = torch.tensor(
            [x[3] for x in self.buffer],
            dtype=torch.float32,
            device=self.device
        )
        advantages = self.advantage_buffer
        returns = self.return_buffer 
        
        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            log_action_probs.to(self.device),
            torch.from_numpy(advantages).to(self.device),
            torch.tensor(returns, dtype=torch.float32).to(self.device)
        )