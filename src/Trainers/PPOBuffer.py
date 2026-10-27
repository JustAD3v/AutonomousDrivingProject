import numpy as np
import torch

class PPOBuffer:
    def __init__(self, buffer_size, device, gamma, lm):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(self.buffer_size)
        self.advantage_buffer = np.zeros(self.buffer_size)
        self.return_buffer = np.zeros(self.buffer_size)
        self.ptr = 0
        self.episode_start = 0
        self.gamma = gamma
        self.lm = lm
        self.device = device
        self.full = False

    def push(self, state_tuple):
        # state_tuple = (state, action, reward, log_p, critic_value)
        self.buffer.append(state_tuple)
        self.ptr = min(self.ptr + 1, self.buffer_size-1)
        self.full = self.ptr == self.buffer_size-1

    def finish_episode(self, extra_critic_value):
        # calculate the advantages using GAE method
        advantages = np.zeros(self.ptr - self.episode_start)
        for start in range(self.episode_start, self.ptr):
            episode_slice = slice(start, self.ptr)
            episode_steps = self.buffer[episode_slice]
            rewards = np.array([e[2] for e in episode_steps])
            critic_values = np.array([e[-1] for e in episode_steps] + [extra_critic_value])
            delta = rewards + self.gamma * critic_values[1:] - critic_values[:-1]
            factors = np.array([pow(self.gamma * self.lm, i) for i in range(self.buffer_size)])
            advantage = np.sum(delta * factors)
            advantages.append(advantage)
        self.advantage_buffer[ep_slice] = advantages
        # calculate return
        ep_slice = slice(self.episode_start, self.ptr)
        gamma_factors = np.array([pow(self.gamma, i) for i in range(self.buffer_size)])
        episode_steps = self.buffer[ep_slice]
        rewards = np.array([e[2] for e in episode_steps])
        self.return_buffer[ep_slice] = [
            np.sum(gamma_factors * rewards) 
            for _ in range(self.ptr-self.episode_start)
        ] 
        
    def get_batch(self):
        states = np.stack([x[0] for x in self.buffer])
        actions = np.array([x[1] for x in self.buffer]) 
        log_action_probs = np.array([x[3] for x in self.buffer])
        critic_values = np.array([x[-1] for x in self.buffer])
        advantages = self.advantage_buffer
        returns = self.return_buffer 
        
        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(log_action_probs).to(self.device),
            torch.from_numpy(critic_values).to(self.device),
            torch.from_numpy(advantages).to(self.device),
            torch.from_numpy(returns).to(self.device)
        )