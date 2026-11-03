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
        # subclasses may append extra elements after critic_value
        if self.ptr >= self.buffer_size:
            raise IndexError("PPO buffer is full")
        self.buffer.append(state_tuple)
        self.ptr += 1
        self.full = self.ptr >= self.buffer_size

    def merge(self, PPOBuffer_list):
        self.advantage_buffer = np.array([])
        self.return_buffer = np.array([])
        for worker_PPOBuffer in PPOBuffer_list:
            self.buffer += worker_PPOBuffer.buffer
            self.advantage_buffer = np.concatenate((self.advantage_buffer, 
                                                    worker_PPOBuffer.advantage_buffer))
            self.return_buffer = np.concatenate((self.return_buffer, 
                                                worker_PPOBuffer.return_buffer))
            self.ptr += worker_PPOBuffer.ptr

    def finish_episode(self, extra_critic_value):
        segment_len = self.ptr - self.episode_start
        segment_slice = slice(self.episode_start, self.ptr)
        segment = self.buffer[segment_slice]

        advantages = np.zeros(segment_len)
        returns = np.zeros(segment_len)

        rewards = np.array([e[2] for e in segment])
        critic_values = np.array([e[4] for e in segment] + [extra_critic_value])        

        # calculate the advantages using GAE method + calculate return
        gae = 0

        # bootstrap rewards with the extra critic value as done in spinning up library
        # The extra critic value is used as an estimator for futur rewards 
        # starting from the last state (the one which was truncated) 
        total_return = extra_critic_value
        
        # backward calculation
        for t in reversed(range(segment_len)):
            delta = rewards[t] + self.gamma * critic_values[t+1] - critic_values[t] 
            gae = delta + self.lm * self.gamma * gae
            total_return = rewards[t] + self.gamma * total_return # add bootstrapped critic value for next state ?
            advantages[t] = gae
            returns[t] = total_return

        self.return_buffer[segment_slice] = returns
        self.advantage_buffer[segment_slice] = advantages
        self.episode_start = self.ptr
         
        
    def batch_tensors(self):
        states = np.stack([x[0] for x in self.buffer])
        actions = np.array([x[1] for x in self.buffer]) 
        log_action_probs = torch.tensor(
            [x[3] for x in self.buffer],
            dtype=torch.float32,
            device=self.device
        )
        advantages = (self.advantage_buffer - self.advantage_buffer.mean()) / (self.advantage_buffer.std() + 1e-8)
        returns = self.return_buffer 

        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            log_action_probs.to(self.device),
            torch.from_numpy(advantages).to(self.device),
            torch.tensor(returns, dtype=torch.float32).to(self.device)
        )

    def get_batch(self, minibatch_size=1):
        assert self.ptr == self.buffer_size
        result = self.batch_tensors()

        # split_tensors = [
        # (states_split, ...), 
        # (actions_split, ...), 
        # (log_prob_split, ...), 
        # (adv_split, ...), 
        # (returns_split, ...)
        #]
        split_tensors = [
            torch.split(tensor, minibatch_size, dim=0)
            for tensor in result
        ]

        # zip(*split_tensors) = zip(states_split, actions_split, log_prob_split, adv_split, returns_split)
        # return [result_split1, result_split2, ...] with minibatch_size * result_plit
        return [tuple(parts) for parts in zip(*split_tensors)]