import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
import time

from Trainers.TrainConfig import TrainConfig
from Trainers.ReplayBuffer import ReplayBuffer
from Trainers.TrainResult import TrainResult

class DQNTrainer:
    def __init__(self, 
                 config: TrainConfig, 
                 target_network: nn.Module,
                 main_network: nn.Module, 
                 policy, 
                 env_name = "highway-v0",  
                 env_config = None,
                 env = None,
                 use_gpu = False):
        if env is not None:
            self.env = env
        else:
            self.env = gym.make(env_name, config=env_config)
            
        self.config = config
        self.policy = policy
        self.main_net = main_network
        self.target_net = target_network
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.r_buffer = ReplayBuffer(self.config.buffer_size, 
                                     config.batch_size,
                                     config.warmup_size,
                                     self.device)
        self.record = TrainResult(self.config.iterations)

    def optimize_main(self, loss):
        self.main_net.optimizer.zero_grad()
        loss.backward()
        self.main_net.optimizer.step()

    def learn(self) -> TrainResult:
        start = time.time()
        print("[+] === Warm up starts ===")
        warmup_samples = 0
        while warmup_samples < self.config.warmup_size:
            state, _ = self.env.reset()
            done = False
            while not done:
                 action, _ = self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)
                 new_state, reward, term, trunc, _ = self.env.step(action)
                 done = term | trunc
                 self.r_buffer.push(
                   (
                      np.asarray(state, dtype=np.float32).flatten(),
                      action, 
                      reward, 
                      np.asarray(new_state, dtype=np.float32).flatten(),
                      done
                    )
                  )
                 warmup_samples += 1
                 state = new_state
            print(f"buffer filled with : {warmup_samples}/{self.config.warmup_size}")
        warmup_duration = time.time()-start
        print(f"[+] Warmup done ! (Duration = {round(warmup_duration, 2)}s)")

        start = time.time()
        print("[+] === Training starts ===")
        iteration = 0
        episode_count = 0
        while iteration < self.config.iterations:
          state, _ = self.env.reset()
          done = False
          total_loss = 0
          episode_return = 0
          nb_steps = 0
          while not done:
            action, _ = self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)
            new_state, reward, term, trunc, _ = self.env.step(action)
            episode_return += reward
            done = term | trunc
            nb_steps += 1
            self.r_buffer.push(
               (
                  np.asarray(state, dtype=np.float32).flatten(),
                  action, 
                  reward, 
                  np.asarray(new_state, dtype=np.float32).flatten(),
                  done
                )
              )
            if (iteration % self.config.train_freq == 0):
              # uniform draw of a batch
              states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
              # calc y_target for every sample of batch using TARGET NETWORK
              qvalues = self.target_net.getMaxQ(next_states, self.env.action_space.n, self.device)
              # rewards, dones and qvalues should all be on same device
              y_target = rewards + self.config.gamma * qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)
              # calc y_main using MAIN NETWORK (network of the agent)
              y_main = self.main_net.process_batch((states, actions, rewards, next_states, dones), self.device) 
              # calc loss
              loss = self.main_net.loss_fn(y_main.squeeze(), y_target)
              total_loss += loss.item()
              # optimize
              self.optimize_main(loss)
            if iteration % self.config.copy_freq == 0:
              self.target_net.load_state_dict(self.main_net.state_dict())
              print("[!!] Weights of Target Network updated !")
            iteration += 1
            state = new_state
        
          print(f"[*] ({iteration}/{self.config.iterations}) | Episode finished (len={nb_steps}), avg_loss = {total_loss/nb_steps}, tot_return={episode_return}, epsilon={self.config.epsilon}")
          self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)
          episode_count += 1
          self.record.add_loss(total_loss/nb_steps)
          self.record.add_episode_return(episode_return)
          self.record.add_epsilon(self.config.epsilon)
          self.record.register_episode_count(episode_count)

        training_duration = time.time()-start
        print(f"[+] Total duration (warmup + training) = {warmup_duration + training_duration}s")

        torch.save(self.target_net.state_dict(), save_file)
        return self.record