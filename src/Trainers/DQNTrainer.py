import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np

from Trainers.TrainConfig import TrainConfig
from Trainers.ReplayBuffer import ReplayBuffer
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

class DQNTrainer:
    def __init__(self, 
                 config: TrainConfig, 
                 target_network: nn.Module,
                 main_network: nn.Module, 
                 policy, 
                 env_name = "highway-fast-v0",  
                 env_config = None,
                 env = None,
                 use_gpu = False):
        self.timer = Timer()

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
        self.record = TrainResult(self.config.iterations, self.config.log_freq, name="DQN training", config=self.config, timer=self.timer)
        self.epsilon_step = (self.config.epsilon_min - self.config.epsilon_start)/(self.config.exploration_frac*self.config.iterations)

    def epsilon_update(self, epsilon):
      return max(self.config.epsilon_min, epsilon + self.epsilon_step)
      #return max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)

    def optimize_main(self, loss):
        self.main_net.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
          self.main_net.parameters(),
          max_norm=self.config.max_grad_norm
        )
        self.main_net.optimizer.step()       

    @Timer.measure("Prediction calculation")
    def _predict(self, state):
        return self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)

    @Timer.measure("Environment processing")
    def _environment_step(self, action):
        return self.env.step(action)

    @Timer.measure("Network processing")
    def _train_main_network(self, states, actions, rewards, next_states, dones):
        # calc y_target for every sample of batch using TARGET NETWORK
        with torch.no_grad():
          qvalues = self.target_net.getMaxQ(next_states, self.env.action_space.n, self.device)
        # rewards, dones and qvalues should all be on same device
        y_target = rewards + self.config.gamma * qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)
        # calc y_main using MAIN NETWORK (network of the agent)
        y_main = self.main_net.get_qvalues((states, actions, rewards, next_states, dones), self.device)
        # calc loss
        loss = self.main_net.loss_fn(y_main.squeeze(), y_target)
        loss_value = loss.item()
        # optimize
        self.optimize_main(loss)

        return loss_value

    @Timer.measure("Warmup duration")
    def _warmup(self):
        print("[+] === Warm up starts ===")
        warmup_samples = 0
        while warmup_samples < self.config.warmup_size:
            state, _ = self.env.reset()
            done = False
            while not done:
                 action, _ = self._predict(state)
                 new_state, reward, term, trunc, _ = self._environment_step(action)
                 done = term | trunc
                 self.r_buffer.push(
                   (
                      np.asarray(state, dtype=np.float32).flatten(),
                      action, 
                      reward, 
                      np.asarray(new_state, dtype=np.float32).flatten(),
                      term
                    )
                  )
                 warmup_samples += 1
                 state = new_state
            print(f"buffer filled with : {warmup_samples}/{self.config.warmup_size}")

    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0
        while iteration < self.config.iterations:
          state, _ = self.env.reset()
          done = False
          total_loss = 0
          episode_return = 0
          nb_steps = 0
          while not done:
            action, _ = self._predict(state)
            new_state, reward, term, trunc, _ = self._environment_step(action)
            episode_return += reward
            done = term | trunc
            nb_steps += 1
            self.r_buffer.push(
               (
                  np.asarray(state, dtype=np.float32).flatten(),
                  action, 
                  reward, 
                  np.asarray(new_state, dtype=np.float32).flatten(),
                  term
                )
              )
            if (iteration % self.config.train_freq == 0):
              # uniform draw of a batch
              states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
              total_loss += self._train_main_network(states, actions, rewards, next_states, dones)
            if iteration % self.config.copy_freq == 0:
              self.target_net.load_state_dict(self.main_net.state_dict())
              print("[!!] Weights of Target Network updated !")
            iteration += 1
            state = new_state
            self.config.epsilon = self.epsilon_update(self.config.epsilon)        
          
          self.record.add_episode(nb_steps,
                                  episode_return,
                                  loss=total_loss/nb_steps,
                                  epsilon=self.config.epsilon,
                                  iteration=iteration)

    def learn(self) -> TrainResult:
        self.timer.reset()

        self._warmup()
        warmup_duration = self.timer.counters["Warmup duration"]
        print(f"[+] Warmup done ! (Duration = {round(warmup_duration, 2)}s)")

        self._train()
        self.record.report_final()

        training_duration = self.timer.counters["Training duration"]

        print(f"[+] Total duration (warmup + training) = {warmup_duration + training_duration}s")

        print(f"[+] Environment processing duration : = {self.timer.counters['Environment processing']}s")
        print(f"[+] Network processing duration : = {self.timer.counters['Network processing']}s")
        print(f"[+] Prediction calculation duration : = {self.timer.counters['Prediction calculation']}s")

        self.timer.report()

        return self.record