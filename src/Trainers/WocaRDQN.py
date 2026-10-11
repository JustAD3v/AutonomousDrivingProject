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
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

class WocaRDQN:
    def __init__(self, 
                 config: TrainConfig, 
                 target_network: nn.Module,
                 main_network: nn.Module, 
                 woca_network: nn.Module, 
                 robust_network: nn.Module, 
                 policy, 
                 env_name = "highway-fast-v0",  
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
        self.woca_net = woca_network
        self.robust_net = robust_network
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.r_buffer = ReplayBuffer(self.config.buffer_size, 
                                     config.batch_size,
                                     config.warmup_size,
                                     self.device)
        self.record = TrainResult(self.config.iterations)
        self.epsilon_step = (self.config.epsilon_min - self.config.epsilon_start)/(self.config.exploration_frac*self.config.iterations)

        # wocaR params
        self.k_wst = 0.5
        self.k_reg = 0.2
        self.attack_bound = 0.5
        self.grad_step_size = 5e-4
        self.grad_steps = 100

    def epsilon_update(self, epsilon):
      return max(self.config.epsilon_min, epsilon + self.epsilon_step)
      #return max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)

    def optimize_main(self, loss):
        self.main_net.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
          self.main_net.parameters(),
          max_norm=10.0
        )
        self.main_net.optimizer.step()

    def optimize_woca(self, loss):
      self.woca_net.optimizer.zero_grad()
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(
        self.woca_net.parameters(),
        max_norm=10.0
      )
      self.woca_net.optimizer.step()
    
    def optimize_robust(self, loss):
      self.robust_net.optimizer.zero_grad()
      loss.backward()
      
      torch.nn.utils.clip_grad_norm_(
        self.robust_net.parameters(),
        max_norm=10.0
      )
      self.robust_net.optimizer.step()

    def getWocaMinQ(self, subset_actions, states):
      qvalues = self.woca_net(states.to(self.device)) # (B, 5)
      # for each set of 5 qvalues only keep the qvalues associated
      # to actions that are in subset_actions
      qvalues = [
         qvalues[row, idx]
         for row, idx in enumerate(subset_actions)
      ]
      # keep only the minimum qvalue
      minQValues = torch.min(qvalues, dim=1).values
      return minQValues

    def filter_actions(self, lb_q, ub_q):
      actions = []
      for (lb, ub) in zip(lb_q, ub_q):
          actions.append([
             # for each action i,
             i for i in range(len(lb)) 
             # if there is an action j (i != j) with a lower bound lower than ub[i]
             # then this action might be chosen 
             # because it's qvalue domain covers a part of the domain of another action
             # You can think it the opposite way :
             # if all lb of other actions j are greater than ub[i],
             # this action will never be chosen because its qvalue will always be lower
             # than the one of the other actions 
             if any([ub[i] > lb[j] for j in range(len(lb)) if j != i])  
          ])
      return actions
    
    def getSubsetActions(self, states):
      # get lower and upper bound of perturbated states
      lb_states = torch.tensor(np.add(states, -self.attack_bound), dtype=torch.float32, device=self.device)
      ub_states = torch.tensor(np.add(states, self.attack_bound), dtype=torch.float32, device=self.device)
      # wrap model in auto_LiRPA framework
      lirpa_model = BoundedModule(self.robust_net, torch.empty_like(states))

      # Compute bounds using LiRPA using the given lower and upper bounds.
      norm = float("inf")
      ptb = PerturbationLpNorm(norm = norm, x_L=lb_states, x_U=ub_states)
      bounded_qvalues = BoundedTensor(states, ptb)

      # Compute bounds using CROWN
      lb_qvalues, ub_qvalues = lirpa_model.compute_bounds(x=(bounded_qvalues,), method='CROWN')
      print(f'CROWN bounds: lower={lb_qvalues.item()}, upper={ub_qvalues.item()}')

      # filter actions
      return self.filter_actions(lb_qvalues, ub_qvalues)

    def getDistMax(self, states, grad_step_size,grad_steps):
      # use require grad on states
      states = states.requires_grad_(True)
    
      # get distribution from qvalues
      qvalues = self.robust_net(states)
      probs = F.softmax(states, dim=1)

      # init delta with random pertubration in [-eps; eps]
      delta = torch.empty_like(states).uniform_(-self.attack_bound, self.attack_bound).requires_grad_(True)

      # find perturbation delta that maximizes the distance using gradient ascent
      dist = None
      for _ in range(grad_steps):
        p_states = torch.add(states, delta)
        p_qvalues = self.robust_net(p_states)
        p_probs = F.softmax(p_qvalues, dim=1)
        dist = F.kl_div(p_probs, probs)
        grad = torch.autograd.grad(outputs=dist, inputs=p_states)
        delta = delta + grad_step_size*grad      
      return dist

    def getStateWeight(self, states):
      max_qvalues = self.robust_net.getMaxQ(states, self.env.action_space.n, self.device)
      min_qvalues = self.robust_net.getMinQ(states, self.env.action_space.n, self.device)
      return torch.sub(max_qvalues, min_qvalues)
    
    def calcRegularization(self, states, grad_step_size, grad_steps):
      w_states = self.getStateWeight(states)
      max_dist = self.getDistMax(states, grad_step_size, grad_steps)
      reg = torch.mul(w_states, max_dist)
      return reg.mean()
      
    def learn(self) -> TrainResult:
        start = time.time()
        env_time = 0
        network_time = 0
        predict_time = 0
        print("[+] === Warm up starts ===")
        warmup_samples = 0
        while warmup_samples < self.config.warmup_size:
            state, _ = self.env.reset()
            done = False
            while not done:
                 predict_time_start = time.time()
                 action, _ = self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)
                 predict_time += time.time() - predict_time_start
                 env_time_start = time.time()
                 new_state, reward, term, trunc, _ = self.env.step(action)
                 env_time += time.time() - env_time_start
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
            predict_time_start = time.time()
            action, _ = self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)
            predict_time += time.time() - predict_time_start
            env_time_start = time.time()
            new_state, reward, term, trunc, _ = self.env.step(action)
            env_time += time.time() - env_time_start
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
              # ============ TRAIN MAIN NETWORK ============
              # uniform draw of a batch
              states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
              # calc y_target for every sample of batch using TARGET NETWORK
              network_time_start = time.time()
              with torch.no_grad():
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
              network_time += time.time() - network_time_start

              # ============ TRAIN WOCA NETWORK ============
              subset_actions = self.getSubsetActions(next_states)
              woca_qvalues = self.getWocaMinQ(subset_actions, next_states)
              y_woca_target = rewards + self.config.gamma * woca_qvalues * (1-dones)
              y_woca_predicted = self.woca_net.process_batch((states, actions, rewards, next_states, dones), self.device)
              woca_loss = self.woca_net.loss_fn(y_woca_predicted.squeeze(), y_woca_target)
              self.optimize_woca(woca_loss)

              # ============ CALCULATE REGULARIZATION ============ 
              reg_loss = self.calcRegularization(states, self.grad_step_size, self.grad_steps)

              # ============ TRAIN ROBUST NETWORK ============ 
              q_target = self.target_net.getMaxQ(next_states, self.env.action_space.n, self.device)
              q_woca = self.woca_net.getMaxQ(next_states, self.env.action_space.n, self.device)
              y_robust = rewards + self.config.gamma * (self.k_wst * q_target + (1 - self.k_wst) * q_woca) * (1-dones)
              predicted_robust = self.robust_net.process_batch((states, actions, rewards, next_states, dones), self.device)
              robust_loss = self.robust_net.loss_fn(predicted_robust.squeeze(), y_robust) + self.k_reg * reg_loss
              self.optimize_robust(robust_loss)

            if iteration % self.config.copy_freq == 0:
              self.target_net.load_state_dict(self.main_net.state_dict())
              print("[!!] Weights of Target Network updated !")
            iteration += 1
            state = new_state
            self.config.epsilon = self.epsilon_update(self.config.epsilon)        
          
          print(f"[*] ({iteration}/{self.config.iterations}) | Episode finished (len={nb_steps}), avg_loss = {total_loss/nb_steps}, tot_return={episode_return}, epsilon={self.config.epsilon}")
          episode_count += 1
          self.record.add_loss(total_loss/nb_steps)
          self.record.add_episode_return(episode_return)
          self.record.add_epsilon(self.config.epsilon)
          self.record.register_episode_count(episode_count)

        training_duration = time.time()-start
        print(f"[+] Total duration (warmup + training) = {warmup_duration + training_duration}s")
      
        print(f"[+] Environment processing duration : = {env_time}s")
        print(f"[+] Network processing duration : = {network_time}s")
        print(f"[+] Prediction calculation duration : = {predict_time}s")

        return self.record