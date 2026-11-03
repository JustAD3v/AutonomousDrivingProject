
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
        self.timer = Timer()

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
        self.record = TrainResult(self.config.iterations, self.config.log_freq, name="WocaR-DQN training")
        self.epsilon_step = (self.config.epsilon_min - self.config.epsilon_start)/(self.config.exploration_frac*self.config.iterations)

        self._bounded_robust_net = None

        # wocaR params
        self.k_wst = config.k_wst
        self.k_reg = config.k_reg
        self.attack_budget = config.attack_budget
        self.grad_step_size = config.wocar_grad_step_size
        self.grad_steps = config.wocar_grad_steps

    def epsilon_update(self, epsilon):
      return max(self.config.epsilon_min, epsilon + self.epsilon_step)
      #return max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)

    def optimize_model(self, loss, model):
       model.optimizer.zero_grad()
       loss.backward()
       
       torch.nn.utils.clip_grad_norm_(
         model.parameters(),
         max_norm=10.0
       )
       model.optimizer.step()

    def getWocaMinQ(self, subset_actions, states):
      with torch.no_grad():
        # don't create a graph for qvalues either way it will bind qvalues to the graph
        # and when the auto grad releases the tensor we can't use them anymore
        # even though we return it (we return minQvalues which is bound to qvalues)
        qvalues = self.woca_net(states.to(self.device)) # (B, 5)
        # for each set of 5 qvalues only keep the qvalue associated
        # to the action from subset_actions
        # that has the minimum value
        minQValues = torch.stack([
          qvalues[row, idx].min()
          for row, idx in enumerate(subset_actions)
        ])
      return minQValues

    def filter_actions(self, lb_q, ub_q):
      actions = []
      for (lb, ub) in zip(lb_q, ub_q):
          actions.append([
             # for each action i,
             i for i in range(len(lb)) 
             # if there is an action j (i != j) with a lower bound lower than ub[i]
             # then this action might be chosen 
             # because its qvalue domain covers a part of the domain of another action
             # You can think it the opposite way :
             # if all lb of other actions j are greater than ub[i],
             # this action will never be chosen because its qvalue will always be lower
             # than the one of the other actions 
             if any([ub[i] > lb[j] for j in range(len(lb)) if j != i])  
          ])
      return actions
    
    def getSubsetActions(self, states):
      # get lower and upper bound of perturbated states
      lb_states = (states - states.abs() * self.attack_budget).to(
          device=self.device,
          dtype=torch.float32
      )
      ub_states = (states + states.abs() * self.attack_budget).to(
          device=self.device,
          dtype=torch.float32
      )
      # wrap model in auto_LiRPA framework
      # use _bounded_robust_net as a cache 
      # to avoid calculating it at every round
      if self._bounded_robust_net is None:
          self._bounded_robust_net = BoundedModule(
              self.robust_net,
              torch.empty_like(states)
          )
      # Compute bounds using LiRPA using the given lower and upper bounds.
      norm = float("inf")
      ptb = PerturbationLpNorm(norm=norm, x_L=lb_states, x_U=ub_states)
      bounded_qvalues = BoundedTensor(states, ptb)
      # Compute bounds using CROWN
      lb_qvalues, ub_qvalues = self._bounded_robust_net.compute_bounds(
          x=(bounded_qvalues,),
          method='CROWN'
      )
      # print(f'CROWN bounds: lower={lb_qvalues}, upper={ub_qvalues}')
      # filter actions
      return self.filter_actions(lb_qvalues, ub_qvalues)

    def getDistMax(self, states, grad_step_size,grad_steps):
      # detach states to not change the computational graph on it
      states = states.detach()
    
      # get distribution from qvalues
      with torch.no_grad():
        # don't create a graph either way it will bind probs to the robust_net graph
        # and at the end it will release the tensors used
        # so next use of probs in the loop after will cause errors (reuse of released tensors)
        qvalues = self.robust_net(states)
        probs = F.log_softmax(qvalues, dim=1)

      # init delta with random pertubration in [-attack; attack], element-wise
      attack = self.attack_budget * states.abs()
      # rescale the uniform sample from [0, 1] to [-1, 1]
      delta = (2.0 * torch.rand_like(states) - 1.0) * attack

      # find perturbation delta that maximizes the distance using gradient ascent
      dist = None
      for _ in range(grad_steps):
        # attach delta to the graph to register operations on it
        # in order to compute the auto grad
        delta.requires_grad_(True)

        # operations recorded in the graph
        p_states = states + delta
        p_qvalues = self.robust_net(p_states)
        p_probs = F.log_softmax(p_qvalues, dim=1)

        dist = F.kl_div(
            p_probs,
            probs,
            reduction="batchmean",
            # this indicates that the target param (probs.log()) is passed in the log space
            # it is recommended for soft_max log output in the documentation :
            # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.kl_div.html
            log_target=True
        )

        # uses the graph to compute gradient
        grad = torch.autograd.grad(dist, delta)[0]

        with torch.no_grad():
            # don't include those calculations in the comutational graph
            delta = torch.clamp(
                delta + grad_step_size * grad,
                min=-attack,
                max=attack
            )
        # detach delta from the graph so next gradient calculation 
        # is a fresh new gradient independent from the previous one
        delta = delta.detach()

      # now we have the delta that maximizes the distance
      # compute the actual distance outside of the loop so it is associated 
      # to a fresh graph that will be used by the robust_net to calc loss and optimize the net      
      p_states = states + delta
      p_qvalues = self.robust_net(p_states)
      p_probs = F.log_softmax(p_qvalues, dim=1)
      final_dist = F.kl_div(
        p_probs,
        probs,
        reduction="batchmean",
        log_target=True
      )
      
      return final_dist

    def getStateWeight(self, states):
      # use main_net to calculate the state weight as the paper says
      max_qvalues = self.main_net.getMaxQ(states, self.env.action_space.n, self.device)
      min_qvalues = self.main_net.getMinQ(states, self.env.action_space.n, self.device)
      return torch.sub(max_qvalues, min_qvalues)
    
    def calcRegularization(self, states, grad_step_size, grad_steps):
      with torch.no_grad():
        # w(s) does not need to be part of the graph for backprop
        # It can be treated as a constant 
        w_states = self.getStateWeight(states)
      max_dist = self.getDistMax(states, grad_step_size, grad_steps)
      reg = torch.mul(w_states, max_dist)
      return reg.mean()

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
        self.optimize_model(loss, self.main_net)

        return qvalues, loss_value

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

    @Timer.measure("WOCA network update")
    def _train_woca_network(self, states, actions, rewards, next_states, dones):
              subset_actions = self.getSubsetActions(next_states)
              with torch.no_grad():
                # no need to build a graph for calculating target
                # (no backprop is computed)
                woca_qvalues = self.getWocaMinQ(subset_actions, next_states)
                y_woca_target = rewards + self.config.gamma * woca_qvalues * (1-dones)

              y_woca_predicted = self.woca_net.get_qvalues((states, actions, rewards, next_states, dones), self.device)
              woca_loss = self.woca_net.loss_fn(y_woca_predicted.squeeze(), y_woca_target)
              self.optimize_model(woca_loss, self.woca_net)

    @Timer.measure("Robust network update")
    def _train_robust_network(self, states, actions, rewards, next_states, dones, qvalues, reg_loss):
              q_target = qvalues
              q_woca = self.woca_net.getMaxQ(next_states, self.env.action_space.n, self.device)
              y_robust = rewards + self.config.gamma * (self.k_wst * q_target + (1 - self.k_wst) * q_woca) * (1-dones)
              predicted_robust = self.robust_net.get_qvalues((states, actions, rewards, next_states, dones), self.device)
              robust_loss = self.robust_net.loss_fn(predicted_robust.squeeze(), y_robust) + self.k_reg * reg_loss
              self.optimize_model(robust_loss, self.robust_net)

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
              # ============ TRAIN MAIN NETWORK ============
              # uniform draw of a batch
              states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
              # calc y_target for every sample of batch using TARGET NETWORK
              qvalues, loss_value = self._train_main_network(states, actions, rewards, next_states, dones)
              total_loss += loss_value

              # ============ TRAIN WOCA NETWORK ============
              self._train_woca_network(states, actions, rewards, next_states, dones)

              # ============ CALCULATE REGULARIZATION ============ 
              reg_loss = self.calcRegularization(states, self.grad_step_size, self.grad_steps)

              # ============ TRAIN ROBUST NETWORK ============ 
              self._train_robust_network(states, actions, rewards, next_states, dones, qvalues, reg_loss)

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

        print("[+] === Detailed timing breakdown ===")
        for name, elapsed in self.timer.counters.items():
            if name not in (
                "Warmup duration",
                "Training duration",
                "Environment processing",
                "Network processing",
                "Prediction calculation"
            ):
                print(f"{name}: {elapsed:.3f}s")

        return self.record
