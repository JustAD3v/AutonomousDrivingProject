
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np

from Adversarial.AbstractAdversarial import Adversarial
from Trainers.TrainConfig import TrainConfig
from Trainers.ReplayBuffer import ReplayBuffer
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
DEBUG_BOUNDS = True
# ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

class WocaRDQN:
    def __init__(self, 
                 config: TrainConfig, 
                 target_network: nn.Module,
                 main_network: nn.Module, 
                 woca_network: nn.Module,
                 woca_target_network: nn.Module,
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
        self.woca_target_net = woca_target_network
        self.robust_net = robust_network
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.woca_target_net.load_state_dict(self.woca_net.state_dict())
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.r_buffer = ReplayBuffer(self.config.buffer_size, 
                                     config.batch_size,
                                     config.warmup_size,
                                     self.device)
        self.record = TrainResult(self.config.iterations, self.config.log_freq, name="WocaR-DQN training", config=self.config, timer=self.timer)
        self.epsilon_step = (self.config.epsilon_min - self.config.epsilon_start)/(self.config.exploration_frac*self.config.iterations)

        self._bounded_robust_net = None

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        self._debug_calls = 0
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        # wocaR params
        # k_wst starts at config.k_wst_start and moves towards config.k_wst
        self.k_wst = config.k_wst_start
        self.k_wst_step = (config.k_wst - config.k_wst_start)/(config.k_wst_frac*config.iterations)
        self.k_reg = config.k_reg
        self.attack_budget = config.attack_budget_start
        self.attack_budget_step = (config.attack_budget - config.attack_budget_start)/(config.attack_budget_frac*config.iterations)
        self.grad_steps = config.wocar_grad_steps
        # perturbation set of the training: the same one the evaluation attacks use,
        # a fraction of the half-range of each feature (see Adversarial.feature_scale_of)
        # tiled over the rows of the observation to get (5,) => (25,)
        # needed because during training we work with a flattened state : (25,) = concatenation of each row (5,)
        self.attack_scale = torch.as_tensor(
            np.tile(Adversarial.feature_scale_of(self.env), self.env.observation_space.shape[0]),
            dtype=torch.float32,
            device=self.device
        )

    def epsilon_update(self, epsilon):
      return max(self.config.epsilon_min, epsilon + self.epsilon_step)
      #return max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)

    def k_wst_update(self, k_wst):
      # here k_wst weighs the main qvalues and (1 - k_wst) the worst-case ones 
      # (the convention from the paper)
      # so it starts at k_wst_start = 1 (no worst-case) and DECREASES down to its
      # target to give the worst-case qvalues more and more weight
      return max(self.config.k_wst, k_wst + self.k_wst_step)

    def attack_budget_update(self, attack_budget):
      return min(self.config.attack_budget, attack_budget + self.attack_budget_step)

    def optimize_model(self, loss, model):
       model.optimizer.zero_grad()
       loss.backward()
       
       torch.nn.utils.clip_grad_norm_(
         model.parameters(),
         max_norm=self.config.max_grad_norm
       )
       model.optimizer.step()

    def filter_actions(self, lb_q, ub_q):
      actions = []
      for (lb, ub) in zip(lb_q, ub_q):
          actions.append([
             # for each action i,
             i for i in range(len(lb)) 
             # an action might be selected 
             # if it's upper bound is greater than ALL of the lower bounds of other actions
             # because its qvalue domain covers a part of the domain of other POSSIBLE actions
             # Previous error: only being greater than one of the other lower bound
             #                is not enough because if the action considered overlaps with
             #                an action that can't be chosen 
             #                then the considered action can't be chosen either 
             #                See explanation on paper, Algo 2, line 12 (https://arxiv.org/pdf/2210.05927)
             # You can think it the opposite way :
             # if all lb of other actions j are greater than ub[i],
             # this action will never be chosen because its qvalue will always be lower
             # than the one of the other actions 
             if all([ub[i] > lb[j] for j in range(len(lb))])  
          ])
      return actions
    
    # Untargeted PGD variant: a single attack per state, pushing away from the clean
    # greedy action, collecting every greedy action met along the path. Cheaper than the
    # targeted version (one block instead of n_actions) but it only explores one direction,
    # so it tends to report smaller subsets.
    def getSubsetActionsPgdUntargeted(self, states):
      states = states.detach().to(device=self.device, dtype=torch.float32)

      attack = self.attack_budget * self.attack_scale
      grad_steps = self.grad_steps
      grad_step_size = 2 * attack / grad_steps

      with torch.no_grad():
        clean_qvalues = self.robust_net(states)
      n_actions = clean_qvalues.shape[1]
      # the clean greedy action is always reachable (delta = 0)
      best_actions = clean_qvalues.argmax(dim=1)
      reached = F.one_hot(best_actions, n_actions).bool()

      delta = (2.0 * torch.rand_like(states) - 1.0) * attack

      for _ in range(grad_steps):
        # every action already found: nothing left to search for, and masking them all
        # would make q_other -inf and the gradient NaN
        if reached.all():
          break

        delta.requires_grad_(True)

        p_qvalues = self.robust_net(states + delta)
        # CW-style margin of the clean greedy action against its best rival, the
        # actions already reached being excluded so the attack aims at a new one
        q_best = p_qvalues.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        q_other = p_qvalues.masked_fill(reached, float("-inf")).max(dim=1).values
        # rows that have reached every action have no rival left, ignore them
        valid = ~reached.all(dim=1)
        # descending the margin pushes the ranking towards the other actions
        margin = ((q_best - q_other) * valid).sum()

        grad = torch.autograd.grad(margin, delta)[0]

        with torch.no_grad():
            delta = torch.clamp(
                delta - grad_step_size * grad.sign(),
                min=-attack,
                max=attack
            )
            # record the greedy action reached at this step
            reached |= F.one_hot(
                self.robust_net(states + delta).argmax(dim=1),
                n_actions
            ).bool()
        delta = delta.detach()

      subset_actions = [row.nonzero(as_tuple=True)[0].tolist() for row in reached]

      # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
      if DEBUG_BOUNDS:
        self._debug_calls += 1
        if self._debug_calls % self.config.log_freq == 0:
          spread = (clean_qvalues.max(dim=1).values - clean_qvalues.min(dim=1).values).mean().item()
          subset = sum(len(e) for e in subset_actions) / len(subset_actions)
          print(f"[DEBUG-PGD-UNTARGETED] call={self._debug_calls} spread={spread:.4f} subset={subset:.3f}")
      # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

      return subset_actions

    # PGD variant of the subset estimation
    # Instead of certified CROWN bounds we run an untargeted attack on robust_net from each
    # state and collect every greedy action reached along the path: those are the actions the
    # adversary is empirically able to force.
    def getSubsetActions(self, states):
      states = states.detach().to(device=self.device, dtype=torch.float32)

      attack = self.attack_budget * self.attack_scale
      grad_steps = self.grad_steps
      grad_step_size = 2 * attack / grad_steps

      with torch.no_grad():
        clean_qvalues = self.robust_net(states)
      batch_size, n_actions = clean_qvalues.shape
      # the clean greedy action is always reachable (delta = 0)
      best_actions = clean_qvalues.argmax(dim=1)
      reached = F.one_hot(best_actions, n_actions).bool()

      # one targeted attack per candidate action, all stacked in a single batch:
      # block t holds the whole state batch attacked towards action t.
      # An untargeted attack only ever explores one direction and systematically
      # under-estimates the subset, which makes the worst-case value optimistic.
      t_states = states.repeat(n_actions, 1)
      targets = torch.arange(n_actions, device=self.device).repeat_interleave(batch_size)
      t_best_actions = best_actions.repeat(n_actions)

      delta = (2.0 * torch.rand_like(t_states) - 1.0) * attack

      for _ in range(grad_steps):
        delta.requires_grad_(True)

        p_qvalues = self.robust_net(t_states + delta)
    
        q_target = p_qvalues.gather(1, targets.unsqueeze(1)).squeeze(1)
        q_best = p_qvalues.gather(1, t_best_actions.unsqueeze(1)).squeeze(1)
        margin = (q_target - q_best).sum()

        grad = torch.autograd.grad(margin, delta)[0]

        with torch.no_grad():
            delta = torch.clamp(
                delta + grad_step_size * grad.sign(),
                min=-attack,
                max=attack
            )
            # record the greedy action reached at this step, for every target block
            p_best = self.robust_net(t_states + delta).argmax(dim=1)
            reached |= F.one_hot(p_best, n_actions).bool().view(
                n_actions, batch_size, n_actions
            ).any(dim=0)
        delta = delta.detach()

      subset_actions = [row.nonzero(as_tuple=True)[0].tolist() for row in reached]

      # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
      if DEBUG_BOUNDS:
        self._debug_calls += 1
        if self._debug_calls % self.config.log_freq == 0:
          spread = (clean_qvalues.max(dim=1).values - clean_qvalues.min(dim=1).values).mean().item()
          subset = sum(len(e) for e in subset_actions) / len(subset_actions)
          print(f"[DEBUG-PGD] call={self._debug_calls} spread={spread:.4f} subset={subset:.3f}")
      # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

      return subset_actions

    def getSubsetActionsCrown(self, states):
      # get lower and upper bound of perturbated states
      attack = self.attack_budget * self.attack_scale
      lb_states = (states - attack).to(
          device=self.device,
          dtype=torch.float32
      )
      ub_states = (states + attack).to(
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
      subset_actions = self.filter_actions(lb_qvalues, ub_qvalues)

      # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
      if DEBUG_BOUNDS:
        self._debug_bounds(states, lb_qvalues, ub_qvalues, subset_actions)
      # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

      return subset_actions

    # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
    # width  : mean CROWN interval width, how blurry the perturbed qvalues are
    # spread : mean (max - min) of the clean qvalues, how far apart the actions are
    # an action is pruned only if its gap to the best one exceeds width, so the
    # subset stays stuck at 5 as long as spread < width (ratio < 1)
    def _debug_bounds(self, states, lb_qvalues, ub_qvalues, subset_actions):
      self._debug_calls += 1
      if self._debug_calls % self.config.log_freq:
        return
      with torch.no_grad():
        qvalues = self.robust_net(states)
      width = (ub_qvalues - lb_qvalues).mean().item()
      spread = (qvalues.max(dim=1).values - qvalues.min(dim=1).values).mean().item()
      subset = sum(len(e) for e in subset_actions) / len(subset_actions)
      print(f"[DEBUG] call={self._debug_calls} width={width:.4f} spread={spread:.4f} "
            f"ratio={spread / max(width, 1e-12):.3f} subset={subset:.3f}")
    # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

    def getDistMax(self, states,grad_steps):
      # detach states to not change the computational graph on it
      states = states.detach()
    
      # get distribution from qvalues
      with torch.no_grad():
        # don't create a graph either way it will bind probs to the robust_net graph
        # and at the end it will release the tensors used
        # so next use of probs in the loop after will cause errors (reuse of released tensors)
        qvalues = self.robust_net(states)
        probs = F.log_softmax(qvalues, dim=1) # size (B, 5)

      # init delta with random perturbation in [-attack; attack], element-wise
      attack = self.attack_budget * self.attack_scale
      # use an attack-relative grad step 
      # so after all the steps the total perturbation 
      # gets a chance of testing the whole attack space (|[-attack;attack]| = 2*attack) 
      grad_step_size = 2 * attack / grad_steps
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
        p_probs = F.log_softmax(p_qvalues, dim=1) # size (B, 5)

        dist = F.kl_div(
            p_probs,
            probs,
            # we reduce to batchmean to have a single scalar value to compute the gradient
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
                delta + grad_step_size * grad.sign(),
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
        # for the final distance WE DON'T REDUCE
        # we need to keep a vector
        # to then multiply element wise with the state weights
        # and compute the mean over that weighted sum
        # See paper Lreg calculation (https://arxiv.org/pdf/2210.05927) 
        reduction="none",
        log_target=True
      )

      # We still have to reduce because final_dist returns (B, 5) (dist elem wise on logits)
      # but we need (B,) to multiply element wise 
      # with state weights vector (B,)
      # So we sum on each element dist to get max dist per row
      return final_dist.sum(dim=1)

    def getStateWeight(self, states):
      # use main_net to calculate the state weight as the paper says
      max_qvalues = self.main_net.getMaxQ(states, self.env.action_space.n, self.device)
      min_qvalues = self.main_net.getMinQ(states, self.env.action_space.n, self.device)
      return torch.sub(max_qvalues, min_qvalues)
    
    def calcRegularizationKL(self, states, grad_steps):
      with torch.no_grad():
        # w(s) does not need to be part of the graph for backprop
        # It can be treated as a constant
        w_states = self.getStateWeight(states)
      max_dists = self.getDistMax(states, grad_steps)
      reg = torch.mul(w_states, max_dists)
      return reg.mean()

    # PGD attack maximizing the margin (i.e the distance between the best qvalue and the second best one)
    # margin(s) = max_{a != a*} Q(s,a) - Q(s,a*)
    def getAdvMargin(self, states, grad_steps):
      states = states.detach()

      with torch.no_grad():
        clean_qvalues = self.robust_net(states)
        best_actions = clean_qvalues.argmax(dim=1)
      best_mask = F.one_hot(best_actions, clean_qvalues.shape[1]).bool()

      attack = self.attack_budget * self.attack_scale
      grad_step_size = 2 * attack / grad_steps
      delta = (2.0 * torch.rand_like(states) - 1.0) * attack

      def margin_of(qvalues):
        q_best = qvalues.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        q_other = qvalues.masked_fill(best_mask, float("-inf")).max(dim=1).values
        return q_other - q_best

      # find the perturbation that maximizes the margin, by gradient ascent
      for _ in range(grad_steps):
        delta.requires_grad_(True)

        p_qvalues = self.robust_net(states + delta)
        margin = margin_of(p_qvalues).sum()

        grad = torch.autograd.grad(margin, delta)[0]

        with torch.no_grad():
            delta = torch.clamp(
                delta + grad_step_size * grad.sign(),
                min=-attack,
                max=attack
            )
        delta = delta.detach()

      # recompute outside of the loop, on a fresh graph the robust_net optimization can use
      final_margin = margin_of(self.robust_net(states + delta))
      # we clamp the margin to avoid widening the margin too much 
      return torch.clamp(final_margin, min=-self.config.wocar_hinge_c)

    def calcRegularization(self, states, grad_steps):
      with torch.no_grad():
        # w(s) does not need to be part of the graph for backprop
        # It can be treated as a constant
        w_states = self.getStateWeight(states)
      adv_margins = self.getAdvMargin(states, grad_steps)
      reg = torch.mul(w_states, adv_margins)
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

    @Timer.measure("WOCA network update")
    def _train_woca_network(self, states, actions, rewards, next_states, dones):
              # the actions an attacker can force the agent to take from next_states
              subset_actions = self.getSubsetActions(next_states)
              # calc y_woca_target for every sample of batch using WOCA TARGET NETWORK
              # same as the main network, with the min over the attacker actions
              # instead of the max over all the actions
              with torch.no_grad():
                # no need to build a graph for calculating target
                # (no backprop is computed)
                woca_qvalues = self.woca_target_net.getMinQ(next_states, self.env.action_space.n, self.device, subset_actions)

              y_woca_target = rewards + self.config.gamma * woca_qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)
              # calc y_woca using WOCA NETWORK
              y_woca = self.woca_net.get_qvalues((states, actions, rewards, next_states, dones), self.device)
              # calc loss
              woca_loss = self.woca_net.loss_fn(y_woca.squeeze(), y_woca_target)
              # optimize
              self.optimize_model(woca_loss, self.woca_net)

    @Timer.measure("Robust network update")
    def _train_robust_network(self, states, actions, rewards, next_states, dones, reg_loss):
              # max_q_sum calculates the maximum of the sum of the qvalues from main network and woca network
              # see paper page 20, Algo 2, line 17 (https://arxiv.org/pdf/2210.05927) 
              # we have to take the max over the sum
              # and not the sum of the max (previous error)
              with torch.no_grad():
                q_target = self.target_net(next_states)
                q_woca = self.woca_target_net(next_states)
              max_q_sum = (self.k_wst * q_target + (1 - self.k_wst) * q_woca).max(dim=1).values
              y_robust = rewards + self.config.gamma * max_q_sum * (1-dones)
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
              loss_value = self._train_main_network(states, actions, rewards, next_states, dones)
              total_loss += loss_value

              # ============ TRAIN WOCA NETWORK ============
              self._train_woca_network(states, actions, rewards, next_states, dones)

              # ============ CALCULATE REGULARIZATION ============ 
              reg_loss = self.calcRegularization(states, self.grad_steps)

              # ============ TRAIN ROBUST NETWORK ============ 
              self._train_robust_network(states, actions, rewards, next_states, dones, reg_loss)

            if iteration % self.config.copy_freq == 0:
              self.target_net.load_state_dict(self.main_net.state_dict())
              self.woca_target_net.load_state_dict(self.woca_net.state_dict())
              print("[!!] Weights of Target Networks updated !")
            iteration += 1
            state = new_state
            self.config.epsilon = self.epsilon_update(self.config.epsilon)
            self.k_wst = self.k_wst_update(self.k_wst)
            self.attack_budget = self.attack_budget_update(self.attack_budget)
          
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

        self.timer.report()

        return self.record
