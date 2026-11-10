import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from Trainers.TrainConfig import TrainConfig
from Trainers.PPOBufferExtended import PPOBufferExtended
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

PARALLEL = True

_worker_env = None

def init_collector_worker():
    # Workers only run single-state inference
    # by default pytorch creates multithreads to run calculations
    # and divide the work
    # but here the work down is too small so the thread spawning overhead
    # is bigger than its benefit
    # so restrict it to only one thread 
    torch.set_num_threads(1)

def get_worker_env(env_name, env_config):
    # Workers are reused across iterations, so the env is built once per process
    # so cache the env to not create it at every call
    global _worker_env
    if _worker_env is None:
        _worker_env = gym.make(env_name, config=env_config)
    return _worker_env

def collector_worker(env_name, env_config, buffer_size, policy, device, critic_network, gamma, ppo_lm):
    env = get_worker_env(env_name, env_config)
    state, _ = env.reset()
    buffer = PPOBufferExtended(buffer_size, device, gamma, ppo_lm)
    for _ in range(buffer_size):
        action, log_action_prob = policy.predict(state, device=device)
        next_state, reward, term, trunc, _ = env.step(policy.to_env_action(action))
        state_flat = np.asarray(state, dtype=np.float32).flatten()
        next_state_flat = np.asarray(next_state, dtype=np.float32).flatten()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state_flat,
                dtype=torch.float32,
                device=device
            )
            critic_value = critic_network(state_tensor).item()
        # Store a detached scalar old log probability.
        if torch.is_tensor(log_action_prob):
            log_action_prob = log_action_prob.detach().item()
        buffer.push((
            state_flat,
            action,
            reward,
            log_action_prob,
            critic_value,
            next_state_flat,
            float(term)
        ))
        end = term | trunc | buffer.full
        if end:
            if term:
                extra_critic_value = 0.0
            else:
                with torch.no_grad():
                    next_state_tensor = torch.as_tensor(
                        next_state_flat,
                        dtype=torch.float32,
                        device=device
                    )
                    extra_critic_value = critic_network(next_state_tensor).item()
            buffer.finish_episode(extra_critic_value)
            # Reset the environment only if the episode ended.
            if term or trunc:
                state, _ = env.reset()
            else:
                # The buffer is full, but the episode may continue.
                state = next_state
        else:
            state = next_state
    return buffer

class WocaRPPO:
    def __init__(self,
                 config: TrainConfig,
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 woca_network: nn.Module,
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

        self.env_name = env_name
        # parallel workers build their own env, they must use the same config (e.g. action type)
        self.env_config = env_config if env_config is not None else self.env.unwrapped.config

        # the worst-case network is a Q network over a finite set of actions,
        # so the WocaR framework only applies to the discrete action space
        # if config.ppo_continuous:
        #     raise ValueError("WocaRPPO only supports the discrete action space")

        self.config = config
        self.policy = policy
        self.clip_epsilon = self.config.ppo_clip_epsilon
        self.actor_update_steps = self.config.ppo_actor_update_steps
        self.critic_update_steps = self.config.ppo_critic_update_steps
        self.robust_actor_network = actor_network
        self.critic_network = critic_network
        self.woca_network = woca_network
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.buffer = PPOBufferExtended(self.config.buffer_size,
                                        self.device,
                                        self.config.gamma,
                                        self.config.ppo_lm)
        self.record = TrainResult(self.config.iterations, self.config.log_freq, name="WocaR-PPO training", config=self.config, timer=self.timer)

        self._bounded_robust_net = None

        # created on the first parallel collection and kept for the whole training
        self.executor = None
        self.nworkers = None

        # wocaR params
        self.k_wst = config.k_wst
        self.k_reg = config.k_reg
        self.attack_budget = config.attack_budget
        self.grad_step_size = config.wocar_grad_step_size
        self.grad_steps = config.wocar_grad_steps

    @Timer.measure("Optimization duration")
    def optimize_model(self, loss, model):
       model.optimizer.zero_grad()
       loss.backward()

       torch.nn.utils.clip_grad_norm_(
         model.parameters(),
         max_norm=self.config.max_grad_norm
       )
       model.optimizer.step()

    def rounded_nworkers(self):
        if self.config.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.config.ppo_nworkers < 1:
            raise ValueError("max_workers must be at least 1")
        for nworkers in range(min(self.config.buffer_size, self.config.ppo_nworkers), 0, -1):
            if self.config.buffer_size % nworkers == 0:
                return nworkers
        return 1

    def getWocaMinQ(self, states, subset_actions=None):
        with torch.no_grad():
            # don't create a graph for qvalues either way it will bind qvalues to the graph
            # and when the auto grad releases the tensor we can't use them anymore
            # even though we return it (we return minQvalues which is bound to qvalues)
            qvalues = self.woca_network(states.to(self.device)) # (B, 5)
            # for each set of 5 qvalues only keep the qvalue associated
            # to the action from subset_actions
            # that has the minimum value
            if subset_actions is not None:
                minQValues = torch.stack([
                    qvalues[row, idx].min()
                    for row, idx in enumerate(subset_actions)
                ])
            else:
                minQValues = qvalues.min(dim=1).values
        return minQValues

    @Timer.measure("Prediction calculation")
    def _predict(self, state):
        return self.policy.predict(state, device=self.device)

    @Timer.measure("Environment processing")
    def _environment_step(self, action):
        return self.env.step(self.policy.to_env_action(action))

    def getDistMax(self, states, grad_step_size,grad_steps):
        # detach states to not change the computational graph on it
        states = states.detach()

        # init delta with random pertubration in [-attack; attack], element-wise
        attack = self.attack_budget * states.abs()

        if self.config.ppo_continuous:
            with torch.no_grad():
                # kl_divergence only supports calculation between Normal objects
                # but distributions wraps Normal into :
                # - a TransformedDistribution(Normal, Tanh)
                # - then an Independent(transformed) object which automates log prob summing
                # Hopefully the kl_divergence is invariant under the tanh transformation fo the variable
                # So we can compute the kl_div on the base Normal by unwrapping it
                # We unwrap the distribution output 2 times :
                # - one to unwrapp the Independent
                # - the second to unwrap the tanh transformation  
                clean_base = self.robust_actor_network.distribution(states).base_dist.base_dist

            # rescale the uniform sample from [0, 1] to [-1, 1]
            delta = (2.0 * torch.rand_like(states) - 1.0) * attack

            # find perturbation delta that maximizes the distance using gradient ascent
            for _ in range(grad_steps):
                delta.requires_grad_(True)
                p_base = self.robust_actor_network.distribution(states + delta).base_dist.base_dist
                # we have to do the sum() because we unwrapped clean_base and p_base from Independent
                # which was responsible for doing that automatically
                # .mean() is equivalent to reduction="batchmean" in kl_div for discrete actions
                dist = torch.distributions.kl_divergence(clean_base, p_base).sum(dim=-1).mean()
                grad = torch.autograd.grad(dist, delta)[0]
                with torch.no_grad():
                    delta = torch.clamp(delta + grad_step_size * grad, min=-attack, max=attack)
                delta = delta.detach()

            # recompute on a fresh graph so the actor can backprop through it
            p_base = self.robust_actor_network.distribution(states + delta).base_dist.base_dist
            return torch.distributions.kl_divergence(clean_base, p_base).sum(dim=-1).mean()

        # get distribution from the actor logits
        with torch.no_grad():
          # don't create a graph either way it will bind probs to the actor graph
          # and at the end it will release the tensors used
          # so next use of probs in the loop after will cause errors (reuse of released tensors)
          logits = self.robust_actor_network(states)
          probs = F.log_softmax(logits, dim=1)

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
          p_logits = self.robust_actor_network(p_states)
          p_probs = F.log_softmax(p_logits, dim=1)

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
        # to a fresh graph that will be used by the actor to calc loss and optimize the net
        p_states = states + delta
        p_logits = self.robust_actor_network(p_states)
        p_probs = F.log_softmax(p_logits, dim=1)
        final_dist = F.kl_div(
          p_probs,
          probs,
          reduction="batchmean",
          log_target=True
        )

        return final_dist

    def getStateWeight(self, states):
        # use wocar network to calculate minQvalue
        # use critic network to calculate the critic value
        # calculate the weight as stated in the paper : V - minQ
        if self.config.ppo_continuous:
            action_lb, action_ub = torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high)
            with torch.no_grad():
                action_means = self.robust_actor_network(states)
            var_actions = action_means.requires_grad_()
            step_eps = (action_ub - action_lb) / self.grad_steps
            
            # find action that minimizes the value of woca network using gradient descent
            for _ in range(self.grad_steps):
              # attach delta to the graph to register operations on it
              # in order to compute the auto grad
              var_actions.requires_grad_(True)
    
              # operations recorded in the graph
              worst_q = self.woca_network(torch.cat((states, var_actions), dim=1))

              # uses the graph to compute gradient
              grad = torch.autograd.grad(worst_q.sum(), var_actions)[0]
    
              with torch.no_grad():
                  # don't include those calculations in the comutational graph
                  var_actions = torch.clamp(
                      var_actions - step_eps * torch.sign(grad),
                      min=action_lb,
                      max=action_ub
                  )
              # detach var_action from the graph so next gradient calculation
              # is a fresh new gradient independent from the previous one
              var_actions = var_actions.detach()
            self.woca_network.zero_grad()
            with torch.no_grad():
                min_qvalues = self.woca_network(torch.cat((states, var_actions), dim=1)).squeeze(-1)
        else:
            min_qvalues = self.getWocaMinQ(states)

        critic_values = self.critic_network(states).squeeze(-1)
        # detach because w(s) does not need to be part of the graph for backprop
        # It can be treated as a constant
        return torch.sub(critic_values, min_qvalues).detach()

    def calcRegularization(self, states, grad_step_size, grad_steps):
        w_states = self.getStateWeight(states)
        max_dist = self.getDistMax(states, grad_step_size, grad_steps)
        reg = torch.mul(w_states, max_dist)
        return reg.mean()

    def getSubsetActions(self, lb_q, ub_q):
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

    def getActionBounds(self, states):
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
                self.robust_actor_network,
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
        return lb_qvalues, ub_qvalues

    @Timer.measure("Critic network update")
    def _train_critic_network(self, states, returns):
        critic_values = self.critic_network(states).squeeze(-1)
        loss = self.critic_network.loss_fn(critic_values, returns)
        self.optimize_model(loss, self.critic_network)

    @Timer.measure("WOCA network update")
    def _train_woca_network(self, states, actions, rewards, next_states, dones):
        if self.config.ppo_continuous:
            action_lb, action_ub = torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high)
            with torch.no_grad():
                action_means = self.robust_actor_network(next_states)
            var_actions = action_means.requires_grad_()
            step_eps = (action_ub - action_lb) / self.grad_steps

            # find action that minimizes the value of woca network using gradient descent
            for _ in range(self.grad_steps):
              # attach delta to the graph to register operations on it
              # in order to compute the auto grad
              var_actions.requires_grad_(True)

              # operations recorded in the graph
              worst_q = self.woca_network(torch.cat((next_states, var_actions), dim=1))

              # uses the graph to compute gradient
              grad = torch.autograd.grad(worst_q.sum(), var_actions)[0]

              with torch.no_grad():
                  # don't include those calculations in the comutational graph
                  var_actions = torch.clamp(
                      var_actions - step_eps * torch.sign(grad),
                      min=action_lb,
                      max=action_ub
                  )
              # detach var_action from the graph so next gradient calculation
              # is a fresh new gradient independent from the previous one
              var_actions = var_actions.detach()
            self.woca_network.zero_grad()
            with torch.no_grad():
                woca_qvalues = self.woca_network(torch.cat((next_states, var_actions), dim=1)).squeeze(1)
        else:
            action_lb, action_ub = self.getActionBounds(states)
            subset_actions = self.getSubsetActions(action_lb, action_ub)
            with torch.no_grad():
                # no need to build a graph for calculating target
                # (no backprop is computed)
                woca_qvalues = self.getWocaMinQ(next_states, subset_actions)

        with torch.no_grad():        
            y_woca_target = rewards + self.config.gamma * woca_qvalues * (1-dones)
        y_woca_predicted = self.woca_network.get_qvalues((states, actions, rewards, next_states, dones), self.device)
        woca_loss = self.woca_network.loss_fn(y_woca_predicted.squeeze(), y_woca_target)
        self.optimize_model(woca_loss, self.woca_network)

    @Timer.measure("Robust actor network update")
    def _train_robust_actor_network(self, states, actions, old_log_probs, advantages, rewards, next_states, dones, reg_loss):
        distribution = self.robust_actor_network.distribution(states)
        current_log_probs = distribution.log_prob(actions)

        ratio = torch.exp(current_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        )
        
        if self.config.ppo_continuous:
            woca_qvalues = self.woca_network(torch.cat((states, actions), dim=1)).squeeze(-1)
        else:
            woca_qvalues = self.woca_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        woca_advantages = advantages + self.k_wst * woca_qvalues
        unclipped_objective = ratio * woca_advantages
        clipped_objective = clipped_ratio * woca_advantages

        loss = -torch.minimum(unclipped_objective, clipped_objective).mean() + self.k_reg * reg_loss
        self.optimize_model(loss, self.robust_actor_network)

    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0

        while iteration < self.config.iterations:

            if PARALLEL:
                if self.executor is None:
                    print("using parallel !")
                    self.nworkers = self.rounded_nworkers()
                    self.executor = ProcessPoolExecutor(max_workers=self.nworkers,
                                                        initializer=init_collector_worker)
                buff_size = self.config.buffer_size // self.nworkers
                # the networks are serialized on every submit, so workers always
                # collect with the current parameters
                futures = [self.executor.submit(collector_worker,
                                         self.env_name,
                                         self.env_config,
                                         buff_size,
                                         self.policy,
                                         self.device,
                                         self.critic_network,
                                         self.config.gamma,
                                         self.config.ppo_lm)
                            for _ in range(self.nworkers)
                        ]
                res = [f.result() for f in futures]
                self.buffer.merge(res)
            else:
                state, _ = self.env.reset()

                for _ in range(self.config.buffer_size):
                    # ========= SAMPLE COLLECTION =============
                    action, log_action_prob = self._predict(state)
                    next_state, reward, term, trunc, _ = self._environment_step(action)

                    state_flat = np.asarray(state, dtype=np.float32).flatten()
                    next_state_flat = np.asarray(next_state, dtype=np.float32).flatten()
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            state_flat,
                            dtype=torch.float32,
                            device=self.device
                        )
                        critic_value = self.critic_network(state_tensor).item()

                    # Store a detached scalar old log probability.
                    if torch.is_tensor(log_action_prob):
                        log_action_prob = log_action_prob.detach().item()

                    self.buffer.push((
                        state_flat,
                        action,
                        reward,
                        log_action_prob,
                        critic_value,
                        next_state_flat,
                        float(term)
                    ))

                    end = term | trunc | self.buffer.full
                    if end:
                        if term:
                            extra_critic_value = 0.0
                        else:
                            with torch.no_grad():
                                next_state_tensor = torch.as_tensor(
                                    next_state_flat,
                                    dtype=torch.float32,
                                    device=self.device
                                )
                                extra_critic_value = self.critic_network(next_state_tensor).item()

                        self.buffer.finish_episode(extra_critic_value)

                        # Reset the environment only if the episode ended.
                        if term or trunc:
                            state, _ = self.env.reset()
                        else:
                            # The buffer is full, but the episode may continue.
                            state = next_state
                    else:
                        state = next_state

            self.record.add_episodes(self.buffer.episode_stats, iteration=iteration)

            # ============ GET BATCH =====================
            minibatches = self.buffer.get_batch(self.config.ppo_minibatch_size)
            for _ in range(self.config.ppo_update_epochs):
                for minibatch in minibatches:
                    states, actions, old_log_probs, advantages, returns, rewards, next_states, dones = minibatch
                    states = states.float()
                    next_states = next_states.float()
                    actions = actions.float() if self.config.ppo_continuous else actions.long()
                    old_log_probs = old_log_probs.float().detach()
                    advantages = advantages.float()
                    returns = returns.float()
                    rewards = rewards.float()
                    dones = dones.float()

                # ============ OPTIMIZE CRITIC ===============
                    self._train_critic_network(states, returns)

                # ============ TRAIN WOCA NETWORK ============
                    self._train_woca_network(states, actions, rewards, next_states, dones)

                # ============ CALCULATE REGULARIZATION ============
                    reg_loss = self.calcRegularization(states, self.grad_step_size, self.grad_steps)

                # ============ OPTIMIZE ROBUST ACTOR =================
                    self._train_robust_actor_network(states, actions, old_log_probs, advantages, rewards, next_states, dones, reg_loss)

            self.buffer.reset()
            iteration += 1
            self.record.report_iteration(iteration)

    def learn(self) -> TrainResult:
        self.timer.reset()
        try:
            self._train()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None
        self.record.report_final()
        self.timer.report()
        return self.record
