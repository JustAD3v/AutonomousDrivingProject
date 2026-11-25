import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from Adversarial.AbstractAdversarial import Adversarial
from Trainers.TrainConfig import TrainConfig
from Trainers.PPOBufferExtended import PPOBufferExtended
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
DEBUG = True
# ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

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

def collector_worker(env_name, env_config, buffer_size, policy, device, critic_network, gamma, ppo_lm, seed=None):
    env = get_worker_env(env_name, env_config)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    state, _ = env.reset(seed=seed)
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
        self.nworkers = 1

        # wocaR params
        # k_wst starts low and grows towards config.k_wst: the paper increases the
        # worst-case weight slowly, so the actor is not pushed by a worst-case
        # network that is still untrained
        self.k_wst = config.k_wst_start
        self.k_wst_step = (config.k_wst - config.k_wst_start)/(config.k_wst_frac*config.iterations)
        self.k_reg = config.k_reg
        self.attack_budget = config.attack_budget_start
        self.attack_budget_step = (config.attack_budget - config.attack_budget_start)/(config.attack_budget_frac*config.iterations)
        self.grad_step_size = config.wocar_grad_step_size
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

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        # one counter per tag, each tag is printed every log_freq calls
        self._debug_calls = {}
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

    def k_wst_update(self, k_wst):
        # k_wst weighs Q_wst, 
        # so it grows from k_wst_start
        # up to its target
        return min(self.config.k_wst, k_wst + self.k_wst_step)

    def attack_budget_update(self, attack_budget):
        return min(self.config.attack_budget, attack_budget + self.attack_budget_step)

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

    def worker_seed(self, iteration, worker):
        if self.config.seed is None:
            return None
        # a distinct seed per worker and per iteration, so workers don't collect the same rollout
        return self.config.seed + iteration * self.nworkers + worker

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
        # absolute budget scaled by the range of each feature, not relative to |s|:
        # a feature close to 0 would otherwise get almost no perturbation, and the
        # budget would not be comparable to the one the evaluation attacks apply
        attack = self.attack_budget * self.attack_scale

        # attack-relative grad step
        pgd_step_size = 2 * attack / grad_steps

        if self.config.ppo_continuous:
            # Dist is the squared difference of the gaussian mean rather than the KL.
            # The KL takes into account the standard deviation, 
            # adding a 1/sigma^2 factor.
            # Problem is that increasing sigma becomes a very good strategy
            # to decrease the Dist. 
            # But that's not what we want, 
            # it would just spread the policy by a lot and break it
            # So we only compute the Dist using the mean 
            with torch.no_grad():
                clean_means = self.robust_actor_network(states)

            # rescale the uniform sample from [0, 1] to [-1, 1]
            delta = (2.0 * torch.rand_like(states) - 1.0) * attack

            # find perturbation delta that maximizes the distance using gradient ascent
            for _ in range(grad_steps):
                delta.requires_grad_(True)
                p_means = self.robust_actor_network(states + delta)
                # mean of squarred differences
                dist = (p_means - clean_means).pow(2).sum(dim=-1).mean()
                grad = torch.autograd.grad(dist, delta)[0]
                with torch.no_grad():
                    delta = torch.clamp(delta + pgd_step_size * grad.sign(), min=-attack, max=attack)
                delta = delta.detach()

            # recompute on a fresh graph so the actor can backprop through it
            # returned per sample, so calcRegularization can weigh each state by w(s)
            p_means = self.robust_actor_network(states + delta)
            return (p_means - clean_means).pow(2).sum(dim=-1)

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
                  delta + pgd_step_size * grad.sign(),
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
          reduction="none",
          log_target=True
        ).sum(dim=1)

        return final_dist

    # Continuous version of getSubsetActions + woca_network.getMinQ : 
    # the bounds of the subset of actions is found 
    # using the relaxation bounds of the gaussian mean,
    # Then the worst one is found by gradient descent inside those bounds
    @Timer.measure("WOCA continuous min Q")
    def getMinQContinuous(self, states):
        states = states.detach()
        # the actor output is the gaussian mean, so bounding the actor bounds the forced action
        action_lb, action_ub = self.getActionBoundsPGD(states)
        space_lb = torch.as_tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        space_ub = torch.as_tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
        raw_width = (action_ub - action_lb).detach()
        action_lb = action_lb.detach().clamp(min=space_lb, max=space_ub)
        action_ub = action_ub.detach().clamp(min=space_lb, max=space_ub)

        with torch.no_grad():
            # s is in its own epsilon-ball, so the clean mean is always inside the box
            action_means = self.robust_actor_network(states)
        var_actions = action_means.clone()
        # cover the whole width of the box in grad_steps steps
        step_eps = (action_ub - action_lb) / self.grad_steps

        # the optimization needs a graph on var_actions
        with torch.enable_grad():
            for _ in range(self.grad_steps):
                var_actions.requires_grad_(True)

                # uses the graph to compute gradient
                worst_q = self.woca_network(torch.cat((states, var_actions), dim=1))
                grad = torch.autograd.grad(worst_q.sum(), var_actions)[0]

                with torch.no_grad():
                    var_actions = torch.clamp(
                        var_actions - step_eps * torch.sign(grad),
                        min=action_lb,
                        max=action_ub
                    )
                # detach so the next gradient is independent from the previous one
                var_actions = var_actions.detach()

        with torch.no_grad():
            min_qvalues = self.woca_network(torch.cat((states, var_actions), dim=1)).squeeze(-1)

            # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
            # width     : box width after the intersection, the continuous equivalent of
            #             the discrete "subset" size. 0 means the attacker can force nothing
            # raw_width : width before the intersection with the action space. Reaching 2
            #             (the action space width) means the bounds constrain nothing
            # gap       : Q_wst(s, mean) - minQ, 0 means the descent found nothing
            # width_sig : box width in units of the policy sigma. Q_wst is only ever fit
            #             on actions drawn at that sigma, so this is how far outside its
            #             training support the descent is allowed to go
            # dev_sig   : distance of the minimizing action from the clean mean, same units
            # oos_gap   : min over actions sampled from the policy itself, minus minQ.
            #             Positive means the box descent reached values no in-support
            #             action can reach, which is extrapolation rather than worst case
            if DEBUG:
                clean_qvalues = self.woca_network(torch.cat((states, action_means), dim=1)).squeeze(-1)
                box_width = action_ub - action_lb
                sigma = self.robust_actor_network.log_std.detach().clamp(self.robust_actor_network.log_std_min, 1.0).exp()

                # fixed, not woca_baseline_samples: this is a measurement called once
                # per minibatch and must not get more expensive when that is raised
                nsamples = 8
                noise = torch.randn(nsamples, *action_means.shape, device=self.device)
                in_support = (action_means.unsqueeze(0) + sigma * noise).clamp(min=space_lb, max=space_ub)
                repeated_states = states.unsqueeze(0).expand(nsamples, -1, -1)
                in_support_qvalues = self.woca_network(torch.cat(
                    (repeated_states.flatten(0, 1), in_support.flatten(0, 1)),
                    dim=1
                )).view(nsamples, states.shape[0])
                min_in_support = in_support_qvalues.min(dim=0).values

                self._debug_log("MINQ",
                                width=box_width.mean().item(),
                                width_max=box_width.max().item(),
                                raw_width=raw_width.mean().item(),
                                gap=(clean_qvalues - min_qvalues).mean().item(),
                                width_sig=(box_width / sigma).mean().item(),
                                dev_sig=((var_actions - action_means).abs() / sigma).mean().item(),
                                oos_gap=(min_in_support - min_qvalues).mean().item(),
                                eps=self.attack_budget)
            # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        return min_qvalues

    def getStateWeight(self, states):
        # use wocar network to calculate minQvalue
        # use critic network to calculate the critic value
        # calculate the weight as stated in the paper : V - minQ
        if self.config.ppo_continuous:
            min_qvalues = self.getMinQContinuous(states)
        else:
            min_qvalues = self.woca_network.getMinQ(states, self.env.action_space.n, self.device)

        critic_values = self.critic_network(states).squeeze(-1)
        # detach because w(s) does not need to be part of the graph for backprop
        # It can be treated as a constant
        w_states = torch.sub(critic_values, min_qvalues).detach()
        # V - minQ is only meant to rank states by how costly a wrong action is
        # If it is negative it will apply opposite effect
        # basically lowering the total regularization 
        # and potentially lowering the effect of other terms
        # If the weight is negative, better ignoring it because the info is not meaningful
        w_states = w_states.clamp(min=0.0)
        # V directly depends on the scale of the rewards. 
        # if the reward is not normalized (the case in some of our envs)
        # then the values can be absurdly big compared to the rest of the loss
        # so we need to rescale (+1e-8 to avoid division by zero)
        normalized_w_states = w_states / (w_states.mean() + 1e-8)

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        # raw_mean/raw_min : V - minQ before the clamp. A negative raw_min is expected
        #                    while the woca network is untrained, a negative raw_mean
        #                    means minQ is above V for most of the batch
        # clamped : share of the batch zeroed by the clamp, 1.0 kills the regularization
        if DEBUG:
            raw_w_states = torch.sub(critic_values, min_qvalues).detach()
            self._debug_log("WEIGHT",
                            raw_mean=raw_w_states.mean().item(),
                            raw_min=raw_w_states.min().item(),
                            clamped=(raw_w_states <= 0).float().mean().item(),
                            w_max=normalized_w_states.max().item())
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        return normalized_w_states

    def calcRegularization(self, states, grad_step_size, grad_steps):
        with torch.no_grad():
            # w(s) does not need to be part of the graph for backprop
            # It can be treated as a constant
            w_states = self.getStateWeight(states)
        max_dist = self.getDistMax(states, grad_step_size, grad_steps)
        reg = torch.mul(w_states, max_dist)

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        # dist : KL between the clean and the perturbed policy, 0 means the attack
        #        found nothing (budget still too small or PGD too short)
        if DEBUG:
            self._debug_log("REG",
                            reg=reg.mean().item(),
                            dist_mean=max_dist.mean().item(),
                            dist_max=max_dist.max().item(),
                            w_mean=w_states.mean().item())
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        return reg.mean()

    def filter_actions(self, lb_q, ub_q):
        actions = []
        for (lb, ub) in zip(lb_q, ub_q):
            actions.append([
               # for each action i,
               i for i in range(len(lb))
               # an action might be selected
               # if it's upper bound is greater than ALL of the lower bounds of other actions
               # because its logit domain covers a part of the domain of other POSSIBLE actions
               # Previous error: only being greater than one of the other lower bound
               #                is not enough because if the action considered overlaps with
               #                an action that can't be chosen
               #                then the considered action can't be chosen either
               #                See explanation on paper, Algo 2, line 12 (https://arxiv.org/pdf/2210.05927)
               # You can think it the opposite way :
               # if all lb of other actions j are greater than ub[i],
               # this action will never be chosen because its logit will always be lower
               # than the one of the other actions
               if all([ub[i] > lb[j] for j in range(len(lb))])
            ])
        return actions

    def getActionBounds(self, states, method="CROWN"):
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
            method=method
        )
        return lb_qvalues, ub_qvalues

    # Targeted PGD counterpart of getActionBounds, for continuous actions : the reachable
    # interval of each action coordinate is estimated by attacking the state towards both
    # of its extremes instead of relaxing the actor. The relaxation bounds grow with the
    # weights, so they end up looser than the action space itself, while the PGD estimate
    # only depends on the actual sensitivity of the mean. It under-approximates the
    # reachable set instead of covering it, which is the deviation the discrete path
    # already makes by using getSubsetActions over getSubsetActionsCrown
    def getActionBoundsPGD(self, states):
        states = states.detach().to(device=self.device, dtype=torch.float32)

        attack = self.attack_budget * self.attack_scale
        grad_steps = self.grad_steps
        # same relative step as getDistMax: cover the whole [-attack; attack] span
        grad_step_size = 2 * attack / grad_steps

        with torch.no_grad():
            clean_means = self.robust_actor_network(states)
        batch_size, n_dims = clean_means.shape

        # one targeted attack per coordinate and per direction, all stacked in a single
        # batch as getSubsetActions does: block d pushes coordinate d up, block
        # (n_dims + d) pushes it down
        t_states = states.repeat(2 * n_dims, 1)
        dims = torch.arange(n_dims, device=self.device).repeat(2).repeat_interleave(batch_size)
        signs = torch.cat((
            torch.ones(n_dims * batch_size, device=self.device),
            -torch.ones(n_dims * batch_size, device=self.device)
        ))

        delta = (2.0 * torch.rand_like(t_states) - 1.0) * attack
        # the clean mean is reached with delta = 0 so it always belongs to the box,
        # which getMinQContinuous relies on to start its descent from it
        action_lb = clean_means.clone()
        action_ub = clean_means.clone()

        with torch.enable_grad():
            for _ in range(grad_steps):
                delta.requires_grad_(True)

                p_means = self.robust_actor_network(t_states + delta)
                targeted = p_means.gather(1, dims.unsqueeze(1)).squeeze(1)
                grad = torch.autograd.grad((signs * targeted).sum(), delta)[0]

                with torch.no_grad():
                    delta = torch.clamp(
                        delta + grad_step_size * grad.sign(),
                        min=-attack,
                        max=attack
                    )
                    
                    reached = self.robust_actor_network(t_states + delta)
                    reached = reached.gather(1, dims.unsqueeze(1)).squeeze(1)
                    reached = reached.view(2, n_dims, batch_size)
                    action_ub = torch.maximum(action_ub, reached[0].t())
                    action_lb = torch.minimum(action_lb, reached[1].t())
                delta = delta.detach()

        return action_lb, action_ub

    # Targeted PGD variant
    def getSubsetActions(self, states):
        states = states.detach().to(device=self.device, dtype=torch.float32)

        attack = self.attack_budget * self.attack_scale
        grad_steps = self.grad_steps
        # same relative step as getDistMax: cover the whole [-attack; attack] span
        grad_step_size = 2 * attack / grad_steps

        with torch.no_grad():
            clean_values = self.robust_actor_network(states)
        batch_size, n_actions = clean_values.shape
        state_indices = torch.arange(batch_size, device=self.device)

        # reached[state, action] is True once the adversary managed to force that action.
        # The clean preferred action is always reachable (delta = 0)
        best_actions = clean_values.argmax(dim=1)
        reached = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=self.device)
        reached[state_indices, best_actions] = True

        # one targeted attack per candidate action
        # An untargeted attack only explores one direction and systematically
        # under-estimates the subset, which makes the worst-case value optimistic.
        t_states = states.repeat(n_actions, 1) # process all 5 actions at once in a batch
        targets = torch.arange(n_actions, device=self.device).repeat_interleave(batch_size)
        t_best_actions = best_actions.repeat(n_actions)
        # row (t * batch_size + b) holds states[b] attacked towards action t, so the state
        # a stacked row belongs to is simply its index modulo batch_size
        t_state_indices = state_indices.repeat(n_actions)

        delta = (2.0 * torch.rand_like(t_states) - 1.0) * attack

        for _ in range(grad_steps):
            delta.requires_grad_(True)

            p_values = self.robust_actor_network(t_states + delta)

            values_target = p_values.gather(1, targets.unsqueeze(1)).squeeze(1)
            values_best = p_values.gather(1, t_best_actions.unsqueeze(1)).squeeze(1)
            margin = (values_target - values_best).sum()

            grad = torch.autograd.grad(margin, delta)[0]

            with torch.no_grad():
                delta = torch.clamp(
                    delta + grad_step_size * grad.sign(),
                    min=-attack,
                    max=attack
                )
                # record the preferred action reached at this step, for every target block
                greedy_actions = self.robust_actor_network(t_states + delta).argmax(dim=1)
                reached[t_state_indices, greedy_actions] = True
            delta = delta.detach()

        subset_actions = [
            [action for action, is_reached in enumerate(row) if is_reached]
            for row in reached.tolist()
        ]

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        if DEBUG:
            self._debug_subset(clean_values, subset_actions, "PGD")
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        return subset_actions

    # Certified variant: CROWN bounds on the actor values, then filter.
    def getSubsetActionsCrown(self, states):
        action_lb, action_ub = self.getActionBounds(states)
        subset_actions = self.filter_actions(action_lb, action_ub)

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        if DEBUG:
            with torch.no_grad():
                clean_logits = self.robust_actor_network(states)
            self._debug_subset(clean_logits, subset_actions, "CROWN",
                               width=(action_ub - action_lb).mean().item())
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        return subset_actions

    # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
    # one line per tag, printed every log_freq calls of that tag
    # (every=1 for a tag already called once per iteration)
    def _debug_log(self, tag, every=None, **values):
        calls = self._debug_calls.get(tag, 0) + 1
        self._debug_calls[tag] = calls
        if calls % (self.config.log_freq if every is None else every):
            return
        fields = " ".join(f"{name}={value:+.4f}" for name, value in values.items())
        print(f"[DEBUG-{tag}] call={calls} {fields}")

    # spread : mean (max - min) of the clean logits, how far apart the actions are
    # subset : mean size of the perturbed action set, the quantity to watch. A subset
    #          stuck at 1 means the worst case is unconstrained, stuck at n_actions
    #          means the constraint vanished
    # width  : CROWN variant only, mean interval width. An action is pruned only if its
    #          gap to the best one exceeds width, so the subset saturates while spread < width
    def _debug_subset(self, clean_logits, subset_actions, tag, width=None):
        values = {
            "spread": (clean_logits.max(dim=1).values - clean_logits.min(dim=1).values).mean().item(),
            "subset": sum(len(e) for e in subset_actions) / len(subset_actions),
            "eps": self.attack_budget
        }
        if width is not None:
            values["width"] = width
            values["ratio"] = values["spread"] / max(width, 1e-12)
        self._debug_log(tag, **values)
    # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

    @Timer.measure("Critic network update")
    def _train_critic_network(self, states, returns):
        critic_values = self.critic_network(states).squeeze(-1)
        loss = self.critic_network.loss_fn(critic_values, returns)

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        if DEBUG:
            self._debug_log("CRITIC",
                            loss=loss.item(),
                            v_mean=critic_values.mean().item(),
                            ret_mean=returns.mean().item(),
                            ret_std=returns.std().item())
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        self.optimize_model(loss, self.critic_network)

    @Timer.measure("WOCA target calculation")
    def _woca_target(self, rewards, next_states, dones):
        if self.config.ppo_continuous:
            woca_qvalues = self.getMinQContinuous(next_states)
        else:
            # the actions an attacker can force the agent to take from next_states
            subset_actions = self.getSubsetActions(next_states)
            with torch.no_grad():
                woca_qvalues = self.woca_network.getMinQ(next_states, self.env.action_space.n, self.device, subset_actions)

        return rewards + self.config.gamma * woca_qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)

    @Timer.measure("WOCA network update")
    def _train_woca_network(self, states, actions, rewards, next_states, dones, y_woca_target):
        y_woca = self.woca_network.get_qvalues((states, actions, rewards, next_states, dones), self.device)
        woca_loss = self.woca_network.loss_fn(y_woca.squeeze(), y_woca_target)

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        if DEBUG:
            self._debug_log("WOCA",
                            loss=woca_loss.item(),
                            q_mean=y_woca.mean().item(),
                            y_mean=y_woca_target.mean().item(),
                            y_std=y_woca_target.std().item())
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        self.optimize_model(woca_loss, self.woca_network)

    @Timer.measure("WOCA advantage calculation")
    def _woca_advantage(self, states, actions):
        with torch.no_grad():
            if self.config.ppo_continuous:
                woca_qvalues = self.woca_network(torch.cat((states, actions), dim=1)).squeeze(-1)
                # same baseline as the discrete branch below, but
                # Independent(Normal) has no .probs so the expectation over the policy
                # cannot be written as a finite sum : it is estimated by sampling
                # nsamples actions per state and averaging Q_wst over them.
                nsamples = self.config.woca_baseline_samples
                sampled_actions = self.robust_actor_network.distribution(states).sample((nsamples,))
                # (nsamples,B,obs) and (nsamples,B,act) flattened to a single (nsamples*B,...) forward
                repeated_states = states.unsqueeze(0).expand(nsamples, -1, -1)
                baseline_qvalues = self.woca_network(torch.cat(
                    (repeated_states.flatten(0, 1), sampled_actions.flatten(0, 1)),
                    dim=1
                ))
                # instead of calculating the weighted sum of qvalues using the distribution
                # we sample actual qvalues which follow the distribution
                # so it gives an approximation of the weighted sum (Monte Carlo method)
                baseline = baseline_qvalues.view(nsamples, states.shape[0]).mean(dim=0)

                # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
                # Does Q_wst use its action input at all, and on what support was it fit
                # q_std_a : spread of Q_wst over the actions sampled at a fixed state,
                #           averaged over states. 0 means the action input carries nothing
                # q_std_s : spread over states of the per-state average. The ratio
                #           q_std_a / q_std_s is the share of the variation Q_wst
                #           explains with the action rather than with the state alone
                # buf_dev : |a_buf - mu(s)| in sigma units, the support the regression
                #           actually sees. To be read against MINQ's width_sig
                # sigma   : that support in action units, same comparison
                if DEBUG:
                    q_per_sample = baseline_qvalues.view(nsamples, states.shape[0])
                    q_std_a = q_per_sample.std(dim=0).mean()
                    q_std_s = baseline.std()
                    sigma = self.robust_actor_network.log_std.detach().clamp(self.robust_actor_network.log_std_min, 1.0).exp()
                    action_means = self.robust_actor_network(states)
                    self._debug_log("QACT",
                                    q_std_a=q_std_a.item(),
                                    q_std_s=q_std_s.item(),
                                    ratio=(q_std_a / (q_std_s + 1e-8)).item(),
                                    buf_dev=((actions - action_means).abs() / sigma).mean().item(),
                                    sigma=sigma.mean().item())
                # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

                return woca_qvalues - baseline
            else:
                qvalues = self.woca_network(states)
                # Q_wst is a value, not an advantage. 
                # So it contains a baseline value + some info about that action
                # basically : A = Q - V => Q = A + V
                # to keep distribution Q but recenter it around A 
                # so it has the scale of an Advantage
                # we calculate the WEIGHTED SUM FOLLOWING the distribution
                # (a flat .mean() would loose the information on the difference between states)
                # and substract it to the value so we keep only the meaningful part of each qvalue
                probs = self.robust_actor_network.distribution(states).probs
                baseline = (probs * qvalues).sum(dim=1)
                return qvalues.gather(1, actions.unsqueeze(1)).squeeze(1) - baseline

    @Timer.measure("Robust actor network update")
    def _train_robust_actor_network(self, states, actions, old_log_probs, advantages, rewards, next_states, dones, reg_loss, woca_advantage):
        distribution = self.robust_actor_network.distribution(states)
        entropy = distribution.entropy()
        current_log_probs = distribution.log_prob(actions)

        ratio = torch.exp(current_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        )
        
        woca_advantages = advantages + self.k_wst * woca_advantage
        unclipped_objective = ratio * woca_advantages
        clipped_objective = clipped_ratio * woca_advantages

        surrogate = -torch.minimum(unclipped_objective, clipped_objective).mean()
        reg_term = self.k_reg * reg_loss
        # entropy is added to favor exploration
        entropy_term = self.config.ent_coef * -torch.mean(entropy)

        loss = surrogate + reg_term + entropy_term

        # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
        # the three terms are the ones summed into the actor loss: if one of them
        # dominates the others by orders of magnitude, its coefficient is off
        # adv/qwst/wadv std show whether Q_wst keeps the same scale as the advantages
        if DEBUG:
            self._debug_log("ACTOR",
                            loss=loss.item(),
                            surrogate=surrogate.item(),
                            reg_term=reg_term.item(),
                            entropy_term=entropy_term.item(),
                            adv_std=advantages.std().item(),
                            qwst_mean=woca_advantage.mean().item(),
                            qwst_std=woca_advantage.std().item(),
                            wadv_std=woca_advantages.std().item(),
                            ratio_mean=ratio.mean().item(),
                            k_wst=self.k_wst)
        # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

        self.optimize_model(loss, self.robust_actor_network)

    @Timer.measure("Sample collection")
    def _collect_samples(self, parallel, iteration):
        if parallel:
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
                                     self.config.ppo_lm,
                                     self.worker_seed(iteration, worker))
                        for worker in range(self.nworkers)
                    ]
            res = [f.result() for f in futures]
            self.buffer.merge(res)
        else:
            state, _ = self.env.reset(seed=self.worker_seed(iteration, 0))

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

    def _cast_minibatch(self, minibatch):
        states, actions, old_log_probs, advantages, returns, rewards, next_states, dones = minibatch
        return (
            states.float(),
            actions.float() if self.config.ppo_continuous else actions.long(),
            old_log_probs.float().detach(),
            advantages.float(),
            returns.float(),
            rewards.float(),
            next_states.float(),
            dones.float()
        )

    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0

        while iteration < self.config.iterations:
            # ============ COLLECT SAMPLES INTO BATCHES =========
            self._collect_samples(self.config.parallel, iteration)
            self.record.add_episodes(self.buffer.episode_stats, iteration=iteration)

            # ============ GET BATCH =====================
            minibatches = self.buffer.get_batch(self.config.ppo_minibatch_size)

            # ============ TRAIN WOCA NETWORK ============
            # the worst-case network is a Q network trained by regression, it does not
            # depend on the PPO ratio, so it gets its own passes over the batch before
            # the policy update : Q_wst then stays fixed while the actor epochs run
            # the targets are calculated once, before any update, so they stay constant
            # for every epoch and the regression has a fixed objective
            woca_targets = []
            for minibatch in minibatches:
                _, _, _, _, _, rewards, next_states, dones = self._cast_minibatch(minibatch)
                woca_targets.append(self._woca_target(rewards, next_states, dones))

            for _ in range(self.config.woca_update_epochs):
                for minibatch, y_woca_target in zip(minibatches, woca_targets):
                    states, actions, old_log_probs, advantages, returns, rewards, next_states, dones = self._cast_minibatch(minibatch)
                    self._train_woca_network(states, actions, rewards, next_states, dones, y_woca_target)

            # Q_wst are qvalues not advantages. Overmore advantages are normalized
            # So Q_wst must be "normalized"/rescaled before added to the advantages
            # For that we recenter the qvalues by substracting the basleine value (see _woca_advantages)
            # then we rescale by dividing by the std over the whole buffer as done with advantages
            woca_advantages = []
            for minibatch in minibatches:
                states, actions, _, _, _, _, _, _ = self._cast_minibatch(minibatch)
                woca_advantages.append(self._woca_advantage(states, actions))
            woca_adv_scale = torch.cat(woca_advantages).std() + 1e-8
            woca_advantages = [advantage/woca_adv_scale for advantage in woca_advantages]

            # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
            # adv_scale : action dependence of Q_wst before the rescale. Close to 0 means
            #             the woca network ignores its action inputs and the rescale turns
            #             numerical noise into a full strength signal
            if DEBUG:
                self._debug_log("WADV",
                                every=1,
                                adv_scale=woca_adv_scale.item(),
                                target_scale=torch.cat(woca_targets).std().item())
            # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

            for _ in range(self.config.ppo_update_epochs):
                for minibatch, woca_advantage in zip(minibatches, woca_advantages):
                    states, actions, old_log_probs, advantages, returns, rewards, next_states, dones = self._cast_minibatch(minibatch)

                # ============ OPTIMIZE CRITIC ===============
                    self._train_critic_network(states, returns)

                # ============ CALCULATE REGULARIZATION ============
                    reg_loss = self.calcRegularization(states, self.grad_step_size, self.grad_steps)

                # ============ OPTIMIZE ROBUST ACTOR =================
                    self._train_robust_actor_network(states, actions, old_log_probs, advantages, rewards, next_states, dones, reg_loss, woca_advantage)

            # ============ DEBUG ONLY - REMOVE AFTER DEBUG PHASE ============
            # log_std is shared by every state and divides the KL regularization, so the
            # actor can shrink that term by widening the policy instead of stabilizing
            # its mean. Climbing towards the clamp (1.0, std 2.7 on a Box(-1,1)) means
            # the policy degenerates into clipped noise
            if DEBUG and self.config.ppo_continuous:
                log_std = self.robust_actor_network.log_std.detach()
                self._debug_log("POLICY",
                                every=1,
                                log_std_mean=log_std.mean().item(),
                                log_std_max=log_std.max().item(),
                                std_mean=log_std.clamp(self.robust_actor_network.log_std_min, 1.0).exp().mean().item())
            # ========== END DEBUG ONLY - REMOVE AFTER DEBUG PHASE ==========

            self.buffer.reset()
            iteration += 1
            self.k_wst = self.k_wst_update(self.k_wst)
            self.attack_budget = self.attack_budget_update(self.attack_budget)
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
