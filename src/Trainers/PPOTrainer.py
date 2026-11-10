import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from Trainers.TrainConfig import TrainConfig
from Trainers.PPOBuffer import PPOBuffer
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

def collector_worker(env_name, env_config, buffer_size, policy, device, critic_network, gamma, ppo_lm):
    env = gym.make(env_name, config=env_config)
    state, _ = env.reset()
    buffer = PPOBuffer(buffer_size, device, gamma, ppo_lm)
    for _ in range(buffer_size):
        action, log_action_prob = policy.predict(state, device=device)
        next_state, reward, term, trunc, _ = env.step(policy.to_env_action(action))
        state_flat = np.asarray(state, dtype=np.float32).flatten()
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
            critic_value
        ))
        end = term | trunc | buffer.full
        if end:
            if term:
                extra_critic_value = 0.0
            else:
                next_state_array = np.asarray(next_state, dtype=np.float32).flatten()
                with torch.no_grad():
                    next_state_tensor = torch.as_tensor(
                        next_state_array,
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

class PPOTrainer:
    def __init__(self, 
                 config: TrainConfig, 
                 actor_network: nn.Module,
                 critic_network: nn.Module, 
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

        self.config = config
        self.policy = policy
        self.clip_epsilon = self.config.ppo_clip_epsilon
        self.actor_update_steps = self.config.ppo_actor_update_steps
        self.critic_update_steps = self.config.ppo_critic_update_steps
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.buffer = PPOBuffer(self.config.buffer_size, 
                                self.device, 
                                self.config.gamma, 
                                self.config.ppo_lm)
        self.record = TrainResult(self.config.iterations, self.config.log_freq, name="PPO training", config=self.config, timer=self.timer)

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

    @Timer.measure("Prediction calculation")
    def _predict(self, state):
        return self.policy.predict(state, device=self.device)

    @Timer.measure("Environment processing")
    def _environment_step(self, action):
        return self.env.step(self.policy.to_env_action(action))
    
    @Timer.measure("Sample collection")
    def _collect_samples(self, parallel):
        if parallel:
            print("using parallel !")
            nworkers = self.rounded_nworkers()
            buff_size = self.config.buffer_size // nworkers
            with ProcessPoolExecutor(max_workers=nworkers) as executor:
                futures = [executor.submit(collector_worker, 
                                         self.env_name, 
                                         self.env_config, 
                                         buff_size, 
                                         self.policy, 
                                         self.device, 
                                         self.critic_network, 
                                         self.config.gamma, 
                                         self.config.ppo_lm)
                            for _ in range(nworkers)
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
                    critic_value
                ))

                end = term | trunc | self.buffer.full
                if end:
                    if term:
                        extra_critic_value = 0.0
                    else:
                        next_state_array = np.asarray(next_state, dtype=np.float32).flatten()
                        with torch.no_grad():
                            next_state_tensor = torch.as_tensor(
                                next_state_array,
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

    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0

        while iteration < self.config.iterations:
            # ============ COLLECT SAMPLES INTO BATCHES =========
            self._collect_samples(self.config.parallel)
            self.record.add_episodes(self.buffer.episode_stats, iteration=iteration)
            
            # ============ GET BATCH =====================
            minibatches = self.buffer.get_batch()
            for _ in range(self.config.ppo_update_epochs):
                for minibatch in minibatches:
                    states, actions, old_log_probs, advantages, returns = minibatch
                    states = states.float()
                    actions = actions.float() if actions.is_floating_point() else actions.long()
                    old_log_probs = old_log_probs.float().detach()
                    advantages = advantages.float()
                    returns = returns.float()


                # ============ OPTIMIZE CRITIC ===============
                    critic_values = self.critic_network(states).squeeze(-1)

                    loss = self.critic_network.loss_fn(critic_values, returns)
                    self.optimize_model(loss, self.critic_network)

                # ============ OPTIMIZE ACTOR =================
                    distribution = self.actor_network.distribution(states)
                    entropy = distribution.entropy()
                    current_log_probs = distribution.log_prob(actions)

                    ratio = torch.exp(current_log_probs - old_log_probs)

                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    )

                    unclipped_objective = ratio * advantages
                    clipped_objective = clipped_ratio * advantages

                    loss = -torch.minimum(unclipped_objective,clipped_objective).mean()

                    # add entropy to favor exploration
                    entropyloss = -torch.mean(entropy)
                    loss += self.config.ent_coef * entropyloss
                    
                    self.optimize_model(loss, self.actor_network)
            
            self.buffer.reset()
            iteration += 1
            self.record.report_iteration(iteration)
    
    def learn(self) -> TrainResult:
        self.timer.reset()
        self._train()
        self.record.report_final()
        self.timer.report()
        return self.record
