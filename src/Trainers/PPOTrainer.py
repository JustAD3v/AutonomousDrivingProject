import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np

from Trainers.TrainConfig import TrainConfig
from Trainers.PPOBuffer import PPOBuffer
from Trainers.TrainResult import TrainResult
from Utils.Timer import Timer

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
        self.record = TrainResult(self.config.iterations)
        
    def optimize_model(self, loss, model):
       model.optimizer.zero_grad()
       loss.backward()
       
       torch.nn.utils.clip_grad_norm_(
         model.parameters(),
         max_norm=10.0
       )
       model.optimizer.step()

    @Timer.measure("Prediction calculation")
    def _predict(self, state):
        return self.policy.predict(state, device=self.device)

    @Timer.measure("Environment processing")
    def _environment_step(self, action):
        return self.env.step(action)

    
    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0

        while iteration < self.config.iterations:
            state, _ = self.env.reset()
            episode_return = 0.0
            episode_length = 0

            for _ in range(self.config.buffer_size):
                # ========= SAMPLE COLLECTION =============
                action, log_action_prob = self._predict(state)
                next_state, reward, term, trunc, _ = self._environment_step(action)

                episode_return += reward
                episode_length += 1

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

                    print(f"Episode finished (len={episode_length}), tot_return={episode_return}")

                    episode_return = 0.0
                    episode_length = 0

                    # Reset the environment only if the episode ended.
                    if term or trunc:
                        state, _ = self.env.reset()
                    else:
                        # The buffer is full, but the episode may continue.
                        state = next_state
                else:
                    state = next_state

            # ============ GET BATCH =====================
            minibatches = self.buffer.get_batch()
            for _ in range(self.ppo_update_epochs):
                for minibatch in minibatches:
                    states, actions, old_log_probs, advantages, returns = minibatch
                    states = states.float()
                    actions = actions.long()
                    old_log_probs = old_log_probs.float().detach()
                    advantages = advantages.float()
                    returns = returns.float()

                    # Normalize advantages ?
                
                # ============ OPTIMIZE CRITIC ===============
                
                    critic_values = self.critic_network(states).squeeze(-1)

                    loss = self.critic_network.loss_fn(critic_values, returns)
                    self.optimize_model(loss, self.critic_network)

            # ============ OPTIMIZE ACTOR =================
                    action_values = self.actor_network(states)
                    # Categorical applies softmax on network output
                    # to get a probability distribution from the output vector
                    # it also wraps it into an interface that allows for
                    # sampling that distribution
                    # and finding the log_prob for a given action
                    distribution = torch.distributions.Categorical(logits=action_values)
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
                    self.optimize_model(loss, self.actor_network)
            
            self.buffer.reset()
            iteration += 1
            print(f"[*] Iteration Finished ({iteration}/{self.config.iterations})")
    
    def learn(self) -> TrainResult:
        self.timer.reset()
        self._train()
        self.timer.report()
        return self.record
