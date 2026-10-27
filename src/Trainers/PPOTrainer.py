# big shoutout to johnny code the goaaaat
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
        self.update_steps = self.config.ppo_update_steps
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
        return self.policy.predict(state, epsilon=self.config.epsilon, device=self.device)

    @Timer.measure("Environment processing")
    def _environment_step(self, action):
        return self.env.step(action)

    @Timer.measure("Training duration")
    def _train(self):
        print("[+] === Training starts ===")
        iteration = 0
        episode_count = 0
        while iteration < self.config.iterations:
            state, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            for _ in range(self.config.batch_size):
                # ========= SAMPLE COLLECTION =============#
                action, log_action_prob = self._predict(state)
                next_state, reward, term, trunc, _ = self._environment_step(action)
                # episode_return += reward
                # episode_length += 1
                end = term | trunc | self.buffer.full
                # no need to use np ?? use torch.as_tensor instead ?
                state_tensor = torch.tensor(np.asarray(state, dtype=np.float32).flatten(), dtype=torch.float32)
                critic_value = self.critic_network(state_tensor)
                self.buffer.push(
                    (
                        state,
                        action,
                        reward,
                        log_action_prob,
                        critic_value
                    )
                )
                state = next_state
                if end:
                    if term:
                        extra_critic_value = 0
                    else:
                        action, log_action_prob = self._predict(state)
                        next_next_state, _, _, _, _ = self._environment_step(action)
                        next_state_tensor = torch.tensor(np.asarray(next_next_state, dtype=np.float32).flatten(), dtype=torch.float32)
                        extra_critic_value = self.critic_network(next_state_tensor)
                    self.buffer.finish_episode(extra_critic_value)
                    state, _ = self.env.reset()
                    # episode_return = 0
                    # episode_length = 0

            states, actions, log_action_probs, critic_values, advantages, returns = self.buffer.get_batch()
            #============ OPTIMIZE CRITIC ===============#
            loss = self.critic_network.loss_fn(critic_values, returns)
            self.optimize_model(loss, self.critic_network)
            #============ OPTIMIZE ACTOR ===============#
            for _ in range(self.update_steps):
                action_values = self.actor_network(states)
                # Categorical applies softmax on network output
                # to get a probability distribution from the output vector
                # it also wraps it into an interface that allows for
                # sampling that distribution
                # and finding the log_prob for a given action
                distribution = torch.distributions.Categorical(action_values)
                current_log_action_probs = distribution.log_prob(actions)
                ratio = torch.exp(current_log_action_probs - log_action_probs)
                left_loss = ratio * advantages 
                right_loss = torch.maximum(1-self.clip_epsilon, torch.minimum(ratio, 1+self.clip_epsilon)) * advantages        
                loss = -torch.minimum(left_loss, right_loss).mean()  
                self.optimize_model(loss, self.actor_network)
            
    def learn(self) -> TrainResult:
        self.timer.reset()

        self._warmup()
        warmup_duration = self.timer.counters["Warmup duration"]
        print(f"[+] Warmup done ! (Duration = {round(warmup_duration, 2)}s)")

        self._train()

        return self.record
