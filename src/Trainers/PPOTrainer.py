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

class WocaRDQN:
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
        self.clip_epsilon = 0.2
        self.actor_network = actor_network
        self.old_actor_weights = self.actor_network.state_dict()
        self.critic_network = critic_network
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.r_buffer = PPOBuffer(self.config.buffer_size, 
                                     config.batch_size,
                                     config.warmup_size,
                                     self.device)
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
            nb_steps = 0
            for _ in range(self.config.batch_size):
                action, action_probability = self._predict(state)
                new_state, reward, term, trunc, _ = self._environment_step(action)
                episode_return += reward
                nb_steps += 1
                state_tensor = torch.tensor(np.asarray(state, dtype=np.float32).flatten(), dtype=torch.float32)
                new_state_tensor = torch.tensor(np.asarray(new_state, dtype=np.float32).flatten(), dtype=torch.float32)
                value = self.critic_network(state_tensor)
                value_next = self.critic_network(new_state_tensor)
                rtg = reward + self.config.gamma * value_next
                advantage = rtg - value 
                self.r_buffer.push(
                (
                    np.asarray(state, dtype=np.float32).flatten(),
                    action,
                    action_probability, 
                    reward, 
                    value,
                    rtg,
                    advantage,
                    term
                    )
                )
            
            state, _, action_probability, _, value, rtg, _, _ = self.r_buffer.get_batch()
            #============ OPTIMIZE CRITIC ===============#
            loss = self.critic_network.loss_fn(value, rtg)
            self.optimize_model(loss, self.critic_network)
            #============ OPTIMIZE ACTOR ===============#
            saved_current_weights = self.actor_network.state_dict()
            # switching/copying state is a performance bottleneck ?
            self.actor_network.load_state_dict(self.old_actor_weights)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            old_action_prob = self.actor_network(state_tensor)
            ratio = action_probability / old_action_prob
            left_loss = ratio * action_probability 
            right_loss = max(1-self.clip_epsilon, min(ratio, 1+self.clip_epsilon))
            self.actor_network.load_state_dict(saved_current_weights)            
            loss = -min(left_loss, right_loss)
            self.old_actor_weights = self.actor_network.state_dict()
            self.optimize_model(loss, self.actor_network)
            
    def learn(self) -> TrainResult:
        self.timer.reset()

        self._warmup()
        warmup_duration = self.timer.counters["Warmup duration"]
        print(f"[+] Warmup done ! (Duration = {round(warmup_duration, 2)}s)")

        self._train()

        return self.record
