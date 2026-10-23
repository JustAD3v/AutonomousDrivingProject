import json
import random

class TrainConfig:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.exploration_frac = 0.1
        self.epsilon_decay = 0.994
        self.epsilon_min = 0.01
        self.buffer_size = 5000
        self.warmup_size = 800
        self.batch_size = 64
        self.copy_freq = 800
        self.train_freq = 1
        self.iterations = 500
        self.learning_rate = 5e-4
        self.policy_kwargs = dict(net_arch=[256, 256])
        self.seed = random.getrandbits(32)
        self.k_wst = 0.5
        self.k_reg = 0.2
        self.attack_budget = 0.5
        self.wocar_grad_step_size = 5e-4
        self.wocar_grad_steps = 100
        self.ppo_update_steps = 4
        self.ppo_clip_epsilon = 0.2
        self.ppo_lm = 0.8


    def to_json(self):
        return json.dumps(self.__dict__)

    def from_json(self, value):
        self.__dict__.update(json.loads(value))
        return self

    def to_sb3_kwargs(self):
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.warmup_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": 1,
            "target_update_interval": self.copy_freq,
            "verbose": 1,
            "policy_kwargs": self.policy_kwargs,
            "exploration_final_eps": self.epsilon_min,
            "seed": self.seed
        }

class TrainConfigBuilder:
    def __init__(self):
        self._config = TrainConfig()

    def gamma(self, value):
        self._config.gamma = value
        return self

    def epsilon_start(self, value):
        self._config.epsilon_start = value
        self._config.epsilon = self._config.epsilon_start
        return self
    
    def epsilon_decay(self, value):
        self._config.epsilon_decay = value
        return self
    
    def epsilon_min(self, value):
        self._config.epsilon_min = value
        return self
    
    def warmup_size(self, value):
        self._config.warmup_size = value
        return self
    
    def batch_size(self, value):
        self._config.batch_size = value
        return self
    
    def copy_freq(self, value):
        self._config.copy_freq = value
        return self
    
    def iterations(self, value):
        self._config.iterations = value
        return self

    def buffer_size(self, value):
        self._config.buffer_size = value
        return self

    def learning_rate(self, value):
        self._config.learning_rate = value
        return self

    def train_freq(self, value):
        self._config.train_freq = value
        return self

    def gradient_steps(self, value):
        self._config.gradient_steps = value
        return self

    def policy_kwargs(self, value):
        self._config.policy_kwargs = value
        return self
    
    def exploration_frac(self, value):
        self._config.exploration_frac = value
        return self
    
    def seed(self, value):
        self._config.seed = value
        return self
    
    def k_wst(self, value):
        self._config.k_wst = value
        return self
    
    def k_reg(self, value):
        self._config.k_reg = value
        return self
    
    def attack_budget(self, value):
        self._config.attack_budget = value
        return self
    
    def wocar_grad_step_size(self, value):
        self._config.wocar_grad_step_size = value
        return self
    
    def wocar_grad_steps(self, value):
        self._config.wocar_grad_steps = value
        return self
    
    def ppo_clip_epsilon(self, value):
        self._config.ppo_clip_epsilon = value
        return self
    
    def ppo_update_steps(self, value):
        self._config.ppo_update_steps = value
        return self
    
    def ppo_lm(self, value):
        self._config.ppo_lm = value
        return self
    
    def build(self):
        return self._config