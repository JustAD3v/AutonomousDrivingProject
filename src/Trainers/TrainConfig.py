import itertools
import json
import random

class TrainConfig:
    def __init__(self):
        self.env_name = "highway-fast-v0"
        self.env_config = None
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
        self.max_grad_norm = 0.5
        self.ent_coef = 0.02
        self.policy_kwargs = dict(net_arch=[256, 256])
        self.seed = random.getrandbits(32)
        self.k_wst = 0.5
        self.k_reg = 0.2
        self.attack_budget = 0.05
        self.wocar_grad_step_size = 5e-4
        self.wocar_grad_steps = 100
        self.ppo_actor_update_steps = 4
        self.ppo_critic_update_steps = 4
        self.ppo_update_epochs = 4
        self.ppo_clip_epsilon = 0.2
        self.ppo_lm = 0.8
        self.ppo_nworkers = 8
        self.ppo_minibatch_size = self.ppo_nworkers * self.buffer_size # default takes the whole buffer
        self.ppo_continuous = False
        self.log_freq = 10
        self.parallel = False

        # PPO params (used by the "ppo" trainer). PPO is on-policy, so its training
        # length is counted in environment steps (total_timesteps), not `iterations`.
        self.total_timesteps = 200000
        self.n_steps = 2048          # steps collected per rollout
        self.ppo_batch_size = 64     # minibatch size (separate from the DQN batch_size)
        self.n_epochs = 10           # passes over each rollout
        self.ppo_gamma = 0.90
        self.gae_lambda = 0.8
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_net_arch = [256, 256]   # same width as the DQN network, for a fair comparison
        self.tensorboard_log = None      # e.g. "./tb_logs/" (needs tensorboard installed)
        self.n_envs = 8 #number of envs. 
        self.eval_freq = 5000
        self.best_model_save_path = "./logs/"
        self.log_path = "./logs/"

    def to_json(self, pretty=False):
        return json.dumps(self.__dict__, indent=4 if pretty else None)            

    def from_json(self, value):
        self.__dict__.update(json.loads(value))
        return self

    def copy(self):
        return TrainConfig().from_json(self.to_json())

    def with_overrides(self, overrides):
        config = self.copy()
        for name, value in overrides.items():
            if not hasattr(config, name):
                raise AttributeError(f"Unknown TrainConfig parameter: {name}")
            setattr(config, name, value)
        if "epsilon_start" in overrides:
            config.epsilon = config.epsilon_start
        return config

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

    def to_ppo_kwargs(self):
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.ppo_batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.ppo_gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            # separate actor (pi) and critic (vf) networks of the same size
            "policy_kwargs": dict(net_arch=dict(pi=list(self.ppo_net_arch), vf=list(self.ppo_net_arch))),
            "tensorboard_log": self.tensorboard_log,
            "verbose": 1,
            "seed": self.seed,
            "device": "cpu",
        }

class TrainConfigSweep:

    def __init__(self, base_config=None):
        self.base_config = base_config if base_config is not None else TrainConfig()
        self.varied = {}

    def vary(self, name, values):
        if not hasattr(self.base_config, name):
            raise AttributeError(f"Unknown TrainConfig parameter: {name}")
        self.varied[name] = list(values)
        return self

    def seeds(self, values):
        if isinstance(values, int):
            values = [random.getrandbits(32) for _ in range(values)]
        return self.vary("seed", values)

    def points(self):
        # a point represent a set of concrete values for parameters to train on
        names = list(self.varied)
        # product(*([a, b, c])) unpacks into -> product(a, b, c)
        # product gives cartesian product of attributes to test every possible combination
        for combination in itertools.product(*(self.varied[name] for name in names)):
            # yield returns the next value of the iterator at each call
            yield dict(zip(names, combination))

    def configs(self):
        for overrides in self.points():
            # yield returns the next value of the iterator at each call
            yield overrides, self.base_config.with_overrides(overrides)

    def __len__(self):
        count = 1
        for values in self.varied.values():
            count *= len(values)
        return count

class TrainConfigBuilder:
    def __init__(self):
        self._config = TrainConfig()

    def env_name(self, value):
        self._config.env_name = value
        return self

    def env_config(self, value):
        self._config.env_config = value
        return self

    def gamma(self, value):
        self._config.gamma = value
        return self

    def epsilon_start(self, value):
        self._config.epsilon_start = value
        self._config.epsilon = self._config.epsilon_start
        return self
    
    def exploration_frac(self, value):
        self._config.exploration_frac = value
        return self
    
    def epsilon_decay(self, value):
        self._config.epsilon_decay = value
        return self
    
    def epsilon_min(self, value):
        self._config.epsilon_min = value
        return self
    
    def buffer_size(self, value):
        self._config.buffer_size = value
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
    
    def train_freq(self, value):
        self._config.train_freq = value
        return self
    
    def iterations(self, value):
        self._config.iterations = value
        return self
    
    def learning_rate(self, value):
        self._config.learning_rate = value
        return self
    
    def max_grad_norm(self, value):
        self._config.max_grad_norm = value
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
    
    def ppo_actor_update_steps(self, value):
        self._config.ppo_actor_update_steps = value
        return self
    
    def ppo_critic_update_steps(self, value):
        self._config.ppo_critic_update_steps = value
        return self
    
    def ppo_update_epochs(self, value):
        self._config.ppo_update_epochs = value
        return self
    
    def ppo_clip_epsilon(self, value):
        self._config.ppo_clip_epsilon = value
        return self
    
    def ppo_lm(self, value):
        self._config.ppo_lm = value
        return self
    
    def ppo_nworkers(self, value):
        self._config.ppo_nworkers = value
        return self
    
    def ppo_minibatch_size(self, value):
        self._config.ppo_minibatch_size = value
        return self

    def ppo_continuous(self, value):
        self._config.ppo_continuous = value
        return self

    def log_freq(self, value):
        self._config.log_freq = value
        return self

    def gradient_steps(self, value):
        self._config.gradient_steps = value
        return self

    def policy_kwargs(self, value):
        self._config.policy_kwargs = value
        return self
    
    def seed(self, value):
        self._config.seed = value
        return self

    def parallel(self, value):
        self._config.parallel = value
        return self
    
    # ---- PPO ----
    def total_timesteps(self, value):
        self._config.total_timesteps = value
        return self

    def n_steps(self, value):
        self._config.n_steps = value
        return self

    def ppo_batch_size(self, value):
        self._config.ppo_batch_size = value
        return self

    def n_epochs(self, value):
        self._config.n_epochs = value
        return self

    def ppo_gamma(self, value):
        self._config.ppo_gamma = value
        return self

    def gae_lambda(self, value):
        self._config.gae_lambda = value
        return self

    def clip_range(self, value):
        self._config.clip_range = value
        return self

    def ent_coef(self, value):
        self._config.ent_coef = value
        return self

    def vf_coef(self, value):
        self._config.vf_coef = value
        return self

    def max_grad_norm(self, value):
        self._config.max_grad_norm = value
        return self

    def ppo_net_arch(self, value):
        self._config.ppo_net_arch = value
        return self

    def tensorboard_log(self, value):
        self._config.tensorboard_log = value
        return self 
    
    def eval_freq(self, value):
        self._config.eval_freq = value
        return self

    def best_model_save_path(self, value):
        self._config.best_model_save_path = value
        return self

    def log_path(self, value):
        self._config.log_path = value
        return self

    def build(self):
        return self._config