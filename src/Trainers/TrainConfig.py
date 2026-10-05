class TrainConfig:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
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
    
    def build(self):
        return self._config