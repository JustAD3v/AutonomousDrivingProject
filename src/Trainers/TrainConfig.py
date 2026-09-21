class TrainConfig:
    def __init__(self):
        self.gamma = 0.95 # reward discount
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.epsilon_decay = 0.994
        self.epsilon_min = 0.01
        self.buffer_size = 5000
        self.warmup_size = 800
        self.batch_size = 64
        self.update_freq = 800
        self.nb_episodes = 500

class TrainConfigBuilder:
    def __init__(self):
        self._config = TrainConfig()

    def gamma(self, value):
        self._config.gamma = value;
        return self

    def epsilon_start(self, value):
        self._config.epsilon_start = value;
        self._config.epsilon = self._config.epsilon_start
        return self
    
    def epsilon_decay(self, value):
        self._config.epsilon_decay = value;
        return self
    
    def epsilon_min(self, value):
        self._config.epsilon_min = value;
        return self
    
    def warmup_size(self, value):
        self._config.warmup_size = value;
        return self
    
    def batch_size(self, value):
        self._config.batch_size = value;
        return self
    
    def update_freq(self, value):
        self._config.update_freq = value;
        return self
    
    def nb_episodes(self, value):
        self._config.nb_episodes = value;
        return self

    def buffer_size(self, value):
        self._config.buffer_size = value;
        return self
    
    def build(self):
        return self._config