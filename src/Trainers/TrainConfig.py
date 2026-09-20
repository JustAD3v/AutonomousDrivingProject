class TrainConfig:
    def __init__(self):
        self.gamma = 0.95 # reward discount
        self.epsilon = 1.0
        self.epislon_decay = 0.994
        self.epsilon_min = 0.01
        self.buffer_size = 5000
        self.warmup_size = 800
        self.batch_size = 64
        self.update_freq = 800
        self.nb_epidodes = 500

class TrainConfigBuilder:
    def __init__(self):
        self._config = TrainConfig()

    def gamma(self, value):
        self._config.gamma = value;

    def epsilon(self, value):
        self._config.epsilon = value;

    def epsilon_decay(self, value):
        self._config.epislon_decay = value;

    def epsilon_min(self, value):
        self._config.epsilon_min = value;

    def warmup_size(self, value):
        self._config.warmup_size = value;

    def batch_size(self, value):
        self._config.batch_size = value;

    def update_freq(self, value):
        self._config.update_freq = value;

    def nb_episodes(self, value):
        self._config.nb_epidodes = value;

    def build(self):
        return self._config