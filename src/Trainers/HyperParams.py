class HyperParams:
    def __init__(self):
        self.gamma = 0.95 # reward discount
        self.epsilon = 1.0
        self.epislon_decay = 0.994
        self.epsilon_min = 0.01
        self.warmup_size = 800
        self.batch_size = 64
        self.update_freq = 800
        self.nb_epidodes = 500

class HyperParamsBuilder:
    def __init__(self):
        self._hyperparams = HyperParams()

    def gamma(self, value):
        self._hyperparams.gamma = value;

    def epsilon(self, value):
        self._hyperparams.epsilon = value;

    def epsilon_decay(self, value):
        self._hyperparams.epislon_decay = value;

    def epsilon_min(self, value):
        self._hyperparams.epsilon_min = value;

    def warmup_size(self, value):
        self._hyperparams.warmup_size = value;

    def batch_size(self, value):
        self._hyperparams.batch_size = value;

    def update_freq(self, value):
        self._hyperparams.update_freq = value;

    def nb_episodes(self, value):
        self._hyperparams.nb_epidodes = value;

    def build(self):
        return self._hyperparams