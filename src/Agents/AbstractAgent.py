from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        super().__init__()

    def init_agent(self, env, model, params, device):
        self.env = env
        self.model = model
        self.params = params
        self.device = device

    @property
    @abstractmethod
    def env(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def process_batch(self, batch):
        pass

