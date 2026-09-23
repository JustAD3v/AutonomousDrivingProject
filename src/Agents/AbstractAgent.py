from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        self.env = None
        self.model = None
        self.config = None
        self.device = None

    def init_agent(self, env, model, config, device):
        self.env = env
        self.model = model
        self.config = config
        self.device = device

    @abstractmethod
    def policy(self):
        pass