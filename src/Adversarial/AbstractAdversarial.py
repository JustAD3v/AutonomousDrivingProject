from abc import ABC, abstractmethod

class Adversarial(ABC):
    @abstractmethod
    def __init__(self, attack_bound=0.5):
        self.attack_bound = attack_bound

    @abstractmethod
    def disrupt(self, state):
        pass