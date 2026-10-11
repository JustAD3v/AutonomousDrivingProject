from abc import ABC, abstractmethod

class Adversarial(ABC):
    @abstractmethod
    def __init__(self, policy, attack_bound: float):
        self.attack_bound = attack_bound
        self.policy = policy

    @abstractmethod
    def disrupt(self, state):
        pass