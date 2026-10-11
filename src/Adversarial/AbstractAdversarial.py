from abc import ABC, abstractmethod

class Adversarial(ABC):
    @abstractmethod
    def __init__(self, policy, attack_bound: float):
        self.attack_bound = attack_bound
        # currently only supports custom implementation of policy
        self.policy = policy

    @abstractmethod
    def disrupt(self, state):
        pass