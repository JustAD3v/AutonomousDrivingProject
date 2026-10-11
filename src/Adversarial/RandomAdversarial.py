import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class RandomAdversarial(Adversarial):
    def __init__(self, attack_bound):
        super().__init__(attack_bound)

    def disrupt(self, state):
        perturbations = [np.random.uniform(-self.attack_bound, self.attack_bound, len(state))]
        return np.add(state, perturbations)