import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class RandomAdversarial(Adversarial):
    def __init__(self, policy, attack_bound=0.5, env=None):
        super().__init__(policy, attack_bound, env)

    def disrupt(self, state):
        # each column of the observation is one feature, so the bound is scaled
        # by the range of that feature (see Adversarial.feature_scale)
        bound = self.attack_bound * self.feature_scale()
        perturbations = np.random.uniform(low=-bound, high=bound, size=state.shape)
        return np.add(state, perturbations)
