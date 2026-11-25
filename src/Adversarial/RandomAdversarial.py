import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class RandomAdversarial(Adversarial):
    def __init__(self, policy, attack_bound=0.5):
        super().__init__(policy, attack_bound)
        self.mean_per_feature = None

    def disrupt(self, state):
        if self.mean_per_feature is None: 
            feature_means = state.mean(axis=0)
            self.mean_per_feature = np.tile(feature_means, (5, 1))
        
        perturbations = np.random.uniform(
            low=-self.attack_bound*self.mean_per_feature, 
            high=self.attack_bound*self.mean_per_feature, 
            size=state.shape
        )
        return np.add(state, perturbations)