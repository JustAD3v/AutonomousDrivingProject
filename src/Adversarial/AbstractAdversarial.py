from abc import ABC, abstractmethod

import numpy as np

class Adversarial(ABC):
    @abstractmethod
    def __init__(self, policy, attack_bound: float, env=None):
        self.attack_bound = attack_bound
        # currently only supports custom implementation of policy
        self.policy = policy
        # environment the perturbed observations come from, used for the feature ranges
        self.env = env
        self.scale = None

    @abstractmethod
    def disrupt(self, state):
        pass

    def feature_scale(self):
        # Calculate Half-range of every observation feature 
        # to scale the attack_bound with it 
        # that makes the bound a fraction of the
        # range of each feature instead of an absolute value shared by features 
        # living on different scales.
        if self.scale is None:
            observation = self.env.unwrapped.observation_type
            # the environment normalizes every feature of features_range to [-1, 1],
            # so their half-range is 1. The others keep their own values: "presence"
            # is a flag in [0, 1], so its half-range is 0.5
            self.scale = np.array([1.0 if feature in observation.features_range else 0.5
                                   for feature in observation.features], dtype=np.float32)
        return self.scale
