import torch
import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class FGSMAdversarial(Adversarial):
    def __init__(self, policy, attack_bound=0.5):
        super().__init__(policy, attack_bound)
        self.mean_per_feature = None

    def disrupt(self, state):
        shape = state.shape
        state_tensor = torch.tensor(np.ravel(state), dtype=torch.float32, requires_grad=True)
        if self.mean_per_feature is None: 
            # each column correspond to the same feature in observation
            # cache the mean per feature 
            # This allows for an attack with fix bound
            # but with relative value meaningful for each feature 
            self.mean_per_feature = state_tensor.reshape(shape).mean(dim=0).repeat(5,1)        
        q_values = self.policy.network(state_tensor)
        action = q_values.argmax()
        loss = q_values[action]
        gradients = torch.autograd.grad(outputs=loss, inputs=state_tensor)[0]
        perturbations = self.attack_bound * self.mean_per_feature * torch.sign(gradients).reshape(shape)
        return np.add(state, perturbations.detach().numpy())