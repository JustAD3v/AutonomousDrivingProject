import torch
import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class FGSMAdversarial(Adversarial):
    def __init__(self, policy, attack_bound=0.5):
        super().__init__(policy, attack_bound)

    def disrupt(self, state):
        shape = state.shape
        state_tensor = torch.tensor(np.ravel(state), dtype=torch.float32, requires_grad=True)
        q_values = self.policy.network(state_tensor)
        action = q_values.argmax()
        loss = q_values[action]
        gradients = torch.autograd.grad(outputs=loss, inputs=state_tensor)[0]
        perturbations = self.attack_bound * state * torch.sign(gradients).reshape(shape)
        return np.add(state, perturbations.numpy())