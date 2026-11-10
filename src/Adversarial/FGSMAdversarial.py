import torch
import numpy as np
from Adversarial.AbstractAdversarial import Adversarial

class FGSMAdversarial(Adversarial):
    def __init__(self, policy, attack_bound=0.5, env=None):
        super().__init__(policy, attack_bound, env)

    def disrupt(self, state):
        shape = state.shape
        state_tensor = torch.tensor(np.ravel(state), dtype=torch.float32, requires_grad=True)
        outputs = self.policy.network(state_tensor)
        action = int(outputs.argmax())
        # in both cases the loss is decreased to bring the agent away from its
        # preferred action, but what has to be decreased depends on the network
        if hasattr(self.policy.network, "distribution"):
            # policy network: the outputs are logits, a common shift leaves the
            # softmax unchanged, so lowering the preferred logit directly lowers
            # the probability of that action
            loss = outputs[action]
        else:
            # value network: the outputs are Q values, where the state value shared
            # by all the actions dominates the gradient. Lowering the margin with the
            # best other action instead keeps the step on the ranking of the actions
            others = torch.cat([outputs[:action], outputs[action + 1:]])
            loss = outputs[action] - others.max()
        gradients = torch.autograd.grad(outputs=loss, inputs=state_tensor)[0]
        # descent on the loss, each column of the observation being scaled by the
        # range of its feature (see Adversarial.feature_scale)
        gradient_sign = np.sign(gradients.numpy()).reshape(shape)
        perturbations = -self.attack_bound * self.feature_scale() * gradient_sign
        return np.add(state, perturbations)
