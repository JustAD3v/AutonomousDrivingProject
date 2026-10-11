from Adversarial.AbstractAdversarial import Adversarial

class NeutralAdversarial(Adversarial):
    def __init__(self, policy=None, attack_bound=0.0):
        super().__init__(policy, attack_bound)

    def disrupt(self, state):
        return state