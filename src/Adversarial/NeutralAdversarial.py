from Adversarial.AbstractAdversarial import Adversarial

class NeutralAdversarial(Adversarial):
    def __init__(self, attack_bound):
        super().__init__(attack_bound)

    def disrupt(self, state):
        return state