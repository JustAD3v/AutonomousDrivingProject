from matplotlib import pyplot as plt

class TrainResult:
    def __init__(self, iterations):
        self.iterations = iterations
        self.episode_count = 1
        self.epsilons = []
        self.total_loss = []
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = []

    def add_loss(self, value):
        self.losses.append(value)
    
    def add_epsilon(self, value):
        self.epsilons.append(value)

    def add_episode_return(self, value):
        self.episode_returns.append(value)

    def register_episode_count(self, value):
        self.episode_count = value

    def plot(self):
        x = range(self.episode_count)
        plt.plot(x, self.losses, label="Loss", color="red")
        plt.plot(x, self.epsilons, label="Epsilon", color="green")
        plt.plot(x, self.episode_returns, label="Return", color="blue")
        plt.xlabel("nb episodes")
        plt.legend()
        plt.show()