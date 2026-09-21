from matplotlib import pyplot as plt

class TrainResult:
    def __init__(self, nb_episodes):
        self.nb_episodes = nb_episodes
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

    def plot(self):
        x = range(self.nb_episodes)
        plt.plot(x, self.losses, label="Loss", color="red")
        plt.plot(x, self.epsilons, label="Epsilon", color="green")
        plt.plot(x, self.episode_returns, label="Return", color="blue")
        plt.xlabel("nb episodes")
        plt.legend()
        plt.show()