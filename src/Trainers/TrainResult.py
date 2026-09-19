from matplotlib import pyplot as plt

class TrainResult:
    def __init__(self, nb_episodes):
        self.nb_episodes = nb_episodes
        self.epsilons = []
        self.total_loss = []
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = []

    def plot(self):
        x = range(self.nb_episode)
        plt.plot(x, self.losses, label="Loss", color="red")
        plt.plot(x, self.epsilons, label="Epsilon", color="green")
        plt.plot(x, self.episode_returns, label="Return", color="blue")
        plt.xlabel("nb episodes")
        plt.legend()
        plt.show()