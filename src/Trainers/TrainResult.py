from matplotlib import pyplot as plt

def _avg(values):
    return sum(values)/len(values) if values else float("nan")

class TrainResult:
    def __init__(self, iterations, log_freq=10, name="Training"):
        self.iterations = iterations
        self.log_freq = log_freq
        self.name = name
        self.episode_count = 0
        self.epsilons = []
        self.total_loss = []
        self.episode_returns = []
        self.episode_lengths = []
        self.losses = []
        self.iteration = 0
        self.last_report = 0

    def add_episode(self, length, episode_return, loss=None, epsilon=None, iteration=None):
        """Record one finished episode and report progress every log_freq episodes."""
        self.episode_lengths.append(length)
        self.episode_returns.append(episode_return)
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        self.episode_count += 1
        if iteration is not None:
            self.iteration = iteration
        if self.log_freq > 0 and self.episode_count - self.last_report >= self.log_freq:
            self.report_progress()

    def add_episodes(self, episodes, iteration=None):
        """Record a batch of (length, return) episodes, e.g. collected by workers."""
        for length, episode_return in episodes:
            self.add_episode(length, episode_return, iteration=iteration)

    def report_progress(self):
        """Print the averages over the episodes finished since the last report."""
        window = self.episode_count - self.last_report
        if window <= 0:
            return
        self.last_report = self.episode_count
        message = (f"[*] ({self.iteration}/{self.iterations}) | episodes {self.episode_count-window+1}-{self.episode_count}"
                   f" | avg_return={_avg(self.episode_returns[-window:]):.3f}"
                   f" | avg_len={_avg(self.episode_lengths[-window:]):.1f}")
        if self.losses:
            message += f" | avg_loss={_avg(self.losses[-window:]):.5f}"
        if self.epsilons:
            message += f" | epsilon={self.epsilons[-1]:.3f}"
        print(message)

    def report_iteration(self, iteration):
        self.iteration = iteration
        print(f"[*] Iteration Finished ({iteration}/{self.iterations})")

    def report_final(self):
        """Print the end of training summary (flushes the pending episodes first)."""
        self.report_progress()
        print(f"[+] === {self.name} finished ===")
        print(f"[+] Episodes = {self.episode_count} | Iterations = {self.iteration}/{self.iterations}")
        print(f"[+] Avg return = {_avg(self.episode_returns):.3f} | Avg episode length = {_avg(self.episode_lengths):.1f}")
        if self.losses:
            print(f"[+] Avg loss = {_avg(self.losses):.5f}")
        if self.epsilons:
            print(f"[+] Final epsilon = {self.epsilons[-1]:.3f}")

    def plot(self):
        x = range(self.episode_count)
        plt.plot(x, self.losses, label="Loss", color="red")
        plt.plot(x, self.epsilons, label="Epsilon", color="green")
        plt.plot(x, self.episode_returns, label="Return", color="blue")
        plt.xlabel("nb episodes")
        plt.legend()
        plt.show()
