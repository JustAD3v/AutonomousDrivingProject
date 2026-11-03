from matplotlib import pyplot as plt

def _avg(values):
    return sum(values)/len(values) if values else float("nan")

def _std(values):
    if not values:
        return float("nan")
    if len(values) < 2:
        return 0.0
    mean = _avg(values)
    return (sum((value - mean)**2 for value in values) / (len(values) - 1)) ** 0.5

class TrainResult:
    def __init__(self, iterations, log_freq=10, name="Training", config=None):
        self.iterations = iterations
        self.log_freq = log_freq
        self.name = name
        self.config = config
        self.episode_count = 0
        self.epsilons = []
        self.total_loss = []
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_iterations = []
        self.losses = []
        self.iteration = 0
        self.last_report = 0

    def add_episode(self, length, episode_return, loss=None, epsilon=None, iteration=None):
        self.episode_lengths.append(length)
        self.episode_returns.append(episode_return)
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        self.episode_count += 1
        if iteration is not None:
            self.iteration = iteration
        self.episode_iterations.append(self.iteration)
        if self.log_freq > 0 and self.episode_count - self.last_report >= self.log_freq:
            self.report_progress()

    def add_episodes(self, episodes, iteration=None):
        """Record a batch of (length, return) episodes, e.g. collected by workers."""
        for length, episode_return in episodes:
            self.add_episode(length, episode_return, iteration=iteration)

    def summary(self):
        data = {
            "name": self.name,
            "episodes": self.episode_count,
            "iterations": self.iteration,
            "planned_iterations": self.iterations,
            "return_mean": _avg(self.episode_returns),
            "return_std": _std(self.episode_returns),
            "return_min": min(self.episode_returns) if self.episode_returns else float("nan"),
            "return_max": max(self.episode_returns) if self.episode_returns else float("nan"),
            "length_mean": _avg(self.episode_lengths),
        }
        if self.losses:
            data["loss_mean"] = _avg(self.losses)
        if self.epsilons:
            data["final_epsilon"] = self.epsilons[-1]
        return data

    def parameters(self):
        return dict(self.config.__dict__) if self.config is not None else {}

    def report_progress(self):
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
        self.report_progress()
        data = self.summary()
        print(f"[+] === {self.name} finished ===")
        print(f"[+] Episodes = {data['episodes']} | Iterations = {data['iterations']}/{data['planned_iterations']}")
        print(f"[+] Return = {data['return_mean']:.3f} +- {data['return_std']:.3f}"
              f" (min {data['return_min']:.3f}, max {data['return_max']:.3f})")
        print(f"[+] Avg episode length = {data['length_mean']:.1f}")
        if "loss_mean" in data:
            print(f"[+] Avg loss = {data['loss_mean']:.5f}")
        if "final_epsilon" in data:
            print(f"[+] Final epsilon = {data['final_epsilon']:.3f}")
        self.report_parameters()

    def report_parameters(self):
        parameters = self.parameters()
        if not parameters:
            return
        print("[+] Parameters :")
        for name, value in parameters.items():
            print(f"    - {name} = {value}")

    @staticmethod
    def moving_average(values, window):
        averaged = []
        total = 0.0
        for index, value in enumerate(values):
            total += value
            if index >= window:
                total -= values[index - window]
            if index >= window - 1:
                averaged.append(total / window)
        return averaged

    def plot(self, window=None, path=None):
        if self.episode_count == 0:
            print("[-] Nothing to plot : no episode recorded")
            return
        if window is None:
            # at least 5 episodes to smooth anything, but never more than a quarter of the run
            window = min(max(5, self.episode_count // 20), max(1, self.episode_count // 4))

        # (label, values, color, smooth)
        series = [
            ("Return", self.episode_returns, "blue", True),
            ("Episode length", self.episode_lengths, "orange", True),
        ]
        if self.losses:
            series.append(("Loss", self.losses, "red", True))
        if self.epsilons:
            series.append(("Epsilon", self.epsilons, "green", False))

        figure, axes = plt.subplots(len(series), 1, sharex=True)
        if len(series) == 1:
            axes = [axes]
        for axis, (label, values, color, smooth) in zip(axes, series):
            episodes = range(1, len(values) + 1)
            axis.plot(episodes, values, color=color, linewidth=1, alpha=0.25 if smooth else 1.0)
            if smooth and window > 1 and len(values) >= window:
                averaged = self.moving_average(values, window)
                axis.plot(range(window, len(values) + 1), 
                          averaged, 
                          color=color,
                          label=f"moving average ({window} episodes)")
                axis.legend(loc="best", fontsize="small")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
        axes[-1].set_xlabel("episode")
        figure.suptitle(f"{self.name} | {self.episode_count} episodes"
                        f" | {self.iteration}/{self.iterations} iterations")
        figure.tight_layout()
        if path is not None:
            figure.savefig(path)
            plt.close(figure)
            print(f"[+] Training curves saved to file : {path}")
        else:
            plt.show()
