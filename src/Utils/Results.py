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

class Results:

    def __init__(self, name="Run", config=None, timer=None):
        self.name = name
        self.config = config
        self.timer = timer
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []

    def add_episode(self, length, episode_return):
        self.episode_lengths.append(length)
        self.episode_returns.append(episode_return)
        self.episode_count += 1

    def summary(self):
        data = {
            "name": self.name,
            "episodes": self.episode_count,
            "return_mean": _avg(self.episode_returns),
            "return_std": _std(self.episode_returns),
            "return_min": min(self.episode_returns) if self.episode_returns else float("nan"),
            "return_max": max(self.episode_returns) if self.episode_returns else float("nan"),
            "length_mean": _avg(self.episode_lengths),
        }
        data.update(self.durations())
        return data

    def durations(self):
        if self.timer is None:
            return {}
        return {f"duration_{name}": elapsed for name, elapsed in self.timer.counters.items()}

    def parameters(self):
        return dict(self.config.__dict__) if self.config is not None else {}

    def report_lines(self, data):
        lines = [f"Episodes = {data['episodes']}"]
        lines.append(f"Return = {data['return_mean']:.3f} +- {data['return_std']:.3f}"
                     f" (min {data['return_min']:.3f}, max {data['return_max']:.3f})")
        lines.append(f"Avg episode length = {data['length_mean']:.1f}")
        return lines

    def report_final(self):
        data = self.summary()
        print(f"[+] === {self.name} finished ===")
        for line in self.report_lines(data):
            print(f"[+] {line}")
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

    def series(self):
        # (label, values, color, smooth)
        return [
            ("Return", self.episode_returns, "blue", True),
            ("Episode length", self.episode_lengths, "orange", True),
        ]

    def plot_title(self):
        return f"{self.name} | {self.episode_count} episodes"

    def plot(self, window=None, path=None):
        if self.episode_count == 0:
            print("[-] Nothing to plot : no episode recorded")
            return
        if window is None:
            # window = how many consecutive episodes get averaged into one point of the smoothed curve
            # at least 5 episodes to smooth anything, but never more than a quarter of the run
            window = min(max(5, self.episode_count // 20), max(1, self.episode_count // 4))

        series = self.series()
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
        figure.suptitle(self.plot_title())
        figure.tight_layout()
        if path is not None:
            figure.savefig(path)
            plt.close(figure)
            print(f"[+] Curves saved to file : {path}")
        else:
            plt.show()
