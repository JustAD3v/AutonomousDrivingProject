from Utils.Results import Results, _avg

class TrainResult(Results):
    def __init__(self, iterations, log_freq=10, name="Training", config=None, timer=None):
        super().__init__(name=name, config=config, timer=timer)
        self.iterations = iterations
        self.log_freq = log_freq
        self.epsilons = []
        self.total_loss = []
        self.episode_iterations = []
        self.losses = []
        self.iteration = 0
        self.last_report = 0

    def add_episode(self, length, episode_return, loss=None, epsilon=None, iteration=None):
        super().add_episode(length, episode_return)
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        if iteration is not None:
            self.iteration = iteration
        self.episode_iterations.append(self.iteration)
        if self.log_freq > 0 and self.episode_count - self.last_report >= self.log_freq:
            self.report_progress()

    def add_episodes(self, episodes, iteration=None):
        for length, episode_return in episodes:
            self.add_episode(length, episode_return, iteration=iteration)

    def summary(self):
        data = super().summary()
        data["iterations"] = self.iteration
        data["planned_iterations"] = self.iterations
        if self.losses:
            data["loss_mean"] = _avg(self.losses)
        if self.epsilons:
            data["final_epsilon"] = self.epsilons[-1]
        return data

    def report_lines(self, data):
        lines = super().report_lines(data)
        lines[0] += f" | Iterations = {data['iterations']}/{data['planned_iterations']}"
        if "loss_mean" in data:
            lines.append(f"Avg loss = {data['loss_mean']:.5f}")
        if "final_epsilon" in data:
            lines.append(f"Final epsilon = {data['final_epsilon']:.3f}")
        return lines

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
        super().report_final()

    def series(self):
        series = super().series()
        if self.losses:
            series.append(("Loss", self.losses, "red", True))
        if self.epsilons:
            series.append(("Epsilon", self.epsilons, "green", False))
        return series

    def plot_title(self):
        return f"{super().plot_title()} | {self.iteration}/{self.iterations} iterations"
