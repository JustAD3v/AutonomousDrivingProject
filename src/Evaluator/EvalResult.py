from Utils.Results import Results, _avg, _std

class EvalResult(Results):

    def __init__(self, nb_episodes, name="Evaluation", config=None, timer=None,
                 model=None, attack=None, attack_bound=0.0, seed=None):
        super().__init__(name=name, config=config, timer=timer)
        self.nb_episodes = nb_episodes
        self.model = model
        self.attack = attack
        self.attack_bound = attack_bound
        self.seed = seed
        self.successes = []
        self.episode_speeds = []

    def add_episode(self, length, episode_return, success=False, avg_speed=float("nan"), details=""):
        super().add_episode(length, episode_return)
        self.successes.append(bool(success))
        self.episode_speeds.append(avg_speed)
        self.report_episode(details)

    def success_count(self):
        return sum(self.successes)

    def add_choices(self, choices):
        self.choices_summary = choices

    def summary(self):
        data = super().summary()
        data["model"] = self.model
        data["attack"] = self.attack
        data["attack_bound"] = self.attack_bound
        data["seed"] = self.seed
        data["planned_episodes"] = self.nb_episodes
        data["successes"] = self.success_count()
        data["crashes"] = self.episode_count - self.success_count()
        data["success_rate"] = self.success_count()/self.episode_count if self.episode_count else float("nan")
        data["choices"] = self.choices_summary
        data["speed_mean"] = _avg(self.episode_speeds)
        data["speed_std"] = _std(self.episode_speeds)
        return data

    def series(self):
        return super().series() + [("Avg speed", self.episode_speeds, "green", True)]

    def report_episode(self, details=""):
        marker = "[+]" if self.successes[-1] else "[-]"
        outcome = "Episode ended with success !" if self.successes[-1] else "Driving failed :("
        print(f"{marker} {outcome} - ({self.episode_count}/{self.nb_episodes})"
              f" - return={self.episode_returns[-1]:.3f} - steps={self.episode_lengths[-1]}"
              f" - avg speed={self.episode_speeds[-1]:.2f}"
              f"{' - ' + details if details else ''}")

    def report_lines(self, data):
        lines = super().report_lines(data)
        lines[0] += f"/{data['planned_episodes']}"
        lines.append(f"Success = {data['successes']} | Crash = {data['crashes']}"
                     f" | Success rate = {data['success_rate']*100:.1f}%")
        lines.append(f"Avg speed = {data['speed_mean']:.2f} +- {data['speed_std']:.2f}")
        lines.append(f"Model = {data['model']}")
        if data['attack'] != "NeutralAdversarial":
            lines.append(f"Attack = {data['attack']} | Attack bound = {data['attack_bound']}")
        lines.append(f"Action choices : {data['choices']}")
        return lines
