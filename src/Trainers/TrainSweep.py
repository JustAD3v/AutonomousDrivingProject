import os

from Trainers.TrainerFactory import TrainerFactory


class TrainSweep:
    def __init__(self, sweep, trainer_type="dqn", model_dir=None, name="Sweep"):
        self.sweep = sweep
        self.trainer_type = trainer_type
        self.model_dir = model_dir
        self.name = name
        self.runs = []

    def run(self):
        self.runs = []
        total = len(self.sweep)
        for index, (overrides, config) in enumerate(self.sweep.configs(), start=1):
            label = self.label(overrides)
            print(f"[+] === {self.name} run {index}/{total} : {label} ===")
            trainer = TrainerFactory.create_from_config(config=config, trainer_type=self.trainer_type)
            result = trainer.learn()
            if result is not None:
                result.name = f"{self.name} [{label}]"
            if self.model_dir is not None:
                trainer.save(self.model_path(index, overrides))
            self.runs.append((overrides, config, result))
        # return the [(overrides, config, result)] runs
        return self.runs

    def label(self, overrides):
        return ", ".join(f"{name}={value}" for name, value in overrides.items())

    def model_path(self, index, overrides):
        parts = [self.name, self.trainer_type] + [f"{name}-{value}" for name, value in overrides.items()]
        file_name = "_".join(str(part) for part in parts)
        file_name = "".join(char if char.isalnum() or char in "-_." else "-" for char in file_name)
        return os.path.join(self.model_dir, f"{file_name}_{index}.pt")
