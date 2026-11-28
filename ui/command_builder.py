import sys

MAIN_SCRIPT = "src/main.py"


def _base_command(setup):
    cmd = [sys.executable, MAIN_SCRIPT, "--model_type", setup["model_type"]]
    if setup["config"]:
        cmd.extend(["--config", setup["config"]])
    return cmd


def build_train_command(setup, train):
    """setup: SetupPanel.values(), train: TrainTab.values()."""
    cmd = _base_command(setup)
    cmd.append("--train")

    if train["seed"]:
        cmd.extend(["--seed", train["seed"]])
    if train["parallel"]:
        cmd.append("--parallel")

    if train["sweep"]:
        cmd.append("--sweep")
        if train["model_dir"]:
            cmd.extend(["--model_dir", train["model_dir"]])
    else:
        # Without a path, main.py would overwrite its default model.
        if not train["net_file"]:
            raise ValueError("Choose where to save the model.")
        cmd.extend(["--net_file", train["net_file"]])

    return cmd


def build_eval_command(setup, evaluation):
    cmd = _base_command(setup)

    if not evaluation["net_file"]:
        raise ValueError("Choose a model to load.")
    cmd.extend(["--net_file", evaluation["net_file"]])

    if evaluation["seed"]:
        cmd.extend(["--seed", evaluation["seed"]])
    if evaluation["render"]:
        cmd.append("--render_human")
        cmd.extend(["--render_delay", str(evaluation["render_delay"])])
    if evaluation["adversarial"]:
        cmd.append("--adversarial")
        cmd.extend(["--attack_type", evaluation["attack_type"]])
        cmd.extend(["--attack_bound", str(evaluation["attack_bound"])])

    return cmd
