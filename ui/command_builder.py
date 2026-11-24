import sys

MAIN_SCRIPT = "src/main.py"


def build_command(mode, evaluation):
    """mode: ModeTab.values(), evaluation: EvalTab.values()."""
    cmd = [sys.executable, MAIN_SCRIPT]

    if mode["train"]:
        cmd.append("--train")

    cmd.extend(["--model_type", mode["model_type"]])

    if mode["net_file"]:
        cmd.extend(["--net_file", mode["net_file"]])
    if mode["config"]:
        cmd.extend(["--config", mode["config"]])
    if mode["seed"]:
        cmd.extend(["--seed", mode["seed"]])

    # Rendering and attacks only apply to evaluation, not training.
    if not mode["train"]:
        if evaluation["render"]:
            cmd.append("--render_human")
            cmd.extend(["--render_delay", str(evaluation["render_delay"])])
        if evaluation["adversarial"]:
            cmd.append("--adversarial")
            cmd.extend(["--attack_bound", str(evaluation["attack_bound"])])

    return cmd
