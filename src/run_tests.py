"""
Examples:
    python src/run_tests.py --list
    python src/run_tests.py
    python src/run_tests.py --only ppo ppo_continuous
    python src/run_tests.py --exclude wocar_dqn --skip-train
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from stable_baselines3 import DQN, PPO

from Trainers.TrainConfig import TrainConfig
from Evaluator.Evaluator import Evaluator
from Policies.CustomPolicy import CustomPolicy
from Policies.PPOPolicy import PPOPolicy
from Networks.StateOnlyNN import StateOnlyNN
from Networks.ActorNetwork import ActorNetwork
from Utils.EnvFactory import EnvFactory
from Adversarial.RandomAdversarial import RandomAdversarial
from Adversarial.FGSMAdversarial import FGSMAdversarial

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "tests" / "configs"
MODEL_DIR = ROOT / "tests" / "models"
RESULT_DIR = ROOT / "tests" / "results"

EVAL_EPISODES = 100
EVAL_SEED = 123123
# these model types are trained inside the docker compose container
CONTAINER_MODEL_TYPES = ("ppo", "wocar_ppo")
# a scenario fails when its success rate moves further than this from the baseline
TOLERANCE = 0.10

# Evaluation variants, reused across scenarios so the attacks stay identical.
CLEAN = {"name": "clean", "attack": None, "attack_bound": 0.0}
FGSM_50 = {"name": "fgsm_50", "attack": "fgsm", "attack_bound": 0.5}
FGSM_10 = {"name": "fgsm_10", "attack": "fgsm", "attack_bound": 0.1}
RANDOM_80 = {"name": "random_80", "attack": "random", "attack_bound": 0.8}

SCENARIOS = {
    "dqn": {
        "model_type": "dqn",
        "config": "dqn.json",
        "evals": [CLEAN, FGSM_10, RANDOM_80],
        "references": {"clean": 0.99, "fgsm_10": 0.47, "random_80": 0.19},
    },
    "sb3_dqn": {
        "model_type": "sb3_dqn",
        "config": "sb3_dqn.json",
        "evals": [CLEAN, RANDOM_80],
        "references": {"clean": 0.94, "random_80": None},
    },
    "wocar_dqn": {
        "model_type": "wocar_dqn",
        "config": "wocar_dqn.json",
        "evals": [CLEAN, FGSM_10, RANDOM_80],
        "references": {"clean": 0.96, "random_80": 0.70, "fgsm_10": 0.86},
    },
    "ppo": {
        "model_type": "ppo",
        "config": "ppo.json",
        "evals": [CLEAN, FGSM_50, FGSM_10, RANDOM_80],
        "references": {"clean": 0.95, "fgsm_50": 0.04, "fgsm_10": None, "random_80": 0.74},
    },
    "ppo_continuous": {
        "model_type": "ppo",
        "config": "ppo_continuous.json",
        "evals": [CLEAN, FGSM_50],
        "references": {"clean": 0.77, "fgsm_50": 0.19},
    },
    "wocar_ppo": {
        "model_type": "wocar_ppo",
        "config": "wocar_ppo.json",
        "evals": [CLEAN, FGSM_10, RANDOM_80],
        "references": {"clean": 0.98, "fgsm_10": 85, "random_80": 90},
    },
}


def load_config(scenario):
    return TrainConfig().from_json((CONFIG_DIR / scenario["config"]).read_text())


def init_worker():
    sys.stdout.reconfigure(line_buffering=True)


def load_policy(scenario, config, model_path, env):
    if scenario["model_type"] in ("ppo", "wocar_ppo"):
        if config.ppo_continuous:
            network = ActorNetwork(out_features=env.action_space.shape[0], continuous=True)
        else:
            network = ActorNetwork()
        network.load_state_dict(torch.load(model_path))
        return PPOPolicy(network=network, action_space=env.action_space)
    elif scenario["model_type"] == "sb3_dqn":
        return DQN.load(model_path.with_suffix(".zip"))
    elif scenario["model_type"] == "sb3-ppo":
        return PPO.load(model_path.with_suffix(".zip")) 
    network = StateOnlyNN()
    network.load_state_dict(torch.load(model_path))
    return CustomPolicy(network=network, action_space=env.action_space)


def build_adversarial(variant, policy, env):
    match variant["attack"]:
        case "fgsm":
            return FGSMAdversarial(policy, variant["attack_bound"], env)
        case "random":
            return RandomAdversarial(policy, variant["attack_bound"], env)
        case _:
            return None


def train_in_container(name, scenario, config, model_path):
    command = [
        "docker", "compose", "run", "--rm", "train",
        "python", "src/main.py", "--container", "--train",
        "--model_type", scenario["model_type"],
        "--config", f"tests/configs/{scenario['config']}",
        "--net_file", f"{model_path.relative_to(ROOT).as_posix()}",
    ]
    if config.parallel:
        command.append("--parallel")
    print(f"[+] {name}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def train_scenario(scenario, config, model_path):
    # imported here so --list and --skip-train runs do not need the training
    from Trainers.TrainerFactory import TrainerFactory

    trainer = TrainerFactory.create_from_config(
        config=config,
        trainer_type=scenario["model_type"],
        env=EnvFactory.create(config),
        policy_str="MlpPolicy",
    )
    results = trainer.learn()
    trainer.save(str(model_path))
    if results is not None:
        results.plot(path=f"{model_path}_curves.png")


def evaluate_variant(name, variant_name, episodes):
    scenario = SCENARIOS[name]
    variant = next(item for item in scenario["evals"] if item["name"] == variant_name)
    config = load_config(scenario)
    env = EnvFactory.create(config)
    policy = load_policy(scenario, config, MODEL_DIR / name, env)
    evaluator = Evaluator(
        policy=policy,
        nb_episodes=episodes,
        env=env,
        continuous=config.ppo_continuous,
        adversarial=build_adversarial(variant, policy, env),
        model=scenario["model_type"],
        seed=EVAL_SEED,
    )
    return name, variant_name, evaluator.eval().summary()


def train_scenario_result(name, skip_train):
    scenario = SCENARIOS[name]
    config = load_config(scenario)
    model_path = MODEL_DIR / name
    if "sb3" in name:
        model_path = model_path.with_suffix(".zip")
    started = time.perf_counter()

    if skip_train:
        if not model_path.exists():
            return {"scenario": name, "error": f"no checkpoint at {model_path}"}
    elif scenario["model_type"] in CONTAINER_MODEL_TYPES:
        train_in_container(name, scenario, config, model_path)
    else:
        train_scenario(scenario, config, model_path)

    return {
        "scenario": name,
        "model_type": scenario["model_type"],
        "config": scenario["config"],
        "seed": config.seed,
        "trained": not skip_train,
        "duration": time.perf_counter() - started,
        "evals": {},
    }


def evaluate_all(results, episodes, workers):
    tasks = [(scenario_result["scenario"], variant["name"])
             for scenario_result in results if "error" not in scenario_result
             for variant in SCENARIOS[scenario_result["scenario"]]["evals"]]
    if not tasks:
        return
    by_name = {scenario_result["scenario"]: scenario_result for scenario_result in results}
    with ProcessPoolExecutor(max_workers=workers or len(tasks),
                             initializer=init_worker) as executor:
        futures = [executor.submit(evaluate_variant, name, variant_name, episodes)
                   for name, variant_name in tasks]
        for future in futures:
            name, variant_name, summary = future.result()
            by_name[name]["evals"][variant_name] = summary

def result(name, variant_name, success_rate):
    baseline = SCENARIOS[name]["references"].get(variant_name)
    if baseline is None:
        return "NEW", baseline
    return ("OK" if abs(success_rate - baseline) <= TOLERANCE else "DRIFT"), baseline


def report(results):
    print("\n[+] === Scenario results ===")
    print(f"{'scenario':<16}{'variant':<12}{'success':>9}{'baseline':>10}{'return':>10}{'result':>10}   choices")
    failures = 0
    for scenario_result in results:
        if "error" in scenario_result:
            print(f"{scenario_result['scenario']:<16}{'-':<12}{'-':>9}{'-':>10}{'-':>10}"
                  f"   ERROR: {scenario_result['error']}")
            failures += 1
            continue
        for variant_name, summary in scenario_result["evals"].items():
            rate = summary["success_rate"]
            status, baseline = result(scenario_result["scenario"], variant_name, rate)
            shown = f"{baseline*100:.0f}%" if baseline is not None else "-"
            print(f"{scenario_result['scenario']:<16}{variant_name:<12}{rate*100:>8.1f}%{shown:>10}"
                  f"{summary['return_mean']:>10.2f}{status:>10}   {summary['choices']}")
            failures += status == "DRIFT"
    print(f"[+] {len(results)} scenarios, {failures} drifted or failed")
    return failures


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_episodes": EVAL_EPISODES,
        "tolerance": TOLERANCE,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=4))
    print(f"[+] Results written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the training/evaluation scenarios and compare them to their references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--only", nargs="+", metavar="NAME", choices=list(SCENARIOS),
                        help="Run only these scenarios (default: all)")
    parser.add_argument("--exclude", nargs="+", metavar="NAME", choices=list(SCENARIOS), default=[],
                        help="Skip these scenarios")
    parser.add_argument("--list", action="store_true", help="List the scenarios and exit")
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse the checkpoints in tests/models instead of training")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel evaluation workers (default: one per scenario variant)")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes per variant (default: {EVAL_EPISODES})")
    parser.add_argument("--out", type=str, metavar="PATH", default=None,
                        help="Where to write the result file (default: tests/results/<timestamp>.json)")
    args = parser.parse_args()

    if args.list:
        for name, scenario in SCENARIOS.items():
            variants = ", ".join(variant["name"] for variant in scenario["evals"])
            print(f"{name:<16}{scenario['model_type']:<12}{scenario['config']:<22}{variants}")
        return

    selected = [name for name in (args.only or SCENARIOS) if name not in args.exclude]
    if not selected:
        print("[-] No scenario selected")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Running {len(selected)} scenarios: {', '.join(selected)}")

    started = time.perf_counter()
    results = [train_scenario_result(name, args.skip_train) for name in selected]
    print(f"[+] Training done, evaluating {args.episodes} episodes per variant")
    evaluate_all(results, args.episodes, args.workers)
    print(f"[+] Total duration = {time.perf_counter() - started:.1f}s")

    report(results)
    out = Path(args.out) if args.out else RESULT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    save_results(results, out)

if __name__ == "__main__":
    main()
