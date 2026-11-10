import argparse

import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO

from Trainers.TrainConfig import TrainConfigBuilder
from Trainers.TrainConfig import TrainConfigSweep
from Trainers.TrainerFactory import TrainerFactory
from Trainers.TrainSweep import TrainSweep
from Evaluator.Evaluator import Evaluator
from Policies.CustomPolicy import CustomPolicy
from Policies.PPOPolicy import PPOPolicy
from Networks.StateOnlyNN import StateOnlyNN
from Networks.ActorNetwork import ActorNetwork
from Utils.EnvFactory import EnvFactory

from Adversarial.RandomAdversarial import RandomAdversarial
from Adversarial.FGSMAdversarial import FGSMAdversarial

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate a reinforcement-learning agent in the "
            "highway environment. Evaluation is the default; use --train "
            "to train and save a model."
        ),
        epilog=(
            "Examples:\n"
            "  python src/main.py --train --model_type ppo --net_file models\\ppo-agent.pt\n"
            "  python src/main.py --model_type dqn --net_file models\\targetNetwork_trained --render_human\n"
            "  python src/main.py --model_type dqn --net_file models\\targetNetwork_trained --adversarial --attack_bound 0.1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Train the selected model instead of evaluating it",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["dqn", "sb3_dqn", "wocar_dqn", "ppo", "wocar_ppo", "sb3-ppo"],
        default="dqn",
        metavar="{dqn,sb3_dqn,wocar_dqn,ppo,wocar_ppo,sb3-ppo}",
    )
    parser.add_argument(
        "--net_file",
        type=str,
        metavar="PATH",
        help=(
            "Path of model file to load for evaluation or save after training "
            "(default: ./models/targetNetwork_trained)."
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        metavar="PATH",
        help=(
            "Existing directory where the sweep saves one model file per run "
            "(default: ./models). Only used with --sweep."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="JSON file with training configuration values that override the defaults.",
    )
    parser.add_argument(
        "--render_delay",
        type=float,
        metavar="SECONDS",
        help="Pause between evaluation steps in seconds (used with evaluation).",
    )
    parser.add_argument(
        "--render_human",
        action="store_true",
        default=False,
        help="Show a human-rendered environment window during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set the random seed for training or evaluation.",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="During evaluation, perturb observations with adversarial method.",
    )
    parser.add_argument(
        "--attack_bound",
        type=float,
        metavar="FLOAT",
        help="Maximum perturbation bound during adversarial evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["fgsm", "random"],
        default="fgsm",
        metavar="{fgsm, random}",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=False,
        help="Use the continuous action space (PPO only), for training and evaluation.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Launch sweep runs (experimental feature, test only)",
    )

    args = parser.parse_args()

    fast_env_config = {
        "lanes_count": 4,
        "vehicles_count": 40,
        "duration": 40,
        "ego_spacing": 2,
        "vehicles_density": 1.1,
        "collision_reward": -5,
        "right_lane_reward": 0.0,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.02,
        "reward_speed_range": [25, 40],
        "normalize_reward": True,
        "offroad_terminal": False,
        "simulation_frequency": 5,
        "policy_frequency": 1,
    }

    TRAIN = args.train
    SWEEP = args.sweep
    MODEL_TYPE = args.model_type
    NETWORK_FILE = args.net_file if args.net_file else "./models/targetNetwork_trained"
    MODEL_DIR = args.model_dir if args.model_dir else "./models"
    ATTACK_TYPE = args.attack_type

    # PPO / Default Config setup
    train_config = (
        TrainConfigBuilder()
        .buffer_size(2048)
        .ppo_update_epochs(5)
        .ppo_minibatch_size(64)
        .ppo_critic_update_steps(10)
        .ppo_actor_update_steps(10)
        .ent_coef(0.02)
        .max_grad_norm(0.5)
        .gamma(0.92)
        .ppo_lm(0.8)
        .ppo_nworkers(8)
        .learning_rate(5e-4)
        .iterations(15)
        .log_freq(20)
        .build()
    )

    if args.config:
        with open(args.config) as f:
            train_config.from_json(f.read())

    # overwrite seed if in args
    if args.seed:
        train_config.seed = args.seed

    # overwrite continuous mode if in args
    if args.continuous:
        train_config.ppo_continuous = True

    env_config = EnvFactory.env_config(train_config)
    env = EnvFactory.create(train_config)

    if TRAIN:
        if SWEEP:
            sweep = TrainConfigSweep(train_config).vary("attack_budget", [0.0, 0.05, 0.1]).seeds(2)
            runs = TrainSweep(sweep, trainer_type=MODEL_TYPE, model_dir=MODEL_DIR, name="eps-sweep").run()
            print(f"Runs: {runs}")
        else:
            trainer = TrainerFactory.create_from_config(
                config=train_config,
                trainer_type=MODEL_TYPE,
                env=env,
                policy_str="MlpPolicy"
            )
            results = trainer.learn()
            trainer.save(NETWORK_FILE)
            if results is not None:
                results.plot()

    else:
        match MODEL_TYPE:
            case "dqn" | "wocar_dqn":
                model = StateOnlyNN()
                model.load_state_dict(torch.load(NETWORK_FILE))
                policy = CustomPolicy(network=model, action_space=env.action_space)
            case "ppo" | "wocar_ppo":
                if train_config.ppo_continuous:
                    model = ActorNetwork(out_features=env.action_space.shape[0], continuous=True)
                else:
                    model = ActorNetwork()
                model.load_state_dict(torch.load(NETWORK_FILE))
                policy = PPOPolicy(network=model, action_space=env.action_space)
            case "sb3_dqn":
                policy = DQN.load(NETWORK_FILE)
            case "sb3-ppo":
                policy = PPO.load(NETWORK_FILE)
            case _:
                policy = DQN.load(NETWORK_FILE)

        render_mode = None
        render_delay = 0.0
        if args.render_human:
            render_mode = "human"
        if args.render_delay:
            render_delay = args.render_delay

        adversarial = None
        attack_bound = 0.2
        if args.attack_bound:
            attack_bound = args.attack_bound
        if args.adversarial:
            match ATTACK_TYPE:
                case "random":
                    adversarial = RandomAdversarial(policy, attack_bound)
                case "fgsm":
                    adversarial = FGSMAdversarial(policy, attack_bound)

        evaluator = Evaluator(policy=policy,
                            nb_episodes=100,
                            env_config=env_config,
                            continuous=args.continuous,
                            render_delay=render_delay,
                            render_mode=render_mode,
                            adversarial=adversarial,
                            model=MODEL_TYPE,
                            seed=args.seed if args.seed else None)
        results = evaluator.eval()

if __name__ == "__main__":
    main()