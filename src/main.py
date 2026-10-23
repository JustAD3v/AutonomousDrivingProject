import argparse

import torch
import gymnasium as gym
from stable_baselines3 import DQN
import random

from Trainers.TrainConfig import TrainConfigBuilder
from Trainers.TrainerFactory import TrainerFactory
from Evaluator.Evaluator import Evaluator
from Policies.CustomPolicy import CustomPolicy
from Networks.StateOnlyNN import StateOnlyNN
from Networks.NetworkFactory import NetworkFactory

from Adversarial.RandomAdversarial import RandomAdversarial
from Adversarial.FGSMAdversarial import FGSMAdversarial

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
    choices=["dqn", "sb3_dqn", "wocar", "ppo"],
    default="dqn",
    metavar="{dqn,sb3_dqn,wocar,ppo}",
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

args = parser.parse_args()

fast_env_config = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.5,
    "collision_reward": -10,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "offroad_terminal": False,
    "simulation_frequency": 5,
    "policy_frequency": 1,
}

TRAIN = args.train
USE_CUSTOM = args.use_custom
TRAINER = args.trainer
NETWORK_FILE = args.net_file if args.net_file else "./models/targetNetwork_trained" 

train_config = (
    TrainConfigBuilder()
    .epsilon_start(1.0)
    .epsilon_min(0.01)
    .epsilon_decay(0.997)
    .buffer_size(15000)
    .warmup_size(400)
    .batch_size(32)
    .gamma(0.95)
    .copy_freq(50)
    .iterations(200)
    .exploration_frac(0.1)
    .learning_rate(5e-4)
    .train_freq(1)
    .gradient_steps(1)
    .policy_kwargs(dict(net_arch=[256, 256]))
    .build()
)

if args.config:
    with open(args.config) as f:
        train_config.from_json(f.read())

# overwrite seed if in args
if args.seed:
    train_config.seed = args.seed

env = gym.make("highway-fast-v0", config=None)

if TRAIN:
    network_factory=NetworkFactory(train_config=train_config, network_type=StateOnlyNN)
    trainer = TrainerFactory.create_trainer(
        env=env,
        config=train_config,
        trainer_type=TRAINER,
        network_factory=network_factory,
        policy_str="MlpPolicy",
        seed=train_config.seed
    )
    results = trainer.learn()
    trainer.save(NETWORK_FILE)
    if results is not None:
        results.plot()

else:
    if USE_CUSTOM:
        model = StateOnlyNN()
        model.load_state_dict(torch.load(NETWORK_FILE))
        policy = CustomPolicy(network=model, action_space=env.action_space)
    else:
        policy = DQN.load(NETWORK_FILE)

    eval_config = {
        **fast_env_config,
        "lanes_count": 3,
        "vehicles_density": 1.3,
        "duration": 30
    }

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
        # adversarial = RandomAdversarial(policy, attack_bound)
        adversarial = FGSMAdversarial(policy, attack_bound)

    evaluator = Evaluator(policy=policy,
                          nb_episodes=100,
                          env_config=None,
                          render_delay=render_delay,
                          render_mode=render_mode,
                          adversarial=adversarial,
                          seed=args.seed if args.seed else None)
    evaluator.eval()
