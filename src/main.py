import torch
import gymnasium as gym
from stable_baselines3 import DQN

from Trainers.TrainConfig import TrainConfigBuilder
from Trainers.TrainerFactory import TrainerFactory
from Evaluator.Evaluator import Evaluator 
from Policies.CustomPolicy import CustomPolicy
from Networks.StateOnlyNN import StateOnlyNN 

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

TRAIN = False
USE_CUSTOM = False

train_config = (
    TrainConfigBuilder()
    .epsilon_start(1.0)
    .epsilon_decay(0.997)
    .buffer_size(15000)
    .warmup_size(400)
    .batch_size(32)
    .gamma(0.95)
    .copy_freq(50)
    .iterations(500)
    .learning_rate(5e-4)
    .train_freq(1)
    .gradient_steps(1)
    .policy_kwargs(dict(net_arch=[256, 256]))
    .build()
)

env = gym.make("highway-fast-v0", config=fast_env_config)

if TRAIN:
    main_network = StateOnlyNN()
    target_network = StateOnlyNN()

    trainer = TrainerFactory.create_trainer(
        env=env,
        config=train_config,
        use_custom=USE_CUSTOM,
        main_network=main_network,
        target_network=target_network,
        policy_str="MlpPolicy"
    )

    results = trainer.learn()
    trainer.save("./models/targetNetwork_trained")

else:
    # model = StateOnlyNN()
    # model.load_state_dict(torch.load('./models/targetNetwork_test.pt'))
    # policy = CustomPolicy(main_network=model, action_space=env.action_space)
    
    # Or for using DQN lib
    policy = DQN.load("./models/final_model")

    eval_config = {
        **fast_env_config,
        "lanes_count": 3,
        "vehicles_density": 1.3,
        "duration": 30
    }
    evaluator = Evaluator(policy=policy,
                          nb_episodes=100,
                          env_config=None,
                          render_delay=0.0)
    evaluator.eval()