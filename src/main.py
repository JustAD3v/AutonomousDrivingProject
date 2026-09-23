import torch

from Trainers.DQNTrainer import DQNTrainer
from Trainers.TrainConfig import TrainConfigBuilder
from Evaluator.Evaluator import Evaluator 
from Agents.LearningAgent import LearningAgent
from Agents.OptimalAgent import OptimalAgent
from Networks.BasicNN import BasicNN

env_config = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1.5,  # => changed from 1 to 1.5
    "collision_reward": -10,  # => changed from -1 to -10 The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes. => chnaged from 0.1 to 0.2
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,  # The reward received at each lane change action.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, 1].
    "normalize_reward": True,
    "offroad_terminal": False,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": None
}

# Change this variable to switch between Training phase and Evaluation phase 
TRAIN = False

train_config = (
    TrainConfigBuilder()
    .epsilon_start(0.01)
    .epsilon_decay(1.0) #0.997
    .buffer_size(15000)
    .warmup_size(400)
    .batch_size(32)
    .gamma(0.8)
    .update_freq(50)
    .nb_episodes(250)
    .build()
)

if TRAIN:
    agent = LearningAgent()
    main_network = BasicNN()
    # too load an already trained model to train it more
    main_network.load_state_dict(torch.load('./models/targetNetwork2.pt'))
    target_network = BasicNN()
    trainer = DQNTrainer(train_config,
                        target_network,
                        main_network,
                        agent,
                        env_config=env_config)

    results = trainer.train()
    results.plot()

else:
    agent = OptimalAgent()
    model = BasicNN()
    model.load_state_dict(torch.load('./models/targetNetwork2_70.pt'))
    evaluator = Evaluator(agent,
                          model,
                          nb_episodes=50,
                          env_config=env_config)
    evaluator.eval()