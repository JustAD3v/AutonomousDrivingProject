import numpy as np
import gymnasium as gym
from highway_env.envs.highway_env import HighwayEnv
from highway_env import utils

class CustomHighwayEnv(HighwayEnv):

    @classmethod
    def default_config(cls):
        config = super().default_config()

        # map speed range to [-1,1] instead of [0,1] so low speed gets negative reward
        # to get positive mapping as before in [25, 40] 
        # and negative below that, we need 25 to be the middle value
        # So the range should be [10, 40]
        config.update({
            "reward_speed_range": [10, 40],
            "reward_speed_min": -1.0,
            "reward_speed_max": 1.0,
        })

        return config

    def _rewards(self, action):
        # Keep all original reward components
        rewards = super()._rewards(action)

        # same as original env
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)

        # Configurable linear mapping
        scaled_speed = utils.lmap(
            forward_speed,
            self.config["reward_speed_range"],
            [
                self.config["reward_speed_min"],
                self.config["reward_speed_max"],
            ],
        )

        # Bound reward to configured limits
        rewards["high_speed_reward"] = float(
            np.clip(
                scaled_speed,
                self.config["reward_speed_min"],
                self.config["reward_speed_max"],
            )
        )

        return rewards

# Register our environment
if "custom-highway-v0" not in gym.registry:
    gym.register(id="custom-highway-v0", entry_point=CustomHighwayEnv)
