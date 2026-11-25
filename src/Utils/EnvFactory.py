import gymnasium as gym
import highway_env

from Environments.CustomHighwayEnv import CustomHighwayEnv

class EnvFactory:
    @staticmethod
    def create(config, render_mode=None):
        # env_name = "custom-highway-v0" if config.custom_env else config.env_name
        return gym.make(config.env_name, config=EnvFactory.env_config(config), render_mode=render_mode)

    @staticmethod
    def env_config(config):
        env_config = dict(config.env_config) if config.env_config is not None else {}
        if config.ppo_continuous:
            # keep the rest of the environment settings
            # only change the action type and offroad_terminal
            env_config.setdefault("action", {"type": "ContinuousAction"})
            env_config.setdefault("offroad_terminal", True)
        return env_config or None
