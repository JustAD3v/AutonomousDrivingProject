import gymnasium as gym
import highway_env


class EnvFactory:
    @staticmethod
    def create(config, render_mode=None):
        return gym.make(config.env_name, config=EnvFactory.env_config(config), render_mode=render_mode)

    @staticmethod
    def env_config(config):
        if config.env_config is not None:
            return config.env_config
        if config.ppo_continuous:
            return {"action": {"type": "ContinuousAction"}, "offroad_terminal": True}
        return None
