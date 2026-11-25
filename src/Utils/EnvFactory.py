import gymnasium as gym
import highway_env

from Environments.CustomHighwayEnv import CustomHighwayEnv
from Environments.EnvWrapper import EnvWrapper
from types import MethodType

class EnvFactory:
    @staticmethod
    def create(config, render_mode=None):
        env = gym.make(config.env_name, config=EnvFactory.env_config(config), render_mode=render_mode)
        # env_name = "custom-highway-v0" if config.custom_env else config.env_name
        return EnvFactory.wrap(env)

    @staticmethod
    def wrap(env):
        match env.spec.id:
            case "merge-v1":
                success_condition = lambda *_: (
                    env.unwrapped.vehicle.position[0] > 370
                    and not env.unwrapped.vehicle.crashed
                )
            case "exit-v0":
                success_condition = lambda obs, reward, term, trunc, info: (
                    info.get("is_success", False)
                    and not env.unwrapped.vehicle.crashed
                )
            case "intersection-v0":
                success_condition = lambda *_: (
                    env.unwrapped.has_arrived(env.unwrapped.vehicle)
                    and not env.unwrapped.vehicle.crashed
                )
            case _:
                success_condition = EnvWrapper.default_success
        return EnvWrapper(env, success_condition)

    @staticmethod
    def env_config(config):
        env_config = dict(config.env_config) if config.env_config is not None else {}
        if config.ppo_continuous:
            # keep the rest of the environment settings
            # only change the action type and offroad_terminal
            env_config.setdefault("action", {"type": "ContinuousAction"})
            env_config.setdefault("offroad_terminal", True)
        return env_config or None
