import gymnasium as gym

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, success_condition=None):
        super().__init__(env)
        self.success_condition = success_condition if success_condition is not None else EnvWrapper.default_success

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        info["success"] = bool(self.success_condition(obs, reward, term, trunc, info))
        return obs, reward, term, trunc, info

    @staticmethod
    def default_success(obs, reward, term, trunc, info):
        return trunc
