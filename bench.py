import time
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

env = gym.make("highway-fast-v0")
obs, _ = env.reset(seed=1)
n = 500

t = time.time()
for _ in range(n):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()
print(f"environment only: {n / (time.time() - t):.0f} steps/s")

model = PPO("MlpPolicy", env, device="cpu", verbose=0)
t = time.time()
for _ in range(n):
    model.predict(obs, deterministic=True)
print(f"policy only: {n / (time.time() - t):.0f} predictions/s")
