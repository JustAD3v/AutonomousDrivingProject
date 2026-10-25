import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

def main():
    # 1. Initialize the Highway Environment with human rendering
    env = gym.make("highway-fast-v0", render_mode="human")

    # 2. LOAD YOUR TRAINED AGENT DIRECTLY FROM DISK
    # This completely replaces the "model = PPO(...)" and "model.learn(...)" code
    print("=== Loading Trained Model Weights ===")
    model = PPO.load("ppo_highway_model", env=env)

    # 3. Test and Evaluate the Loaded Agent
    print("=== Testing Loaded Agent (5 Episodes) ===")
    for episode in range(10):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Predict actions using the loaded weights
            action, _states = model.predict(obs, deterministic=True)
            
            # Advance the simulator step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
        print(f"Episode {episode + 1} finished | Total Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
