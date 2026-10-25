import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

def main():
    # 1. Initialize the Highway Environment for Training
    # We use "highway-fast-v0" for much quicker training iterations
    env = gym.make("highway-fast-v0", render_mode="rgb_array")

    # 2. Define and Configure the PPO Model
    # 'MlpPolicy' is used because highway-env's default state is a vector array
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Minor entropy prevents premature exploitation
        verbose=1,
        tensorboard_log="./ppo_highway_tensorboard/"
    )

    # 3. Train the Agent
    print("\n=== Starting Training ===")
    model.learn(total_timesteps=20000)
    print("=== Training Complete ===\n")

    # Save the trained model weights to disk
    model.save("ppo_highway_model")
    env.close()

    # 4. Test and Evaluate the Trained Agent with human graphics
    eval_env = gym.make("highway-fast-v0", render_mode="human")
    
    print("=== Testing Trained Agent (5 Episodes) ===")
    for episode in range(5):
        obs, info = eval_env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Predict the next best action using the trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Advance the environment step
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            # The 'human' render_mode displays the UI window automatically on step()
            
        print(f"Episode {episode + 1} finished | Total Reward: {episode_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()
