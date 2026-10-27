import os
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

def main():
    # 1. Create a Vectorized Environment (4 parallel highway tracks)
    # This collects training data 4x faster than a single environment.
    env = make_vec_env("highway-fast-v0", n_envs=4)

    # Set up a separate evaluation environment for background testing
    eval_env = gym.make("highway-fast-v0", render_mode="rgb_array")

    # Automatically test performance every 5000 steps and save the best checkpoint
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model_path = "ppo_highway_model.zip"

    # 2. Check if a saved model already exists on disk
    if os.path.exists(model_path):
        print("\n=== Loading Existing Model to Continue Training ===")
        model = PPO.load("ppo_highway_model", env=env)
    else:
        print("\n=== Creating New Model with Upgraded Brain ===")
        # Configure model with a larger 256x256 neural network architecture
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./ppo_highway_tensorboard/",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        )

    # 3. Train (or continue training) with the EvalCallback attached
    print("\n=== Starting Upgraded Training Session ===")
    model.learn(
        total_timesteps=50000, 
        reset_num_timesteps=False,
        callback=eval_callback
    )
    print("=== Training Complete ===\n")

    # Save the final weights to disk
    model.save("ppo_highway_model")
    env.close()
    eval_env.close()

    # 4. Final human-rendered evaluation loop to watch the agent drive
    display_env = gym.make(
        "highway-fast-v0", 
        render_mode="human", 
        config={"real_time_rendering": True}
    )
    
    print("=== Testing Final Agent (3 Episodes) ===")
    for episode in range(3):
        obs, info = display_env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = display_env.step(action)
            episode_reward += reward
            
        print(f"Episode {episode + 1} finished | Total Reward: {episode_reward:.2f}")

    display_env.close()

if __name__ == "__main__":
    main()