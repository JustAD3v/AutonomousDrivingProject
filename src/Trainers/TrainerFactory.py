from Trainers.TrainerWrapper import TrainerWrapper
import random
import sys
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class TrainerFactory:
    @staticmethod
    def create_network_factory(env, config, trainer_type="dqn"):
        from Networks.NetworkFactory import NetworkFactory
        from Networks.StateOnlyNN import StateOnlyNN

        if trainer_type in ("ppo", "wocar_ppo"):
            from Networks.ActorNetwork import ActorNetwork
            from Networks.CriticNetwork import CriticNetwork
            from Networks.WocaPPONN import WocaPPONN

            network_types = {
                "actor": ActorNetwork,
                "critic": CriticNetwork,
                # the worst-case network is a Q network, as in the DQN version
                "woca": WocaPPONN if config.ppo_continuous else StateOnlyNN
            }
            configs = {
                "actor": {"out_features": env.action_space.shape[0], "continuous": True}
            } if config.ppo_continuous else None
            return NetworkFactory(train_config=config, network_types=network_types, configs=configs)

        return NetworkFactory(train_config=config, network_types=StateOnlyNN)

    @staticmethod
    def create_from_config(
        config,
        trainer_type="dqn",
        env=None,
        policy_str="MlpPolicy"
    ) -> TrainerWrapper:
        """Build the environment, the networks and the trainer from a single TrainConfig."""
        from Utils.EnvFactory import EnvFactory

        if env is None:
            env = EnvFactory.create(config)
        return TrainerFactory.create_trainer(
            env=env,
            config=config,
            network_factory=TrainerFactory.create_network_factory(env, config, trainer_type),
            trainer_type=trainer_type,
            policy_str=policy_str,
            seed=config.seed
        )

    @staticmethod
    def create_trainer(
        env,
        config,
        network_factory,
        trainer_type="dqn",
        policy_str="MlpPolicy",
        seed=random.randint(0, 2**32 - 1)
    ) -> TrainerWrapper:
        match trainer_type:
            case "dqn":
                from Trainers.DQNTrainer import DQNTrainer
                from Policies.CustomPolicy import CustomPolicy

                main_network = network_factory.create()
                target_network = network_factory.create()

                policy = CustomPolicy(network=main_network, action_space=env.action_space)

                trainer = DQNTrainer(
                    config=config,
                    target_network=target_network,
                    main_network=main_network,
                    policy=policy,
                    env=env
                )
                return TrainerWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)
            
            case "wocar_dqn":
                from Trainers.WocaRDQN import WocaRDQN
                from Policies.CustomPolicy import CustomPolicy

                robust_network = network_factory.create()
                target_network = network_factory.create()
                main_network = network_factory.create()
                woca_network = network_factory.create()

                policy = CustomPolicy(network=robust_network, action_space=env.action_space)

                trainer = WocaRDQN(
                    config=config,
                    target_network=target_network,
                    main_network=main_network,
                    woca_network=woca_network,
                    robust_network=robust_network,
                    policy=policy,
                    env=env
                )
                return TrainerWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)
            
            case "sb3-ppo":
                from stable_baselines3 import PPO
                ppo_kwargs = config.to_ppo_kwargs()
                ppo_kwargs["seed"] = seed
                train_env = env
                if config.n_envs > 1:
                    # Windows doesn't support 'fork' multiprocessing. Use DummyVecEnv on Windows, SubprocVecEnv on Linux/Mac.
                    vec_env_cls = DummyVecEnv if sys.platform == "win32" else SubprocVecEnv
                    vec_env_kwargs = {} if sys.platform == "win32" else dict(start_method="fork")
                
                    train_env = make_vec_env(
                        env.spec.id, 
                        n_envs=config.n_envs, 
                        seed=seed,
                        vec_env_cls=vec_env_cls,
                        vec_env_kwargs=vec_env_kwargs
                    )
                    # n_steps counts per environment: split it so one rollout stays 2048 steps
                    ppo_kwargs["n_steps"] = max(1, config.n_steps // config.n_envs)
                
                ppo_model = PPO(policy_str, train_env, **ppo_kwargs)

                # Setup EvalCallback for best model saving
                eval_env = gym.make(env.spec.id, config=env.unwrapped.config)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=config.best_model_save_path,
                    log_path=config.log_path,
                    eval_freq=max(1, config.eval_freq // config.n_envs),
                    deterministic=True,
                    render=False,
                    verbose=1
                )

                return TrainerWrapper(trainer=ppo_model, trainer_type=trainer_type, config=config, seed=seed, callback=eval_callback)

            case "ppo":
                from Trainers.PPOTrainer import PPOTrainer
                from Policies.PPOPolicy import PPOPolicy

                actor_network = network_factory.create("actor")
                critic_network = network_factory.create("critic")

                policy = PPOPolicy(network=actor_network, action_space=env.action_space)

                trainer = PPOTrainer(
                    config=config,
                    actor_network=actor_network,
                    critic_network=critic_network,
                    policy=policy,
                    env=env
                )
                return TrainerWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)
            
            case "wocar_ppo":
                from Trainers.WocaRPPO import WocaRPPO
                from Policies.PPOPolicy import PPOPolicy

                actor_network = network_factory.create("actor")
                critic_network = network_factory.create("critic")
                woca_network = network_factory.create("woca")

                policy = PPOPolicy(network=actor_network, action_space=env.action_space)

                trainer = WocaRPPO(
                    config=config,
                    actor_network=actor_network,
                    critic_network=critic_network,
                    woca_network=woca_network,
                    policy=policy,
                    env=env
                )
                return TrainerWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)

            case _:
                from stable_baselines3 import DQN
 
                sb3_kwargs = config.to_sb3_kwargs() if config else {}
                sb3_model = DQN(policy_str, env, **sb3_kwargs)

                # Setup EvalCallback for best model saving
                eval_env = gym.make(env.spec.id, config=env.unwrapped.config)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=config.best_model_save_path,
                    log_path=config.log_path,
                    eval_freq=config.eval_freq,
                    deterministic=True,
                    render=False,
                    verbose=1
                )

                return TrainerWrapper(trainer=sb3_model, trainer_type=trainer_type, config=config, seed=seed, callback=eval_callback)