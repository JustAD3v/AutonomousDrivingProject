from Trainers.DQNWrapper import DQNWrapper
import random

class TrainerFactory:
    @staticmethod
    def create_trainer(
        env,
        config,
        network_factory,
        trainer_type="dqn",
        policy_str="MlpPolicy",
        seed=random.randint(0, 2**32 - 1)
    ) -> DQNWrapper:
        match trainer_type:
            case "dqn":
                from Trainers.DQNTrainer import DQNTrainer
                from Policies.CustomPolicy import CustomPolicy

                policy = CustomPolicy(network=network_factory.create(), action_space=env.action_space)

                trainer = DQNTrainer(
                    config=config,
                    target_network=network_factory.create(),
                    main_network=network_factory.create(),
                    policy=policy,
                    env=env
                )
                return DQNWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)
            
            case "wocar":
                from Trainers.WocaRDQN import WocaRDQN
                from Policies.CustomPolicy import CustomPolicy

                policy = CustomPolicy(network=network_factory.create(), action_space=env.action_space)

                trainer = WocaRDQN(
                    config=config,
                    target_network=network_factory.create(),
                    main_network=network_factory.create(),
                    woca_network=network_factory.create(),
                    robust_network=network_factory.create(),
                    policy=policy,
                    env=env
                )
                return DQNWrapper(trainer=trainer, trainer_type=trainer_type, config=config, seed=seed)
            
            case "ppo":
                # PPO is on-policy and uses SB3's own rollout collection, so the model
                # is built directly (no custom network/policy objects), like sb3_dqn.
                from stable_baselines3 import PPO

                ppo_kwargs = config.to_ppo_kwargs()
                ppo_kwargs["seed"] = seed
                ppo_model = PPO(policy_str, env, **ppo_kwargs)
                return DQNWrapper(trainer=ppo_model, trainer_type=trainer_type, config=config, seed=seed)

            case _:
                from stable_baselines3 import DQN

                sb3_kwargs = config.to_sb3_kwargs() if config else {}
                sb3_model = DQN(policy_str, env, **sb3_kwargs)
                return DQNWrapper(trainer=sb3_model, trainer_type=trainer_type, config=config, seed=seed)