from Trainers.DQNWrapper import DQNWrapper
import random

class TrainerFactory:
    @staticmethod
    def create_trainer(
        env,
        config,
        trainer="dqn",
        main_network=None,
        target_network=None,
        woca_network=None,
        robust_network=None,
        policy_str="MlpPolicy",
        seed=random.randint(0, 2**32 - 1)
    ) -> DQNWrapper:
        match trainer:
            case "dqn":
                from Trainers.DQNTrainer import DQNTrainer
                from Policies.CustomPolicy import CustomPolicy

                policy = CustomPolicy(network=main_network, action_space=env.action_space)

                trainer = DQNTrainer(
                    config=config,
                    target_network=target_network,
                    main_network=main_network,
                    policy=policy,
                    env=env
                )
                return DQNWrapper(model=trainer, use_custom=True, config=config, seed=seed)
            
            case "wocar":
                from Trainers.WocaRDQN import WocaRDQN
                from Policies.CustomPolicy import CustomPolicy

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
                return DQNWrapper(model=trainer, use_custom=True, config=config, seed=seed)
            
            case _:
                from stable_baselines3 import DQN

                sb3_kwargs = config.to_sb3_kwargs() if config else {}
                sb3_model = DQN(policy_str, env, **sb3_kwargs)
                return DQNWrapper(model=sb3_model, use_custom=False, config=config, seed=seed)