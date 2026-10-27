from Trainers.TrainerWrapper import TrainerWrapper
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
            
            case "wocar":
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
                return TrainerWrapper(trainer=sb3_model, trainer_type=trainer_type, config=config, seed=seed)