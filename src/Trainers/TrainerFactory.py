from Trainers.TrainerWrapper import TrainerWrapper
import random

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
                from stable_baselines3.common.env_util import make_vec_env

                # Create vectorized environment using your config's n_envs
                vec_env = make_vec_env(
                    env_id="highway-fast-v0",  # or your environment getter
                    n_envs=config.n_envs,
                    seed=seed
                )

                ppo_kwargs = config.to_ppo_kwargs()
                ppo_model = PPO("MlpPolicy", vec_env, **ppo_kwargs)
                return TrainerWrapper(trainer=ppo_model, trainer_type=trainer_type, config=config, seed=seed)

            
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