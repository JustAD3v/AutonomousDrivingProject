from Trainers.DQNWrapper import DQNWrapper

class TrainerFactory:
    @staticmethod
    def create_trainer(
        env,
        config,
        use_custom=True,
        main_network=None,
        target_network=None,
        policy_str="MlpPolicy"
    ) -> DQNWrapper:
        if use_custom:
            from Trainers.DQNTrainer import DQNTrainer
            from Policies.CustomPolicy import CustomPolicy

            policy = CustomPolicy(main_network=main_network, action_space=env.action_space)

            trainer = DQNTrainer(
                config=config,
                target_network=target_network,
                main_network=main_network,
                policy=policy,
                env=env
            )
            return DQNWrapper(model=trainer, use_custom=True, config=config)
        else:
            from stable_baselines3 import DQN

            sb3_kwargs = config.to_sb3_kwargs() if config else {}
            sb3_model = DQN(policy_str, env, **sb3_kwargs)
            return DQNWrapper(model=sb3_model, use_custom=False, config=config)