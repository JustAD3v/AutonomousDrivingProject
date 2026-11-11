from Adversarial.AbstractAdversarial import Adversarial
from Adversarial.NeutralAdversarial import NeutralAdversarial
from Evaluator.EvalResult import EvalResult

import torch
import gymnasium as gym
import highway_env
import time
import random
import numpy as np

class Evaluator:
    def __init__(self, 
                 policy, 
                 nb_episodes, 
                 env=None,
                 render_mode=None, 
                 render_delay=0.05,
                 use_gpu=False,
                 adversarial:Adversarial = None,
                 attack_bound=0.5,
                 seed=None,
                 continuous=False,
                 model=None):
        self.policy = policy
        self.nb_episodes = nb_episodes
        self.seed = seed
        self.model = model
        self.env = gym.make('highway-fast-v0', config=env.spec.kwargs if env is not None else None, render_mode=render_mode)
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
            
        self.continuous = continuous
        if not self.continuous:
            self.action_labels = self.env.unwrapped.action_type.actions.values()
        self.render_delay = render_delay
        self.adversarial = adversarial if adversarial is not None else NeutralAdversarial(attack_bound)
        if self.adversarial.env is None:
            # the attacks need the evaluated environment to scale their bound per feature
            self.adversarial.env = self.env
        self.record = EvalResult(nb_episodes=self.nb_episodes,
                                 name="Evaluation",
                                 model=self.model,
                                 attack=type(self.adversarial).__name__,
                                 attack_bound=self.adversarial.attack_bound,
                                 seed=self.seed)
        if seed is not None:
            self.set_seed(seed)

    def eval(self):
        # discrete: count of each action, continuous: sum of actions (averaged for display)
        action_shape = self.env.action_space.shape[0] if self.continuous else self.env.action_space.n
        choices = np.zeros(action_shape) if self.continuous else [0 for i in range(action_shape)]
        steps = 0
        for i in range(self.nb_episodes):
            episode_choices = np.zeros(action_shape) if self.continuous else [0 for i in range(action_shape)]
            episode_steps = 0
            episode_return = 0.0
            state, _ = self.env.reset()
            term = trunc = False
            while not (term | trunc):
                # this line disrupt the current state observed by the agent but does not change the actual state of the environment
                state = self.adversarial.disrupt(state)
                action, _ = self.policy.predict(state, deterministic=True)
                if self.continuous:
                    action = self.policy.to_env_action(action)
                    choices += action
                    episode_choices += action
                else:
                    choices[action] += 1
                    episode_choices[action] += 1
                steps += 1
                episode_steps += 1
                new_state, reward, term, trunc, _ = self.env.step(action)
                episode_return += reward
                state = new_state
                time.sleep(self.render_delay)
            choices_dict = self.choices_summary(episode_choices, episode_steps)
            self.record.add_episode(episode_steps, episode_return, success=trunc, details=str(choices_dict))
        self.record.report_final()
        choices_dict = self.choices_summary(choices, steps)
        print(f"[+] Action choices : {choices_dict}")
        self.env.close()
        return self.record

    def choices_summary(self, choices, steps):
        if self.continuous:
            mean_action = choices / max(steps, 1)
            return {"mean_action": np.round(mean_action, 3).tolist()}
        return dict(zip(self.action_labels, choices))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        