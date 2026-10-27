from Adversarial.AbstractAdversarial import Adversarial
from Adversarial.NeutralAdversarial import NeutralAdversarial

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
                 env_config,
                 render_mode=None, 
                 render_delay=0.05,
                 use_gpu=False,
                 adversarial:Adversarial = None,
                 attack_bound=0.5,
                 seed=None,
                 continuous=False):
        self.policy = policy
        self.nb_episodes = nb_episodes
        self.env = gym.make('highway-fast-v0', config=env_config, render_mode=render_mode)
        # self.env = gym.make('highway-fast-v0')
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
            
        self.continuous = continuous
        if not self.continuous:
            self.action_labels = self.env.unwrapped.action_type.actions.values()
        self.render_delay = render_delay
        self.adversarial = adversarial if adversarial is not None else NeutralAdversarial(attack_bound) 
        if seed is not None:
            self.set_seed(seed)

    def eval(self):
        success = 0
        # discrete: count of each action, continuous: sum of actions (averaged for display)
        action_shape = self.env.action_space.shape[0] if self.continuous else self.env.action_space.n
        choices = np.zeros(action_shape) if self.continuous else [0 for i in range(action_shape)]
        steps = 0
        for i in range(self.nb_episodes):
            episode_choices = np.zeros(action_shape) if self.continuous else [0 for i in range(action_shape)]
            episode_steps = 0
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
                state = new_state
                time.sleep(self.render_delay)
            choices_dict = self.choices_summary(episode_choices, episode_steps)
            if trunc:
              success += 1
              print(f"[+] Episode ended with success ! - ({i}/{self.nb_episodes}) - {choices_dict}")
            else:
              print(f"[-] Driving failed :( - ({i}/{self.nb_episodes}) - {choices_dict}")
        print("[*] Final result :")
        print(f"- Success : {success}")
        print(f"- Crash : {self.nb_episodes-success}")
        print(f"Success rate : {(success/self.nb_episodes)*100}%")
        choices_dict = self.choices_summary(choices, steps)
        print(f"Action choices : {choices_dict}")
        self.env.close()

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
        