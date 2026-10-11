from Adversarial.AbstractAdversarial import Adversarial
from Adversarial.NeutralAdversarial import NeutralAdversarial

import torch
import gymnasium as gym
import highway_env
import time

class Evaluator:
    def __init__(self, 
                 policy, 
                 nb_episodes, 
                 env_config,
                 render_mode=None, 
                 render_delay=0.05,
                 use_gpu=False,
                 adversarial:Adversarial = None,
                 attack_bound=0.5):
        self.policy = policy
        self.nb_episodes = nb_episodes
        self.env = gym.make('highway-fast-v0', config=env_config, render_mode=render_mode)
        # self.env = gym.make('highway-fast-v0')
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
            
        self.action_labels = self.env.unwrapped.action_type.actions.values()
        self.render_delay = render_delay
        self.adversarial = adversarial if adversarial is not None else NeutralAdversarial(attack_bound) 

    def eval(self):
        success = 0
        choices = [0 for i in range(self.env.action_space.n)]
        for i in range(self.nb_episodes):
            episode_choices = [0 for i in range(self.env.action_space.n)]
            state, _ = self.env.reset()
            term = trunc = False
            while not (term | trunc):
                # this line disrupt the current state observed by the agent but does not change the actual state of the environment
                state = self.adversarial.disrupt(state)
                action, _ = self.policy.predict(state, deterministic=True)
                choices[action] += 1
                episode_choices[action] += 1
                new_state, reward, term, trunc, _ = self.env.step(action)
                state = new_state
                time.sleep(self.render_delay)
            choices_dict = dict(zip(self.action_labels, episode_choices))
            if trunc:
              success += 1
              print(f"[+] Episode ended with success ! - ({i}/{self.nb_episodes}) - {choices_dict}")
            else:
              print(f"[-] Driving failed :( - ({i}/{self.nb_episodes}) - {choices_dict}")
        print("[*] Final result :")
        print(f"- Success : {success}")
        print(f"- Crash : {self.nb_episodes-success}")
        print(f"Success rate : {(success/self.nb_episodes)*100}%")
        choices_dict = dict(zip(self.action_labels, choices))
        print(f"Action choices : {choices_dict}")
        self.env.close()