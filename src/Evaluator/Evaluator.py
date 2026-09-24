import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import time

from Agents.AbstractAgent import Agent

class Evaluator:
    def __init__(self, 
                 agent: Agent, 
                 model, 
                 nb_episodes, 
                 env_config, 
                 render_delay=0.05,
                 use_gpu=False):
        self.agent = agent
        self.model = model
        self.nb_episodes = nb_episodes
        self.env = gym.make('highway-fast-v0', config=env_config, render_mode="human")
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.agent.init_agent(self.env,
                              self.model,
                              None,
                              self.device)
        self.action_labels = self.env.unwrapped.action_type.actions.values()
        self.render_delay = render_delay


    def eval(self):
        success = 0
        choices = [0 for i in range(self.env.action_space.n)]
        for i in range(self.nb_episodes):
            episode_choices = [0 for i in range(self.env.action_space.n)]
            state, _ = self.env.reset()
            term = trunc = False
            while not (term | trunc):
                action = self.agent.policy(state)
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
                
        