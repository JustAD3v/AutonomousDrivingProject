import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
import random

from HyperParams import HyperParams
from ReplayBuffer import ReplayBuffer

class DQNTrainer:
    def __init__(self, 
                 hyperparams: HyperParams, 
                 target_network: nn.Module, 
                 main_network: nn.Module, 
                 r_buffer: ReplayBuffer,
                 env_name: str,  
                 env_config = None,
                 use_gpu = False):
        self.env = gym.make(env_name, config=env_config)
        self.hyperparams = hyperparams
        self.r_buffer = r_buffer
        self.target_net = target_network
        self.main_net = main_network
        self.target_network.load_state_dict(self.main_net.state_dict())
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")

    def optimize_main(self, loss):
        """ Method to optimize the main networks weights given the loss """
        self.mainNet.optimizer.zero_grad()
        loss.backward()
        self.mainNet.optimizer.step()

    def b_policy(self, state): 
        if random.uniform(0,1) < self.hyperparams.epsilon:
            return self.env.action_space.sample()
        else:
            return self.getArgMaxQMain(state)

    def one_hot(x):
        l = [0.0,0.0,0.0,0.0,0.0]
        l[x] = 1.0
        return l

    def getArgMaxQMain(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(np.ravel(state)) # (25,)
            # unsqueeze and expand to have => (5, 25)
            state_t_exp = torch.unsqueeze(state_t, 0).expand(5, 25)
            # append actions => (5,5) + (5, 25) => (5, 30)
            actions = torch.tensor([self.one_hot(x) for x in range(self.env.action_space.n)])
            input_tensor = torch.cat((state_t_exp,actions), dim=1)
            # get qvalues from model
            qvalues = self.mainNet.forward(input_tensor.to(self.device)) # out => (5,)
            return torch.argmax(qvalues).item()

    def getMaxQTarget(self, states):
        with torch.no_grad():
            batchsize = len(states) # batchsize = B
            nb_actions = self.env.action_space.n # 5
            # prepare actions => (5,5) => (B,5,5)
            actions = torch.tensor([self.one_hot(x) for x in range(self.env.action_space.n)]).expand(batchsize,nb_actions,nb_actions)
            # reshape to have the states repeated for every possible action a => (B, 1, 25) => (B, 5, 25)
            states = torch.unsqueeze(states, 1).expand(batchsize, nb_actions, 25)
            # add actions to every state vector => (B, 5, 5) + (B, 5, 25) = (B, 5, 30)
            input_t = torch.cat((states, actions), dim=2)
            # turn it into a matrix usable as input to the network => (B*5, 30) [Network takes (30,) as input]
            input_t = torch.flatten(input_t, start_dim=0, end_dim=1)
            # process all the inputs in one batch, doing only one forward call
            qvalues = self.targetNet.forward(input_t.to(self.device)) # (B*5,1)
            # reshape to (B, 5)
            qvalues = torch.reshape(qvalues, (batchsize, nb_actions))
            # reduce the QValues to only the max among the 5 actions for every state => (B, 5) => (B,)
            maxQValues = torch.max(qvalues, dim=1).values
            return maxQValues

    def trainLoop(self, save_file="targetNetwork.pt", stats_logfile=""):
        losses = []
        episode_returns = []
        epsilons = []
        for _ in range(self.hyperparams.nb_episode):
          state, info = self.env.reset()
          done = False
          total_loss = 0
          episode_return = 0
          nb_steps = 0
          while not done:
            action = self.b_policy(state)
            new_state, reward, term, trunc, _ = self.env.step(action)
            episode_return += reward
            done = term | trunc
            nb_steps += 1
            self.r_buffer.push((state, action, reward, new_state, done))
            if len(self.r_buffer) >= self.hyperparams.warmup_size:
              # uniform draw of a batch
              shuffled_idx = np.random.choice(np.arange(len(self.r_buffer)), size=self.batch_size, replace=False)
              batch = [self.r_buffer[i] for i in shuffled_idx]
              # extract components
              states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
              # calc y_target for every sample of batch using TARGET NETWORK
              qvalues = self.getMaxQTarget(next_states)
              # rewards, dones and qvalues should all be on same device
              y_target = rewards + self.hyperparams.gamma * qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)
              # calc y_main using MAIN NETWORK
              one_hot_actions = torch.tensor([self.one_hot(x) for x in actions]) # (B,5)
              batch_input = torch.cat((states, one_hot_actions), dim=1) # turn states = (B, 25) to => (B, 30)
              y_main = self.mainNet.forward(batch_input.to(self.device))
              # calc loss
              loss = self.mainNet.loss_fn(y_main.squeeze(), y_target)
              total_loss += loss.item()
              # optimize
              self.optimize_main(loss)
              if iteration % self.hyperparams.update_freq == 0:
                # copy mainNet weights to targetNet
                self.targetNet.load_state_dict(self.mainNet.state_dict())
                print("[!!] Weights of Target Network updated !")
            iteration += 1
            state = new_state
        
          print(f"episode {episode_count}/{self.hyperparams.nb_episode} finished (length={nb_steps}), avg_loss = {total_loss/nb_steps}, tot_return={episode_return}, epsilon={epsilon}, warmup_done = {len(r_buffer) >= warmup_size}")
          epsilon = max(self.hyperparams.epsilon_min, epsilon*self.hyperparams.epsilon_decay)
          episode_count += 1
          losses.append(total_loss/nb_steps)
          episode_returns.append(episode_return)
          epsilons.append(epsilon)
    
        print("[+] Training finished !")

        # TODO: implement conditional saving
        torch.save(self.hyperparams.targetNet.state_dict(), save_file)