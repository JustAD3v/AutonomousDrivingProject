import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env

from Trainers.TrainConfig import TrainConfig
from Trainers.ReplayBuffer import ReplayBuffer
from Agents.AbstractAgent import Agent
from Trainers.TrainResult import TrainResult

class DQNTrainer:
    def __init__(self, 
                 config: TrainConfig, 
                 target_network: nn.Module,
                 main_network: nn.Module, 
                 agent: Agent, 
                 env_name = "highway-v0",  
                 env_config = None,
                 use_gpu = False):
        self.env = gym.make(env_name, config=env_config)
        self.config = config
        self.agent = agent
        self.main_net = main_network
        self.target_net = target_network
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.r_buffer = ReplayBuffer(self.config.buffer_size, 
                                             config.batch_size,
                                             config.warmup_size,
                                             self.device)
        self.record = TrainResult(self.config.nb_episodes)
        self.agent.init_agent(self.env, 
                              self.main_net,
                              self.config,
                              self.device)

    def optimize_main(self, loss):
        """ Method to optimize the main networks weights given the loss """
        self.main_net.optimizer.zero_grad()
        loss.backward()
        self.main_net.optimizer.step()

    def one_hot(self, x):
        l = [0.0,0.0,0.0,0.0,0.0]
        l[x] = 1.0
        return l

    """ 
    TODO: 
    Move reshaping out of this method
    Trainer should not depend on a Network specific encoding       
    """
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
            # forward is done internally by calling target_net(...) because we called the method "forward"
            qvalues = self.target_net(input_t.to(self.device)) # (B*5,1)
            # reshape to (B, 5)
            qvalues = torch.reshape(qvalues, (batchsize, nb_actions))
            # reduce the QValues to only the max among the 5 actions for every state => (B, 5) => (B,)
            maxQValues = torch.max(qvalues, dim=1).values
            return maxQValues

    def trainLoop(self, save_file="targetNetwork.pt", stats_logfile="") -> TrainResult:
        """ Warm up """
        print("[+] === Warm up starts ===")
        warmup_samples = 0
        while warmup_samples < self.config.warmup_size:
            state, _ = self.env.reset()
            done = False
            while not done:
                 action = self.agent.policy(state)
                 new_state, reward, term, trunc, _ = self.env.step(action)
                 done = term | trunc
                 self.r_buffer.push((state, action, reward, new_state, done))
                 warmup_samples += 1
                 state = new_state
            print(f"buffer filled with : {warmup_samples}/{self.config.warmup_size}")
        print("[+] Warmup done !")
        """ Training """
        print("[+] === Training starts ===")
        iteration = 0
        episode_count = 0
        for _ in range(self.config.nb_episodes):
          state, _ = self.env.reset()
          done = False
          total_loss = 0
          episode_return = 0
          nb_steps = 0
          while not done:
            action = self.agent.policy(state)
            new_state, reward, term, trunc, _ = self.env.step(action)
            episode_return += reward
            done = term | trunc
            nb_steps += 1
            self.r_buffer.push((state, action, reward, new_state, done))
            # uniform draw of a batch
            states, actions, rewards, next_states, dones = self.r_buffer.get_batch()
            # calc y_target for every sample of batch using TARGET NETWORK
            qvalues = self.getMaxQTarget(next_states)
            # rewards, dones and qvalues should all be on same device
            y_target = rewards + self.config.gamma * qvalues * (1-dones) # only add the estimated next_state qvalue if there is a next state (= not done)
            # calc y_main using MAIN NETWORK (network of the agent)
            y_main = self.agent.process_batch((states, actions, rewards, next_states, dones)) 
            # calc loss
            loss = self.main_net.loss_fn(y_main.squeeze(), y_target)
            total_loss += loss.item()
            # optimize
            self.optimize_main(loss)
            if iteration % self.config.update_freq == 0:
              # copy main_net weights to target_net
              self.target_net.load_state_dict(self.main_net.state_dict())
              print("[!!] Weights of Target Network updated !")
            iteration += 1
            state = new_state
        
          print(f"episode {episode_count}/{self.config.nb_episodes} finished (length={nb_steps}), avg_loss = {total_loss/nb_steps}, tot_return={episode_return}, epsilon={self.config.epsilon}")
          self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon*self.config.epsilon_decay)
          episode_count += 1
          self.record.add_loss(total_loss/nb_steps)
          self.record.add_episode_return(episode_return)
          self.record.add_epsilon(self.config.epsilon)
    
        print(f"[+] Training finished ! Target network saved to file : {save_file}")

        # TODO: implement conditional saving
        torch.save(self.config.target_net.state_dict(), save_file)
        return self.record
    