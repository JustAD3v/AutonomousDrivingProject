import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicNN(nn.Module):
  def __init__(self, in_features=30, h1=64, h2=64, out_features=1):
    # init internal pytorch nn.Module parent class
    super().__init__()

    # network structure
    self.l1 = nn.Linear(in_features, h1)
    self.l2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

    # other params
    self.lr = 1e-4
    self.loss_fn = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.out(x) # don't apply relu on the output to allow for negative values
    return x

  def getMaxQ(self, states, nb_actions, device):
    with torch.no_grad():
        batchsize = len(states) # batchsize = B
        # prepare actions => (5,5) => (B,5,5)
        actions = torch.tensor([self.one_hot(x) for x in range(nb_actions)]).expand(batchsize,nb_actions,nb_actions)
        # reshape to have the states repeated for every possible action a => (B, 1, 25) => (B, 5, 25)
        states = torch.unsqueeze(states, 1).expand(batchsize, nb_actions, 25)
        # add actions to every state vector => (B, 5, 5) + (B, 5, 25) = (B, 5, 30)
        input_t = torch.cat((states, actions), dim=2)
        # turn it into a matrix usable as input to the network => (B*5, 30) [Network takes (30,) as input]
        input_t = torch.flatten(input_t, start_dim=0, end_dim=1)
        # process all the inputs in one batch, doing only one forward call 
        # forward is done internally by calling self(...) because we called the method "forward"
        qvalues = self(input_t.to(device)) # (B*5,1)
        # reshape to (B, 5)
        qvalues = torch.reshape(qvalues, (batchsize, nb_actions))
        # reduce the QValues to only the max among the 5 actions for every state => (B, 5) => (B,)
        maxQValues = torch.max(qvalues, dim=1).values
        return maxQValues

  def getArgMaxQPolicy(self, state, nb_actions, device):
    with torch.no_grad():
      state_t = torch.FloatTensor(np.ravel(state)) # (25,)
      # unsqueeze and expand to have => (5, 25)
      state_t_exp = torch.unsqueeze(state_t, 0).expand(5, 25)
      # append actions => (5,5) + (5, 25) => (5, 30)
      actions = torch.tensor([self.one_hot(x) for x in range(nb_actions)])
      input_tensor = torch.cat((state_t_exp,actions), dim=1)
      # get qvalues from model
      # calling model(...) instead of model.forward(..) works and is better because we named it forward
      qvalues = self(input_tensor.to(device)) # out => (5,)
      return torch.argmax(qvalues).item()
  
  def get_qvalues(self, batch_tuple, device):
    states, actions, _, _, _ = batch_tuple
    one_hot_actions = torch.tensor([self.one_hot(x) for x in actions]) # (B,5)
    batch_input = torch.cat((states, one_hot_actions), dim=1) # turn states = (B, 25) to => (B, 30)
    # calling model(...) instead of model.forward(..) works and is better because we named it forward
    return self(batch_input.to(device))
    
  def one_hot(self, x):
    l = [0.0,0.0,0.0,0.0,0.0]
    l[x] = 1.0
    return l