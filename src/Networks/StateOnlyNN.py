import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StateOnlyNN(nn.Module):
  def __init__(self, in_features=25, h1=64, h2=64, out_features=5):
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
        print(states)
        # process all the inputs in one batch, doing only one forward call 
        # forward is done internally by calling self(...) because we called the method "forward"
        qvalues = self(states.to(device)) # (B, 5)
        # reduce the QValues to only the max among the 5 actions for every state => (B, 5) => (B,)
        maxQValues = torch.max(qvalues, dim=1).values
        return maxQValues
  
  def getArgMaxQPolicy(self, state, nb_actions, device):
    with torch.no_grad():
      state_t = torch.FloatTensor(np.ravel(state)) # (25,)
      qvalues = self(state_t.to(device)) # out => (5,)
      return torch.argmax(qvalues).item()
  
  def process_batch(self, batch_tuple):
    states, _, _, _, _ = batch_tuple
    # calling model(...) instead of model.forward(..) works and is better because we named it forward
    return self(states.to(self.device))