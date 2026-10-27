import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WocaContNN(nn.Module):
  def __init__(self, in_features=25, h1=256, h2=256, out_features=2, learning_rate=5e-4, continuous=True):
    # init internal pytorch nn.Module parent class
    super().__init__()

    # network structure
    self.l1 = nn.Linear(in_features, h1)
    self.l2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

    if continuous:
      # state-independent log standard deviation, learned with the network
      self.log_std = nn.Parameter(-0.5 * torch.ones(out_features))

    # other params
    self.lr = learning_rate
    self.loss_fn = nn.SmoothL1Loss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.out(x) # don't apply relu on the output to allow for negative values
    return x
  
  def process_batch(self, batch_tuple, device):
    states, actions, _, _, _ = batch_tuple
    # calling self(...) instead of model.forward(..) works and is better because we named it forward
    qvalues = self(states.to(device)) # => (B,5)
    # Select Q(s, action_chosen)
    actions = actions.to(device).long().unsqueeze(1) # => (B,1)
    selected_qvalues = qvalues.gather(1, actions).squeeze(1) # gather taking only the qvalue of chosen action => (B,1) => squeeze => (B,)

    return selected_qvalues