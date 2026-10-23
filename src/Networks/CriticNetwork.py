import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CriticNetwork(nn.Module):
  # out_features = 1 = critic_value
  def __init__(self, in_features=25, h1=256, h2=256, out_features=1, learning_rate=5e-4):
    # init internal pytorch nn.Module parent class
    super().__init__()

    # network structure
    self.l1 = nn.Linear(in_features, h1)
    self.l2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

    # other params
    self.lr = learning_rate
    # use MSE for critic network
    self.loss_fn = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.out(x) # don't apply relu on the output to allow for negative values
    return x