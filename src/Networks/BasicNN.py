import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNetwork(nn.Module):
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