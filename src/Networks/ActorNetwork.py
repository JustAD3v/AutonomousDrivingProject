import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
  def __init__(self, in_features=25, h1=256, h2=256, out_features=5, learning_rate=5e-4, continuous=False):
    # init internal pytorch nn.Module parent class
    super().__init__()

    # network structure
    self.l1 = nn.Linear(in_features, h1)
    self.l2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

    self.continuous = continuous
    if continuous:
      # state-independent log standard deviation, learned with the network
      self.log_std = nn.Parameter(-0.5 * torch.ones(out_features))

    self.lr = learning_rate
    self.loss_fn = nn.SmoothL1Loss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = self.out(x) # don't apply relu on the output to allow for negative values
    return x

  def distribution(self, x):
    output = self(x)
    if self.continuous:
      # clamp log_std to avoid unstable training
      # if clamp is too low the gaussian will be very "precise" and so there is no exploration
      # if clamp is too high the gaussian will be too "flat" and so mostly noise
      std = torch.exp(self.log_std.clamp(-5.0, 1.0))
      # we expand to match the size of the output 
      # (which can be different from just out_features initialized in the constructor if we use mutliple minibatches)
      # (with minibatches we have multiple elements of size out_features)
      std = std.expand_as(output)
      gaussian = torch.distributions.Normal(output, std)
      tanh_gossian = torch.distributions.TransformedDistribution(
          gaussian,
          [torch.distributions.trasnforms.TanhTransform(cache_size=1)],
      )
      # Independent wraps the Normal distribution into an object 
      # capable of sampling and calculating the log probability "automatically"
      # That means by using .sample() on it we just get a sample of the Normal distribution
      # And when calling .log_prob(v) on it, with v = (X, Y) a vector of action (with 2 components X and Y)
      # It does calculate the sum of log probabilities of both components to get the log probability of v
      # It's the same as calculating : sum(Normal.log_prob(v)) 
      # or log(prob(X)) + log(prog(Y)) = log(prob(v))      
      return torch.distributions.Independent(tanh_gossian, 1)
    # Categorical applies softmax on network output
    # to get a probability distribution from the output vector
    # it also wraps it into an interface that allows for
    # sampling that distribution
    # and finding the log_prob for a given action
    return torch.distributions.Categorical(logits=output)

  def distribution(self, x):
    output = self(x)
    if self.continuous:
      
      return torch.distributions.Independent(torch.distributions.Normal(output, torch.exp(self.log_std)), 1)
    
    # Categorical applies softmax on network output
    # to get a probability distribution from the output vector
    # it also wraps it into an interface that allows for
    # sampling that distribution
    # and finding the log_prob for a given action
    return torch.distributions.Categorical(logits=output)