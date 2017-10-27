import torch

#pytorch stuff
tdtype = torch.FloatTensor

#Two dimensional plot stuff.
upper_lim = 2.
lower_lim = -1.*upper_lim
points_per_dim = 70

#Hyperparameters.
bias_variance = 0.2
weight_variance = 0.8
noise_variance = 1e-1
shared_depth = 3
hidden_units = 50

#Text for figures
deep_net_name = 'Bayesian deep network'
