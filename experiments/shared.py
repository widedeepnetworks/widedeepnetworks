import sys
sys.path.append('../')
import time
import numpy as np

#from IPython import embed

import torch
import gpflow

from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu
from ess_torch import ESS
from hmc_torch import HMC

import defaults

def get_kernel(input_dim,depth):
    kern = DeepArcCosine(input_dim = input_dim, num_steps = depth, variance = defaults.weight_variance, bias_variance = defaults.bias_variance ) + gpflow.kernels.White(input_dim,variance=1e-6)
    return kern

def get_gp_model(X,Y, input_dim, depth):
    kern = get_kernel( input_dim, depth )
    model = gpflow.gpr.GPR(X,Y, kern = kern )
    model.likelihood.variance= defaults.noise_variance
    return model

def get_intermediate_layers(H, num_layers, bias):
    intermediate_layers = []
    for layer_index in range(num_layers-1):
        intermediate_layers+= [GaussLinearStandardized(H, H, bias=bias, raw_weight_variance = defaults.weight_variance, raw_bias_variance = defaults.bias_variance),
        ScaledRelu() ]
    return intermediate_layers
    
def get_nn_model(D_IN,H,D_OUT, num_layers):
    intermediate_layers = get_intermediate_layers(H, num_layers, True)
    model = torch.nn.Sequential(
        GaussLinearStandardized(D_IN, H, bias=True, raw_weight_variance = defaults.weight_variance, raw_bias_variance = defaults.bias_variance),
        ScaledRelu(),
        *intermediate_layers,
        GaussLinearStandardized(H, D_OUT, bias=True, raw_weight_variance = defaults.weight_variance, raw_bias_variance = defaults.bias_variance)
    )
    return model 

def draw_from_gp_prior(rng=np.random.RandomState(1)):
    grid_points = get_grid()
    kern = get_kernel()
    K = kern.compute_K_symm( grid_points ) 
    L = np.linalg.cholesky(K)
    standard_normal = rng.randn( grid_points.shape[0] )
    sample = np.dot(L, standard_normal )
    return sample

def draw_sample_from_nn_prior(model, grid_points):
    pred = model(grid_points)
    return pred

def nn_model_regression(X,Y, test_X, model, num_samples, burn_in, epsilon, beta, leap_frog_iters ):
    test_size = test_X.size()[0]

    #run sampler
    #at the same time mantain online estimated of mean
    #and marginal variance
    #this stops us having to store large numbers of samples.
    
    num_points = 0
    online_mean = np.zeros(test_size) 
    online_squares = np.zeros(test_size)
    criterion = torch.nn.MSELoss(size_average=False)
    
    sampler = HMC(model.parameters(), np.random.RandomState(1), epsilon = epsilon , beta = beta, leap_frog_iters = leap_frog_iters )
    samplerB = ESS(model.parameters(), np.random.RandomState(2) )
    
    energies = np.zeros(num_samples)
    
    start_time = time.time()
    for sample_index in range(num_samples):
        def closure():
            sampler.zero_grad()
            pred = model( X )
            energy = 0.5*criterion(pred, Y )/ defaults.noise_variance
            energy.backward()
            return energy
        sampler.step( closure )
        
        def closureB():
            pred = model( X )
            energy = 0.5*criterion(pred, Y )/ defaults.noise_variance
            return energy    
        energies[sample_index] = samplerB.step( closureB )
                
        if sample_index > burn_in:
            #get prediction
            pred = model(test_X).data.cpu().numpy().flatten()
            
            #do online updates.
            num_points+=1
            delta = pred-online_mean
            online_mean = online_mean + delta/num_points
            delta2 = pred-online_mean
            online_squares = online_squares + delta * delta2
    end_time = time.time()
    print('Total time' , end_time - start_time )
    print('iterations per second', num_samples*1./(end_time - start_time))
    
    #embed()
    return online_mean, online_squares / (num_points + 1)       

def xor_data_np():
    X = np.array( [[1. , 1.] , [1.,-1], [-1.,1.], [-1., -1] ], np.float32 )
    Y = np.array( [ [-1.], [1.], [1.],  [-1.] ], np.float32 )
    return X, Y
