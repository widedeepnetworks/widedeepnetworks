import sys
sys.path.append('../')
import itertools
from matplotlib import pylab as plt
import numpy as np
import math
import time
import pickle

import torch
from torch.autograd import Variable

from matplotlib2tikz import save as save_tikz

import gpflow
import cProfile

from ess_torch import ESS
from hmc_torch import HMC
from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu
import shared
import defaults

from IPython import embed

#One dimensional plot stuff.
ylim = [-2.75, 2.75]
xlim = [-2.,2]    
results_file_name = 'results/comparison_bias.pickle'

def get_XY_gridpoints():
    X = np.atleast_2d( np.array([-1.,0., 1.], np.float32 ) ).T
    Y = np.array( [ [-1.], [-0.], [1.] ], np.float32 )
    grid_points = np.atleast_2d( np.linspace( -2. , 2., 100 ) ).T    
    return X, Y, grid_points

def plot_mean_and_stds( X, Y, ax, grid_points, means, stds, title ):
    ax.plot( grid_points, means,'b' )
    ax.plot( grid_points, means+2.*stds,'g--')
    ax.plot( grid_points, means-2.*stds,'g--')
    ax.plot( X, Y, 'rx')
    ax.set_xlim(xlim)
    ax.set_xlabel('Input')
    ax.set_ylim(ylim)
    ax.set_ylabel('Output')
    ax.set_title(title)

def gp_experiments(X,Y,grid_points):#, fig, axes):
    
    gp_model = shared.get_gp_model(X,Y,input_dim=1,depth=defaults.shared_depth)
    gp_mean, gp_var = gp_model.predict_f( grid_points )
    
    return gp_mean, gp_var 

def nn_experiments(X,Y,grid_points):#, fig, axes):
    H = defaults.hidden_units
    num_layers = defaults.shared_depth
    D_IN = X.shape[1]
    D_OUT = 1
    X_var = Variable( torch.from_numpy(X).type(defaults.tdtype), requires_grad=False)
    Y_var = Variable( torch.from_numpy(Y).type(defaults.tdtype), requires_grad=False)
    grid_var = Variable( torch.from_numpy(grid_points).type(defaults.tdtype), requires_grad=False)
    
    model = shared.get_nn_model(D_IN,H,D_OUT, num_layers)
    #model.cuda()

    nn_sample = shared.draw_sample_from_nn_prior(model, grid_var)

    pred_mean, pred_var = shared.nn_model_regression(X_var,Y_var,grid_var, model, num_samples = 3000, burn_in = 50, epsilon = 0.1, beta = 1, leap_frog_iters = 10  )
    
    return pred_mean, pred_var
    
def main():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    X,Y, grid_points = get_XY_gridpoints()
    
    gp_mean, gp_var = gp_experiments(X,Y,grid_points)#, fig, axes )
    nn_mean, nn_var = nn_experiments(X,Y,grid_points)#, fig, axes )

    results = { 'gp_mean' : gp_mean.flatten() , 'gp_var' : gp_var.flatten() , 'nn_mean' : nn_mean.flatten(), 'nn_var' : nn_var.flatten() }
    
    pickle.dump( results, open(results_file_name,'wb') )
    
if __name__ == '__main__':
    main()
