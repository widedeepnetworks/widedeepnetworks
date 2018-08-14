# Copyright 2018 Alexander Matthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import gpflow
import cProfile

from ess_torch import ESS
from hmc_torch import HMC
from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu

import shared
import defaults

#from IPython import embed
results_file_name = 'results/comparison_bias_2d.pickle'

def get_grid():
    linspaced = np.linspace( defaults.lower_lim, defaults.upper_lim, defaults.points_per_dim, dtype = np.float32 )
    grid = np.array( [*itertools.product( linspaced, linspaced )] )
    return grid

def reshape_and_plot(fig, ax, input_array, vmin, vmax, colorbar=True ):
    assert( (input_array>=vmin).all() )
    assert( (input_array<=vmax).all() )
    reshaped_array = np.reshape( input_array, (defaults.points_per_dim,defaults.points_per_dim) ) 
    im = ax.imshow( reshaped_array, extent = [defaults.upper_lim,defaults.lower_lim,defaults.upper_lim,defaults.lower_lim], vmin=vmin, vmax=vmax )
    if colorbar:
        fig.colorbar( im, ax= ax)
    return im

def gp_experiments(X,Y,grid_points):
    
    gp_model = shared.get_gp_model(X,Y, input_dim = 2, depth = defaults.shared_depth)
    gp_mean, gp_var = gp_model.predict_f( grid_points )
    
    return gp_mean, gp_var

def nn_experiments(X,Y,grid_points):
    H = defaults.hidden_units
    num_layers = defaults.shared_depth
    D_IN = X.shape[1]
    D_OUT = 1
    X_var = Variable( torch.from_numpy(X).type(defaults.tdtype), requires_grad=False)
    Y_var = Variable( torch.from_numpy(Y).type(defaults.tdtype), requires_grad=False)
    grid_var = Variable( torch.from_numpy(grid_points).type(defaults.tdtype), requires_grad=False)
    
    model = shared.get_nn_model(D_IN,H,D_OUT, num_layers)
    #model.cuda()
    #3000
    nthin = 1
    burn_in = 50
    results_manager = shared.ResultsManager(None, burn_in, nthin, False, False, True, False, True)
    shared.nn_model_regression(X_var,Y_var,grid_var, model, num_samples = 30000, epsilon = 0.05, beta = 1., leap_frog_iters=10, results_manager = results_manager )
    
    return results_manager

def get_plot_min_and_max_mean( mean_a, mean_b ): #rounds to the nearest 0.1
    abs_max = np.maximum( np.abs(mean_a).max(), np.abs(mean_b).max() )
    round_max = np.ceil( abs_max * 10. )/ 10.
    return round_max

def get_plot_max_sqrt( sqrt_a, sqrt_b ): #rounds to the nearest 0.1
    sqrt_max = np.maximum( sqrt_a.max(), sqrt_b.max() )
    round_max = np.ceil( sqrt_max * 10. )/ 10.
    return round_max

def main():
    torch.manual_seed(2)
    torch.set_num_threads(1)

    X,Y = shared.xor_data_np()
    grid_points = get_grid()
    
    gp_mean, gp_var = gp_experiments(X,Y,grid_points)
    nn_results = nn_experiments(X,Y,grid_points)
    
    results = { 'gp_mean' : gp_mean.flatten() , 'gp_var' : gp_var.flatten() }
    results.update(nn_results.to_dict())
    
    pickle.dump( results, open(results_file_name,'wb') )
    
if __name__ == '__main__':
    main()
