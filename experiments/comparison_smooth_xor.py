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
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pylab as plt
import numpy as np
import math
import time
import pickle

import torch
from torch.autograd import Variable

import gpflow
import cProfile

from IPython import embed

from sklearn import datasets
from ess_torch import ESS
from hmc_torch import HMC
from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu

import shared
import defaults

#from IPython import embed
results_file_name = 'results/comparison_smooth_xor.pickle'

def plotPredictions( ax, color, label, predMean, predVar, xtest ):
    handle = ax.plot( xtest, predMean, color, linewidth=1)#, label=label )
    ax.plot( xtest, predMean + 2.*np.sqrt(predVar),color, linewidth=1 )
    ax.plot( xtest, predMean - 2.*np.sqrt(predVar), color, linewidth=1 )  
    #ax.text( 6., 2.5, label )
    standardPlotLimits(ax)
    return handle

def standardPlotLimits(ax):
    ax.set_xlim( [-3., 3. ] )
    ax.set_ylim( [-4.,4. ] )


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
    gp_model.optimize()
    gp_mean, gp_var = gp_model.predict_f( grid_points )
    
    return gp_mean, gp_var, gp_model

def nn_experiments(X,Y,grid_points, weight_variance, bias_variance, noise_variance):
    num_layers = defaults.shared_depth
    D_IN = X.shape[1]
    D_OUT = 1
    X_var = Variable( torch.from_numpy(X).type(defaults.tdtype), requires_grad=False)
    Y_var = Variable( torch.from_numpy(Y).type(defaults.tdtype), requires_grad=False)
    grid_var = Variable( torch.from_numpy(grid_points).type(defaults.tdtype), requires_grad=False)
    
    H = defaults.hidden_units
    model = shared.get_nn_model(D_IN,H,D_OUT, num_layers,'identity', weight_variance, bias_variance)
    nthin = 50 
    burn_in = 100
    results_manager = shared.ResultsManager(None, burn_in, nthin, False, False, True, False, True)
    shared.nn_model_regression(X_var,Y_var,grid_var, model, num_samples = 2000000, epsilon = 0.0005, beta = 0.1, leap_frog_iters=10, noise_variance=noise_variance, results_manager=results_manager )
    
    return results_manager

def get_plot_min_and_max_mean( mean_a, mean_b ): #rounds to the nearest 0.1
    abs_max = np.maximum( np.abs(mean_a).max(), np.abs(mean_b).max() )
    round_max = np.ceil( abs_max * 10. )/ 10.
    return round_max

def get_plot_max_sqrt( sqrt_a, sqrt_b ): #rounds to the nearest 0.1
    sqrt_max = np.maximum( sqrt_a.max(), sqrt_b.max() )
    round_max = np.ceil( sqrt_max * 10. )/ 10.
    return round_max

def smooth_xor(x):
    beta = np.sqrt(2.)
    gamma = np.exp(beta)    
    sq_dist = x[:,0]**2 + x[:,1]**2
    exp_term = np.exp(-sq_dist/beta)
    y = - x[:,0] * x[:,1] * exp_term * gamma
    return y

def make_smooth_xor_data():
    n_samples = 100
    dimensionality = 2
    true_noise_std = 0.1
    rng = np.random.RandomState(1)
    x = rng.randn(n_samples,dimensionality) 
    noise = rng.randn(n_samples) * true_noise_std
    y = smooth_xor(x) + noise
    return x, np.atleast_2d(y).T

def get_base_grid(ngrid):
    return np.linspace(-3.,3.,ngrid)
    
def get_grid_two_d(ntest_per_dim):
    xA = get_base_grid(ntest_per_dim)
    xB = get_base_grid(ntest_per_dim)
    xComb = np.array( [elem for elem in itertools.product(xA, xB)] )
    return xComb
        
def plot_smooth_xor_data_2d(ax):
    ngrid = 1000
    xA = get_base_grid(ngrid)
    xB = get_base_grid(ngrid)
    
    #xAv, yBv = np.meshgrid(xA, xB)
    
    xComb = np.array( [elem for elem in itertools.product(xA, xB)] )
    y = smooth_xor(xComb)
    
    mappable = ax.imshow(-y.reshape(ngrid,ngrid), extent = [-3.,3.,-3.,3.])
    plt.colorbar(mappable)

    
    x,not_used = make_smooth_xor_data() 
    ax.scatter(x[:,0],x[:,1], color='r', marker='+')    
    
    testA,testB, not_used = get_test_points()
    ax.plot(testA[:250],testB[:250], 'k--')
    ax.plot(testA[250:],testB[250:], 'k--')
    ax.text(-2.8,0.75,'Cross-section 1', fontsize=12)
    ax.text(-2.25,-2.5,'Cross-section 2', fontsize=12)
    
    
    #zv = y.reshape(xA.shape[0],xB.shape[0])
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(xAv, yBv, zv)

def get_test_points():
    #test set A
    ncross_section = 250
    betas = np.linspace(-3.,3. ,ncross_section)
    xA = np.vstack([betas,betas]).T

    #test set B
    const = np.ones_like(betas)
    xB = np.vstack([betas,const]).T

    return xA, xB, betas

def main():
    torch.manual_seed(2)
    X,Y = make_smooth_xor_data()
    ntest_per_dim = 30
    grid_points = np.vstack( get_test_points()[0:-1] )

    gp_mean, gp_var, gp_model = gp_experiments(X,Y,grid_points)
    bias_variance = gp_model.kern.bias_variance.value[0]
    weight_variance = gp_model.kern.variance.value[0]
    noise_variance = gp_model.likelihood.variance.value[0]
    nn_results = nn_experiments(X,Y,grid_points, weight_variance, bias_variance, noise_variance)
    
    results = { 'gp_mean' : gp_mean.flatten(), 'gp_var' : gp_var.flatten()}
    results.update(nn_results.to_dict())
    
    pickle.dump( results, open(results_file_name,'wb') )

if __name__ == '__main__':
    main()
