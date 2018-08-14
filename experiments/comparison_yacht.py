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
from scipy.misc import logsumexp
import pickle

import torch
from torch.autograd import Variable

import gpflow
import cProfile

from ess_torch import ESS
from hmc_torch import HMC
from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu

from IPython import embed

import shared
import defaults

results_file_name = 'results/comparison_yacht.pickle'
input_data_file_name = 'datasets/yacht_hydrodynamics.csv'

def getData():
    full_array = np.genfromtxt(input_data_file_name, delimiter=',')
    num_data = full_array.shape[0]
    num_train = 100
    rng = np.random.RandomState(1)
    permutation = rng.permutation(num_data)
    full_targets = np.atleast_2d(full_array[permutation,-1]).T
    full_covariates = full_array[permutation,:-1]
    
    X_train = full_covariates[:num_train,:]
    X_test = full_covariates[num_train:,:]
    
    def standardizeArray(input_array, mean, std):
        return (input_array - mean[None,:])/std[None,:]
    
    def add_noise(input_array):
        return input_array + rng.randn(*input_array.shape)*0.1
    
    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)
    
    X_train = standardizeArray(X_train, train_means, train_stds)
    X_test = standardizeArray(X_test, train_means, train_stds)
    
    Y_train = full_targets[:num_train,:]
    Y_test = full_targets[num_train:,:]
    
    target_mean = Y_train.mean(axis=0)
    target_std = Y_train.std(axis=0)
    
    Y_train = standardizeArray(Y_train, target_mean, target_std)
    Y_test = standardizeArray(Y_test, target_mean, target_std)
    
    Y_train = add_noise(Y_train)
    Y_test = add_noise(Y_test)
    
    return X_train, Y_train, X_test, Y_test
    
def gp_experiments(X_train, Y_train, X_test, Y_test):
    gp_model = shared.get_gp_model(X_train,Y_train,input_dim=X_train.shape[1],depth=defaults.shared_depth)
    gp_model.optimize()
    individual_log_densities  = gp_model.predict_density(X_test,Y_test)
    holdout_ll = individual_log_densities.mean()
    pred_mean, pred_var = gp_model.predict_f(X_test)
    norm_rmse = shared.nrmse(Y_test, pred_mean)
    return holdout_ll, norm_rmse, gp_model, individual_log_densities

def nn_experiments(X_train, Y_train, X_test, Y_test, weight_variance, bias_variance, noise_variance):
    H = defaults.hidden_units
    num_layers = defaults.shared_depth
    D_IN = X_train.shape[1]
    D_OUT = 1
    X_train_var = Variable( torch.from_numpy(X_train).type(defaults.tdtype), requires_grad=False)
    Y_train_var = Variable( torch.from_numpy(Y_train).type(defaults.tdtype), requires_grad=False)
    X_test_var = Variable( torch.from_numpy(X_test).type(defaults.tdtype), requires_grad=False)
    
    H = defaults.hidden_units
    model = shared.get_nn_model(D_IN,H,D_OUT, num_layers,'identity', weight_variance, bias_variance)
    burn_in = 200
    nthin = 50
    results_manager = shared.ResultsManager(Y_test, burn_in, nthin, True, True, True, True, True)
    num_samples = 1000000
    shared.nn_model_regression(X_train_var, Y_train_var, X_test_var, model, num_samples = num_samples, epsilon = 0.0005, beta = 0.1, leap_frog_iters=10, noise_variance=noise_variance, results_manager=results_manager  )
    return results_manager

def main():
    torch.manual_seed(2)

    X_train, Y_train, X_test, Y_test = getData()
    holdout_ll_gp, nrmse_gp, gp_model, gp_log_likelihoods = gp_experiments(X_train, Y_train, X_test, Y_test)
    #print('hold_out_gp ', hold_out_gp )
    
    baseline_ll = shared.baseline_log_density(Y_test)

    bias_variance = gp_model.kern.bias_variance.value[0]
    weight_variance = gp_model.kern.variance.value[0]
    noise_variance = gp_model.likelihood.variance.value[0]
    
    nn_results_manager = nn_experiments(X_train, Y_train, X_test, Y_test, weight_variance, bias_variance, noise_variance)

    results = {'holdout_ll_gp': holdout_ll_gp, 'nrmse_gp':nrmse_gp, 'baseline_ll':baseline_ll, 'log_densities_GP': gp_log_likelihoods, 'bias_variance': bias_variance, 'weight_variance': weight_variance, 'noise_variance': noise_variance}
    nn_results = nn_results_manager.to_dict()
    
    results.update(nn_results)
    
    pickle.dump( results, open(results_file_name,'wb') )
    
    plt.show()
    
if __name__ == '__main__':
    main()
