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
import time
import numpy as np
from scipy.misc import logsumexp
from scipy.stats import norm


from IPython import embed

import torch
from torch.autograd import Variable
from tqdm import tqdm
import gpflow

from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu
from ess_torch import ESS
from hmc_torch import HMC

import defaults

valid_width_classes = ['identity','largest_first','largest_last']

class ResultsManager(object):
    def __init__(self, Y_test, num_burn_in, nthin, track_params, track_preds, track_moments, track_lls, track_energy):
        self.Y_test = Y_test
        self.num_burn_in = num_burn_in
        self.nthin = nthin
        self.track_params = track_params
        self.track_preds = track_preds
        self.track_moments = track_moments
        self.track_lls = track_lls
        self.track_energy = track_energy
    
    def get_array_size(self, num_samples, nthin, burn_in):
        return (num_samples-burn_in-1)//nthin
    
    def initialize(self, num_samples, model, test_size, noise_variance):
        self.num_samples = num_samples
        self.test_size = test_size
        self.num_points = 0
        self.noise_variance = noise_variance
        
        self.thinned_array_size = self.get_array_size(num_samples, self.nthin, self.num_burn_in)
        
        if self.track_params:
            self.num_params = len( [ elem for elem in model.parameters() ] )
            self.param_trackers = np.zeros( (self.thinned_array_size, self.num_params) )        
        if self.track_preds:
            self.pred_trackers = np.zeros( (self.thinned_array_size, self.test_size ) )
        if self.track_moments:
            self.online_mean = np.zeros(self.test_size)
            self.online_squares = np.zeros(self.test_size)
        if self.track_lls:
            self.lls = np.zeros( (self.thinned_array_size, self.test_size ))
        if self.track_energy:
            self.energies = np.zeros(num_samples)
    
    def update(self, sample_index, pred, energy, model ):
        if (sample_index > self.num_burn_in) and (sample_index % self.nthin == 0) :    
            #do updates for thinned quantities
            self.num_points+=1

            if self.track_moments:
                delta = pred-self.online_mean
                self.online_mean = self.online_mean + delta/self.num_points
                delta2 = pred-self.online_mean
                self.online_squares = self.online_squares + delta * delta2    
            if self.track_lls:
                pred_lls = -0.5 * np.square( pred-self.Y_test.flatten() ) / self.noise_variance - 0.5* np.log( 2. * np.pi * self.noise_variance )
                self.lls[self.num_points-1,:] = pred_lls
                
            if self.track_params:
                for param, param_index in zip(model.parameters(), range(self.num_params)):
                    rank = len(param.size())
                    index = [0]*rank
                    if rank==1:
                        self.param_trackers[self.num_points-1, param_index ] = param.data[index[0]]
                    else:
                        self.param_trackers[self.num_points-1, param_index ] = param.data[index[0],index[1]]
        
            if self.track_preds:
                self.pred_trackers[self.num_points-1,:] = pred            
        
        if self.track_energy:
            self.energies[sample_index] = energy
    
    def finalize(self, hmc_acceptance_rate):
        self.hmc_acceptance_rate = hmc_acceptance_rate
        
        if self.track_lls:
            self.log_pred_densities = logsumexp(self.lls,axis=0) - np.log(self.lls.shape[0])
            self.holdout_ll = self.log_pred_densities.mean()
        if self.track_moments:
            self.var = self.online_squares / (self.num_points + 1)
            if self.Y_test is not None:
                self.norm_rmse = nrmse(self.Y_test, np.atleast_2d(self.online_mean).T )
            else:
                self.norm_rmse = None
            
    def to_dict(self):
        output_dict = {}
        
        if self.track_energy:
            output_dict['energies'] = self.energies
        if self.track_moments:
            output_dict['nn_mean'] = self.online_mean
            output_dict['nn_var'] = self.var
            output_dict['nrmse_nn'] = self.norm_rmse
        if self.track_lls:
            output_dict['log_densities_NN'] = self.log_pred_densities
            output_dict['holdout_ll_nn'] = self.holdout_ll
        output_dict['hmc_acceptance_rate'] = self.hmc_acceptance_rate
        if self.track_params:
            output_dict['param_trackers'] = self.param_trackers
        if self.track_preds:
            output_dict['pred_trackers'] = self.pred_trackers
        
        return output_dict 

def nrmse(Y_true, Y_pred, mean_train = 0. ):
    assert(Y_true.shape[1] ==  1 ) #assume 1D targets.
    assert( Y_true.shape == Y_pred.shape )
    #Normalised route mean squared error.
    #The normalization is the RMSE for a regressor
    #that simply predicts the mean of the training output data.
    #Therefore 1 is as good as ignoring the input covariates.
    #Greater than 1 is worse and less than 1 is good with 0 perfect.
    RMSE = np.sqrt( np.mean( np.square(Y_true - Y_pred) ) )
    normaliser =  np.sqrt( np.mean( np.square(Y_true - mean_train) ) )
    return RMSE/normaliser

def baseline_log_density(Y_test, train_mean=0., train_std=1.):
    #Evaluate the log density associated with 
    #a base line model that uses a single normal density with
    #the mean and std of the training data.
    log_pdfs = norm.logpdf(Y_test, loc=train_mean, scale=train_std)
    return log_pdfs.mean()
            
def get_kernel(input_dim,depth):
    kern = DeepArcCosine(input_dim = input_dim, num_steps = depth, variance = defaults.weight_variance, bias_variance = defaults.bias_variance )
    return kern

def get_gp_model(X,Y, input_dim, depth):
    kern = get_kernel( input_dim, depth )
    model = gpflow.gpr.GPR(X,Y, kern = kern )
    model.likelihood.variance= defaults.noise_variance
    return model

def get_intermediate_layers(H, num_layers, bias, width_class, weight_variance, bias_variance):
    intermediate_layers = []
    for layer_index in range(num_layers-1):
        if width_class == 'identity':
            incoming_hidden_size = H
            outgoing_hidden_size = H
        elif width_class == 'largest_last':
            incoming_hidden_size = (layer_index+1)*H
            outgoing_hidden_size = (layer_index+2)*H
        elif width_class == 'largest_first':
            incoming_hidden_size = (num_layers-layer_index)*H
            outgoing_hidden_size = (num_layers-layer_index-1)*H
        else:
            raise NotImplementedError
        intermediate_layers+= [GaussLinearStandardized(incoming_hidden_size, outgoing_hidden_size, bias=bias, raw_weight_variance = weight_variance, raw_bias_variance = bias_variance),
        ScaledRelu() ]
    return intermediate_layers
    
def get_nn_model(D_IN,H,D_OUT, num_layers, width_class='identity', weight_variance=None, bias_variance=None):
    if weight_variance is None:
        weight_variance = defaults.weight_variance
    if bias_variance is None:
        bias_variance = defaults.bias_variance
    assert(width_class in valid_width_classes )
    intermediate_layers = get_intermediate_layers(H, num_layers, True, width_class, weight_variance, bias_variance)
    if width_class == 'identity':
        first_hidden_size = H
        last_hidden_size = H
    elif width_class == 'largest_last':
        first_hidden_size = H
        last_hidden_size = num_layers * H
    elif width_class == 'largest_first':
        first_hidden_size = num_layers*H 
        last_hidden_size = H
    else:
        raise NotImplementedError
    #embed()
    model = torch.nn.Sequential(
        GaussLinearStandardized(D_IN, first_hidden_size, bias=True, raw_weight_variance = weight_variance, raw_bias_variance = bias_variance),
        ScaledRelu(),
        *intermediate_layers,
        GaussLinearStandardized(last_hidden_size, D_OUT, bias=True, raw_weight_variance = weight_variance, raw_bias_variance = bias_variance)
    )
    return model 

def get_middel_nn_model(H, D_OUT, num_layers, weight_variance=None, bias_variance=None):
    if weight_variance is None:
        weight_variance = defaults.weight_variance
    if bias_variance is None:
        bias_variance = defaults.bias_variance
    #for use in the HSIC experiments.
    #width_class is assumed to be identity.
    width_class = 'identity'
    last_hidden_size = H
    #D_IN is assumed to equal H.
    intermediate_layers = get_intermediate_layers(H, num_layers, True, width_class, weight_variance, bias_variance)
    model = torch.nn.Sequential(
        ScaledRelu(),
        *intermediate_layers,
        GaussLinearStandardized(last_hidden_size, D_OUT, bias=True, raw_weight_variance = weight_variance, raw_bias_variance = bias_variance)
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

def nn_model_regression(X,Y, test_X, model, num_samples, epsilon, beta, leap_frog_iters, results_manager, noise_variance = None):
    if noise_variance is None:
        noise_variance = defaults.noise_variance    
    test_size = test_X.size()[0]

    criterion = torch.nn.MSELoss(size_average=False)
    
    sampler = HMC(model.parameters(), np.random.RandomState(1), epsilon = epsilon , beta = beta, leap_frog_iters = leap_frog_iters )
    
    results_manager.initialize(num_samples, model, test_size, noise_variance)
    
    start_time = time.time()
    for sample_index in tqdm(range(num_samples)):
        def closure():
            sampler.zero_grad()
            pred = model( X )
            energy = 0.5*criterion(pred, Y )/ noise_variance
            energy.backward()
            return energy
        energy = sampler.step( closure )

        #get prediction
        pred = model(test_X).data.cpu().numpy().flatten()        
        results_manager.update(sample_index, pred, energy, model)

    end_time = time.time()
    print('Total time' , end_time - start_time )
    print('iterations per second', num_samples*1./(end_time - start_time))
    
    results_manager.finalize(sampler.acceptance_rate())

def xor_data_np():
    X = np.array( [[1. , 1.] , [1.,-1], [-1.,1.], [-1., -1] ], np.float32 )
    Y = np.array( [ [-1.], [1.], [1.],  [-1.] ], np.float32 )
    return X, Y
