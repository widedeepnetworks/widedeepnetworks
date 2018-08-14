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

from IPython import embed

def nn_model_ml(X,Y, model_constructor ):
    
    criterion = torch.nn.MSELoss(size_average=False)
    
    #good settings which give error of 1 nat with 1 extra layer.
    #num_temperatures = 100000
    #num_repeats = 10
    
    num_temperatures = 100000
    num_repeats = 100
    
    num_data = Y.size()[0]
    
    beta_sequence = np.linspace(0., 1. , num_temperatures, endpoint = True)
    
    log_weights = np.zeros(num_repeats)    
    rng = np.random.RandomState(1)
    
    for repeat_index in range(num_repeats):
        model = model_constructor()
        sampler = HMC( model.parameters(), epsilon = 0.01, rng = rng,  beta = None, leap_frog_iters = 1, include_integrator=True)
        for temperature_index in range(num_temperatures):
            def log_energy_final():
                pred = model(X)
                energy = 0.5*criterion(pred, Y )/ defaults.noise_variance + 0.5*num_data* np.log( 2. * np.pi * defaults.noise_variance ) 
                return energy
            def closure():
                sampler.zero_grad()
                beta = Variable(torch.FloatTensor([beta_sequence[temperature_index]]), requires_grad=False)
                energy_f = log_energy_final()
                energy = beta * energy_f
                energy.backward()
                return energy
            if temperature_index==0:
                sampler.weight_accumulator.record_start_energy(closure())
            else:
                sampler.step(closure)
        log_weights[repeat_index] = sampler.weight_accumulator.log_weight.data[0]
    
    embed()

def gp_experiments(X,Y):
    
    gp_model = shared.get_gp_model(X,Y,input_dim=2,depth=defaults.shared_depth)
    return gp_model.compute_log_likelihood()
    
def nn_experiments(X,Y):
    H = defaults.hidden_units
    num_layers = defaults.shared_depth
    D_IN = X.shape[1]
    D_OUT = 1
    X_var = Variable( torch.from_numpy(X).type(defaults.tdtype), requires_grad=False)
    Y_var = Variable( torch.from_numpy(Y).type(defaults.tdtype), requires_grad=False)
    
    def model_constructor():
        return shared.get_nn_model(D_IN,H,D_OUT, num_layers)
    
    nn_model_ml(X_var,Y_var, model_constructor)

def main():
    torch.manual_seed(2)
    fig, axes = plt.subplots( 3, 2 )

    X, Y = shared.xor_data_np()
    
    gp_ml = gp_experiments(X,Y )
    print('gp_ml ', gp_ml)
    nn_experiments(X,Y)

    embed()
    plt.show()
    
if __name__ == '__main__':
    main()
