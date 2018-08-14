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

from IPython import embed

import sys
import pickle
import numpy as np
import torch
from torch.autograd import Variable

import gpflow

import mmd
import shared
import defaults

results_file_stub = 'results/mmd'

#@profile
def mmd_experiments(width_class):
    
    torch.manual_seed(3)
    torch.set_num_threads(1)
    rng = np.random.RandomState(3)
    
    input_dim = 4
    output_dim = 1
    num_data_points = 10
    extra_variance = 1e-3 #for numerical stability of multivariate normal.

    #long params
    hidden_unit_numbers = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,75,100,250,500]
    
    hidden_layer_numbers = [1,2,3]
    num_repeats = 100
    num_callibration_repeats = 50
    
    num_function_samples = 2000
    
    mmd_squareds = np.zeros( (len(hidden_unit_numbers), len(hidden_layer_numbers), num_repeats) )
    mmd_kern = gpflow.kernels.RBF(num_data_points , lengthscales = np.ones(num_data_points)*0.5 )
    
    X_input_np = rng.randn(num_data_points,input_dim)
    X_input_torch = Variable( torch.from_numpy(X_input_np).type(defaults.tdtype), requires_grad=False)

    #callibrate again rbf kernel.
    callibration_mmds = np.zeros( num_callibration_repeats) 
    char_length_scale = np.sqrt( 2. * input_dim )
    kernA = gpflow.kernels.RBF(input_dim = input_dim, lengthscales = np.ones(input_dim)*char_length_scale)
    kernB = gpflow.kernels.RBF(input_dim = input_dim, lengthscales = np.ones(input_dim)*char_length_scale*2.) 
    gramA = kernA.compute_K_symm(X_input_np )+ np.eye(num_data_points) * extra_variance
    gramB = kernB.compute_K_symm(X_input_np )+ np.eye(num_data_points) * extra_variance
    for repeat_index in range(num_callibration_repeats):
        rbf_samples_A = rng.multivariate_normal( mean = np.zeros(num_data_points), cov = gramA, size = num_function_samples )
        rbf_samples_B = rng.multivariate_normal( mean = np.zeros(num_data_points), cov = gramB, size = num_function_samples )
        callibration_mmds[repeat_index] = mmd.mmd( rbf_samples_A, rbf_samples_B, mmd_kern )
    np.savetxt('results/callibration_mmds.csv', callibration_mmds )

    for hidden_layer_index in range(len(hidden_layer_numbers)):
        print('hidden_layer_index',hidden_layer_index)
        hidden_layers = hidden_layer_numbers[hidden_layer_index]    
        kern = shared.get_kernel(input_dim,hidden_layers) 
        K = kern.compute_K_symm( X_input_np ) + np.eye(num_data_points) * extra_variance
        for hidden_unit_index in range(len(hidden_unit_numbers)):
            hidden_units = hidden_unit_numbers[hidden_unit_index]
            for repeat_index in range(num_repeats):
                nn_samples = np.zeros( (num_function_samples, num_data_points ))
                for sample_index in range(num_function_samples):
                    nn_model = shared.get_nn_model( input_dim, hidden_units, output_dim, hidden_layers, width_class)
                    nn_sample = nn_model( X_input_torch ).data.numpy().flatten()
                    noise_sample = rng.randn( *nn_sample.shape )*np.sqrt(extra_variance)# for consistency with GP whitening.
                    nn_samples[sample_index,:] = nn_sample + noise_sample   
                gp_samples = rng.multivariate_normal( mean = np.zeros(num_data_points), cov = K, size = num_function_samples )
                mmd_squareds[hidden_unit_index, hidden_layer_index, repeat_index] = mmd.mmd( gp_samples, nn_samples, mmd_kern )
    
    results_file_name = results_file_stub + "_" +  width_class + ".pickle"
    results = {'hidden_unit_numbers':hidden_unit_numbers, 'hidden_layer_numbers':hidden_layer_numbers, 'num_repeats':num_repeats, 'mmd_squareds':mmd_squareds }
    pickle.dump( results, open(results_file_name,'wb') )
    
if __name__ == '__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in shared.valid_width_classes:
        print("Usage: ", sys.argv[0], " <width_class>")
        sys.exit(-1)
    mmd_experiments(sys.argv[1])
