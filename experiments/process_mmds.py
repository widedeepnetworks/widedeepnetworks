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
import pickle
import numpy as np
from IPython import embed
from matplotlib import pylab as plt

import shared
import mmd_experiment

def process_mmd_experiment(width_class):
    results_file_name = mmd_experiment.results_file_stub + "_" +  width_class + ".pickle"
    results = pickle.load( open(results_file_name,'rb' ) )
    
    callibration_mmds = np.loadtxt('results/callibration_mmds.csv')
    mean_callibration = np.mean(callibration_mmds)
    
    mmd_squareds = results['mmd_squareds']
    hidden_layer_numbers = results['hidden_layer_numbers']
    hidden_unit_numbers = results['hidden_unit_numbers']
    num_repeats = mmd_squareds.shape[2]
    
    mean_mmds = np.mean( mmd_squareds, axis = 2 )
    std_mmds = np.std( mmd_squareds, axis = 2 ) / np.sqrt(num_repeats)
    
    plt.figure()
    
    for hidden_layer_number, index in zip(hidden_layer_numbers,range(len(hidden_layer_numbers))):
        if hidden_layer_number==1:
            layer_string = ' hidden layer'
        else:
            layer_string = ' hidden layers'
        line_name = str(hidden_layer_number) + layer_string
        plt.errorbar( hidden_unit_numbers, mean_mmds[:,index], yerr = 2.*std_mmds[:,index], label = line_name)
    plt.xlabel('Number of hidden units per layer')
    plt.xlim([0,60])
    plt.ylabel('MMD SQUARED(GP, NN)')
    plt.ylim([0.,0.02])
    plt.axhline(y=mean_callibration, color='r', linestyle='--')
    plt.legend()
    output_file_name = "../figures/mmds_" + width_class + ".pdf"
    plt.savefig(output_file_name)
    embed()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in shared.valid_width_classes:
        print("Usage: ", sys.argv[0], " <width_class>")
        sys.exit(-1)
    process_mmd_experiment(sys.argv[1])
