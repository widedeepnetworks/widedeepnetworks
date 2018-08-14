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

def process_mmd_experiment_all():
    
    long_text_names = {'identity': 'Identity width function', 'largest_first': 'Largest first width function', 'largest_last' : 'Largest last width function'}
    
    callibration_mmds = np.loadtxt('results/callibration_mmds.csv')
    mean_callibration = np.mean(callibration_mmds)
    
    [fig, axes] = plt.subplots(3,2, figsize=(12,12))
    
    for width_class_index in range(len(shared.valid_width_classes)-1,-1,-1):
        width_class = shared.valid_width_classes[width_class_index]
        results_file_name = mmd_experiment.results_file_stub + "_" +  width_class + ".pickle"
        results = pickle.load( open(results_file_name,'rb' ) )

        mmd_squareds = results['mmd_squareds']
        hidden_layer_numbers = results['hidden_layer_numbers']
        hidden_unit_numbers = results['hidden_unit_numbers']
        num_repeats = mmd_squareds.shape[2]
    
        mean_mmds = np.mean( mmd_squareds, axis = 2 )
        std_mmds = np.std( mmd_squareds, axis = 2 ) / np.sqrt(num_repeats)
    
        for hidden_layer_number, index in zip(hidden_layer_numbers,range(len(hidden_layer_numbers))):
            if hidden_layer_number==1:
                layer_string = ' hidden layer'
            else:
                layer_string = ' hidden layers'
            line_name = str(hidden_layer_number) + layer_string
            axes[width_class_index,0].errorbar( hidden_unit_numbers, mean_mmds[:,index], yerr = 2.*std_mmds[:,index], label = line_name)
            axes[width_class_index,1].errorbar( hidden_unit_numbers, mean_mmds[:,index], yerr = 2.*std_mmds[:,index], label = line_name)
            text_description = long_text_names[width_class]
            axes[width_class_index,0].text(20,0.015, text_description)

        axes[width_class_index,0].set_ylim([0.,0.02])
        axes[width_class_index,1].set_ylim([0.,0.005])
        axes[width_class_index,0].axhline(y=mean_callibration, color='r', linestyle='--')
        axes[width_class_index,1].axhline(y=mean_callibration, color='r', linestyle='--')
        axes[width_class_index,1].set_ylabel('MMD SQUARED(GP, NN)')
        axes[width_class_index,0].set_ylabel('MMD SQUARED(GP, NN)')
    
        if width_class_index == 0:
            axes[width_class_index,0].set_xlabel('Number of hidden units per layer')
            axes[width_class_index,1].set_xlabel('Number of hidden units per layer')        
        else:
            axes[width_class_index,0].set_xlabel('Smallest number of hidden units per layer')
            axes[width_class_index,1].set_xlabel('Smallest number of hidden units per layer')
        axes[width_class_index,0].set_xlim([0,50])
        axes[width_class_index,1].set_xlim([0,300])
    
    plt.legend()

    
    
    output_file_name = "../figures/mmds_all.pdf"
    plt.savefig(output_file_name,bbox_inches='tight')
    embed()
    plt.show()

if __name__ == '__main__':
    process_mmd_experiment_all()
