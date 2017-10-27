import pickle
import numpy as np
from IPython import embed
from matplotlib import pylab as plt

import mmd_experiment


results = pickle.load( open(mmd_experiment.results_file_name,'rb' ) )

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
    #plt.plot( hidden_unit_numbers, np.sqrt(mean_mmds[:,index]), label = line_name)
plt.xlabel('Number of hidden units per layer')
plt.xlim([0,60])
plt.ylabel('MMD SQUARED(GP, NN)')
plt.ylim([0.,0.02])
plt.axhline(y=mean_callibration, color='r', linestyle='--')
plt.legend()
plt.savefig('../figures/mmds.pdf')
embed()
plt.show()
