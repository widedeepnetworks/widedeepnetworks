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

import pickle
import numpy as np
from matplotlib import pylab as plt
import comparison_smooth_xor
import defaults
from IPython import embed

results = pickle.load( open(comparison_smooth_xor.results_file_name,'rb' ) )

gp_mean = results['gp_mean']
gp_var = results['gp_var']

nn_mean = results['nn_mean']
nn_var = results['nn_var']

def split_array(array_in):
    length = array_in.shape[0]
    assert(length%2==0)
    return array_in[0:length//2], array_in[length//2:]

nn_meanA, nn_meanB = split_array(nn_mean)
nn_varA, nn_varB = split_array(nn_var)
gp_meanA, gp_meanB = split_array(gp_mean)
gp_varA, gp_varB = split_array(gp_var)

betas = comparison_smooth_xor.get_test_points()[-1]

#fig = plt.figure(figsize = (10,16) )

fig = plt.figure(figsize = (9,14) )
grid = plt.GridSpec(4, 2)#, hspace=0.2, wspace=0.2)

#fig, ax = plt.subplots(1,1)
axA   = fig.add_subplot( grid[:2,:] )
comparison_smooth_xor.plot_smooth_xor_data_2d(axA)

#fig, axes = plt.subplots(2,1) 
axB = fig.add_subplot( grid[2,:] )
axC = fig.add_subplot( grid[3,:] )
axB.text(-0.5,3.,'Cross-section 1',fontsize=12)
axC.text(-0.5,3.,'Cross-section 2',fontsize=12)

comparison_smooth_xor.plotPredictions( axC, 'g', 'Gaussian process', gp_meanA, gp_varA, betas)
comparison_smooth_xor.plotPredictions( axC, 'b', 'Bayesian deep network', nn_meanA, nn_varA, betas)
handleA = comparison_smooth_xor.plotPredictions( axB, 'g', 'Gaussian process', gp_meanB, gp_varB, betas)
handleB = comparison_smooth_xor.plotPredictions( axB, 'b', 'Bayesian deep network', nn_meanB, nn_varB, betas)

axC.legend([handleA[0],handleB[0]], ['Gaussian process','Bayesian deep network'], loc='lower right' )
plt.savefig('../figures/comparison_smooth_xor.pdf',bbox_inches='tight')

embed()
#plt.show()
