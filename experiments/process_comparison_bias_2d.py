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
import comparison_bias_2d

from IPython import embed

import defaults

results = pickle.load( open(comparison_bias_2d.results_file_name,'rb' ) )

gp_mean = results['gp_mean']
gp_var = results['gp_var']

nn_mean = results['nn_mean']
nn_var = results['nn_var']

mean_limit = comparison_bias_2d.get_plot_min_and_max_mean( gp_mean, nn_mean )
sqrt_limit = comparison_bias_2d.get_plot_max_sqrt( np.sqrt(gp_var), np.sqrt(nn_var) ) 

fig, axes = plt.subplots( 2, 2 )    
comparison_bias_2d.reshape_and_plot( fig, axes[0,1], nn_mean, -mean_limit, mean_limit )
comparison_bias_2d.reshape_and_plot( fig, axes[0,0], gp_mean, -mean_limit, mean_limit )
comparison_bias_2d.reshape_and_plot( fig, axes[1,1], np.sqrt(nn_var), 0. , sqrt_limit )
comparison_bias_2d.reshape_and_plot( fig, axes[1,0], np.sqrt(gp_var), 0. , sqrt_limit )
axes[0,0].set_ylabel('Mean')
axes[1,0].set_ylabel('Standard deviation')
axes[0,0].set_title('Gaussian process')
axes[0,1].set_title('Neural network')
plt.savefig('../figures/comparison_bias_2d.pdf',bbox_inches='tight')

fig, axes = plt.subplots(1,2, figsize=(6,3) )
comparison_bias_2d.reshape_and_plot( fig, axes[0], nn_mean, -mean_limit, mean_limit, False )
axes[0].set_title('Gaussian process')
axes[1].set_title(defaults.deep_net_name)
im  =comparison_bias_2d.reshape_and_plot( fig, axes[1], gp_mean, -mean_limit, mean_limit, False )
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax)

print('mean_abs_diff ',np.mean(np.abs(nn_mean-gp_mean)))

plt.savefig('../figures/comparison_bias_2d_means.pdf',bbox_inches='tight')

embed()
plt.show()
