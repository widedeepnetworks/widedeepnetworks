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

import numpy as np
import pickle
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
#from matplotlib2tikz import save as save_tikz

import defaults
import comparison_bias

results = pickle.load( open(comparison_bias.results_file_name,'rb' ) )
X,Y,grid_points = comparison_bias.get_XY_gridpoints()
fig, axes = plt.subplots( 1, 2, figsize = (15,3) )
#gs1 = gridspec.GridSpec(1,2)
#gs1.update(wspace=0.1, hspace=0.05)
#axes = [ plt.subplot( elem ) for elem in gs1 ]
#plt.setp(axes[0].get_xticklabels(), visible=False)
comparison_bias.plot_mean_and_stds( X,Y, axes[0], grid_points.flatten(), results['gp_mean'], np.sqrt(results['gp_var']), 'Gaussian process' )
comparison_bias.plot_mean_and_stds( X,Y, axes[1], grid_points.flatten(), results['nn_mean'], np.sqrt(results['nn_var']), defaults.deep_net_name )
plt.savefig('../figures/comparison_bias.pdf',bbox_inches='tight')
plt.show()
