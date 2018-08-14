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
from IPython import embed
from scipy.stats import norm

import shared
import comparison_yacht
import pyhmc

import defaults

results = pickle.load( open(comparison_yacht.results_file_name,'rb' ) )

log_pred_densities_nn = results['log_densities_NN']
log_pred_densities_gp = results['log_densities_GP']
param_trackers = results['param_trackers']
pred_trackers = results['pred_trackers']

[fig, axes] = plt.subplots(1,1, figsize=(4.5,4.5))
#fig = plt.figure()
#ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1)
#ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1)
#plt.

shared_limits = [-5., 1.]
plot_range  = np.linspace( *shared_limits, 100)
axes.plot(plot_range, plot_range,'k')
axes.plot(log_pred_densities_gp, log_pred_densities_nn, 'b+')
axes.set_xlabel('Gaussian process log density')
axes.set_ylabel('Neural network log density')
axes.axis('square')
axes.set_xlim( shared_limits )
axes.set_ylim( shared_limits )
random_point_index = 9

plt.savefig('../figures/comparison_yacht_a.pdf')

X_train, Y_train, X_test, Y_test = comparison_yacht.getData()

[fig, axesB] = plt.subplots(1,1, figsize=(4.5,4.5))
axesB.hist(pred_trackers[:,random_point_index],20,normed=True)
test_point = np.atleast_2d(X_test[random_point_index,:])

#add gp density to plot.
gp_model = shared.get_gp_model(X_train,Y_train,input_dim=X_train.shape[1],depth=defaults.shared_depth)
gp_model.kern.bias_variance = results['bias_variance']
gp_model.kern.variance = results['weight_variance']
gp_model.likelihood.variance = results['noise_variance']

pred_mean, pred_var = gp_model.predict_f( test_point )
pred_std = np.sqrt(pred_var)
plot_range = [-3. , 2.]
x_points = np.linspace(*plot_range,100)
densities = norm.pdf( x_points, loc=pred_mean, scale = pred_std )
axesB.plot( x_points, densities.flatten() )
axesB.set_xlabel('Function value')
axesB.set_ylabel('Density')
plt.savefig('../figures/comparison_yacht_b.pdf')
embed()
plt.show()
