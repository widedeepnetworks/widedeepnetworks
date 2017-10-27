
import pickle
import numpy as np
from scipy.stats import norm

from matplotlib import pylab as plt
from IPython import embed

import shared
import defaults

import comparison_bias_four

results = pickle.load( open(comparison_bias_four.results_file_name,'rb' ) )

log_pred_densities_nn = results['log_densities_NN']
log_pred_densities_gp = results['log_densities_GP']
param_trackers = results['param_trackers']
pred_trackers = results['pred_trackers']
energies = results['energies']
X_train = results['X_train']
Y_train = results['Y_train'].data.numpy()
X_test = results['X_test']
Y_test = results['Y_test'].data.numpy()

[fig, axes] = plt.subplots(1,1, figsize=(4.5,4.5))
#fig = plt.figure()
#ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1)
#ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1)
#plt.

shared_limits = [-8., 0.]
plot_range  = np.linspace( *shared_limits, 100)
axes.plot(plot_range, plot_range,'k')
axes.plot(log_pred_densities_gp, log_pred_densities_nn, 'bo')
axes.set_xlabel('Gaussian process log density')
axes.set_ylabel('Neural network log density')
axes.axis('square')
axes.set_xlim( shared_limits )
axes.set_ylim( shared_limits )
random_point_index = 9

plt.savefig('../figures/comparison_bias_four_a.pdf')

[fig, axesB] = plt.subplots(1,1, figsize=(4.5,4.5))
axesB.hist(pred_trackers[50:,random_point_index],30,normed=True)
test_point = np.atleast_2d(X_test[random_point_index,:])

#add gp density to plot.
gp_model = shared.get_gp_model(X_train,Y_train,input_dim=X_train.shape[1],depth=defaults.shared_depth)
pred_mean, pred_var = gp_model.predict_f( test_point )
pred_std = np.sqrt(pred_var)
plot_range = [-3. , 2.]
x_points = np.linspace(*plot_range,100)
densities = norm.pdf( x_points, loc=pred_mean, scale = pred_std )
axesB.plot( x_points, densities.flatten() )
axesB.set_xlabel('Function value')
axesB.set_ylabel('Density')
plt.savefig('../figures/comparison_bias_four_b.pdf')
embed()
plt.show()
