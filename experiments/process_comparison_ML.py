import numpy as np

from IPython import embed

from scipy.stats import gaussian_kde
from matplotlib import pylab as plt

results_dir = 'results/'
weights_file_name = results_dir + 'log_weights.csv'
gp_file_name = results_dir + 'gp_ml.csv'

log_weights = np.loadtxt(weights_file_name)
kernel = gaussian_kde(log_weights)
plot_range = [-12.5, -10.5]
x_points = np.linspace( *plot_range, 100 )
gp_ml = np.loadtxt(gp_file_name)

fig, ax = plt.subplots(1,figsize=(3.5,3))
ax.plot( x_points, np.maximum(kernel(x_points),np.zeros_like(x_points)) )
ax.set_xlim( *plot_range )
ax.set_ylim( [0,2.2] )
ax.axvline(x=gp_ml,color='r')
ax.set_xlabel('Log marginal likelihood')
#ax.set_ylabel('Density')
#ax.yaxis.tick_right()
ax.get_yaxis().set_visible(False)
plt.savefig('../figures/importance_sampling.pdf',bbox_inches='tight')
embed()
plt.show()
