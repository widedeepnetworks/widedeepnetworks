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

import pyhmc

import defaults
import comparison_snelson

results = pickle.load( open(comparison_snelson.results_file_name,'rb' ) )

num_prior_samples = 8
fig, axes = plt.subplots( 2, 1, figsize = (15,12) ) #first row is prior. second row is posterior.
comparison_snelson.plotPredictions( axes[0], 'g', 'Gaussian process', results['gp_mean'], results['gp_var'], offset=-0.75)
comparison_snelson.plotPredictions( axes[1], 'g', defaults.deep_net_name , results['nn_mean'], results['nn_var'], offset=-1.)
plt.savefig('../figures/comparison_snelson.pdf',bbox_inches='tight')
embed()
plt.show()
