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

from matplotlib import pylab as plt
import sys
sys.path.append('../')
import csv
import pickle

import tensorflow as tf
import numpy as np
import nn
import defaults

import torch
from torch.autograd import Variable

import gpflow

from RecursiveKernel import DeepArcCosine
import shared

from IPython import embed

#Use Radford Neal result on GP convergence to confirm code is working.

snelson_offset = 3. - 6.
snelson_scaling = 1.
np_float = np.float32
tf_float = tf.float32
results_file_name = 'results/comparison_snelson.pickle'


def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )

def plotPredictions( ax, color, title, predMean, predVar, offset ):
  X,Y = getTrainingHoldOutData()
  xtest = getTestData()
  ax.plot( X, Y, 'ro' )
  ax.plot( xtest, predMean, color='b' )
  ax.plot( xtest, predMean + 2.*np.sqrt(predVar),'g--' )
  ax.plot( xtest, predMean - 2.*np.sqrt(predVar),'g--' )  
  ax.text( offset, 3, title, fontsize=12)
  #ax.set_title(title)
  ax.set_xlabel('Input')
  ax.set_ylabel('Output')
  #ax.text( 6., 2.5, label )
  
  standardPlotLimits(ax)


def getTrainingHoldOutData(isTrain=True):
  X = (readCsvFile( 'datasets/train_inputs' ) + snelson_offset)/snelson_scaling
  Y = readCsvFile( 'datasets/train_outputs' ) 
  #embed()
  trainIndeces = []
  testIndeces = []
  nPoints = X.shape[0]
  skip = 2
  for index in range(nPoints):
    if ( (index%skip) == 0):
      trainIndeces.append( index )
    else:
      testIndeces.append( index )
    
  if isTrain:    
    return X[trainIndeces,:],Y[trainIndeces,:]
  else:
    return X[testIndeces,:],Y[testIndeces,:]        

def getTestData():
  xtest = np.linspace( -3.5 + snelson_offset, 9.5+snelson_offset, 1000, endpoint=True ) / snelson_scaling
  return np.atleast_2d(xtest).T

def standardPlotLimits(ax):
  ax.set_xlim( np.array([-3+snelson_offset, 9.+snelson_offset ])/snelson_scaling )
  ax.set_ylim( [-4.0,4.0 ] )

def getRunGPModel():
  kernel = DeepArcCosine(input_dim=1, num_steps=3, variance = 0.8, bias_variance = 0.2)
  X,Y = getTrainingHoldOutData()
  model = gpflow.gpr.GPR( X=X, Y=Y, kern=kernel)
  model.optimize()
  return model

def gpPriorSamples(ax,model,num_samples):
  X_test = getTestData()
  #K = ArcCosine(input_dim=1).compute_K_symm(X_test)
  K = model.kern.compute_K_symm(X_test)
  K = K + np.eye(K.shape[0])*1e-4
  ax.plot(X_test, np.random.multivariate_normal(np.zeros(K.shape[0]), K, num_samples).T)
  standardPlotLimits(ax)  

def nn_experiments(weight_variance,bias_variance,noise_variance):#, fig, axes):
  H = defaults.hidden_units
  num_layers = defaults.shared_depth
  X,Y = getTrainingHoldOutData()
  X_var = Variable( torch.from_numpy(X).type(defaults.tdtype), requires_grad=False)
  Y_var = Variable( torch.from_numpy(Y).type(defaults.tdtype), requires_grad=False)
  X_test = getTestData()
  grid_var = Variable( torch.from_numpy(X_test).type(defaults.tdtype), requires_grad=False)
  
  D_IN = 1
  D_OUT = 1
  H = defaults.hidden_units
  nthin = 50 
  model = shared.get_nn_model(D_IN,H,D_OUT, num_layers,'identity', weight_variance, bias_variance)
  #model.cuda()
  
  nn_sample = shared.draw_sample_from_nn_prior(model, grid_var)
  
  pred_mean, pred_var, pred_trackers, param_trackers, energies, hmc_acceptance_rate = shared.nn_model_regression(X_var,Y_var,grid_var, model, num_samples = 1000000, burn_in = 50, epsilon = 0.0005, beta = 1, leap_frog_iters = 10, noise_variance=noise_variance, return_trackers = True, nthin=nthin  )
  return pred_mean, pred_var, pred_trackers, param_trackers, energies, hmc_acceptance_rate, nthin

def main():
  tf.set_random_seed(1)
  #Run gp regression.
  gpModel = getRunGPModel()
  xtest = getTestData()
  gp_mean, gp_var = gpModel.predict_f(xtest) 
  #Use gpModel hypers for the neural network. 
  bias_variance = gpModel.kern.bias_variance.value[0]
  weight_variance = gpModel.kern.variance.value[0]
  noise_variance = gpModel.likelihood.variance.value[0]
  #Run regression with neural network.
  nn_mean, nn_var, pred_trackers, param_trackers, energies, hmc_acceptance_rate, nthin = nn_experiments(weight_variance, bias_variance, noise_variance)
  
  results = { 'nn_mean' : nn_mean.flatten() , 'nn_var' : nn_var.flatten(), 'gp_mean' : gp_mean.flatten(), 'gp_var' : gp_var.flatten(), 'pred_trackers' : pred_trackers, 'param_trackers' : param_trackers, 'energies' : energies, 'hmc_acceptance_rate':hmc_acceptance_rate, 'nthin':nthin}
  
  pickle.dump( results, open(results_file_name,'wb') )

if __name__ == '__main__':
  main()
