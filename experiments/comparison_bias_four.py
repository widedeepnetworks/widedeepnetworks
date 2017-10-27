import sys
sys.path.append('../')
import itertools
from matplotlib import pylab as plt
import numpy as np
import math
import time
from scipy.misc import logsumexp
import pickle

import torch
from torch.autograd import Variable

import gpflow
import cProfile

from ess_torch import ESS
from hmc_torch import HMC
from RecursiveKernel import DeepArcCosine
from layers_torch import GaussLinearStandardized, ScaledRelu

from IPython import embed

import shared
import defaults

results_file_name = 'results/comparison_four.pickle'

def nn_model_regression(model, X_train,Y_train, X_test, Y_test ):
    #run sampler
    #at the same time mantain online estimated of mean
    #and marginal variance
    #this stops us having to store large numbers of samples.
    
    num_samples = 100000
    burn_in = 50
    n_test = X_test.size()[0]
    
    nlls = np.zeros( (num_samples, X_test.size()[0] ))
    num_params = len( [ elem for elem in model.parameters() ] )
    param_trackers = np.zeros( (num_samples, num_params) )
    pred_trackers = np.zeros( (num_samples, n_test) )
    
    criterion = torch.nn.MSELoss(size_average=False)
    
    sampler = HMC(model.parameters(), np.random.RandomState(1), epsilon = 0.05, beta = 1, leap_frog_iters = 10 )
    samplerB = ESS(model.parameters(), np.random.RandomState(2) )
    
    energies = np.zeros(num_samples)
    
    start_time = time.time()
    for sample_index in range(num_samples):
        def closure():
            sampler.zero_grad()
            pred = model( X_train )
            energy = 0.5*criterion(pred, Y_train )/ defaults.noise_variance
            energy.backward()
            return energy
        sampler.step( closure )
        
        def closureB():
            pred = model( X_train )
            energy = 0.5*criterion(pred, Y_train )/ defaults.noise_variance 
            return energy    
        energies[sample_index] = samplerB.step( closureB )

        for param, param_index in zip(model.parameters(), range(num_params)):
            rank = len(param.size())
            index = [0]*rank
            if rank==1:
                param_trackers[sample_index, param_index ] = param.data[index[0]]
            else:
                param_trackers[sample_index, param_index ] = param.data[index[0],index[1]]
                
            
        #get prediction
        pred_test = model(X_test)
        pred_trackers[sample_index,:] = pred_test.data.numpy().flatten()
        pred_energies = 0.5 * torch.pow( pred_test-Y_test, 2 ) / defaults.noise_variance + 0.5* np.log( 2. * np.pi * defaults.noise_variance )
        nlls[sample_index,:] = pred_energies.data.numpy().flatten() 

    end_time = time.time()
    print('Total time' , end_time - start_time )
    print('iterations per second', num_samples*1./(end_time - start_time))
    
    valid_holdout = nlls[burn_in:,:] 
    #log p(y* | x* ) \roughly \log \sum \exp (- valid_holdout)  -  \log(num_samples)
    #with summation taken over the sample axis. 
    log_pred_densities = (logsumexp(-valid_holdout,axis=0) - np.log(valid_holdout.shape[0]))

    results = {'log_densities_NN' : log_pred_densities , 'pred_trackers':pred_trackers , 'param_trackers':param_trackers, 'energies' : energies }

    return results        

def gp_experiments(X_train, Y_train, X_test, Y_test, num_layers):
    
    gp_model = shared.get_gp_model(X_train,Y_train,input_dim=X_train.shape[1],depth=num_layers)
    individual_densities  = gp_model.predict_density(X_test,Y_test)
    print('individual GP densities ', individual_densities )
    holdout = individual_densities.mean()
    return individual_densities
    

def nn_experiments(X_train, Y_train, X_test, Y_test, H, num_layers):

    D_IN = X_train.size()[1]
    D_OUT = 1
    
    model = shared.get_nn_model(D_IN, H, D_OUT, num_layers)
    #model.cuda()

    results = nn_model_regression(model, X_train,Y_train, X_test, Y_test )
    return results

def main():
    torch.manual_seed(2)
    torch.set_num_threads(1)

    num_dim = 4
    num_train = 10
    num_test = num_train

    #randomly generate X and Y
    rng = np.random.RandomState(3)
    X_train = rng.randn(num_train,num_dim)
    X_test = rng.randn(num_test,num_dim)
    #create a random network.
    H = defaults.hidden_units
    num_layers = defaults.shared_depth
    D_IN = X_train.shape[1]
    D_OUT = 1
    model = shared.get_nn_model(D_IN,H,D_OUT, num_layers)  
    
    #create torch version of inputs
    X_train_t = Variable( torch.from_numpy(X_train).type(defaults.tdtype), requires_grad=False)
    X_test_t = Variable( torch.from_numpy(X_test).type(defaults.tdtype), requires_grad=False)
    
    #create train and test data.
    f_train = model(X_train_t)
    f_test = model(X_test_t)

    #corrupt with correct level of noise.
    Y_train = (f_train + Variable( torch.randn(*f_train.size()).type(defaults.tdtype).mul( math.sqrt(defaults.noise_variance) ) , requires_grad = False ) ).detach()
    Y_test = (f_test + Variable( torch.randn(*f_test.size()).type(defaults.tdtype).mul( math.sqrt(defaults.noise_variance) ), requires_grad = False )).detach()

    densities_gp = gp_experiments(X_train, Y_train.data.numpy(), X_test, Y_test.data.numpy(), num_layers )
    #print('hold_out_gp ', hold_out_gp )
    
    results = nn_experiments(X_train_t, Y_train, X_test_t, Y_test, H, num_layers)
    results['log_densities_GP'] = densities_gp
    results['X_train']=X_train
    results['X_test']=X_test
    results['Y_train']=Y_train
    results['Y_test']=Y_train

    pickle.dump( results, open(results_file_name,'wb') )
    embed()
    plt.show()
    
if __name__ == '__main__':
    main()
