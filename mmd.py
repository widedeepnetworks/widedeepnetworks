from IPython import embed

import numpy as np
from scipy.stats import multivariate_normal

import gpflow

def mmd(datasetA, datasetB, kernel ):
    #here we use a biased but consistent estimator 
    #corresponding to equation 5 of 
    #Gretton et al 2012 JMLR.
    #The bias will should be negligible with the large number of samples
    #we use.
    KAA = kernel.compute_K_symm(datasetA)
    KAA_corrected = KAA - np.diag(np.diag(KAA))
    KBB = kernel.compute_K_symm(datasetB)
    KBB_corrected = KBB - np.diag(np.diag(KBB))
    KAB = kernel.compute_K(datasetA,datasetB)
    M = KAA.shape[0]
    return np.sum( KAA_corrected/M/(M-1) + KBB_corrected/M/(M-1) - 2*KAB/M/M)

def test_mmd():
    from matplotlib import pylab as plt
    np.random.seed(1)

    num_dim = 10
    kern = gpflow.kernels.RBF(num_dim, lengthscales = np.ones(num_dim) )
    #embed()
    #kern.lengscales = np.ones( num_dim ) 
    
    meanA = np.zeros(num_dim)
    covA = np.eye(num_dim)
    covB = covA
    
    num_test = 30
    num_repeats = 20

    num_samples = 2000
    
    betas = np.linspace(0., 1. , num_test ) 
    mmd_squareds = np.zeros(( num_test, num_repeats ))
    
    for repeat_index in range(num_repeats):
        for beta, index in zip(betas,range(len(betas))):
            meanB = np.ones_like( meanA )
            meanB = beta*meanB/ np.sqrt( np.sum( meanB ** 2 ) )
            samplesA = multivariate_normal.rvs( size = num_samples, mean = meanA, cov=covA )
            samplesB = multivariate_normal.rvs( size = num_samples, mean = meanB, cov=covB )
            mmd_squareds[index,repeat_index]  = mmd(samplesA, samplesB, kern )
        #stop
    
    mean_mmd_squared = np.mean( mmd_squareds, axis = 1)
    std_mmd_squared = np.std( mmd_squareds, axis = 1 ) / np.sqrt( num_repeats-1 )
    plt.errorbar(betas,mean_mmd_squared, yerr = 2.*std_mmd_squared)
    plt.figure()
    #plt.errorbar(beta, np.sqrt( mean_mmd_squared ) 
    embed()
        
    
if __name__ == '__main__':
    test_mmd()
