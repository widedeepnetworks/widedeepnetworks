import numpy as np
import itertools
import tensorflow as tf
from tensorflow.python.framework import ops
from IPython import embed

import gpflow

class DeepArcCosine(gpflow.kernels.Kern):

    def __init__(self, input_dim, num_steps, base_kernel=None,
                 variance=1.0, bias_variance=0., active_dims=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)
        
        if base_kernel is None:
            # corresponds to using "Radford" scaled input to hidden weights
            base_kernel = gpflow.kernels.Linear(
                input_dim=input_dim, variance=1./input_dim
            ) + gpflow.kernels.Constant( input_dim = input_dim, variance = bias_variance )
        
        self.num_steps = num_steps
        self.base_kernel = base_kernel

        self.variance = gpflow.param.Param(variance, gpflow.transforms.positive)
        self.bias_variance = gpflow.param.Param(bias_variance, gpflow.transforms.positive)
        
    def K(self, X, X2=None):
        # the linear kernel
        if X2 is None:
            X2 = X
        K = self.base_kernel.K( X, X2 )
        kxdiag  = self.base_kernel.Kdiag(X)
        kx2diag = self.base_kernel.Kdiag(X2) 

        for step_index in range(self.num_steps):
            # recursively compute (scaled) relu kernel
            K = self.recurseK( K, kxdiag, kx2diag )
            kxdiag = self.recurseKdiag( kxdiag )
            kx2diag = self.recurseKdiag( kx2diag )

        return K

    def Kdiag(self, X):
        Kdiag = self.base_kernel.Kdiag( X )

        for step_index in range(self.num_steps):
            Kdiag = self.recurseKdiag( Kdiag )

        return Kdiag

    def recurseK(self, K, kxdiag, kx2diag):
        norms = tf.sqrt(kxdiag)
        norms2 = tf.sqrt(kx2diag)
        norms_prod = norms[:, None] * norms2[None, : ]
        
        
        cos_theta = tf.clip_by_value(K / norms_prod, -1., 1.)
        theta = tf.acos(cos_theta)  # angle wrt the previous RKHS
        
        # J(theta) = sin(theta) + (pi - theta) * cos(theta)
        J = tf.sqrt(1. - cos_theta**2) + (np.pi - theta) * cos_theta
        # Note: sin(acos(cos_theta)) = sqrt(1 - cos_theta**2)
        
        # norm(x) norm(y) / pi * J(theta)
        cho_saul = norms_prod / np.pi * J
        
        return self.variance * cho_saul + self.bias_variance * tf.ones_like(cho_saul)
    
    def recurseKdiag(self, Kdiag):
        # angle is zero, hence the diagonal stays the same (if scaled relu is used)        
        return self.variance * Kdiag  + self.bias_variance * tf.ones_like(Kdiag)      

def demo_deep_arcosine_kernel():
    X = np.atleast_2d( np.linspace( -0.5 , 0.5 , 4 ) ).T
    bias_variance = 0.5
    reference_kernel = gpflow.kernels.ArcCosine( input_dim = 1, order = 1, variance = 1., weight_variances = 1., bias_variance = bias_variance )
    reference_K = reference_kernel.compute_K_symm(X)
    test_kernel = DeepArcCosine(1,1, variance=1., bias_variance = bias_variance)
    test_K = test_kernel.compute_K_symm(X)
    print('reference_K + bias', reference_K+np.ones_like(reference_K)*bias_variance)
    print('test_K ',test_K)
    embed()
    
if __name__ == "__main__":
    demo_deep_arcosine_kernel()
