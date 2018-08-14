# Copyright 2018 Alexander Matthews and Jiri Hron
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
import itertools
import tensorflow as tf
from tensorflow.python.framework import ops
from IPython import embed

import gpflow

class DeepArcCosine(gpflow.kernels.Kern):

    def __init__(self, input_dim, num_steps,
                 variance=1.0, bias_variance=0., active_dims=None):
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)
        
        self.num_steps = num_steps
        
        self.variance = gpflow.param.Param(variance, gpflow.transforms.positive)
        self.bias_variance = gpflow.param.Param(bias_variance, gpflow.transforms.positive)
    
    def baseK(self, X, X2):
        inner = tf.matmul(X * self.variance, X2, transpose_b=True)/self.input_dim 
        return inner + tf.ones_like(inner) * self.bias_variance
        
    def baseKdiag(self, X):
        inner = tf.reduce_sum(tf.square(X) * self.variance, 1)/self.input_dim
        return inner + tf.ones_like(inner) * self.bias_variance
        
    def K(self, X, X2=None):
        # the linear kernel
        if X2 is None:
            X2 = X
        K = self.baseK( X, X2 )
        kxdiag  = self.baseKdiag(X)
        kx2diag = self.baseKdiag(X2) 

        for step_index in range(self.num_steps):
            # recursively compute (scaled) relu kernel
            K = self.recurseK( K, kxdiag, kx2diag )
            kxdiag = self.recurseKdiag( kxdiag )
            kx2diag = self.recurseKdiag( kx2diag )

        return K

    def Kdiag(self, X):
        Kdiag = self.baseKdiag( X )

        for step_index in range(self.num_steps):
            Kdiag = self.recurseKdiag( Kdiag )

        return Kdiag

    def recurseK(self, K, kxdiag, kx2diag):
        norms = tf.sqrt(kxdiag)
        norms_rec = tf.rsqrt(kxdiag)
        norms2 = tf.sqrt(kx2diag)
        norms2_rec = tf.rsqrt(kx2diag)
        
        jitter = 1e-7
        scaled_numerator = K * (1.-jitter)
                
        cos_theta = scaled_numerator * norms_rec[:,None] *  norms2_rec[None,:]
        theta = tf.acos(cos_theta) 
        return self.variance / np.pi * ( tf.sqrt(kxdiag[:,None] * kx2diag[None,:] - tf.square(scaled_numerator) ) + (np.pi - theta) * scaled_numerator ) + self.bias_variance*tf.ones_like(K)
    
    def recurseKdiag(self, Kdiag):
        # angle is zero, hence the diagonal stays the same (if scaled relu is used)        
        return self.variance * Kdiag  + self.bias_variance * tf.ones_like(Kdiag)      

def demo_deep_arcosine_kernel():
    bias_variance = 0.5
    variance = 0.7
    order = 1
    num_hidden_layers = 1

    def test_one_dim():
        print("\n Test one dim: \n")
        input_dim = 1
        X = np.atleast_2d( np.linspace( -0.5 , 0.5 , 4 ) ).T
        reference_kernel = gpflow.kernels.ArcCosine( input_dim = input_dim, order = order, variance = variance, weight_variances = variance, bias_variance = bias_variance )
        reference_K = reference_kernel.compute_K_symm(X)
        test_kernel = DeepArcCosine(input_dim = input_dim, num_steps = num_hidden_layers, variance=variance, bias_variance = bias_variance)
        test_K = test_kernel.compute_K_symm(X)
        print('reference_K + bias', reference_K+np.ones_like(reference_K)*bias_variance)
        print('test_K ',test_K)
    
    def test_two_dim():
        print("\n Test two dim: \n")
        input_dim = 2
        rng = np.random.RandomState(1)
        X = rng.randn(4,input_dim)
        #Factor of 1/2 is because we scale the weight variance.
        reference_kernel = gpflow.kernels.ArcCosine( input_dim = input_dim, order = order, variance = variance, weight_variances = variance/2, bias_variance = bias_variance )
        reference_K = reference_kernel.compute_K_symm(X) 
        test_kernel = DeepArcCosine(input_dim = input_dim, num_steps = num_hidden_layers, variance=variance, bias_variance = bias_variance)
        test_K = test_kernel.compute_K_symm(X)
        print('reference_K + bias', reference_K+np.ones_like(reference_K)*bias_variance)
        print('test_K ',test_K)        
    
    test_one_dim()
    test_two_dim()

    embed()
    
if __name__ == "__main__":
    demo_deep_arcosine_kernel()
