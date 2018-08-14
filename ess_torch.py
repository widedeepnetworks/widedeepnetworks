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

import math
import numpy as np

from torch.optim import Optimizer
import torch
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn as nn

from IPython import embed

#Elliptical slice sampling in PyTorch
#Based on code by Jo Bovy https://github.com/jobovy/bovy_mcmc
#This was in turn based on code in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m
#The paper is by Murray, Adams and MacKay. AISTATS 2010.

class ESS(Optimizer):
    #Assumes that the prior normal distribution has zero mean and unit covariance.
    #Obviously not an optimizer in a strict sense but has common code requirements.
    def __init__(self, params, rng):
        self.rng = rng
        super(ESS, self).__init__(params, {})

    def __setstate__(self, state):
        super(ESS, self).__setstate__(state)
 
    def get_random_angle(self):
        phi= self.rng.uniform()*2.*math.pi
        phi_min = phi-2.*math.pi
        phi_max = phi
        return phi, phi_min, phi_max
    
    #Assumes that the closure returns -ln_pdf i.e the energy.
    def step(self, closure):
        """Performs a single ESS step"""
        cur_lnpdf = -1.*closure().data[0]

        #initial loop 
        group_original_params = []
        group_normal_samples = []
        for group in self.param_groups: 
            #take copy of parameters and create normal vector
            original_params = []
            normal_samples = []
            
            for p in group['params']:
                original_params.append( p.clone() )
                normal_samples.append( Variable(torch.randn( p.size() ).type(p.data.type()) ) )
            group_original_params.append(original_params)
            group_normal_samples.append(normal_samples)

        #get starting random angle.
        phi, phi_min, phi_max = self.get_random_angle()
        
        #Gibbs step under curve
        hh = math.log(self.rng.uniform()) + cur_lnpdf
        
        while True:
            #get next proposed point
            for group, original_group, normal_group in zip(self.param_groups, group_original_params, group_normal_samples ):
                for param, original_param, normal_sample in zip(group['params'], original_group, normal_group):
                    param.data = (math.cos(phi)*original_param + normal_sample * math.sin(phi)).data
            cur_lnpdf = -1.*closure().data[0]
            
            #slice sampling logic
            if cur_lnpdf > hh:
                break
            
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
            phi = self.rng.uniform()*(phi_max - phi_min) + phi_min
            
        return -cur_lnpdf

class NormalTestModule(nn.Module):
    #implement prior and likelihood for a multivariate normal model.
    #assume zero prior mean
    #assume unit noise variance.
    
    def __init__(self, num_dimensions):
        super(NormalTestModule, self).__init__()
        self.num_dimensions = num_dimensions
        self.epsilons = nn.Parameter(torch.randn(self.num_dimensions,1))
        
    def forward(self, L, y):
        #change normal samples into a function sample
        f = L.mm( self.epsilons )
        criterion = torch.nn.MSELoss(size_average=False)
        energy = 0.5 * criterion( f, y )
        return energy

def test_ess():
    num_dimensions = 3
    num_samples = 10000
    L_np = np.array( [[ 1., 0., 0  ], [1., 1., 0], [1., 1., 1, ] ], dtype=np.float32 )
    y_np = np.atleast_2d( np.array( np.array( [ 1. , 1., 1. ], dtype=np.float32 ) ) ).T
    L = Variable( torch.from_numpy( L_np ), requires_grad=False)
    y = Variable( torch.from_numpy( y_np  ) )
    model = NormalTestModule(num_dimensions) 
    rng = np.random.RandomState(1)
    
    samples = np.zeros( (num_samples, num_dimensions ))
    sampler = ESS( model.parameters(), rng )
    energies = np.zeros( num_samples )
    
    for sample_index in range(num_samples):
        def closure():
            energy = model(L, y)
            return energy
        sampler.step( closure )
        samples[sample_index] = model.epsilons.data.numpy().flatten()
        energies[sample_index] = closure().data[0]
    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov( samples.T )
    sigma = np.eye(num_dimensions) + np.dot(L_np.T,L_np)
    posterior_mean = np.linalg.solve(sigma, np.dot(L_np.T, y_np ) )
    posterior_cov = np.linalg.inv( sigma )
    print('posterior mean',posterior_mean)
    print('posterior cov', posterior_cov )
    print('empirical mean', empirical_mean)
    print('empirical cov', empirical_cov )
   
if __name__ == '__main__':
    test_ess()
