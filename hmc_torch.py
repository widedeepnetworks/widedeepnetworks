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
from copy import deepcopy

from IPython import embed

from ess_torch import NormalTestModule

# Hamiltonian Monte Carlo and Hamiltonian Annealed Importance Sampling in PyTorch.
# See MCMC using Hamiltonian dynamics by Neal 2012 for a history of HMC. 
# https://arxiv.org/pdf/1206.1901.pdf
# Hamiltonian Annealed Importance Sampling is discussed in slides by Neal and :
# by Sohl-Dickstein and Culpepper 2012 
# https://arxiv.org/abs/1205.1925

class HAISACC(object):
    #Weight accumulator for Hamiltonian Annealed Importance Sampling.
    def __init__(self):
        self.log_weight = 0.
        self.start_pending = True

    def record_start_energy(self, energy):
        assert(self.start_pending)
        self.start_energy = energy
        self.start_pending = False

    def record_finish_energy_and_increment(self, energy):
        assert( not(self.start_pending) )
        self.log_weight += self.start_energy - energy 
        self.start_pending = True

    

class HMC(Optimizer):
    #Assumes that the prior normal distribution has zero mean and unit covariance.
    #This makes the model spec consistent with the ESS class.
    #Obviously not an optimizer in a strict sense but has common code requirements.
    #We use the HMC variant that uses a single leap frog step and partial momentum refreshment.
    #This is therefore close to the HMC steps in the Hamiltonian Annealed Importance Sampling paper.
    def __init__(self, params, rng, epsilon=0.2, beta=None, leap_frog_iters=1, include_integrator=False):
        self.epsilon = epsilon
        self.leap_frog_iters = leap_frog_iters
        if beta is None:
            self.beta = 1. - np.exp( np.log( 0.5 ) * epsilon ) #default heuristic as in HAIS paper
        else:
            self.beta = beta
        defaults = dict( momentum=1 )
        self.rng = rng
        self.iter_count = 0
        self.accepted_count = 0
        self.include_integrator = include_integrator
        if self.include_integrator:
            self.weight_accumulator = HAISACC()
        super(HMC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HMC, self).__setstate__(state)

    def acceptance_rate(self):
        return self.accepted_count*1. / self.iter_count

    def get_state_norm(self):
        state_norm_squared = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]            
            state_norm_squared += p.data.norm()**2
        return state_norm_squared

    #Assumes that the closure returns -ln_pdf i.e the energy.
    def step(self, closure):
        """Performs a single HMC step with partial momentum refreshment"""
        
        #loop over all params
        #update or initialize the momentum
        #also accumulate the squared momentum and state norms
        old_momentum_norm_squared = 0
        old_state_norm_squared = 0
         
        old_lnpdf = -closure() #old ln pdf of state variables on their own with no prior.

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state: 
                    #this means we don't yet have a momentum
                    #so initialize to a standard normal
                    buf = param_state['momentum_buffer'] = torch.randn( p.size() ).type(p.data.type()) 
                    print('initializing')
                else:
                    #This means we do have a momentum already
                    #So partially refresh it in place.
                    buf = param_state['momentum_buffer']
                    buf.mul_(-np.sqrt( 1. - self.beta )).add_( np.sqrt( self.beta ), torch.randn( p.size() ).type(p.data.type()) )
                #accumulate the norms
                old_momentum_norm_squared += buf.norm()**2
                old_state_norm_squared += p.data.norm()**2
                #store the parameters

        if self.include_integrator:
            self.weight_accumulator.record_finish_energy_and_increment(-old_lnpdf)

                                    
        #also store the current state of the system in case metropolis rejects
        group_original_params = []
        group_original_momenta = []
        for group in self.param_groups:
            original_params = []
            original_momenta = []
            for p in group['params']: #TODO check order.
                original_params.append( p.clone() )
                param_state = self.state[p]
                original_momenta.append( param_state['momentum_buffer'].clone() )
            group_original_params.append( original_params )   
            group_original_momenta.append( original_momenta )

        #loop over all params again 
        #there are several different ways to write leapfrog iterations.
        #we are using the 'position verlet' variant
        #http://physics.ucsc.edu/~peter/242/leapfrog.pdf

        #this time undertake the first part of the leap frog iteration.
        #corresponds to...
        #position_half = position_in + 0.5 * self.epsilon * momentum_in
        #but we will store position_half in params.
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                current_momentum = param_state['momentum_buffer']
                p.data.add_( 0.5*self.epsilon, current_momentum )
        
        #this will clear the gradients and then do back prop with the new position.
        closure()       

        for leap_frog_index in range(self.leap_frog_iters-1):
            for group in self.param_groups:
                for p in group['params']:
                    d_p = p.grad.data
                    param_state = self.state[p]
                    current_momentum = param_state['momentum_buffer']
                    current_momentum.add_( -self.epsilon, p.grad.data ) #minus sign is because we are using energy
                    current_momentum.add_( -self.epsilon, p.data ) # include gradient of unit normal state prior by hand. 
                    p.data.add_( self.epsilon, current_momentum )  # this whole step is composition of two half steps whist we are in model of integration.
                    closure()

        #before we then do the next part of the leap frog iteration.
        #corresponds to...
        #momentum_one = momentum_in + self.epsilon * self.annealed_density.grad_log_prob( n, position_half )
        #and...
        #position_one = position_half + 0.5 * self.epsilon * momentum_one
        #but we will store the new momentum and position in the main params.

        #while we are at it accumulate the new norms.
        new_momentum_norm_squared = 0
        new_state_norm_squared = 0
         
        for group in self.param_groups:
            for p in group['params']:
                d_p = p.grad.data
                param_state = self.state[p]
                current_momentum = param_state['momentum_buffer']
                current_momentum.add_( -self.epsilon, p.grad.data ) #minus sign is because we are using energy
                current_momentum.add_( -self.epsilon, p.data ) # include gradient of unit normal state prior by hand.
                p.data.add_( 0.5*self.epsilon, current_momentum )
                current_momentum.neg_() #negate the proposed momentum
                new_momentum_norm_squared += current_momentum.norm()**2
                new_state_norm_squared += p.data.norm()**2
        
        new_lnpdf = -closure()
        old_total_lnpdf = old_lnpdf - 0.5 * old_momentum_norm_squared - 0.5 * old_state_norm_squared
        new_total_lnpdf = new_lnpdf - 0.5 * new_momentum_norm_squared - 0.5 * new_state_norm_squared
        delta_total_lnpdf = new_total_lnpdf - old_total_lnpdf

        self.iter_count += 1
        #no finally compute the metropolis acceptance.
        if delta_total_lnpdf.data[0] > np.log( self.rng.rand() ):
            #accept proposal
            self.accepted_count += 1
            if self.include_integrator:
                self.weight_accumulator.record_start_energy(-new_lnpdf.data[0])
            return -new_lnpdf.data[0] #could have included prior loss here also.
        else:
            #reject proposal
            #copy states and momenta back over.
            for dest_group, source_group, source_momenta in zip(self.param_groups,group_original_params,group_original_momenta):
                for dest_param, source_param, source_momentum in zip(dest_group['params'],source_group, source_momenta):
                    dest_param.data = source_param.data
                    param_state = self.state[dest_param]
                    param_state['momentum_buffer'] = source_momentum
            if self.include_integrator:
                self.weight_accumulator.record_start_energy(-old_lnpdf.data[0])
            return -old_lnpdf.data[0] #could have included prior loss here also.

def test_hmc():
    torch.manual_seed(2)
    num_dimensions = 3
    num_samples = 20000
    L_np = np.array( [[ 1., 0., 0  ], [1., 1., 0], [1., 1., 1, ] ], dtype=np.float32 )
    y_np = np.atleast_2d( np.array( np.array( [ 1. , 1., 1. ], dtype=np.float32 ) ) ).T
    L = Variable( torch.from_numpy( L_np ), requires_grad=False)
    y = Variable( torch.from_numpy( y_np  ) )
    model = NormalTestModule(num_dimensions) 
    rng = np.random.RandomState(1)
    
    samples = np.zeros( (num_samples, num_dimensions ))
    #sampler = HMC( model.parameters(), rng, epsilon=0.5, beta=0.1 )
    sampler = HMC( model.parameters(), rng, epsilon=0.15, beta=1., leap_frog_iters = 10 )
    energies = np.zeros( num_samples )
    
    import time
    start_time = time.time()

    for sample_index in range(num_samples):
        def closure():
            sampler.zero_grad()
            energy = model(L, y)
            energy.backward()
            return energy
        sampler.step( closure )
        samples[sample_index] = model.epsilons.data.numpy().flatten()
        energies[sample_index] = closure().data[0]

    finish_time = time.time()
    print('samples per second ',(num_samples*1./(finish_time-start_time) ) )
    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov( samples.T )
    sigma = np.eye(num_dimensions) + np.dot(L_np.T,L_np)
    posterior_mean = np.linalg.solve(sigma, np.dot(L_np.T, y_np ) )
    posterior_cov = np.linalg.inv( sigma )
    print('posterior mean',posterior_mean)
    print('posterior cov', posterior_cov )
    print('empirical mean', empirical_mean)
    print('empirical cov', empirical_cov )
    embed()
   
if __name__ == '__main__':
    test_hmc()
