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

import torch
from torch.nn.modules import Linear
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ScaledRelu(Module):
    def __init__(self):
        super(ScaledRelu, self).__init__()
    
    def forward(self, input):
        return torch.mul(F.relu(input),math.sqrt(2))

class GaussLinearStandardized(Module):
    def __init__(self, in_features, out_features, bias=True, raw_weight_variance=1., raw_bias_variance=1.):
        super(GaussLinearStandardized, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight_variance = raw_weight_variance
        self.raw_bias_variance = raw_bias_variance
        self.epsilon_weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.epsilon_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('epsilon_bias', None)
        self.reset_parameters() 

    def reset_parameters(self):
        self.epsilon_weight.data.normal_()
        if self.epsilon_bias is not None:
            self.epsilon_bias.data.normal_()
            
    def forward(self, input):
        stdv = 1. / math.sqrt(self.in_features)
        weight = self.epsilon_weight * stdv * math.sqrt(self.raw_weight_variance)
        if self.epsilon_bias is not None:
            bias = self.epsilon_bias * math.sqrt(self.raw_bias_variance)
        else:
            bias = None
        return F.linear(input, weight, bias)

class GaussLinear(Linear):
    def reset_parameters(self):
        n_nodes = self.weight.size(1)
        if self.bias is not None:
            n_nodes += 1
        stdv = 1. / math.sqrt(n_nodes)
        self.weight.data.normal_(std = stdv )
        if self.bias is not None:
            self.bias.data.normal_(std = stdv )
