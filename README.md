Code for the paper: 

Gaussian Process Behaviour in Wide Deep Neural Networks
Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner and Zoubin Ghahramani

Which can be found at:

https://arxiv.org/abs/1804.11271

Note that this paper is substantially expanded from the original ICLR version. 

The latest version of the experiments uses:

TensorFlow 1.9.0
PyTorch 0.3.0
GPflow 0.5.0

The code is released under the Apache 2.0 license.

The newer code uses a more numerically stable variant of the Deep ReLu kernel built on GPflow. This enables us to backpropagate gradients for type II maximum likelihood. 

The random number generator in PyTorch has changed significantly across versions and this can effect numerical values, though not the qualitative conclusions.

An earlier version of the code had an inconsistency in the scaling of the input weight variances between the neural network and Gaussian process. This had a negligible effect on the experiments where it was used. 

Note that the HMC experiments are particularly slow. Mixing is verified using the autocorrelation estimators included as part of PyHMC. We work in terms of the mixing of the predictive distribution since the weight posterior will have combinatorically many symmetric modes that are predictively equivalent. 
