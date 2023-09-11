"""Graph Neural Networks architectures.

Classes
-------

Functions
---------
build_fnn
    Build multilayer feed-forward neural network.
"""
#
#                                                                       Modules
# =============================================================================
# Standard

# Third-party
import numpy as np
import torch
import torch_geometric.data
# Local

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def build_fnn(input_size, output_size,
              output_activation=torch.nn.Identity,
              hidden_layer_sizes=[],
              hidden_activation=torch.nn.ReLU):
    """Build multilayer feed-forward neural network.
    
    Based on: geoelements/gns/graph_network.py
    
    (1) Changed output_size to mandatory argument
    (2) Changed hidden_layer_sizes to optional argument and default to []
    (3) Add check procedures for activation functions type
    
    Parameters
    ----------
    input_size : int
        Number of neurons of input layer.
    output_size : int
        Number of neurons of output layer.
    output_activation : torch.nn.Module, default=torch.nn.Identity
        Output unit activation function. Defaults to identity (linear) unit
        activation function.
    hidden_layer_sizes : list[int], default=[]
        Number of neurons of hidden layers.
    hidden_activation : torch.nn.Module, default=torch.nn.ReLU
        Hidden unit activation function. Defaults to ReLU (rectified linear
        unit function) unit activation function.

    Returns
    -------
    fnn : torch.nn.Sequential
        Multilayer feed-forward neural network.
    """
    # Set number of neurons of each layer
    layer_sizes = []
    layer_sizes.append(input_size)
    layer_sizes += hidden_layer_sizes
    layer_sizes.append(output_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of layers of adaptive weights
    n_layer = len(layer_sizes) - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hidden and output layers unit activation functions
    if not isinstance(hidden_activation, torch.nn.Module):
        raise RuntimeError('Hidden unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions = [hidden_activation for i in range(n_layer - 1)]
    if not isinstance(output_activation, torch.nn.Module):
        raise RuntimeError('Output unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions.append(output_activation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create multilayer feed-forward neural network:
    # Initialize neural network
    fnn = torch.nn.Sequential()
    # Loop over neural network layers
    for i in range(n_layer):
        # Set layer linear transformation
        fnn.add_module("Layer-" + str(i),
                       torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                       bias=True)
        # Set layer unit activation function
        fnn.add_module("Activation-" + str(i), activation_functions[i]())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return fnn