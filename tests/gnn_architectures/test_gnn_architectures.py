"""Test Graph Neural Networks architectures."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.gnn_model.gnn_architectures import \
    build_fnn, GraphIndependentNetwork, GraphInteractionNetwork
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
@pytest.mark.parametrize('input_size, output_size, output_activation,'
                         'hidden_layer_sizes, hidden_activation',
                         [(1, 1, torch.nn.Identity, [], torch.nn.ReLU),
                          (2, 3, torch.nn.Tanh, [2, 3, 1], torch.nn.LeakyReLU),
                          ])
def test_build_fnn(input_size, output_size, output_activation,
                   hidden_layer_sizes, hidden_activation):
    """Test building of multilayer feed-forward neural network."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build multilayer feed-forward neural network
    fnn = build_fnn(input_size, output_size, output_activation,
                    hidden_layer_sizes, hidden_activation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get feed-forward neural network layers
    layers = [layer for layer in fnn.named_children() if 'Layer-' in layer[0]]
    activations = [layer for layer in fnn.named_children()
                   if 'Activation-' in layer[0]]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check number of layers
    if len(layers) != len(hidden_layer_sizes) + 1:
        errors.append('Number of layers was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over layers
    for i, layer in enumerate(layers):
        # Check input layer features              
        if i == 0:
            if layer[1].in_features != input_size:
                errors.append('Number of input features of input layer was '
                              'not properly set.')  
            if len(layers) == 1:
                if layer[1].out_features != output_size:
                    errors.append('Number of ouput features of input layer '
                                  'was not properly set.')
            else:
                if layer[1].out_features != hidden_layer_sizes[0]:
                    errors.append('Number of ouput features of input layer '
                                  'was not properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check output layer features
        elif i == len(layers) - 1:
            if layer[1].in_features != hidden_layer_sizes[-1]:
                errors.append('Number of input features of output layer was '
                              'not properly set.')
            if layer[1].out_features != output_size:
                errors.append('Number of output features of output layer was '
                              'not properly set.')
            if not isinstance(activations[i][1], output_activation):
                errors.append('Output unit activation function was not '
                              'properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hidden layer features
        else:
            if layer[1].in_features != hidden_layer_sizes[i - 1] \
                    or layer[1].out_features != hidden_layer_sizes[i]:
                errors.append('Number of input/output features of hidden '
                              'layer was not properly set.')
            if not isinstance(activations[i][1], hidden_activation):
                errors.append('Hidden unit activation function was not '
                              'properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('input_size, output_size, output_activation,'
                         'hidden_layer_sizes, hidden_activation',
                         [(1, 1, 'invalid_type', [], torch.nn.ReLU),
                          (2, 3, torch.nn.Tanh, [2, 3, 1], 'invalid_type'),
                          ])
def test_build_fnn_invalid(input_size, output_size, output_activation,
                           hidden_layer_sizes, hidden_activation):
    """Test invalid activation functions."""
    with pytest.raises(RuntimeError):
        fnn = build_fnn(input_size, output_size, output_activation,
                        hidden_layer_sizes, hidden_activation)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(1, 5, 2, 3, 2, 4),
                          (3, 2, 1, 4, 1, 2),
                          (2, 4, 5, 4, 0, 2),
                          ])
def test_graph_independent_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_hidden_layers,
                                        hidden_layer_size):
    """Test Graph Independent Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Independent Network update functions and number of features
    update_functions = ((model._node_fn, n_node_in, n_node_out),
                        (model._edge_fn, n_edge_in, n_edge_out))
    # Loop over update functions
    for update_fn, n_features_in, n_features_out in update_functions:        
        # Loop over update function modules
        for name, module in update_fn.named_children():
            # Check feed-forward neural network
            if name == 'FNN':
                # Get feed-forward neural network layers
                layers = [layer for layer in module.named_children()
                          if 'Layer-' in layer[0]]                
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of layers
                if len(layers) != n_hidden_layers + 1:
                    errors.append('Number of layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over layers
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].in_features != n_features_in:
                            errors.append('Number of input features of '
                                          'input layer was not properly set.')  
                        if len(layers) == 1:
                            if layer[1].out_features != n_features_out:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                        else:
                            if layer[1].out_features != hidden_layer_size:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check output layer features
                    elif i == len(layers) - 1:
                        if layer[1].in_features != hidden_layer_size:
                            errors.append('Number of input features of output '
                                          'layer was not properly set.')
                        if layer[1].out_features != n_features_out:
                            errors.append('Number of output features of '
                                          'output layer was not properly set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check hidden layer features
                    else:
                        if layer[1].in_features != hidden_layer_size \
                                or layer[1].out_features != hidden_layer_size:
                            errors.append('Number of input/output features of '
                                          'hidden layer was not properly set.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check normalization layer
            elif name == 'Norm-Layer':
                if module.normalized_shape[0] != n_features_out:
                    errors.append('Number of features of Normalization Layer '
                                  'was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(10, 1, 5, 20, 2, 3, 2, 4),
                          (1, 3, 2, 1, 1, 4, 1, 2),
                          (3, 2, 4, 6, 5, 4, 0, 2),
                          ])
def test_graph_independent_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_hidden_layers, hidden_layer_size):
    """Test Graph Independent Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = torch.rand(n_edges, n_edge_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out = model(node_features_in,
                                                 edge_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix 
    if not isinstance(node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(node_features_out.size()),
                         torch.tensor([n_nodes, n_node_out])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    # Check edge features output matrix
    if not isinstance(edge_features_out, torch.Tensor):
        errors.append('Edges features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(edge_features_out.size()),
                         torch.tensor([n_edges, n_edge_out])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(1, 5, 2, 3, 2, 4),
                          (3, 2, 1, 4, 1, 2),
                          (2, 4, 5, 4, 0, 2),
                          ])
def test_graph_interaction_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_hidden_layers,
                                        hidden_layer_size):
    """Test Graph Interaction Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Interaction Network update functions and number of features
    update_functions = ((model._node_fn, n_node_in + n_edge_out, n_node_out),
                        (model._edge_fn, n_edge_in + 2*n_node_in, n_edge_out))
    # Loop over update functions
    for update_fn, n_features_in, n_features_out in update_functions:        
        # Loop over update function modules
        for name, module in update_fn.named_children():
            # Check feed-forward neural network
            if name == 'FNN':
                # Get feed-forward neural network layers
                layers = [layer for layer in module.named_children()
                          if 'Layer-' in layer[0]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of layers
                if len(layers) != n_hidden_layers + 1:
                    errors.append('Number of layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over layers
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].in_features != n_features_in:
                            errors.append('Number of input features of '
                                          'input layer was not properly set.')  
                        if len(layers) == 1:
                            if layer[1].out_features != n_features_out:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                        else:
                            if layer[1].out_features != hidden_layer_size:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check output layer features
                    elif i == len(layers) - 1:
                        if layer[1].in_features != hidden_layer_size:
                            errors.append('Number of input features of output '
                                          'layer was not properly set.')
                        if layer[1].out_features != n_features_out:
                            errors.append('Number of output features of '
                                          'output layer was not properly set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check hidden layer features
                    else:
                        if layer[1].in_features != hidden_layer_size \
                                or layer[1].out_features != hidden_layer_size:
                            errors.append('Number of input/output features of '
                                          'hidden layer was not properly set.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check normalization layer
            elif name == 'Norm-Layer':
                if module.normalized_shape[0] != n_features_out:
                    errors.append('Number of features of Normalization Layer '
                                  'was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(10, 1, 5, 20, 2, 3, 2, 4),
                          (1, 3, 2, 1, 1, 4, 1, 2),
                          (3, 2, 4, 6, 5, 4, 0, 2),
                          ])
def test_graph_interaction_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_hidden_layers, hidden_layer_size):
    """Test Graph Interaction Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = torch.rand(n_edges, n_edge_in)
    # Generate random edges indexes
    edges_indexes = torch.randint(low=0, high=n_nodes, size=(2, n_edges),
                                  dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out = model(node_features_in,
                                                 edge_features_in,
                                                 edges_indexes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if not isinstance(node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(node_features_out.size()),
                         torch.tensor([n_nodes, n_node_out])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    # Check edge features output matrix
    if not isinstance(edge_features_out, torch.Tensor):
        errors.append('Edges features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(edge_features_out.size()),
                         torch.tensor([n_edges, n_edge_out])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(10, 1, 5, 20, 2, 3, 2, 4),
                          (1, 3, 2, 1, 1, 4, 1, 2),
                          (3, 2, 4, 6, 5, 4, 0, 2),
                          ])
def test_graph_interaction_network_message(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_hidden_layers, hidden_layer_size):
    """Test Graph Interaction Network message building (edge update)."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random source nodes features input matrix
    node_features_in_i = torch.rand(n_edges, n_node_in)
    # Generate random source nodes features input matrix
    node_features_in_j = torch.rand(n_edges, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = torch.rand(n_edges, n_edge_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (edge update)
    edge_features_out = model.message(node_features_in_i, node_features_in_j,
                                      edge_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check edge features output matrix
    if not isinstance(edge_features_out, torch.Tensor):
        errors.append('Edges features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(edge_features_out.size()),
                         torch.tensor([n_edges, n_edge_out])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(10, 1, 5, 20, 2, 3, 2, 4),
                          (1, 3, 2, 1, 1, 4, 1, 2),
                          (3, 2, 4, 6, 5, 4, 0, 2),
                          ])
def test_graph_interaction_network_update(n_nodes, n_node_in, n_node_out,
                                          n_edges, n_edge_in, n_edge_out,
                                          n_hidden_layers, hidden_layer_size):
    """Test Graph Interaction Network node features update."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(n_node_in=n_node_in, n_node_out=n_node_out,
                                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix (resulting from aggregation)
    node_features_in_aggr = torch.rand(n_nodes, n_edge_out)
    # Generate random nodes features input matrix
    node_features_in = torch.rand(n_nodes, n_node_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (node update)
    node_features_out = model.update(node_features_in_aggr, node_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if not isinstance(node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(node_features_out.size()),
                         torch.tensor([n_nodes, n_node_out])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))