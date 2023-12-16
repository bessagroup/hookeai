"""Test Graph Neural Networks architectures."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.gnn_base_model.model.gnn_architectures import \
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
                         [(1, 1, torch.nn.Identity(), [], torch.nn.ReLU()),
                          (2, 3, torch.nn.Tanh(), [2, 3, 1],
                           torch.nn.LeakyReLU()),
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
    activations = [layer for layer in fnn.named_modules(remove_duplicate=False)
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
            if not isinstance(activations[i][1], type(output_activation)):
                errors.append('Output unit activation function was not '
                              'properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hidden layer features
        else:
            if layer[1].in_features != hidden_layer_sizes[i - 1] \
                    or layer[1].out_features != hidden_layer_sizes[i]:
                errors.append('Number of input/output features of hidden '
                              'layer was not properly set.')
            if not isinstance(activations[i][1], type(hidden_activation)):
                errors.append('Hidden unit activation function was not '
                              'properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('input_size, output_size, output_activation,'
                         'hidden_layer_sizes, hidden_activation',
                         [(1, 1, 'invalid_type', [], torch.nn.ReLU()),
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
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'is_skip_unset_update',
                         [(1, 5, 2, 3, 0, 0, 2, 4, False),
                          (3, 2, 1, 4, 2, 3, 1, 2, False),
                          (2, 4, 5, 4, 0, 0, 0, 2, False),
                          (0, 0, 5, 4, 3, 1, 0, 2, False),
                          (3, 2, 0, 0, 0, 0, 1, 2, False),
                          (1, 5, 2, 3, 0, 0, 2, 4, True),
                          (0, 0, 5, 4, 2, 3, 0, 2, True),
                          (3, 2, 0, 0, 0, 0, 1, 2, True),
                          ])
def test_graph_independent_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_global_in, n_global_out,
                                        n_hidden_layers, hidden_layer_size,
                                        is_skip_unset_update):
    """Test Graph Independent Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_node_out=n_node_out,
        n_edge_in=n_edge_in, n_edge_out=n_edge_out,
        n_global_in=n_global_in, n_global_out=n_global_out,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity(),
        is_skip_unset_update=is_skip_unset_update)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Independent Network update functions and number of features
    update_functions = []
    if model._node_fn is not None:
        update_functions.append((model._node_fn, n_node_in, n_node_out))
    if model._edge_fn is not None:
        update_functions.append((model._edge_fn, n_edge_in, n_edge_out))
    if model._global_fn is not None:
        update_functions.append((model._global_fn, n_global_in, n_global_out))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                if isinstance(module, torch.nn.BatchNorm1d):
                    if module.num_features != n_features_out:
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
                elif isinstance(module, torch.nn.LayerNorm):
                    if module.normalized_shape != (n_features_out,):
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'is_skip_unset_update',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, False),
                          (2, 3, 2, 2, 1, 4, 3, 2, 1, 2, False),
                          (3, 2, 4, 6, 5, 4, 0, 0, 0, 2, False),
                          (3, 0, 0, 6, 5, 4, 1, 3, 0, 2, False),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, False),
                          (3, 0, 0, 6, 5, 4, 2, 3, 0, 2, True),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, True),
                          ])
def test_graph_independent_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_hidden_layers, hidden_layer_size,
                                           is_skip_unset_update):
    """Test Graph Independent Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_node_out=n_node_out,
        n_edge_in=n_edge_in, n_edge_out=n_edge_out,
        n_global_in=n_global_in, n_global_out=n_global_out,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity(),
        is_skip_unset_update=is_skip_unset_update)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = None
    if isinstance(n_node_in, int):
        node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if isinstance(n_edge_in, int):
        edge_features_in = torch.rand(n_edges, n_edge_in)
    # Generate random global features input matrix
    global_features_in = None
    if isinstance(n_global_in, int):
        global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out, global_features_out = model(
        node_features_in=node_features_in, edge_features_in=edge_features_in,
        global_features_in=global_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if model._node_fn is not None:
        if not isinstance(node_features_out, torch.Tensor):
            errors.append('Nodes features output matrix is not torch.Tensor.')
        elif not torch.equal(torch.tensor(node_features_out.size()),
                             torch.tensor([n_nodes, n_node_out])):
            errors.append('Nodes features output matrix is not '
                          'torch.Tensor(2d) of shape (n_nodes, n_features).')
    else:
        if node_features_in is not None and is_skip_unset_update:
            if not torch.allclose(node_features_out, node_features_in):
                errors.append('Nodes features output matrix is not '
                              'equal to nodes features input matrix.')
        else:
            if node_features_out is not None:
                errors.append('Nodes features output matrix is not None.')
    # Check edge features output matrix
    if model._edge_fn is not None:
        if not isinstance(edge_features_out, torch.Tensor):
            errors.append('Edges features output matrix is not torch.Tensor.')
        elif not torch.equal(torch.tensor(edge_features_out.size()),
                             torch.tensor([n_edges, n_edge_out])):
            errors.append('Edges features output matrix is not '
                          'torch.Tensor(2d) of shape (n_edges, n_features).')
    else:
        if edge_features_in is not None and is_skip_unset_update:
            if not torch.allclose(edge_features_out, edge_features_in):
                errors.append('Edges features output matrix is not '
                              'equal to edges features input matrix.')
        else:
            if edge_features_out is not None:
                errors.append('Edges features output matrix is not None.')
    # Check global features output matrix
    if model._global_fn is not None:
        if not isinstance(global_features_out, torch.Tensor):
            errors.append('Global features output matrix is not torch.Tensor.')
        elif not torch.equal(torch.tensor(global_features_out.size()),
                             torch.tensor([1, n_global_out])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, n_features).')
    else:
        if global_features_in is not None and is_skip_unset_update:
            if not torch.allclose(global_features_out, global_features_in):
                errors.append('Global features output matrix is not '
                              'equal to global features input matrix.')
        else:
            if global_features_out is not None:
                errors.append('Global features output matrix is not None.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(1, 5, 2, 3, 0, 0, 2, 4, 'add', 'add'),
                          (3, 2, 1, 4, 0, 2, 1, 2, 'add', 'add'),
                          (2, 4, 5, 4, 2, 3, 0, 2, 'add', 'add'),
                          (0, 4, 5, 4, 1, 0, 0, 2, 'add', 'add'),
                          (2, 4, 0, 4, 0, 0, 0, 2, 'add', 'add'),
                          ])
def test_graph_interaction_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_global_in, n_global_out,
                                        n_hidden_layers, hidden_layer_size,
                                        edge_to_node_aggr,
                                        node_to_global_aggr):
    """Test Graph Interaction Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Interaction Network update functions and number of features
    update_functions = []
    if model._node_fn is not None:
        update_functions.append((model._node_fn, n_node_in+n_edge_out,
                                 n_node_out))
    if model._edge_fn is not None:
        update_functions.append((model._edge_fn, n_edge_in+2*n_node_in,
                                 n_edge_out))
    if model._global_fn is not None:
        update_functions.append((model._global_fn, n_global_in+n_node_out,
                                 n_global_out))
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
                if isinstance(module, torch.nn.BatchNorm1d):
                    if module.num_features != n_features_out:
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
                elif isinstance(module, torch.nn.LayerNorm):
                    if module.normalized_shape != (n_features_out,):
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 'add', 'add'),
                          ])
def test_graph_interaction_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_hidden_layers, hidden_layer_size,
                                           edge_to_node_aggr,
                                           node_to_global_aggr):
    """Test Graph Interaction Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = torch.empty(n_nodes, 0)
    if n_node_in > 0:
        node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if n_edge_in > 0:
        edge_features_in = torch.rand(n_edges, n_edge_in)
    # Generate random edges indexes
    edges_indexes = torch.randint(low=0, high=n_nodes, size=(2, n_edges),
                                  dtype=torch.long)
    # Generate random global features input matrix
    global_features_in = None
    if n_global_in > 0:
        global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out, global_features_out = \
        model(edges_indexes=edges_indexes,
              node_features_in=node_features_in,
              edge_features_in=edge_features_in,
              global_features_in=global_features_in)
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
    # Check global features output matrix
    if model._global_fn is not None:
        if not isinstance(global_features_out, torch.Tensor):
            errors.append('Global features output matrix is not torch.Tensor.')
        elif not torch.equal(torch.tensor(global_features_out.size()),
                             torch.tensor([1, n_global_out])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 'add', 'add'),
                          ])
def test_graph_interaction_network_message(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_hidden_layers, hidden_layer_size,
                                           edge_to_node_aggr,
                                           node_to_global_aggr):
    """Test Graph Interaction Network message building (edge update)."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random source nodes features input matrix
    node_features_in_i = None
    if n_node_in > 0:
        node_features_in_i = torch.rand(n_edges, n_node_in)
    # Generate random source nodes features input matrix
    node_features_in_j = None
    if n_node_in > 0:
        node_features_in_j = torch.rand(n_edges, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if n_edge_in > 0:
        edge_features_in = torch.rand(n_edges, n_edge_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (edge update)
    edge_features_out = model.message(node_features_in_i=node_features_in_i,
                                      node_features_in_j=node_features_in_j,
                                      edge_features_in=edge_features_in)
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
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 'add', 'add'),
                          ])
def test_graph_interaction_network_update(n_nodes, n_node_in, n_node_out,
                                          n_edges, n_edge_in, n_edge_out,
                                          n_global_in, n_global_out,
                                          n_hidden_layers, hidden_layer_size,
                                          edge_to_node_aggr,
                                          node_to_global_aggr):
    """Test Graph Interaction Network node features update."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix (resulting from aggregation)
    node_features_in_aggr = torch.rand(n_nodes, n_edge_out)
    # Generate random nodes features input matrix
    node_features_in = None
    if n_node_in > 0:
        node_features_in = torch.rand(n_nodes, n_node_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (node update)
    node_features_out = \
        model.update(node_features_in_aggr=node_features_in_aggr,
                     node_features_in=node_features_in)
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
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 'add', 'mean'),
                          ])
def test_graph_interaction_network_update_global(
    n_nodes, n_node_in, n_node_out, n_edges, n_edge_in, n_edge_out,
    n_global_in, n_global_out, n_hidden_layers, hidden_layer_size,
    edge_to_node_aggr, node_to_global_aggr):
    """Test Graph Interaction Network global features update."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features output matrix
    node_features_out = torch.rand(n_nodes, n_node_out)
    # Generate random global features input matrix
    global_features_in = None
    if n_global_in > 0:
        global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (global update)
    global_features_out = \
        model.update_global(node_features_out=node_features_out,
                            global_features_in=global_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check global features output matrix
    if not isinstance(global_features_out, torch.Tensor):
        errors.append('Global features output matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(global_features_out.size()),
                         torch.tensor([1, n_global_out])):
        errors.append('Global features output matrix is not torch.Tensor(2d) '
                      'of shape (1, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))