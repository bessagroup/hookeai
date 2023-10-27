"""Test Graph Neural Network based Encoder-Process-Decoder model."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
import torch_geometric.nn
# Local
from src.vegapunk.gnn_model.gnn_epd_model import \
    EncodeProcessDecode, Encoder, Processor, Decoder
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
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_hidden_layers, hidden_layer_size',
                         [(1, 5, 2, 3, 2, 4),
                          (3, 2, 1, 4, 1, 2),
                          (2, 4, 5, 4, 0, 2),
                          ])
def test_encoder_init(n_node_in, n_node_out, n_edge_in, n_edge_out,
                      n_hidden_layers, hidden_layer_size):
    """Test GNN-based encoder constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based encoder
    model = Encoder(n_node_in=n_node_in, n_node_out=n_node_out,
                    n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                    n_hidden_layers=n_hidden_layers,
                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based encoder update functions and number of features
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
def test_encoder_forward(n_nodes, n_node_in, n_node_out, n_edges, n_edge_in,
                         n_edge_out, n_hidden_layers, hidden_layer_size):
    """Test GNN-based encoder forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based encoder
    model = Encoder(n_node_in=n_node_in, n_node_out=n_node_out,
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
@pytest.mark.parametrize('n_message_steps, n_node_in, n_node_out, n_edge_in,'
                         'n_edge_out, n_hidden_layers, hidden_layer_size,'
                         'is_node_res_connect, is_edge_res_connect',
                         [(1, 1, 5, 2, 3, 2, 4, False, False),
                          (1, 3, 3, 1, 4, 1, 2, True, False),
                          (1, 1, 5, 4, 4, 2, 4, False, True),
                          (1, 5, 5, 4, 4, 2, 4, True, True),
                          (2, 3, 3, 4, 4, 1, 2, False, False),
                          (2, 3, 3, 4, 4, 1, 2, True, True),
                          ])
def test_processor_init(n_message_steps, n_node_in, n_node_out, n_edge_in,
                        n_edge_out, n_hidden_layers, hidden_layer_size,
                        is_node_res_connect, is_edge_res_connect):
    """Test GNN-based processor constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based processor
    model = Processor(n_message_steps=n_message_steps, n_node_in=n_node_in,
                      n_node_out=n_node_out, n_edge_in=n_edge_in,
                      n_edge_out=n_edge_out, n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not isinstance(model._processor, torch.nn.ModuleList):
        errors.append('GNN-based processor is not a torch.nn.ModuleList.')
    elif len(model._processor) != n_message_steps:
        errors.append('GNN-based processor number of message-passing steps '
                      'was not properly set.')
    if not all([isinstance(model, torch_geometric.nn.MessagePassing)
                for model in model._processor]):
        errors.append('GNN-based processor is not a sequence of models of '
                      'type torch_geometric.nn.MessagePassing.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_message_steps, n_node_in, n_node_out, n_edge_in,'
                         'n_edge_out, n_hidden_layers, hidden_layer_size,'
                         'is_node_res_connect, is_edge_res_connect',
                         [(0, 1, 5, 2, 3, 2, 4, False, False),
                          (1, 4, 3, 1, 4, 1, 2, True, False),
                          (1, 1, 5, 4, 2, 2, 4, False, True),
                          (1, 2, 5, 4, 3, 2, 4, True, True),
                          (2, 3, 3, 4, 2, 1, 2, False, False),
                          (2, 3, 1, 4, 4, 1, 2, False, False),
                          (2, 3, 3, 3, 4, 1, 2, False, False)
                          ])
def test_processor_init_invalid(n_message_steps, n_node_in, n_node_out,
                                n_edge_in, n_edge_out, n_hidden_layers,
                                hidden_layer_size, is_node_res_connect,
                                is_edge_res_connect):
    """Test detection of invalid input to GNN-based processor constructor."""
    with pytest.raises(RuntimeError):
        # Build GNN-based processor
        _ = Processor(n_message_steps=n_message_steps, n_node_in=n_node_in,
                      n_node_out=n_node_out, n_edge_in=n_edge_in,
                      n_edge_out=n_edge_out, n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_message_steps, n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out, n_hidden_layers,'
                         'hidden_layer_size, is_node_res_connect,'
                         'is_edge_res_connect',
                         [(1, 10, 1, 5, 20, 2, 3, 2, 4, False, False),
                          (1, 10, 3, 3, 20, 1, 4, 1, 2, True, False),
                          (1, 10, 1, 5, 20, 4, 4, 2, 4, False, True),
                          (1, 10, 5, 5, 20, 4, 4, 2, 4, True, True),
                          (2, 10, 3, 3, 20, 4, 4, 1, 2, False, False),
                          (2, 10, 3, 3, 20, 4, 4, 1, 2, True, True),
                          ])
def test_processor_forward(n_message_steps, n_nodes, n_node_in, n_node_out,
                           n_edges, n_edge_in, n_edge_out, n_hidden_layers,
                           hidden_layer_size, is_node_res_connect,
                           is_edge_res_connect):
    """Test GNN-based processor forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based processor
    model = Processor(n_message_steps=n_message_steps, n_node_in=n_node_in,
                      n_node_out=n_node_out, n_edge_in=n_edge_in,
                      n_edge_out=n_edge_out, n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect)
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
@pytest.mark.parametrize('n_node_in, n_node_out, n_hidden_layers,'
                         'hidden_layer_size',
                         [(1, 5, 2, 4),
                          (3, 2, 1, 2),
                          (2, 4, 0, 2),
                          ])
def test_decoder_init(n_node_in, n_node_out, n_hidden_layers,
                      hidden_layer_size):
    """Test GNN-based decoder constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based decoder
    model = Decoder(n_node_in=n_node_in, n_node_out=n_node_out,
                    n_hidden_layers=n_hidden_layers,
                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build hidden layers sizes
    hidden_layer_sizes = n_hidden_layers*[hidden_layer_size,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get feed-forward neural network layers
    layers = [layer for layer in model._node_fn.named_children()
              if 'Layer-' in layer[0]]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check number of layers
    if len(layers) != n_hidden_layers + 1:
        errors.append('Number of layers was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over layers
    for i, layer in enumerate(layers):
        # Check input layer features
        if i == 0:
            if layer[1].in_features != n_node_in:
                errors.append('Number of input features of input layer was '
                              'not properly set.')  
            if len(layers) == 1:
                if layer[1].out_features != n_node_out:
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
            if layer[1].out_features != n_node_out:
                errors.append('Number of output features of output layer was '
                              'not properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hidden layer features
        else:
            if layer[1].in_features != hidden_layer_sizes[i - 1] \
                    or layer[1].out_features != hidden_layer_sizes[i]:
                errors.append('Number of input/output features of hidden '
                              'layer was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out, n_hidden_layers,'
                         'hidden_layer_size',
                         [(1, 1, 5, 2, 4),
                          (10, 3, 2, 1, 2),
                          (5, 2, 4, 0, 2),
                          ])
def test_decoder_forward(n_nodes, n_node_in, n_node_out, n_hidden_layers,
                         hidden_layer_size):
    """Test GNN-based decoder forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based decoder
    model = Decoder(n_node_in=n_node_in, n_node_out=n_node_out,
                    n_hidden_layers=n_hidden_layers,
                    hidden_layer_size=hidden_layer_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = torch.rand(n_nodes, n_node_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out = model(node_features_in)
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
def test_epd_init():
    """Test GNN-based Encoder-Process-Decoder constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based Encoder-Process-Decoder model
    model = EncodeProcessDecode(n_message_steps=5, n_node_in=2, n_node_out=4,
                                n_edge_in=3, enc_n_hidden_layers=2,
                                pro_n_hidden_layers=3, dec_n_hidden_layers=4,
                                hidden_layer_size=5,
                                enc_node_hidden_activation=torch.nn.ReLU,
                                enc_node_output_activation=torch.nn.Identity,
                                enc_edge_hidden_activation=torch.nn.ReLU,
                                enc_edge_output_activation=torch.nn.Identity,
                                pro_node_hidden_activation=torch.nn.ReLU,
                                pro_node_output_activation=torch.nn.Identity,
                                pro_edge_hidden_activation=torch.nn.ReLU,
                                pro_edge_output_activation=torch.nn.Identity,
                                dec_node_hidden_activation=torch.nn.ReLU,
                                dec_node_output_activation=torch.nn.Identity,
                                is_node_res_connect=False,
                                is_edge_res_connect=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not isinstance(model._encoder, torch.nn.Module):
        errors.append('GNN-based encoder is not a torch.nn.Module.')
    if not isinstance(model._processor, torch_geometric.nn.MessagePassing):
        errors.append('GNN-based processor is not a '
                      'torch_geometric.nn.MessagePassing.')
    if not isinstance(model._decoder, torch.nn.Module):
        errors.append('GNN-based decoder is not a torch.nn.Module.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_message_steps, n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_hidden_layers,'
                         'hidden_layer_size, is_node_res_connect,'
                         'is_edge_res_connect',
                         [(4, 10, 5, 5, 20, 4, 6, 3, True, True),
                          ])
def test_epd_forward(n_message_steps, n_nodes, n_node_in, n_node_out, n_edges,
                     n_edge_in, n_hidden_layers, hidden_layer_size,
                     is_node_res_connect, is_edge_res_connect):
    """Test GNN-based Encoder-Process-Decoder forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based Encoder-Process-Decoder model
    model = EncodeProcessDecode(n_message_steps=n_message_steps,
                                n_node_in=n_node_in, n_node_out=n_node_out,
                                n_edge_in=n_edge_in,
                                enc_n_hidden_layers=n_hidden_layers,
                                pro_n_hidden_layers=n_hidden_layers,
                                dec_n_hidden_layers=n_hidden_layers,
                                hidden_layer_size=hidden_layer_size,
                                enc_node_hidden_activation=torch.nn.ReLU,
                                enc_node_output_activation=torch.nn.Identity,
                                enc_edge_hidden_activation=torch.nn.ReLU,
                                enc_edge_output_activation=torch.nn.Identity,
                                pro_node_hidden_activation=torch.nn.ReLU,
                                pro_node_output_activation=torch.nn.Identity,
                                pro_edge_hidden_activation=torch.nn.ReLU,
                                pro_edge_output_activation=torch.nn.Identity,
                                dec_node_hidden_activation=torch.nn.ReLU,
                                dec_node_output_activation=torch.nn.Identity,
                                is_node_res_connect=is_node_res_connect,
                                is_edge_res_connect=is_edge_res_connect)
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
    node_features_out = model(node_features_in, edge_features_in,
                              edges_indexes)
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