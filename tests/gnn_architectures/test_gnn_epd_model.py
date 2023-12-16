"""Test Graph Neural Network based Encoder-Process-Decoder model."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import pytest
import torch
import torch_geometric.nn
# Local
from src.vegapunk.gnn_base_model.model.gnn_epd_model import Processor
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
@pytest.mark.parametrize('n_message_steps, n_node_in, n_node_out, n_edge_in,'
                         'n_edge_out, n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'is_node_res_connect, is_edge_res_connect,'
                         'is_global_res_connect',
                         [(1, 1, 5, 2, 3, 0, 0, 2, 4, False, False, False),
                          (1, 3, 3, 1, 4, 2, 3, 1, 2, True, False, False),
                          (1, 1, 5, 4, 4, 0, 0, 2, 4, False, True, False),
                          (1, 5, 5, 4, 4, 1, 1, 2, 4, True, True, True),
                          (2, 3, 3, 4, 4, 0, 0, 1, 2, False, False, False),
                          (2, 3, 3, 4, 4, 0, 2, 1, 2, True, True, False),
                          (2, 0, 3, 4, 4, 0, 0, 1, 2, False, True, False),
                          (2, 0, 3, 4, 4, 1, 1, 1, 2, False, False, True),
                          (2, 3, 3, 0, 4, 0, 0, 1, 2, True, False, False),
                          (2, 3, 3, 0, 4, 0, 0, 1, 2, False, False, False),
                          ])
def test_processor_init(n_message_steps, n_node_in, n_node_out, n_edge_in,
                        n_edge_out, n_global_in, n_global_out, n_hidden_layers,
                        hidden_layer_size, is_node_res_connect,
                        is_edge_res_connect, is_global_res_connect):
    """Test GNN-based processor constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based processor
    model = Processor(n_message_steps=n_message_steps,
                      n_node_out=n_node_out, n_edge_out=n_edge_out,
                      n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      n_node_in=n_node_in, n_edge_in=n_edge_in,
                      n_global_in=n_global_in, n_global_out=n_global_out,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect,
                      is_global_res_connect=is_global_res_connect)
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
                         'n_edge_out, n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'is_node_res_connect, is_edge_res_connect,'
                         'is_global_res_connect',
                         [(0, 1, 5, 2, 3, 0, 0, 2, 4, False, False, False),
                          (1, 4, 3, 1, 4, 0, 0, 1, 2, True, False, False),
                          (1, 1, 5, 4, 2, 0, 0, 2, 4, False, True, False),
                          (1, 2, 5, 4, 3, 0, 0, 2, 4, True, True, False),
                          (2, 3, 3, 4, 2, 0, 0, 1, 2, False, False, False),
                          (2, 3, 1, 4, 4, 0, 0, 1, 2, False, False, False),
                          (2, 3, 3, 3, 4, 0, 0, 1, 2, False, False, False),
                          (2, 0, 3, 0, 4, 0, 0, 1, 2, False, False, False),
                          (2, 0, 3, 4, 4, 1, 2, 1, 2, False, False, True),
                          ])
def test_processor_init_invalid(n_message_steps, n_node_in, n_node_out,
                                n_edge_in, n_edge_out, n_global_in,
                                n_global_out, n_hidden_layers,
                                hidden_layer_size, is_node_res_connect,
                                is_edge_res_connect, is_global_res_connect):
    """Test detection of invalid input to GNN-based processor constructor."""
    with pytest.raises(RuntimeError):
        # Build GNN-based processor
        _ = Processor(n_message_steps=n_message_steps, 
                      n_node_out=n_node_out, n_edge_out=n_edge_out,
                      n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      n_node_in=n_node_in, n_edge_in=n_edge_in,
                      n_global_in=n_global_in, n_global_out=n_global_out,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect,
                      is_global_res_connect=is_global_res_connect)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    'n_message_steps, n_nodes, n_node_in, n_node_out, n_edges, n_edge_in,'
    'n_edge_out, n_global_in, n_global_out, n_hidden_layers,'
    'hidden_layer_size, is_node_res_connect, is_edge_res_connect,'
    'is_global_res_connect',
    [(1, 10, 1, 5, 20, 2, 3, 0, 0, 2, 4, False, False, False),
     (1, 10, 3, 3, 20, 1, 4, 5, 2, 1, 2, True, False, False),
     (1, 10, 1, 5, 20, 4, 4, 0, 0, 2, 4, False, True, False),
     (1, 10, 5, 5, 20, 4, 4, 3, 3, 2, 4, True, True, True),
     (2, 10, 3, 3, 20, 4, 4, 0, 0, 1, 2, False, False, False),
     (2, 10, 3, 3, 20, 4, 4, 0, 0, 1, 2, True, True, False),
     (2, 10, 0, 3, 20, 4, 4, 0, 0, 1, 2, False, True, False),
     (2, 10, 3, 3, 20, 0, 4, 1, 1, 1, 2, True, False, True),
     (1, 10, 0, 5, 20, 4, 4, 0, 0, 2, 4, False, True, False),
     (1, 10, 3, 3, 20, 0, 4, 0, 0, 1, 2, True, False, False),
     (1, 10, 3, 3, 20, 0, 4, 2, 2, 1, 2, True, False, True),
     ])
def test_processor_forward(n_message_steps, n_nodes, n_node_in, n_node_out,
                           n_edges, n_edge_in, n_edge_out, n_global_in,
                           n_global_out, n_hidden_layers, hidden_layer_size,
                           is_node_res_connect, is_edge_res_connect,
                           is_global_res_connect):
    """Test GNN-based processor forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based processor
    model = Processor(n_message_steps=n_message_steps, 
                      n_node_out=n_node_out, n_edge_out=n_edge_out,
                      n_hidden_layers=n_hidden_layers,
                      hidden_layer_size=hidden_layer_size,
                      n_node_in=n_node_in, n_edge_in=n_edge_in,
                      n_global_in=n_global_in, n_global_out=n_global_out,
                      is_node_res_connect=is_node_res_connect,
                      is_edge_res_connect=is_edge_res_connect,
                      is_global_res_connect=is_global_res_connect)
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
    if n_global_out > 0:
        if not isinstance(global_features_out, torch.Tensor):
            errors.append('Global features output matrix is not torch.Tensor.')
        elif not torch.equal(torch.tensor(global_features_out.size()),
                            torch.tensor([1, n_global_out])):
            errors.append('Global features output matrix is not '
                            'torch.Tensor(2d) of shape (1, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))