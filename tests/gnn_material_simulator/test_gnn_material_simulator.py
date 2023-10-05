"""Test Graph Neural Network based material patch model."""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
# Third-party
import pytest
import torch
import sklearn.preprocessing
# Local
from src.vegapunk.gnn_model.gnn_material_simulator import GNNMaterialPatchModel
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
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_message_steps,'
                         'n_hidden_layers, hidden_layer_size, model_name,'
                         'is_data_normalization, device_type',
                         [(2, 5, 4, 10, 2, 3, 'material_patch_model', True,
                           'cpu'),
                          ])
def test_material_patch_model_init(n_node_in, n_node_out, n_edge_in,
                                   n_message_steps, n_hidden_layers,
                                   hidden_layer_size, model_name,
                                   is_data_normalization, device_type,
                                   tmp_path):
    """Test GNN-based material patch model constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = tmp_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in,
                           n_message_steps=n_message_steps,
                           n_hidden_layers=n_hidden_layers,
                           hidden_layer_size=hidden_layer_size,
                           model_directory=model_directory,
                           model_name=model_name,
                           is_data_normalization=is_data_normalization,
                           device_type=device_type)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model attributes
    if not (model._n_node_in == n_node_in,
            model._n_node_out == n_node_out,
            model._n_edge_in == n_edge_in,
            model._n_message_steps == n_message_steps,
            model._n_hidden_layers == n_hidden_layers,
            model._hidden_layer_size == hidden_layer_size,
            model.model_directory == str(tmp_path),
            model.model_name == model_name,
            model.is_data_normalization == is_data_normalization,
            model._device == device_type):
        errors.append('One or more attributes of GNN-based material patch '
                      'model class were not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check GNN-based material patch model class initialization parameters
    # storage
    model_init_file_path = os.path.join(model_directory,
                                        'model_init_args' + '.pkl')
    with open(model_init_file_path, 'rb') as model_init_file:
        loaded_init_args = pickle.load(model_init_file)
    if loaded_init_args != model_init_args:
        errors.append('GNN-based material patch model class initialization '
                      'parameters were not properly recovered.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model
    if not isinstance(model._gnn_epd_model, torch.nn.Module):
        errors.append('GNN-based Encoder-Process-Decoder model is not a '
                      'torch.nn.Module.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data scalers
    if is_data_normalization:
        if not isinstance(model._data_scalers, dict):
            errors.append('GNN-based Encoder-Process-Decoder model data '
                          'scalers were not initialized.')
        if not all([key in model._data_scalers.keys()
                    for key in ('node_features_in', 'edge_features_in',
                                'node_features_out')]):
            errors.append('One or more GNN-based Encoder-Process-Decoder '
                          'model data scalers were not initialized.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_message_steps,'
                         'n_hidden_layers, hidden_layer_size, model_name,'
                         'is_data_normalization, device_type',
                         [(2, 5, 4, 10, 2, 3, 'material_patch_model', True,
                           'cpu'),
                          ])
def test_material_patch_model_init_invalid(n_node_in, n_node_out, n_edge_in,
                                           n_message_steps, n_hidden_layers,
                                           hidden_layer_size, model_name,
                                           is_data_normalization, device_type,
                                           tmp_path):
    """Test GNN-based material patch model constructor."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = str(tmp_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set valid GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in,
                           n_message_steps=n_message_steps,
                           n_hidden_layers=n_hidden_layers,
                           hidden_layer_size=hidden_layer_size,
                           model_directory=model_directory,
                           model_name=model_name,
                           is_data_normalization=is_data_normalization,
                           device_type=device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    with pytest.raises(RuntimeError):
        # Test detection of unexistent directory
        test_init_args = model_init_args.copy()
        test_init_args['model_directory'] = 'unknown_directory'
        _ = GNNMaterialPatchModel(**test_init_args)
    with pytest.raises(RuntimeError):
        # Test detection of invalid model name
        test_init_args = model_init_args.copy()
        test_init_args['model_name'] = 0
        _ = GNNMaterialPatchModel(**test_init_args)
    with pytest.raises(RuntimeError):
        # Test detection of invalid device
        test_init_args = model_init_args.copy()
        test_init_args['device_type'] = 'unknown_device_type'
        model = GNNMaterialPatchModel(**test_init_args)
# -----------------------------------------------------------------------------
def test_save_init_args_invalid(tmp_path):
    """Test detection of failed save of model initialization parameters."""
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=2, n_node_out=5, n_edge_in=3,
                        n_message_steps=2, n_hidden_layers=2,
                        hidden_layer_size=2, model_directory=str(tmp_path),
                        model_name='material_patch_model',
                        is_data_normalization=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of unexistent material patch model directory
        model.model_directory = 'unknown_dir'
        model._save_model_init_args()
# -----------------------------------------------------------------------------
def test_get_input_features_from_graph(graph_patch_data_2d, tmp_path):
    """Test extraction of input features from material patch graph.
    
    Test is performed assuming no data normalization.
    """
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material patch graph input features matrices
    node_features_in = graph_patch_data_2d.get_node_features_matrix()
    edge_features_in = graph_patch_data_2d.get_edge_features_matrix()
    # Get material patch graph edges indexes
    edges_indexes = graph_patch_data_2d.get_graph_edges_indexes()
    # Get material patch graph nodes targets matrix
    node_targets_matrix = graph_patch_data_2d.get_node_targets_matrix()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_patch_data_2d.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=node_features_in.shape[1],
                           n_node_out=node_targets_matrix.shape[1],
                           n_edge_in=edge_features_in.shape[1],
                           n_message_steps=2,
                           n_hidden_layers=2,
                           hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=False)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input features from material patch graph
    test_node_features_in, test_edge_features_in, test_edges_indexes = \
        model.get_input_features_from_graph(pyg_graph,
                                            is_normalized=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes features input matrix
    if not torch.allclose(test_node_features_in,
                          torch.tensor(node_features_in, dtype=torch.float)):
        errors.append('Extracted nodes features input matrix does not '
                      'match graph nodes features input matrix.')
    # Check edges features input matrix
    if not torch.allclose(test_edge_features_in,
                          torch.tensor(edge_features_in, dtype=torch.float)):
        errors.append('Extracted edges features input matrix does not '
                      'match graph edges features input matrix.')
    # Check edges indexes
    if not torch.allclose(test_edges_indexes,
                          torch.tensor(edges_indexes.T, dtype=torch.long)):
        errors.append('Extracted edges indexes do not match graph edges '
                      'indexes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_input_features_from_graph_invalid(graph_patch_data_2d, tmp_path):
    """Test invalid extraction of input features from material patch graph."""
    # Get material patch graph input features matrices
    node_features_in = graph_patch_data_2d.get_node_features_matrix()
    edge_features_in = graph_patch_data_2d.get_edge_features_matrix()
    # Get material patch graph nodes targets matrix
    node_targets_matrix = graph_patch_data_2d.get_node_targets_matrix()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_patch_data_2d.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set valid GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=node_features_in.shape[1],
                           n_node_out=node_targets_matrix.shape[1],
                           n_edge_in=edge_features_in.shape[1],
                           n_message_steps=2,
                           n_hidden_layers=2,
                           hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test invalid input graph
        model = GNNMaterialPatchModel(**model_init_args)
        _, _, _ = model.get_input_features_from_graph('invalid_type',
                                                      is_normalized=False)
    with pytest.raises(RuntimeError):
        # Test inconsistent number of node input features
        test_init_args = model_init_args.copy()
        test_init_args['n_node_in'] = node_features_in.shape[1] + 1
        model = GNNMaterialPatchModel(**test_init_args)
        _, _, _ = model.get_input_features_from_graph(pyg_graph,
                                                      is_normalized=False)
    with pytest.raises(RuntimeError):
        # Test inconsistent number of edge input features
        test_init_args = model_init_args.copy()
        test_init_args['n_edge_in'] = edge_features_in.shape[1] + 1
        model = GNNMaterialPatchModel(**test_init_args)
        _, _, _ = model.get_input_features_from_graph(pyg_graph,
                                                      is_normalized=False)
# -----------------------------------------------------------------------------
def test_get_output_features_from_graph(graph_patch_data_2d, tmp_path):
    """Test extraction of output features from material patch graph.
    
    Test is performed assuming no data normalization.
    """
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material patch graph input features matrices
    node_features_in = graph_patch_data_2d.get_node_features_matrix()
    edge_features_in = graph_patch_data_2d.get_edge_features_matrix()
    # Get material patch graph nodes targets matrix
    node_targets_matrix = graph_patch_data_2d.get_node_targets_matrix()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_patch_data_2d.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=node_features_in.shape[1],
                           n_node_out=node_targets_matrix.shape[1],
                           n_edge_in=edge_features_in.shape[1],
                           n_message_steps=2,
                           n_hidden_layers=2,
                           hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=False)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input features from material patch graph
    test_node_features_out = \
        model.get_output_features_from_graph(pyg_graph, is_normalized=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes features output matrix
    if not torch.allclose(test_node_features_out,
                          torch.tensor(node_targets_matrix,
                                       dtype=torch.float)):
        errors.append('Extracted nodes features output matrix does not '
                      'match graph nodes features output matrix.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_output_features_from_graph_invalid(graph_patch_data_2d, tmp_path):
    """Test invalid extraction of output features from material patch graph."""
    # Get material patch graph output features matrices
    node_features_in = graph_patch_data_2d.get_node_features_matrix()
    edge_features_in = graph_patch_data_2d.get_edge_features_matrix()
    # Get material patch graph nodes targets matrix
    node_targets_matrix = graph_patch_data_2d.get_node_targets_matrix()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_patch_data_2d.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set valid GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=node_features_in.shape[1],
                           n_node_out=node_targets_matrix.shape[1],
                           n_edge_in=edge_features_in.shape[1],
                           n_message_steps=2,
                           n_hidden_layers=2,
                           hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test invalid input graph
        model = GNNMaterialPatchModel(**model_init_args)
        _, _, _ = model.get_input_features_from_graph('invalid_type',
                                                      is_normalized=False)
# -----------------------------------------------------------------------------
def test_fit_data_scalers(batch_graph_patch_data_2d, tmp_path):
    """Test fitting of GNN-based material patch model data scalers."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of node input and output features
    n_node_in = \
        batch_graph_patch_data_2d[0].get_node_features_matrix().shape[1]
    n_node_out = \
        batch_graph_patch_data_2d[0].get_node_targets_matrix().shape[1]
    # Get number of edge input features
    n_edge_in = \
        batch_graph_patch_data_2d[0].get_edge_features_matrix().shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_message_steps=2,
                           n_hidden_layers=2, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=True)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build dataset
    dataset = [gnn_patch_data.get_torch_data_object()
               for gnn_patch_data in batch_graph_patch_data_2d]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit model data scalers
    model.fit_data_scalers(dataset, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check fitted data scalers
    if not isinstance(model.get_fitted_data_scaler('node_features_in'),
                      sklearn.preprocessing.StandardScaler):
        errors.append('Model data scaler for node input features was not '
                      'properly fitted.')
    if not isinstance(model.get_fitted_data_scaler('edge_features_in'),
                      sklearn.preprocessing.StandardScaler):
        errors.append('Model data scaler for edge input features was not '
                      'properly fitted.')
    if not isinstance(model.get_fitted_data_scaler('node_features_out'),
                      sklearn.preprocessing.StandardScaler):
        errors.append('Model data scaler for node output features was not '
                      'properly fitted.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_fitted_data_scaler_invalid(tmp_path):
    """Test GNN-based material patch model fitted data scaler getter."""
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=2, n_node_out=4, n_edge_in=3,
                           n_message_steps=2, n_hidden_layers=2,
                           hidden_layer_size=2, model_directory=tmp_path,
                           is_data_normalization=True)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test unknown data scaler
        _ = model.get_fitted_data_scaler(features_type='unknown_type')
    with pytest.raises(RuntimeError):
        # Test unfitted data scaler
        _ = model.get_fitted_data_scaler(features_type='node_features_in')
# -----------------------------------------------------------------------------
def test_check_normalized_return(tmp_path):
    """Test detection of unfitted model data scalers."""
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=2, n_node_out=4, n_edge_in=3,
                           n_message_steps=2, n_hidden_layers=2,
                           hidden_layer_size=2, model_directory=str(tmp_path),
                           is_data_normalization=False)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test unknown data scaler
        _ = model._check_normalized_return()
# -----------------------------------------------------------------------------
def test_get_input_features_from_graph_norm(batch_graph_patch_data_2d,
                                            tmp_path):
    """Test extraction of input features from material patch graph.
    
    Test is performed assuming data normalization.
    """
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pick material patch graph data
    graph_data = batch_graph_patch_data_2d[0]
    # Get material patch graph input features matrices
    node_features_in = graph_data.get_node_features_matrix()
    edge_features_in = graph_data.get_edge_features_matrix()
    # Get material patch graph edges indexes
    edges_indexes = graph_data.get_graph_edges_indexes()
    # Get number of node input and output features
    n_node_in = node_features_in.shape[1]
    n_node_out = graph_data.get_node_targets_matrix().shape[1]
    # Get number of edge input features
    n_edge_in = edge_features_in.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_message_steps=2,
                           n_hidden_layers=2, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=True)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build dataset
    dataset = [gnn_patch_data.get_torch_data_object()
               for gnn_patch_data in batch_graph_patch_data_2d]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit model data scalers
    model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input features from material patch graph
    test_node_features_in, test_edge_features_in, test_edges_indexes = \
        model.get_input_features_from_graph(graph=dataset[0],
                                            is_normalized=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes features input matrix
    if not isinstance(test_node_features_in, torch.Tensor):
        errors.append('Nodes features input matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(test_node_features_in.size()),
                         torch.tensor(torch.tensor(node_features_in).size())):
        errors.append('Nodes features input matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    # Check edges features input matrix
    if not isinstance(test_edge_features_in, torch.Tensor):
        errors.append('Nodes features input matrix is not torch.Tensor.')
    elif not torch.equal(torch.tensor(test_edge_features_in.size()),
                         torch.tensor(torch.tensor(edge_features_in).size())):
        errors.append('Edges features input matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    # Check edges indexes
    if not torch.allclose(test_edges_indexes,
                          torch.tensor(edges_indexes.T, dtype=torch.long)):
        errors.append('Extracted edges indexes do not match graph edges '
                      'indexes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_output_features_from_graph_norm(batch_graph_patch_data_2d,
                                             tmp_path):
    """Test extraction of output features from material patch graph.
    
    Test is performed assuming data normalization.
    """
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pick material patch graph data
    graph_data = batch_graph_patch_data_2d[0]
    # Get material patch graph input features matrices
    node_features_in = graph_data.get_node_features_matrix()
    edge_features_in = graph_data.get_edge_features_matrix()
    # Get material patch graph nodes targets matrix
    node_targets_matrix = graph_data.get_node_targets_matrix()
    # Get number of node input and output features
    n_node_in = node_features_in.shape[1]
    n_node_out = graph_data.get_node_targets_matrix().shape[1]
    # Get number of edge input features
    n_edge_in = edge_features_in.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_message_steps=2,
                           n_hidden_layers=2, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=True)
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build dataset
    dataset = [gnn_patch_data.get_torch_data_object()
               for gnn_patch_data in batch_graph_patch_data_2d]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit model data scalers
    model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get output features from material patch graph
    test_node_features_out = \
        model.get_output_features_from_graph(graph=dataset[0],
                                             is_normalized=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes features output matrix
    if not isinstance(test_node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif not torch.equal(
            torch.tensor(test_node_features_out.size()),
            torch.tensor(torch.tensor(node_targets_matrix).size())):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_save_and_load_model_state(tmp_path):
    """Test save and load material patch model state."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=2, n_node_out=5, n_edge_in=3,
                        n_message_steps=2, n_hidden_layers=2,
                        hidden_layer_size=2, model_directory=str(tmp_path),
                        model_name='material_patch_model',
                        is_data_normalization=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # Get model state
    saved_state_dict = model.state_dict()
    # Save material patch model state to file
    model.save_model_state()
    # Load material patch model state from file
    loaded_training_step = model.load_model_state()
    # Get model state
    loaded_state_dict = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step is not None:
        errors.append('GNN-based material patch model unknown training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict) != str(loaded_state_dict):
        errors.append('GNN-based material patch model state was not properly'
                      'recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # Get model state
    saved_state_dict_0 = model.state_dict()
    # Save material patch model state to file
    model.save_model_state(training_step=0)
    # Load material patch model state from file
    loaded_training_step = model.load_model_state(training_step=0)
    # Get model state
    loaded_state_dict_0 = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step != 0:
        errors.append('GNN-based material patch model initial training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict_0) != str(loaded_state_dict_0):
        errors.append('GNN-based material patch model initial state was not '
                      'properly recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model (reinitializing parameters to
    # emulate parameters update)
    model = GNNMaterialPatchModel(**model_init_args)
    # Get model state
    saved_state_dict_1 = model.state_dict()
    # Save material patch model state to file
    model.save_model_state(training_step=1)
    # Load material patch model state from file
    loaded_training_step = model.load_model_state(training_step=1)
    # Get model state
    loaded_state_dict_1 = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step != 1:
        errors.append('GNN-based material patch model first training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict_1) != str(loaded_state_dict_1):
        errors.append('GNN-based material patch model first training step '
                      'state was not properly recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load material patch model state from file
    loaded_training_step = model.load_model_state(training_step=0,
                                                  is_remove_posterior=False)
    # Get model state
    loaded_state_dict_0 = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step != 0:
        errors.append('GNN-based material patch model old training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict_0) != str(loaded_state_dict_0):
        errors.append('GNN-based material patch model old training step '
                      'state was not properly recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load material patch model state from file
    loaded_training_step = model.load_model_state(is_latest=True)
    # Get model state
    loaded_state_dict_1 = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step != 1:
        errors.append('GNN-based material patch model latest training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict_1) != str(loaded_state_dict_1):
        errors.append('GNN-based material patch model latest training step '
                      'state was not properly recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load material patch model state from file
    loaded_training_step = model.load_model_state(training_step=0,
                                                  is_remove_posterior=True)
    # Get model state
    loaded_state_dict_0 = model.state_dict()
    # Check loaded model state training step
    if loaded_training_step != 0:
        errors.append('GNN-based material patch model old training step '
                      'was not properly recovered from file.')
    # Check loaded model state parameters
    if str(saved_state_dict_0) != str(loaded_state_dict_0):
        errors.append('GNN-based material patch model old training step '
                      'state was not properly recovered from file.')   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_save_and_load_model_state_invalid(tmp_path):
    """Test detection of failed save and load material patch model state."""
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=2, n_node_out=5, n_edge_in=3,
                        n_message_steps=2, n_hidden_layers=2,
                        hidden_layer_size=2, model_directory=str(tmp_path),
                        model_name='material_patch_model',
                        is_data_normalization=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test loading unexistent material patch model state file
        _ = model.load_model_state(training_step=0)
    with pytest.raises(RuntimeError):
        # Test loading unexistent material patch model state file
        _ = model.load_model_state(is_latest=True)
    with pytest.raises(RuntimeError):
        # Test detection of unexistent material patch model directory
        model.model_directory = 'unknown_dir'
        _ = model.load_model_state(is_latest=True)
    with pytest.raises(RuntimeError):
        # Test detection of unexistent material patch model directory
        model.model_directory = 'unknown_dir'
        _ = model.save_model_state()
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('is_data_normalization, is_normalized,',
                         [(False, False), (True, True), (True, False)
                          ])
def test_model_forward_propagation(batch_graph_patch_data_2d, tmp_path,
                                   is_data_normalization, is_normalized):
    """Test GNN-based material patch model forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pick material patch graph data
    graph_data = batch_graph_patch_data_2d[0]
    # Get material patch graph input features matrices
    node_features_in = graph_data.get_node_features_matrix()
    edge_features_in = graph_data.get_edge_features_matrix()
    # Get number of nodes
    n_nodes = node_features_in.shape[0]
    # Get number of node input and output features
    n_node_in = node_features_in.shape[1]
    n_node_out = graph_data.get_node_targets_matrix().shape[1]
    # Get number of edge input features
    n_edge_in = edge_features_in.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_message_steps=2,
                           n_hidden_layers=2, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=is_data_normalization)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_data.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data features normalization
    if is_data_normalization:
        # Build dataset
        dataset = [gnn_patch_data.get_torch_data_object()
                   for gnn_patch_data in batch_graph_patch_data_2d]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit model data scalers
        model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Predict internal forces
    node_internal_forces = model(pyg_graph, is_normalized=is_normalized)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes features output matrix
    if not isinstance(node_internal_forces, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif not torch.equal(
            torch.tensor(node_internal_forces.size()),
            torch.tensor((n_nodes, n_node_out))):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).') 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('is_data_normalization, is_normalized,',
                         [(False, True)
                          ])
def test_model_forward_propagation_invalid(batch_graph_patch_data_2d, tmp_path,
                                           is_data_normalization,
                                           is_normalized):
    """Test GNN-based material patch model forward invalid requests."""
    # Pick material patch graph data
    graph_data = batch_graph_patch_data_2d[0]
    # Get material patch graph input features matrices
    node_features_in = graph_data.get_node_features_matrix()
    edge_features_in = graph_data.get_edge_features_matrix()
    # Get number of nodes
    n_nodes = node_features_in.shape[0]
    # Get number of node input and output features
    n_node_in = node_features_in.shape[1]
    n_node_out = graph_data.get_node_targets_matrix().shape[1]
    # Get number of edge input features
    n_edge_in = edge_features_in.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_message_steps=2,
                           n_hidden_layers=2, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_data_normalization=is_data_normalization)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyG homogeneous graph data object
    pyg_graph = graph_data.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data features normalization
    if is_data_normalization:
        # Build dataset
        dataset = [gnn_patch_data.get_torch_data_object()
                   for gnn_patch_data in batch_graph_patch_data_2d]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit model data scalers
        model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Predict internal forces
    with pytest.raises(RuntimeError):
        # Requesting normalized output when model data scalars have not been
        # fitted
        _ = model(pyg_graph, is_normalized=is_normalized)