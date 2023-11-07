"""Test GNNPatchGraphData class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.data
# Local
from src.vegapunk.gnn_model.gnn_patch_data import GNNPatchGraphData
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
def test_gnn_patch_data_init_2d(squad4_regular_mesh):
    """Test 2D GNN-based material patch graph data initialization."""
    nodes_coords, _ = squad4_regular_mesh
    gnn_patch_data = GNNPatchGraphData(2, nodes_coords)
    assert gnn_patch_data._n_dim == 2 and \
        np.allclose(gnn_patch_data._nodes_coords, nodes_coords), \
        'Failed 2D GNN-based material patch graph data initialization.'
# -----------------------------------------------------------------------------
def test_get_nodes_coords(gnn_patch_graph_2d):
    """Test node coordinates getter type."""
    assert isinstance(gnn_patch_graph_2d.get_nodes_coords(), np.ndarray), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_graph_edges_indexes(gnn_patch_graph_2d):
    """Test material patch graph edges indexes getter type."""
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    assert isinstance(edges_indexes, np.ndarray) \
            and edges_indexes.dtype == int, 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_set_graph_edges_indexes(squad4_regular_mesh):
    """Test setup of material patch graph edges indexes."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nodes_coords, connected_nodes = squad4_regular_mesh
    gnn_patch_data = GNNPatchGraphData(2, nodes_coords)
    edges_indexes_mesh = \
        GNNPatchGraphData.get_edges_indexes_mesh(connected_nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test case: no edge data
    gnn_patch_data.set_graph_edges_indexes()
    edges_indexes = gnn_patch_data.get_graph_edges_indexes()
    if not isinstance(edges_indexes, np.ndarray) \
            and edges_indexes.dtype == int:
        errors.append('Material patch graph edges indexes were not properly '
                      'set. Case: No edge data.')
    # Test case: connectivity radius
    gnn_patch_data.set_graph_edges_indexes(connect_radius=1.5)
    edges_indexes = gnn_patch_data.get_graph_edges_indexes()
    if not isinstance(edges_indexes, np.ndarray) \
            and edges_indexes.dtype == int:
        errors.append('Material patch graph edges indexes were not properly '
                      'set. Case: Connectivity radius.')
    # Test case: finite element mesh
    gnn_patch_data.set_graph_edges_indexes(
        edges_indexes_mesh=edges_indexes_mesh)
    edges_indexes = gnn_patch_data.get_graph_edges_indexes()
    if not isinstance(edges_indexes, np.ndarray) \
            and edges_indexes.dtype == int:
        errors.append('Material patch graph edges indexes were not properly '
                      'set. Case: Connectivity radius.')
    # Test case: connectivity radius + finite element mesh
    gnn_patch_data.set_graph_edges_indexes(connect_radius=1.5,
        edges_indexes_mesh=edges_indexes_mesh)
    edges_indexes = gnn_patch_data.get_graph_edges_indexes()
    if not isinstance(edges_indexes, np.ndarray) \
            and edges_indexes.dtype == int:
        errors.append('Material patch graph edges indexes were not properly '
                      'set. Case: Connectivity radius and finite element mesh.'
                      )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_edges_from_local_radius_type(squad4_regular_mesh):
    """Test detection of edges between nodes within a connectivity radius."""
    nodes_coords, _ = squad4_regular_mesh
    edges_indexes_radius = GNNPatchGraphData._get_edges_from_local_radius(
        nodes_coords, connect_radius=1.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(edges_indexes_radius, np.ndarray) \
        and edges_indexes_radius.dtype == int, 'Graph edges indexes ' \
        'must be stored as numpy 2d array of shape (n_edges, 2).'
# -----------------------------------------------------------------------------
def test_get_edges_indexes_mesh(squad4_regular_mesh):
    """Test conversion of mesh connected nodes to edges indexes matrix."""
    _, connected_nodes = squad4_regular_mesh
    edges_indexes_mesh = \
        GNNPatchGraphData.get_edges_indexes_mesh(connected_nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(edges_indexes_mesh, np.ndarray) \
        and edges_indexes_mesh.dtype == int, 'Graph edges indexes ' \
        'must be stored as numpy 2d array of shape (n_edges, 2).'
# -----------------------------------------------------------------------------
def test_check_edges_indexes_matrix():
    """Test checking procedure of edges indexes matrix."""
    with pytest.raises(RuntimeError):
        edges_indexes = 'invalid_type'
        GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes)
    with pytest.raises(RuntimeError):
        edges_indexes = np.array([[0.0, 1.0], [2.0, 3.0]])
        GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes)
    with pytest.raises(RuntimeError):
        edges_indexes = np.zeros((1, 2, 3), dtype=int)
        GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes)
    with pytest.raises(RuntimeError):
        edges_indexes = np.zeros((1, 3), dtype=int)
        GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes)
# -----------------------------------------------------------------------------
def test_set_and_get_node_features_matrix(gnn_patch_graph_2d):
    """Test setter and getter of nodes input features matrix."""
    nodes_coords = gnn_patch_graph_2d.get_nodes_coords()
    n_nodes = nodes_coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_node_features_matrix = np.zeros((n_nodes, 5))
    gnn_patch_graph_2d.set_node_features_matrix(input_node_features_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_node_features_matrix(),
                       input_node_features_matrix), \
        'Node input features matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_node_features_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid nodes input features matrix."""
    nodes_coords = gnn_patch_graph_2d.get_nodes_coords()
    n_nodes = nodes_coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_node_features_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_node_features_matrix(input_node_features_matrix)
    # Test invalid number of nodes
    with pytest.raises(RuntimeError):
        input_node_features_matrix = np.zeros((n_nodes + 1, 5))
        gnn_patch_graph_2d.set_node_features_matrix(input_node_features_matrix)
# -----------------------------------------------------------------------------
def test_set_and_get_edge_features_matrix(gnn_patch_graph_2d):
    """Test setter and getter of edges input features matrix."""
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edges = edges_indexes.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_edge_features_matrix = np.zeros((n_edges, 5))
    gnn_patch_graph_2d.set_edge_features_matrix(input_edge_features_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_edge_features_matrix(),
                       input_edge_features_matrix), \
        'Edge input features matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_edge_features_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid edges input features matrix."""
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edges = edges_indexes.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_edge_features_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_edge_features_matrix(input_edge_features_matrix)
    # Test invalid number of edges
    with pytest.raises(RuntimeError):
        input_edge_features_matrix = np.zeros((n_edges + 1, 5))
        gnn_patch_graph_2d.set_edge_features_matrix(input_edge_features_matrix)
# -----------------------------------------------------------------------------
def test_set_and_get_global_features_matrix(gnn_patch_graph_2d):
    """Test setter and getter of global input features matrix."""
    input_global_features_matrix = np.zeros((1, 5))
    gnn_patch_graph_2d.set_global_features_matrix(input_global_features_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_global_features_matrix(),
                       input_global_features_matrix), \
        'Global input features matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_global_features_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid global input features matrix."""
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_global_features_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_global_features_matrix(
            input_global_features_matrix)
    # Test invalid shape
    with pytest.raises(RuntimeError):
        input_global_features_matrix = np.zeros((2, 5))
        gnn_patch_graph_2d.set_global_features_matrix(
            input_global_features_matrix)
# -----------------------------------------------------------------------------
def test_set_and_get_node_targets_matrix(gnn_patch_graph_2d):
    """Test setter and getter of nodes targets matrix."""
    nodes_coords = gnn_patch_graph_2d.get_nodes_coords()
    n_nodes = nodes_coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_node_targets_matrix = np.zeros((n_nodes, 5))
    gnn_patch_graph_2d.set_node_targets_matrix(input_node_targets_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_node_targets_matrix(),
                       input_node_targets_matrix), \
        'Node targets matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_node_targets_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid nodes targets matrix."""
    nodes_coords = gnn_patch_graph_2d.get_nodes_coords()
    n_nodes = nodes_coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_node_targets_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_node_targets_matrix(input_node_targets_matrix)
    # Test invalid number of nodes
    with pytest.raises(RuntimeError):
        input_node_targets_matrix = np.zeros((n_nodes + 1, 5))
        gnn_patch_graph_2d.set_node_targets_matrix(input_node_targets_matrix)
# -----------------------------------------------------------------------------
def test_set_and_get_edge_targets_matrix(gnn_patch_graph_2d):
    """Test setter and getter of edges targets matrix."""
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edges = edges_indexes.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_edge_targets_matrix = np.zeros((n_edges, 5))
    gnn_patch_graph_2d.set_edge_targets_matrix(input_edge_targets_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_edge_targets_matrix(),
                       input_edge_targets_matrix), \
        'Edge targets matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_edge_targets_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid edges targets matrix."""
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edges = edges_indexes.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_edge_targets_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_edge_targets_matrix(input_edge_targets_matrix)
    # Test invalid number of edges
    with pytest.raises(RuntimeError):
        input_edge_targets_matrix = np.zeros((n_edges + 1, 5))
        gnn_patch_graph_2d.set_edge_targets_matrix(input_edge_targets_matrix)
# -----------------------------------------------------------------------------
def test_set_and_get_global_targets_matrix(gnn_patch_graph_2d):
    """Test setter and getter of global targets matrix."""
    input_global_targets_matrix = np.zeros((1, 5))
    gnn_patch_graph_2d.set_global_targets_matrix(input_global_targets_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(gnn_patch_graph_2d.get_global_targets_matrix(),
                       input_global_targets_matrix), \
        'Global targets matrix was not properly set or returned.'
# -----------------------------------------------------------------------------
def test_set_global_targets_matrix_invalid(gnn_patch_graph_2d):
    """Test detection of invalid global targets matrix."""
    # Test invalid type
    with pytest.raises(RuntimeError):
        input_global_targets_matrix = 'invalid_type'
        gnn_patch_graph_2d.set_global_targets_matrix(
            input_global_targets_matrix)
    # Test invalid shape
    with pytest.raises(RuntimeError):
        input_global_targets_matrix = np.zeros((2, 5))
        gnn_patch_graph_2d.set_global_targets_matrix(
            input_global_targets_matrix)
# -----------------------------------------------------------------------------
def test_get_torch_data_object(gnn_patch_graph_2d):
    """Test PyG homogeneous graph data object type."""
    nodes_coords = gnn_patch_graph_2d.get_nodes_coords()
    n_nodes = nodes_coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    node_features_matrix = np.zeros((n_nodes, 5))
    gnn_patch_graph_2d.set_node_features_matrix(node_features_matrix)    
    node_targets_matrix = np.zeros((n_nodes, 5))
    gnn_patch_graph_2d.set_node_targets_matrix(node_targets_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(gnn_patch_graph_2d.get_torch_data_object(),
                      torch_geometric.data.Data), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_plot_material_patch_graph(gnn_patch_graph_2d, tmp_path, monkeypatch):
    """Test plot of material patch graph."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        gnn_patch_graph_2d.plot_material_patch_graph(
            is_show_plot=True, is_save_plot=True, save_directory=str(tmp_path),
            plot_name='material_patch_graph', is_overwrite_file=True)
        gnn_patch_graph_2d.plot_material_patch_graph(
            is_show_plot=True, is_save_plot=True, save_directory=str(tmp_path),
            plot_name=None, is_overwrite_file=False)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'material patch graph.'