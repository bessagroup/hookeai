"""Test GNNPatchFeaturesGenerator class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
# Local
from src.vegapunk.gnn_material_patch.gnn_model_tools.features import \
    GNNPatchFeaturesGenerator
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
@pytest.mark.parametrize('is_n_edge, is_edges_indexes',
                         [(True, True), (True, False),
                          (False, True), (False, False)
                          ])
def test_gnn_patch_features_init_2d(gnn_patch_graph_2d, is_n_edge,
                                    is_edges_indexes):
    """Test 2D GNN-based material patch features generator."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_nodes = gnn_patch_graph_2d.get_nodes_coords().shape[0]
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edge = edges_indexes.shape[0]
    if is_n_edge and not is_edges_indexes:
        edges_indexes = None
    elif not is_n_edge and is_edges_indexes:
        n_edge = None
    elif not is_n_edge and not is_edges_indexes:
        edges_indexes = None
        n_edge = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_dim = 2
    n_time_steps = 5
    nodes_coords_hist = np.zeros((n_nodes, n_dim, n_time_steps))
    nodes_disps_hist = np.zeros((n_nodes, n_dim, n_time_steps))
    nodes_int_forces_hist = np.zeros((n_nodes, n_dim, n_time_steps))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    features_generator = GNNPatchFeaturesGenerator(
        n_dim, nodes_coords_hist, n_edge=n_edge, edges_indexes=edges_indexes,
        nodes_disps_hist=nodes_disps_hist,
        nodes_int_forces_hist=nodes_int_forces_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not np.allclose(features_generator._nodes_coords_hist,
                       nodes_coords_hist):
        errors.append('Failed to set nodes_coords_hist in 2D GNN-based '
                      'material patch features generator initialization.')
    if not np.allclose(features_generator._nodes_disps_hist,
                       nodes_disps_hist):
        errors.append('Failed to set nodes_disps_hist in 2D GNN-based '
                      'material patch features generator initialization.')
    if not np.allclose(features_generator._nodes_int_forces_hist,
                       nodes_int_forces_hist):
        errors.append('Failed to set nodes_int_forces_hist in 2D GNN-based '
                      'material patch features generator initialization.')
    if (edges_indexes is not None
            and not np.allclose(features_generator._edges_indexes,
                                edges_indexes)):
        errors.append('Failed to set edges_indexes in 2D GNN-based '
                      'material patch features generator initialization.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_available_nodes_features():
    """Test available nodes features getter."""
    assert isinstance(GNNPatchFeaturesGenerator.get_available_nodes_features(),
                      tuple), 'Unexpected return type.'   
# -----------------------------------------------------------------------------
def test_build_nodes_features_matrix(features_generator_2d):
    """Test computation of nodes features matrix with all features."""
    available_features = \
        GNNPatchFeaturesGenerator.get_available_nodes_features()
    node_features_matrix = features_generator_2d.build_nodes_features_matrix(
        features=available_features)
    assert isinstance(node_features_matrix, np.ndarray), \
        'Nodes features matrix was not properly computed.'
# -----------------------------------------------------------------------------
def test_build_nodes_features_matrix_invalid(features_generator_2d):
    """Test detection of invalid inputs to compute nodes features matrix."""
    # Test invalid features type
    with pytest.raises(RuntimeError):
        features = 'invalid_type'
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features)
    # Test unknown feature
    with pytest.raises(RuntimeError):
        features = ('unknown_feature',)
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features)
    # Test invalid number of time steps
    with pytest.raises(RuntimeError):
        n_time_steps = 'invalid_type'
        _ = features_generator_2d.build_nodes_features_matrix(
            n_time_steps=n_time_steps)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test unsufficient history data
    with pytest.raises(RuntimeError):
        features = ('coord_hist',)
        n_time_steps = features_generator_2d._nodes_coords_hist.shape[1] + 1
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features, n_time_steps=n_time_steps)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test unsufficient history data
    with pytest.raises(RuntimeError):
        features = ('disp_hist',)
        n_time_steps = features_generator_2d._nodes_disps_hist.shape[1] + 1
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features, n_time_steps=n_time_steps)
    # Test missing nodes displacement history
    with pytest.raises(RuntimeError):
        features = ('disp_hist',)
        features_generator_2d._nodes_disps_hist = None
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test missing nodes internal forces
    with pytest.raises(RuntimeError):
        features = ('int_force',)
        features_generator_2d._nodes_int_forces_hist = None
        _ = features_generator_2d.build_nodes_features_matrix(
            features=features)
# -----------------------------------------------------------------------------
def test_get_available_edges_features():
    """Test available edges features getter."""
    assert isinstance(GNNPatchFeaturesGenerator.get_available_edges_features(),
                      tuple), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_build_edges_features_matrix(features_generator_2d):
    """Test computation of edges features matrix with all features."""
    available_features = \
        GNNPatchFeaturesGenerator.get_available_edges_features()
    edge_features_matrix = features_generator_2d.build_edges_features_matrix(
        features=available_features)
    assert isinstance(edge_features_matrix, np.ndarray), \
        'Edges features matrix was not properly computed.'
# -----------------------------------------------------------------------------
def test_build_edges_features_matrix_invalid(features_generator_2d):
    """Test detection of invalid inputs to compute edges features matrix."""
    # Test invalid features type
    with pytest.raises(RuntimeError):
        features = 'invalid_type'
        _ = features_generator_2d.build_edges_features_matrix(
            features=features)
    # Test unknown feature
    with pytest.raises(RuntimeError):
        features = ('unknown_feature',)
        _ = features_generator_2d.build_edges_features_matrix(
            features=features)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test missing nodes displacement history
    with pytest.raises(RuntimeError):
        features = ('relative_disp',)
        features_generator_2d._nodes_disps_hist = None
        _ = features_generator_2d.build_edges_features_matrix(
            features=features)
    with pytest.raises(RuntimeError):
        features = ('relative_disp_norm',)
        features_generator_2d._nodes_disps_hist = None
        _ = features_generator_2d.build_edges_features_matrix(
            features=features) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test missing number of edges
    with pytest.raises(RuntimeError):
        features_generator_2d._n_edge = None
        features_generator_2d._edges_indexes = None
        _ = features_generator_2d.build_edges_features_matrix()   