"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
# Local
from src.vegapunk.gnn_model.gnn_patch_data import GNNPatchGraphData, \
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
@pytest.fixture
def squad4_regular_mesh():
    """2D SQUAD4 regular finite element mesh."""
    # Set node coordinates
    nodes_coords = np.array([[0.0, 0.0],
                             [1.0, 0.0],
                             [2.0, 0.0],
                             [3.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 1.0],
                             [2.0, 1.0],
                             [3.0, 1.0],
                             [0.0, 2.0],
                             [1.0, 2.0],
                             [2.0, 2.0],
                             [3.0, 2.0],
                             ])
    # Set node connectivites
    connected_nodes = ((1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                       (9, 10), (10, 11), (11, 12), (1, 5), (5, 9), (2, 6),
                       (6, 10), (3, 7), (7, 11), (4, 8), (8, 12))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords, connected_nodes
# -----------------------------------------------------------------------------
@pytest.fixture
def gnn_patch_graph_2d(squad4_regular_mesh):
    """2D GNN-based material patch graph data."""
    nodes_coords, connected_nodes = squad4_regular_mesh
    gnn_patch_data = GNNPatchGraphData(2, nodes_coords)
    edges_indexes_mesh = \
        GNNPatchGraphData.get_edges_indexes_mesh(connected_nodes)
    gnn_patch_data.set_graph_edges_indexes(
        edges_indexes_mesh=edges_indexes_mesh)
    return gnn_patch_data
# -----------------------------------------------------------------------------
@pytest.fixture
def features_generator_2d(gnn_patch_graph_2d):
    """2D GNN-based material patch features generator."""
    n_nodes = gnn_patch_graph_2d.get_nodes_coords().shape[0]
    edges_indexes = gnn_patch_graph_2d.get_graph_edges_indexes()
    n_edge = edges_indexes.shape[0]
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
    return features_generator