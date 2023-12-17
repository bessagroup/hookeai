"""Test FiniteElementPatch class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.vegapunk.projects.gnn_material_patch.material_patch.patch import \
    FiniteElementPatch
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
def test_get_n_dim(squad4_patch):
    """Test number of spatial dimensions getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_n_dim(), int), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_elem_type(squad4_patch):
    """Test finite element type getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_elem_type(), str), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_n_elems_per_dim(squad4_patch):
    """Test number of finite elements per dimension getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_n_elems_per_dim(), tuple), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_n_edge_nodes_per_dim(squad4_patch):
    """Test number of patch edge nodes along each dimension getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_n_edge_nodes_per_dim(), tuple), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_mesh_nodes_matrix(squad4_patch):
    """Test finite element mesh nodes matrix getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_mesh_nodes_matrix(), np.ndarray), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_mesh_nodes_coords_ref(squad4_patch):
    """Test reference coordinates of finite element mesh nodes getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_mesh_nodes_coords_ref(), dict), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_mesh_boundary_nodes_disps(squad4_patch):
    """Test displacements prescribed on mesh boundary nodes getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_mesh_boundary_nodes_disps(), dict), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_elem_size_dims(squad4_patch):
    """Test finite element size getter type."""
    elem = squad4_patch
    assert isinstance(elem.get_elem_size_dims(), tuple), \
        'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_mesh_connected_nodes(squad4_patch):
    """Test finite element mesh connected nodes pairs."""
    elem = squad4_patch
    assert isinstance(elem.get_mesh_connected_nodes(), tuple), \
        'Unexpected return type.' 
    assert all([isinstance(pair, tuple) and len(pair) == 2
                for pair in elem.get_mesh_connected_nodes()]), \
        'At least one finite element mesh connectivity is not a pair nodes.'
    # Missing 3D implementation
    elem._n_dim = 3
    with pytest.raises(RuntimeError):
        elem.get_mesh_connected_nodes()
# -----------------------------------------------------------------------------
def test_get_boundary_nodes_labels(squad4_patch):
    """Test finite element mesh boundary nodes labels getter."""
    elem = squad4_patch
    assert isinstance(elem._get_boundary_nodes_labels(), tuple), \
        'Unexpected return type.'
    # Missing 3D implementation
    elem._n_dim = 3
    with pytest.raises(RuntimeError):
        elem._get_boundary_nodes_labels()
# -----------------------------------------------------------------------------
def test_non_boundary_prescribed_disps(squad4_patch):
    """Test is prescribed displacements on non-boundary edges are detected."""
    elem = squad4_patch
    # Get non-boundary edge nodes
    boundary_nodes_labels = set(elem._get_boundary_nodes_labels())
    mesh_nodes_labels = set(elem._mesh_nodes_matrix.flatten())
    non_boundary_nodes_labels = mesh_nodes_labels - boundary_nodes_labels
    # Get first non-boundary edge node
    non_boundary_node = list(non_boundary_nodes_labels)[0]
    # Set non-boundary edge node prescribed displacement
    invalid_mesh_boundary_nodes_disps = {}
    invalid_mesh_boundary_nodes_disps[str(non_boundary_node)] = \
        np.zeros((elem._n_dim))
    # Test if prescribed displacement on non-boundary edge node is detected
    with pytest.raises(RuntimeError):
        FiniteElementPatch(
            elem._n_dim, elem._patch_dims, elem._elem_type,
            elem._n_elems_per_dim, elem._mesh_nodes_matrix,
            elem._mesh_nodes_coords_ref,
            mesh_boundary_nodes_disps=invalid_mesh_boundary_nodes_disps)
# -----------------------------------------------------------------------------
def test_plot_deformed_patch(squad4_patch, tmp_path, monkeypatch):
    """Test generation of finite element material patch plot."""
    elem = squad4_patch
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        elem.plot_deformed_patch(is_show_plot=True, is_save_plot=True,
                                 save_directory = str(tmp_path))
        elem.plot_deformed_patch(is_show_plot=True, is_save_plot=True,
                                 save_directory = str(tmp_path))
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'finite element material patch.'
    # Missing 3D implementation
    elem._n_dim = 3
    with pytest.raises(RuntimeError):
        elem.plot_deformed_patch()