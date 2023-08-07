"""Test FiniteElementPatchGenerator class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.vegapunk.patch_generator import FiniteElementPatchGenerator
from tests.finite_element.conftest import elem_type_2d, available_elem_type
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
@pytest.mark.parametrize('corners_lab_bc, corners_bc',
                         [({'1': (1, 1),},
                           ((1, 1), (0, 0), (0, 0), (0, 0))),
                          ({'1': (1, 0), '2': (0, 1)},
                           ((1, 0), (0, 1), (0, 0), (0, 0))),
                          ({'1': (1, 0), '3': (1, 1)},
                           ((1, 0), (0, 0), (1, 1), (0, 0))),
                          ({},
                           ((0, 0), (0, 0), (0, 0), (0, 0)))])
def test_build_corners_bc_2d(patch_generator_2d, corners_lab_bc, corners_bc):
    """Test patch corners boundary conditions generation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test random set of minimal constraints
    corners_bc_min = patch_generator_2d._build_corners_bc(is_random_min=True)
    if np.sum(sum(corners_bc_min, ())) != 3:
        errors.append('Incorrect number of minimal constraints.')
    if not set(sum(corners_bc_min, ())).issubset((0, 1)):
        errors.append('Invalid (non-binary) constraints.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test generic corners boundary conditions
    if patch_generator_2d._build_corners_bc(corners_lab_bc) != corners_bc:
        errors.append('Incorrectly generated boundary conditions.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test that 3D implementation is missing
    patch_generator_2d._n_dim = 3
    with pytest.raises(RuntimeError):
        corners_bc = patch_generator_2d._build_corners_bc(is_random_min=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors)) 
# -----------------------------------------------------------------------------
def test_build_corners_bc_2d_invalid(patch_generator_2d):
    """Test invalid patch corners boundary conditions."""
    with pytest.raises(RuntimeError):
        patch_generator_2d._build_corners_bc({'1': 1,})
    with pytest.raises(RuntimeError):
        patch_generator_2d._build_corners_bc({'1': (1, 0, 1),})
# -----------------------------------------------------------------------------
def test_build_corners_disp_range_2d(patch_generator_2d):
    """Test patch corners displacement range generation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test null displacement range
    corners_disp_range = patch_generator_2d._build_corners_disp_range()
    if not np.allclose(corners_disp_range, np.zeros((4, 2, 2))):
        errors.append('Incorrect null displacement range.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test generic patch corners displacement ranges
    corners_lab_disp_range = {'1': ((0.0, 1.0), (0.0, 1.0)),
                              '3': ((-1.0, 2.0), (0.0, 1.0))}
    corners_disp_range = np.zeros((4, 2, 2))
    corners_disp_range[0, :, 0] = [0.0, 0.0]
    corners_disp_range[0, :, 1] = [1.0, 1.0]
    corners_disp_range[2, :, 0] = [-1.0, 0.0]
    corners_disp_range[2, :, 1] = [2.0, 1.0]    
    if not np.allclose(patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range), corners_disp_range):
        errors.append('Incorrectly generated displacement ranges.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    # Test enforcement of corners boundary conditions
    corners_bc = ((1, 0), (0, 0), (0, 1), (0, 0))
    corners_disp_range[0, :, 0] = [0.0, 0.0]
    corners_disp_range[0, :, 1] = [0.0, 1.0]
    corners_disp_range[2, :, 0] = [-1.0, 0.0]
    corners_disp_range[2, :, 1] = [2.0, 0.0]
    if not np.allclose(patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range, corners_bc), corners_disp_range):
        errors.append('Corners boundary conditions were not properly '
                      'enforced.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_generate_finite_element_mesh(patch_generator_2d, available_elem_type,
                                      n_elems_per_dim_cases):
    """Test finite element mesh generation for different element types."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for elem_type in available_elem_type:
        for n_elems_per_dim in n_elems_per_dim_cases:
            # Testing case identifier
            case_str = str(elem_type) + ' / ' + str(n_elems_per_dim) + ': '
            # Generate finite element mesh
            mesh_nodes_matrix, mesh_nodes_coords = \
                patch_generator_2d._generate_finite_element_mesh(
                    elem_type, n_elems_per_dim)
            # Test return types
            is_valid_mesh_type = True
            is_valid_mesh_coords = True
            if not isinstance(mesh_nodes_matrix, np.ndarray):
                errors.append(case_str + 'Finite element mesh nodes matrix is '
                              'not a numpy.ndarray.')
                is_valid_mesh_type = False
            if not isinstance(mesh_nodes_coords, dict):
                errors.append(case_str + 'Finite element mesh node '
                              'coordinates is not a dictionary.')
                is_valid_mesh_coords = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Test if finite element mesh is compatible with node coordinates
            if is_valid_mesh_type and is_valid_mesh_coords:
                nodes_mesh = set(mesh_nodes_matrix.flatten()[
                    np.flatnonzero(mesh_nodes_matrix)])
                nodes_coords = set([int(node) for node in
                                    mesh_nodes_coords.keys()])
                if nodes_mesh != nodes_coords:
                    errors.append(case_str + 'Finite element mesh matrix and '
                                  'node coordinates are not compatible.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_rotate_coords_array(rotation_tensor, coord_array_2d, coord_array_3d):
    """Test rotation of coordinates array."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test rotation of 2D coordinates array
    rot_coords_array_2d = FiniteElementPatchGenerator._rotate_coords_array(
        coord_array_2d, rotation_tensor)
    if not isinstance(rot_coords_array_2d, np.ndarray):
        errors.append('2D: ' + 'Rotated coordinates array is not a'
                      'numpy.ndarray.')
    elif rot_coords_array_2d.shape[1] != 2:
        errors.append('2D: ' + 'Rotated coordinates array number of columns'
                      'does not match number of spatial dimensions.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test rotation of 3D coordinates array
    rot_coords_array_3d = FiniteElementPatchGenerator._rotate_coords_array(
        coord_array_3d, rotation_tensor)
    if not isinstance(rot_coords_array_3d, np.ndarray):
        errors.append('3D: ' + 'Rotated coordinates array is not a'
                      'numpy.ndarray.')
    elif rot_coords_array_3d.shape[1] != 3:
        errors.append('3D: ' + 'Rotated coordinates array number of columns'
                      'does not match number of spatial dimensions.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('dim, orth_dims', [(0, (1,)), (1, (0,))])
def test_get_orthogonal_dims_2d(patch_generator_2d, dim, orth_dims):
    """Test orthogonal dimensions."""
    assert patch_generator_2d._get_orthogonal_dims(dim) == orth_dims, \
        'Orthogonal dimensions were not correctly identified.'
# -----------------------------------------------------------------------------
def test_node_label_from_coords_not_found(patch_generator_2d,
                                          mesh_2d_squad4):
    """Test if non-existent node in finite element mesh is detected."""
    # Generate 2D finite element mesh
    _, mesh_nodes_coords = mesh_2d_squad4
    # Set non-existent node coordinates
    node_coords = (-1.0, -1.0)
    # Test if non-existent node is detected
    with pytest.raises(RuntimeError):
        patch_generator_2d._get_node_label_from_coords(mesh_nodes_coords,
                                                       node_coords)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('elem_type, n_elems_per_dim, global_index, \
                         local_index',
                         [('SQUAD4', (2, 3), (1, 1), (1, 1)),
                          ('SQUAD4', (2, 3), (2, 1), (1, 1)),
                          ('SQUAD4', (2, 3), (0, 2), (0, 1)),
                          ('SQUAD4', (2, 3), (2, 3), (1, 1)),
                          ('SQUAD8', (2, 3), (2, 1), (2, 1)),
                          ('SQUAD8', (2, 3), (2, 3), (2, 1)),
                          ('SQUAD8', (2, 3), (4, 2), (2, 2)),
                          ('SQUAD8', (2, 3), (1, 6), (1, 2))])
def test_get_elem_node_index(patch_generator_2d, elem_type, n_elems_per_dim,
                             global_index, local_index):
    """Test if element node local index is correctly identified."""
    assert patch_generator_2d._get_elem_node_index(elem_type, n_elems_per_dim,
        global_index) == local_index, 'Element node local index was not ' \
            'correctly identified.'
# -----------------------------------------------------------------------------
def test_missing_3d_implementation():
    """Test that 3D implementation is missing."""
    with pytest.raises(RuntimeError):
        patch_generator = FiniteElementPatchGenerator(
            n_dim=3,patch_dims=(1.0, 1.0, 1.0))
