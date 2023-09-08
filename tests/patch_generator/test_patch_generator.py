"""Test FiniteElementPatchGenerator class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.vegapunk.material_patch.patch_generator import \
    FiniteElementPatchGenerator
from tests.finite_element.conftest import available_elem_type
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
def test_build_corners_disp_range_2d_invalid(patch_generator_2d):
    """Test invalid patch corners displacement range generation."""
    with pytest.raises(RuntimeError):
        corners_lab_disp_range = 'invalid_type'
        _ = patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range)
    with pytest.raises(RuntimeError):
        corners_lab_disp_range = {'1': 'invalid_type',}
        _ = patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range)
    with pytest.raises(RuntimeError):
        corners_lab_disp_range = {'1': (0.0,),}
        _ = patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range)
    with pytest.raises(RuntimeError):
        corners_lab_disp_range = {'1': (0.0, (0.0, 1.0)),}
        _ = patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range)
    with pytest.raises(RuntimeError):
        corners_lab_disp_range = {'1': ((0.0,), (0.0, 1.0)),}
        _ = patch_generator_2d._build_corners_disp_range(
            corners_lab_disp_range)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('edges_lab_def_order, edge_poly_orders',
                         [(None, {'0': (0, 0), '1': (0, 0)}),
                          (1, {'0': (1, 1), '1': (1, 1)}),
                          ({'1': 1, '2': 3, '3': 1, '4': 0},
                           {'0': (1, 3), '1': (1, 0)}),
                          ({'2': 3, '4': 0},
                           {'0': (0, 3), '1': (0, 0)})])
def test_build_edges_poly_orders(patch_generator_2d, edges_lab_def_order,
                                 edge_poly_orders):
    """Test if patch edges deformation polynomial orders are properly set."""
    assert patch_generator_2d._build_edges_poly_orders(edges_lab_def_order) \
        == edge_poly_orders, 'Patch edges deformation polynomial orders ' \
        'were not properly assigned.'
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('edges_lab_def_order',
                         ['1', {'1': '1',}, 'invalid_type'])
def test_build_edges_poly_orders_invalid(patch_generator_2d,
                                         edges_lab_def_order):
    """Test invalid patch edges deformation polynomial order detection."""
    with pytest.raises(RuntimeError):
        patch_generator_2d._build_edges_poly_orders(edges_lab_def_order)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('edges_lab_disp_range, edge_disp_range',
                         [(None, {'0': ((0, 0), (0, 0)),
                                  '1': ((0, 0), (0, 0))}),
                          ({'1': (0.0, 1.0),},
                           {'0': ((0, 1.0), (0, 0)),
                            '1': ((0, 0), (0, 0))}),
                          ({'1': (0.0, 1.0), '2': (-1.0, 0.0),
                            '3': (-1.0, 2.0), '4': (1.0, 3.0),},
                           {'0': ((0, 1.0), (-1.0, 0)),
                            '1': ((-1.0, 2.0), (1.0, 3.0))})])
def test_build_edges_disp_range(patch_generator_2d, edges_lab_disp_range,
                                edge_disp_range):
    """Test if patch edges displacements range is properly set."""
    assert patch_generator_2d._build_edges_disp_range(edges_lab_disp_range) \
        == edge_disp_range, 'Patch edges deformation polynomial orders ' \
        'were not properly assigned.'
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('edges_lab_disp_range',
                         [{'1': 'invalid_type',},
                          {'1': [0.0, 1.0],},
                          'invalid_type'])
def test_build_edges_disp_range_invalid(patch_generator_2d,
                                        edges_lab_disp_range):
    """Test invalid patch edges displacements range detection."""
    with pytest.raises(RuntimeError):
        patch_generator_2d._build_edges_disp_range(edges_lab_disp_range)
# -----------------------------------------------------------------------------
def test_get_corners_random_displacements_2d(patch_generator_2d,
                                             corners_disp_range_2d):
    """Test computation of patch corners random displacements."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test if sampled displacements are within the prescribed bounds
    corners_disp, _ = patch_generator_2d._get_corners_random_displacements(
        corners_disp_range=corners_disp_range_2d)
    for i in range(corners_disp_range_2d.shape[0]):
        for j in range(corners_disp_range_2d.shape[1]):
            disp = corners_disp[i, j]
            lbound = corners_disp_range_2d[i, j, 0]
            ubound = corners_disp_range_2d[i, j, 1]
            if not (disp >= lbound and disp <= ubound):
                errors.append('Sampled displacement is not within prescribed '
                              'bounds.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
    # Test if edges zero order deformation polynomial is properly enforced
    edges_poly_orders = {'0': (0, 3), '1': (0, 1)}
    corners_disp, _ = patch_generator_2d._get_corners_random_displacements(
        corners_disp_range=corners_disp_range_2d,
        edges_poly_orders=edges_poly_orders)
    if not np.allclose(corners_disp[0, 1], corners_disp[1, 1]):
        errors.append('Edge zero order deformation polynomial was not '
                      'properly enforced.')
    if not np.allclose(corners_disp[0, 0], corners_disp[3, 0]):
        errors.append('Edge zero order deformation polynomial was not '
                      'properly enforced.')
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
@pytest.mark.parametrize('poly_order, poly_bounds_range',
                         [(0, (-0.1, 0.2)),
                          (1, (-0.2, -0.1)),
                          (2, (0.2, 0.4)),
                          (3, None)])
def test_get_deformed_boundary_edge_2d(patch_generator_2d, poly_order,
                                       poly_bounds_range):
    """Test generation of randomly deformed boundary edge."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test generation of randomly deformed boundary edge
    nodes_coords_ref = np.array([[0.0, 0.0], [1.0, 0.0],
                                 [2.0, 0.0], [3.0, 0.0]])
    left_node_def = np.array([-0.5, 1.0])
    right_node_def = np.array([3.5, 2.0])
    nodes_coords_def, nodes_disp = \
        patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref, left_node_def, right_node_def, poly_order,
            poly_bounds_range=poly_bounds_range)
    if not (isinstance(nodes_coords_def, np.ndarray)
            and isinstance(nodes_disp, np.ndarray)):
        errors.append('Unexpected return type.')
    else:
        is_same_shape_def = True
        is_same_shape_disp = True
        if nodes_coords_def.shape != nodes_coords_ref.shape:
           errors.append('Different shape between nodes coordinates array '
                         'in reference and deformed configurations.')
           is_same_shape_def = False
        if nodes_disp.shape != nodes_coords_ref.shape:
           errors.append('Different shape between nodes displacements array '
                         'nodes coordinates array in reference configuration.')
           is_same_shape_disp = False
        if is_same_shape_def and is_same_shape_disp:
            if not np.allclose(nodes_disp,
                               nodes_coords_def - nodes_coords_ref):
                errors.append('Displacements are inconsistent with nodes '
                              'positions in the reference and deformed '
                              'configurations.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_deformed_boundary_edge_2d_invalid(patch_generator_2d):
    """Test invalid inputs to generate randomly deformed boundary edge."""
    # Set default inputs
    nodes_coords_ref = np.array([[0.0, 0.0], [1.0, 0.0],
                                 [2.0, 0.0], [3.0, 0.0]])
    left_node_def = np.array([-0.5, 1.0])
    right_node_def = np.array([3.5, 2.0])
    poly_order = 2
    with pytest.raises(RuntimeError):
        inv_right_node_def = np.array([-1.0, 2.0])
        _, _ = patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref=nodes_coords_ref, left_node_def=left_node_def,
            right_node_def=inv_right_node_def, poly_order=poly_order)
    with pytest.raises(RuntimeError):
        inv_nodes_coords_ref = np.array([[1.0, 0.0], [3.0, 0.0],
                                         [0.0, 0.0], [2.0, 0.0]])
        _, _ = patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref=inv_nodes_coords_ref, left_node_def=left_node_def,
            right_node_def=right_node_def, poly_order=poly_order)
    with pytest.raises(RuntimeError):
        poly_bounds_range = 'invalid_type'
        _, _ = patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref=nodes_coords_ref, left_node_def=left_node_def,
            right_node_def=right_node_def, poly_order=poly_order,
            poly_bounds_range=poly_bounds_range)
    with pytest.raises(RuntimeError):
        poly_bounds_range = (-0.1,)
        _, _ = patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref=nodes_coords_ref, left_node_def=left_node_def,
            right_node_def=right_node_def, poly_order=poly_order,
            poly_bounds_range=poly_bounds_range)
# -----------------------------------------------------------------------------
def test_get_deformed_boundary_edge_2d_plot(patch_generator_2d, monkeypatch):
    """Test plot of randomly deformed boundary edge."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        nodes_coords_ref = np.array([[0.0, 0.0], [1.0, 0.0],
                                    [2.0, 0.0], [3.0, 0.0]])
        left_node_def = np.array([-0.5, 1.0])
        right_node_def = np.array([3.5, 2.0])
        poly_order = 2
        _, _ = patch_generator_2d._get_deformed_boundary_edge(
            nodes_coords_ref, left_node_def, right_node_def, poly_order,
            is_plot=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'randomly deformed boundary edge.'
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('order, left_point, right_point, lower_bound, \
                         upper_bound',
                         [(1, (0.0, 1.0), (1.0, 2.0), -1.0, 1.0),
                          (2, (-1.0, 1.0), (1.0, -2.0), None, 1.0),
                          (3, (1.0, -1.0), (2.0, -2.0), -1.0, None),
                          (4, (0.0, 1.0), (1.0, 2.0), None, None)])
def test_polynomial_sampler(order, left_point, right_point, lower_bound,
                            upper_bound):
    """Test generation of random polynomial for different input sets."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test polynomial generation
    coefficients = FiniteElementPatchGenerator._polynomial_sampler(
        order, left_point, right_point, lower_bound, upper_bound)
    if not isinstance(coefficients, tuple):
        errors.append('Unexpected return type.')
    elif len(coefficients) != order + 1:
        errors.append('Incorrect number of coefficients for prescribed '
                      'polynomial order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------   
def test_polynomial_sampler_plot(monkeypatch):
    """Test plot of randomly generated polynomial."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        _ = FiniteElementPatchGenerator._polynomial_sampler(
            order=1, left_point=(0.0, 1.0), right_point=(1.0, 2.0),
            lower_bound=-1.0, upper_bound=1.0, is_plot=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'random polynomial.'
# -----------------------------------------------------------------------------
def test_missing_3d_implementation():
    """Test that 3D implementation is missing."""
    with pytest.raises(RuntimeError):
        patch_generator = FiniteElementPatchGenerator(
            n_dim=3, patch_dims=(1.0, 1.0, 1.0))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize fake 3D generator
    patch_generator = FiniteElementPatchGenerator(n_dim=2,
                                                  patch_dims=(1.0, 1.0))
    patch_generator._n_dim = 3
    patch_generator._patch_dims = (1.0, 1.0, 1.0)
    patch_generator._n_corners = 8
    patch_generator._n_edges = 12
    patch_generator._set_corners_attributes()
    patch_generator._set_corners_coords_ref()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 3D missing implementations
    with pytest.raises(RuntimeError):
        patch_generator._set_edges_attributes()
    with pytest.raises(RuntimeError):
        patch_generator._build_corners_bc(is_random_min=True)
    with pytest.raises(RuntimeError):
        patch_generator._build_edges_poly_orders()
    with pytest.raises(RuntimeError):
        patch_generator._build_edges_disp_range()
    with pytest.raises(RuntimeError):
        patch_generator._is_admissible_geometry(edges_coords=None)
# -----------------------------------------------------------------------------
def test_generate_deformed_patch_2d(patch_generator_2d):
    """Test the full generation process of a finite element deformed patch."""
    # Set finite element discretization
    elem_type = 'SQUAD4'
    n_elems_per_dim = (3, 3)
    # Set corners boundary conditions
    corners_lab_bc = {'1': (1, 0), '3': (1, 1)}
    # Set corners displacement range
    corners_lab_disp_range = {'1': ((-0.1, 0.1), (-0.1, 0.1)),
                              '2': ((-0.1, 0.1), (-0.1, 0.1)),
                              '3': ((-0.1, 0.1), (-0.1, 0.1)),
                              '4': ((-0.1, 0.1), (-0.1, 0.1))}
    # Set edges polynomial deformation order and displacement range
    edges_lab_def_order = {'1': 1, '2': 2, '3': 2, '4': 0}
    edges_lab_disp_range = {'1': (-0.1, 0.1),
                            '2': (-0.1, 0.1),
                            '3': (-0.1, 0.1),
                            '4': (-0.1, 0.1)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate randomly deformed material patch
    is_admissible, patch = patch_generator_2d.generate_deformed_patch(
        elem_type, n_elems_per_dim, corners_lab_bc=corners_lab_bc,
        corners_lab_disp_range=corners_lab_disp_range,
        edges_lab_def_order=edges_lab_def_order,
        edges_lab_disp_range=edges_lab_disp_range, max_iter=10,
        is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert is_admissible and patch is not None, 'A finite element deformed ' \
        'patch was not successfuly generated.'
# -----------------------------------------------------------------------------
def test_generate_deformed_patch_2d_nonadmissible(patch_generator_2d):
    """Test handling of non-admissible finite element deformed patch."""
    """
    # Set finite element discretization
    elem_type = 'SQUAD4'
    n_elems_per_dim = (3, 3)
    # Set corners boundary conditions
    corners_lab_bc = None
    # Set corners displacement range
    corners_lab_disp_range = {'1': ((1.0, 1.0), (0.0, 0.0)),
                              '2': ((-1.0, -1.0), (0.0, 0.0)),
                              '3': ((-1.0, -1.0), (0.0, 0.0)),
                              '4': ((1.0, 1.0), (0.0, 0.0))}
    # Set edges polynomial deformation order and displacement range
    edges_lab_def_order = {'1': 1, '2': 2, '3': 2, '4': 0}
    edges_lab_disp_range = {'1': (-0.1, 0.1),
                            '2': (-0.1, 0.1),
                            '3': (-0.1, 0.1),
                            '4': (-0.1, 0.1)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate randomly deformed material patch
    is_admissible, patch = patch_generator_2d.generate_deformed_patch(
        elem_type, n_elems_per_dim, corners_lab_bc=corners_lab_bc,
        corners_lab_disp_range=corners_lab_disp_range,
        edges_lab_def_order=edges_lab_def_order,
        edges_lab_disp_range=edges_lab_disp_range, max_iter=10,
        is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not is_admissible and patch is None, 'A non-admissible finite ' \
        'element deformed patch was not detected.'
    """
    pass