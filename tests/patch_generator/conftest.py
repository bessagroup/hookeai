"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
# Local
from src.vegapunk.patch_generator import FiniteElementPatchGenerator, \
    rotation_tensor_from_euler_angles
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
def patch_generator_2d():
    """2D finite element material patch generator."""
    # Set number of dimensions
    n_dim = 2
    # Set material patch dimensions
    patch_dims = (1.0, 1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patch generator
    patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_generator
# -----------------------------------------------------------------------------
@pytest.fixture
def n_elems_per_dim_cases():
    """Different cases of number of finite elements per dimension."""
    n_elems_per_dim_cases = []
    n_elems_per_dim_cases.append((1, 1))
    n_elems_per_dim_cases.append((2, 2))
    n_elems_per_dim_cases.append((2, 3))
    n_elems_per_dim_cases.append((3, 2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_elems_per_dim_cases
# -----------------------------------------------------------------------------
@pytest.fixture
def rotation_tensor():
    """Generic 3D rotation tensor."""
    return rotation_tensor_from_euler_angles((30, -20, 60))
# -----------------------------------------------------------------------------
@pytest.fixture
def coord_array_2d():
    """Generic 2D coordinates array."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.0, 2.0]])
# -----------------------------------------------------------------------------
@pytest.fixture
def coord_array_3d():
    """Generic 3D coordinates array."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -2.0], [0.0, 2.0, 1.0],
                     [-1.0, 2.0, 2.0]])
# -----------------------------------------------------------------------------
@pytest.fixture
def mesh_2d_squad4():
    """Simple 2D finite element mesh of SQUAD4."""
    patch_generator = FiniteElementPatchGenerator(n_dim=2,
                                                  patch_dims=(1.0, 1.0))
    mesh_nodes_matrix, mesh_nodes_coords = \
        patch_generator._generate_finite_element_mesh('SQUAD4', (2, 3))
    return mesh_nodes_matrix, mesh_nodes_coords
        