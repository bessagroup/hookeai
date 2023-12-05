"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
# Local
from src.vegapunk.material_patch.patch_generator import \
    FiniteElementPatchGenerator
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
def squad4_patch():
    """SQUAD4 finite element material patch."""
    # Set number of dimensions
    n_dim = 2
    # Set material patch dimensions
    patch_dims = (1.0, 1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patch generator
    patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set finite element discretization
    elem_type = 'SQUAD4'
    n_elems_per_dim = (5, 5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set corners boundary conditions
    corners_lab_bc = None
    # Set corners displacement range
    corners_lab_disp_range = {'1': ((0.0, 0.0), (-0.1, -0.1)),
                              '2': ((0.1, 0.1), (0.1, 0.1)),
                              '3': ((-0.1, -0.1), (-0.1, -0.1)),
                              '4': ((0.1, 0.1), (0.1, 0.1))}
    # Set edges polynomial deformation order and displacement range
    edges_lab_def_order = {'1': 2, '2': 1, '3': 2, '4': 3}
    edges_lab_disp_range = {'1': (-0.1, -0.1),
                            '2': (0.2, 0.2),
                            '3': (0.1, 0.1),
                            '4': (-0.1, -0.1)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set rigid body translation
    translation_range = {'1': (-0.1, 0.1), '2': (-0.1, 0.1)}
    # Set rigid body rotation
    rotation_angles_range = {'alpha': (0.0, 10), 'beta': (0.0, 0.0),
                             'gamma': (0.0, 0.0)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate randomly deformed material patch
    is_admissible, patch = patch_generator.generate_deformed_patch(
        elem_type, n_elems_per_dim, corners_lab_bc=corners_lab_bc,
        corners_lab_disp_range=corners_lab_disp_range,
        edges_lab_def_order=edges_lab_def_order,
        edges_lab_disp_range=edges_lab_disp_range,
        translation_range=translation_range,
        rotation_angles_range=rotation_angles_range)
    if not is_admissible:
        raise RuntimeError('Non-admissible SQUAD4 material patch.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch




    