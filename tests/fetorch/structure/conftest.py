"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.simulators.fetorch.element.type.quad4 import FEQuad4
from src.vegapunk.simulators.fetorch.element.type.hexa8 import FEHexa8
from src.vegapunk.simulators.fetorch.structure.structure_mesh import \
    StructureMesh
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
def toy_uniaxial_specimen_2d_quad4():
    """2D toy uniaxial specimen discretized in QUAD4 finite element mesh."""
    # Set finite element mesh initial nodes coordinates
    nodes_coords_mesh_init = torch.tensor([[0.00, 0.00],
                                           [1.00, 0.00],
                                           [2.00, 0.25],
                                           [3.00, 0.00],
                                           [4.00, 0.00],
                                           [0.00, 1.00],
                                           [1.00, 1.00],
                                           [2.00, 0.75],
                                           [3.00, 1.00],
                                           [4.00, 1.00]], dtype=torch.float)
    # Set finite element mesh elements types
    elements_type = {str(i): FEQuad4() for i in range(1, 5)}
    # Set elements connectivies
    connectivities = {'1': (1, 2, 7, 6),
                      '2': (2, 3, 8, 7),
                      '3': (3, 4, 9, 8),
                      '4': (4, 5, 10, 9)}
    # Set Dirichlet boundary conditions
    dirichlet_bool_mesh = torch.zeros_like(nodes_coords_mesh_init,
                                           dtype=torch.int)
    dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[4, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[5, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create FETorch structure finite element mesh
    structure_mesh = StructureMesh(nodes_coords_mesh_init, elements_type,
                                   connectivities, dirichlet_bool_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return structure_mesh
# -----------------------------------------------------------------------------
@pytest.fixture
def toy_uniaxial_specimen_3d_hexa8():
    """3D toy uniaxial specimen discretized in HEXA8 finite element mesh."""
    # Set finite element mesh initial nodes coordinates
    nodes_coords_mesh_init = \
        torch.tensor([[0.00, 0.00, 0.0],
                      [1.00, 0.00, 0.0],
                      [2.00, 0.25, 0.0],
                      [3.00, 0.00, 0.0],
                      [4.00, 0.00, 0.0],
                      [0.00, 1.00, 0.0],
                      [1.00, 1.00, 0.0],
                      [2.00, 0.75, 0.0],
                      [3.00, 1.00, 0.0],
                      [4.00, 1.00, 0.0],
                      [0.00, 0.00, 1.0],
                      [1.00, 0.00, 1.0],
                      [2.00, 0.25, 1.0],
                      [3.00, 0.00, 1.0],
                      [4.00, 0.00, 1.0],
                      [0.00, 1.00, 1.0],
                      [1.00, 1.00, 1.0],
                      [2.00, 0.75, 1.0],
                      [3.00, 1.00, 1.0],
                      [4.00, 1.00, 1.0]], dtype=torch.float)
    # Set finite element mesh elements types
    elements_type = {str(i): FEHexa8() for i in range(1, 5)}
    # Set elements connectivies
    connectivities = {'1': (1, 11, 12, 2, 6, 16, 17, 7),
                      '2': (2, 12, 13, 3, 7, 17, 18, 8),
                      '3': (3, 13, 14, 4, 8, 18, 19, 9),
                      '4': (4, 14, 15, 5, 9, 19, 20, 10)}
    # Set Dirichlet boundary conditions
    dirichlet_bool_mesh = torch.zeros_like(nodes_coords_mesh_init,
                                           dtype=torch.int)
    dirichlet_bool_mesh[0, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[4, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[5, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[9, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[10, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[14, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[15, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[19, :] = torch.tensor([1, 1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create FETorch structure finite element mesh
    structure_mesh = StructureMesh(nodes_coords_mesh_init, elements_type,
                                   connectivities, dirichlet_bool_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return structure_mesh