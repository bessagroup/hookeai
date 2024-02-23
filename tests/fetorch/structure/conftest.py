"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.simulators.fetorch.element.type.quad4 import FEQuad4
from src.vegapunk.simulators.fetorch.element.type.quad8 import FEQuad8
from src.vegapunk.simulators.fetorch.element.type.hexa8 import FEHexa8
from src.vegapunk.simulators.fetorch.element.type.hexa20 import FEHexa20
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
def toy_uniaxial_specimen_2d_quad8():
    """2D toy uniaxial specimen discretized in QUAD8 finite element mesh."""
    # Set finite element mesh initial nodes coordinates
    nodes_coords_mesh_init = torch.tensor([[0.000, 0.000],
                                           [0.500, 0.000],
                                           [1.000, 0.000],
                                           [1.500, 0.125],
                                           [2.000, 0.250],
                                           [2.500, 0.125],
                                           [3.000, 0.000],
                                           [3.500, 0.000],
                                           [4.000, 0.000],
                                           [0.000, 0.500],
                                           [1.000, 0.500],
                                           [2.000, 0.500],
                                           [3.000, 0.500],
                                           [4.000, 0.500],
                                           [0.000, 1.000],
                                           [0.500, 1.000],
                                           [1.000, 1.000],
                                           [1.500, 0.875],
                                           [2.000, 0.750],
                                           [2.500, 0.875],
                                           [3.000, 1.000],
                                           [3.500, 1.000],
                                           [4.000, 1.000]], dtype=torch.float)
    # Set finite element mesh elements types
    elements_type = {str(i): FEQuad8() for i in range(1, 5)}
    # Set elements connectivies
    connectivities = {'1': (1, 3, 17, 15, 2, 11, 16, 10),
                      '2': (3, 5, 19, 17, 4, 12, 18, 11),
                      '3': (5, 7, 21, 19, 6, 13, 20, 12),
                      '4': (7, 9, 23, 21, 8, 14, 22, 13)}
    # Set Dirichlet boundary conditions
    dirichlet_bool_mesh = torch.zeros_like(nodes_coords_mesh_init,
                                           dtype=torch.int)
    dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[8, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[13, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[14, :] = torch.tensor([1, 1])
    dirichlet_bool_mesh[22, :] = torch.tensor([1, 1])
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
# -----------------------------------------------------------------------------
@pytest.fixture
def toy_uniaxial_specimen_3d_hexa20():
    """3D toy uniaxial specimen discretized in HEXA20 finite element mesh."""
    # Set finite element mesh initial nodes coordinates
    nodes_coords_mesh_init = \
        torch.tensor([[0.000, 0.000, 0.000],
                      [0.500, 0.000, 0.000],
                      [1.000, 0.000, 0.000],
                      [1.500, 0.125, 0.000],
                      [2.000, 0.250, 0.000],
                      [2.500, 0.125, 0.000],
                      [3.000, 0.000, 0.000],
                      [3.500, 0.000, 0.000],
                      [4.000, 0.000, 0.000],
                      [0.000, 0.500, 0.000],
                      [1.000, 0.500, 0.000],
                      [2.000, 0.500, 0.000],
                      [3.000, 0.500, 0.000],
                      [4.000, 0.500, 0.000],
                      [0.000, 1.000, 0.000],
                      [0.500, 1.000, 0.000],
                      [1.000, 1.000, 0.000],
                      [1.500, 0.875, 0.000],
                      [2.000, 0.750, 0.000],
                      [2.500, 0.875, 0.000],
                      [3.000, 1.000, 0.000],
                      [3.500, 1.000, 0.000],
                      [4.000, 1.000, 0.000],
                      [0.000, 0.000, 0.500],
                      [1.000, 0.000, 0.500],
                      [2.000, 0.250, 0.500],
                      [3.000, 0.000, 0.500],
                      [4.000, 0.000, 0.500],
                      [0.000, 1.000, 0.500],
                      [1.000, 1.000, 0.500],
                      [2.000, 0.750, 0.500],
                      [3.000, 1.000, 0.500],
                      [4.000, 1.000, 0.500],
                      [0.000, 0.000, 1.000],
                      [0.500, 0.000, 1.000],
                      [1.000, 0.000, 1.000],
                      [1.500, 0.125, 1.000],
                      [2.000, 0.250, 1.000],
                      [2.500, 0.125, 1.000],
                      [3.000, 0.000, 1.000],
                      [3.500, 0.000, 1.000],
                      [4.000, 0.000, 1.000],
                      [0.000, 0.500, 1.000],
                      [1.000, 0.500, 1.000],
                      [2.000, 0.500, 1.000],
                      [3.000, 0.500, 1.000],
                      [4.000, 0.500, 1.000],
                      [0.000, 1.000, 1.000],
                      [0.500, 1.000, 1.000],
                      [1.000, 1.000, 1.000],
                      [1.500, 0.875, 1.000],
                      [2.000, 0.750, 1.000],
                      [2.500, 0.875, 1.000],
                      [3.000, 1.000, 1.000],
                      [3.500, 1.000, 1.000],
                      [4.000, 1.000, 1.000]], dtype=torch.float)
    # Set finite element mesh elements types
    elements_type = {str(i): FEHexa20() for i in range(1, 5)}
    # Set elements connectivies
    connectivities = {'1': (1, 34, 36, 3, 15, 48, 50, 17, 24, 35, 25, 2,
                            10, 43, 44, 11, 29, 49, 30, 16),
                      '2': (3, 36, 38, 5, 17, 50, 52, 19, 25, 37, 26, 4,
                            11, 44, 45, 12, 30, 51, 31, 18),
                      '3': (5, 38, 40, 7, 19, 52, 54, 21, 26, 39, 27, 6,
                            12, 45, 46, 13, 31, 53, 32, 20),
                      '4': (7, 40, 42, 9, 21, 54, 56, 23, 27, 41, 28, 8,
                            13, 46, 47, 14, 32, 55, 33, 22)}
    # Set Dirichlet boundary conditions
    dirichlet_bool_mesh = torch.zeros_like(nodes_coords_mesh_init,
                                           dtype=torch.int)
    dirichlet_bool_mesh[0, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[9, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[14, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[28, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[47, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[42, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[33, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[23, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[22, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[32, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[55, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[46, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[41, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[27, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[8, :] = torch.tensor([1, 1, 1])
    dirichlet_bool_mesh[13, :] = torch.tensor([1, 1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create FETorch structure finite element mesh
    structure_mesh = StructureMesh(nodes_coords_mesh_init, elements_type,
                                   connectivities, dirichlet_bool_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return structure_mesh