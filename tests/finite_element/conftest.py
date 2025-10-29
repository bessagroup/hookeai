"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
# Local
from src.hookeai.projects.gnn_material_patch.discretization.finite_element \
    import FiniteElement
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
def elem_type_2d():
    """One 2D finite element type."""
    return 'SQUAD4'
# -----------------------------------------------------------------------------
@pytest.fixture
def available_elem_type():
    """Available finite element types."""
    return set(FiniteElement._available_elem_type())
# -----------------------------------------------------------------------------
@pytest.fixture
def available_elems(available_elem_type):
    """Available FiniteElement instances."""
    return tuple([FiniteElement(elem_type)
                  for elem_type in available_elem_type]) 




    