"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
# Local
from src.hookeai.simulators.fetorch.element.type.interface import ElementType
from src.hookeai.simulators.fetorch.element.type.tri3 import FETri3
from src.hookeai.simulators.fetorch.element.type.tri6 import FETri6
from src.hookeai.simulators.fetorch.element.type.quad4 import FEQuad4
from src.hookeai.simulators.fetorch.element.type.quad8 import FEQuad8
from src.hookeai.simulators.fetorch.element.type.tetra4 import FETetra4
from src.hookeai.simulators.fetorch.element.type.tetra10 import FETetra10
from src.hookeai.simulators.fetorch.element.type.hexa8 import FEHexa8
from src.hookeai.simulators.fetorch.element.type.hexa20 import FEHexa20
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
def available_elem_type():
    """Available finite element types."""
    return set((FETri3, FETri6, FEQuad4, FEQuad8, FETetra4, FETetra10, FEHexa8,
               FEHexa20))




    