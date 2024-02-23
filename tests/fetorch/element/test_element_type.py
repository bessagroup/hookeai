"""Test ElementType class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
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
def test_shape_functions_properties(available_elem_type):
    """Test if elements shape functions satisfy known properties."""
    # Loop over available finite elements
    for element_class in available_elem_type:
        element = element_class()
        element.check_shape_functions_properties()