"""Test generation of set of deformed finite element material patches."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
# Local
from src.vegapunk.patch_dataset import get_default_design_parameters
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
# TO-DO:
# Test full generation and simulatation of a set of deformed finite element
# material patches.
#
# Note 1: Basically reproduce patch_dataset.py __main__ test
#
# Note 2: This test is conditional on the existence of the Links binary path
#
# Note 3: Simulation directory must be set as str(tmp_path)
#
# -----------------------------------------------------------------------------
def test_get_default_design_parameters_type():
    """Test design space default parameters type."""
    default_parameters = get_default_design_parameters(2)
    assert isinstance(default_parameters, dict)
# -----------------------------------------------------------------------------
def test_missing_3d_implementation():
    """Test that 3D implementation is missing."""
    with pytest.raises(RuntimeError):
        get_default_design_parameters(3)