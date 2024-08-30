"""Test strain/stress tensors Voigt notation matricial storage."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.simulators.fetorch.math.voigt_notation import \
    get_strain_from_vmf, get_stress_vmf
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
def test_get_strain_from_vmf():
    """Test recovery of strain tensor from Voigt matricial form."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set 2D problem parameters
    n_dim = 2
    comp_order_sym = ('11', '22', '12')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random 2D strain tensor stored in Voigt matricial form
    strain_vfm = torch.rand(len(comp_order_sym))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover strain tensor from Voigt matricial form
    strain = get_strain_from_vmf(strain_vfm, n_dim, comp_order_sym)
    # Set expected strain tensor
    strain_sol = torch.tensor([[strain_vfm[0], 0.5*strain_vfm[2]],
                               [0.5*strain_vfm[2], strain_vfm[1]]])
    # Check recovered strain tensor
    if not torch.allclose(strain, strain_sol):
        errors.append('2D strain tensor was not properly recovered from '
                      'the corresponding Voigt matricial form.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set 3D problem parameters
    n_dim = 3
    comp_order_sym = ('11', '22', '33', '12', '23', '13')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random 3D strain tensor stored in Voigt matricial form
    strain_vfm = torch.rand(len(comp_order_sym))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover strain tensor from Voigt matricial form
    strain = get_strain_from_vmf(strain_vfm, n_dim, comp_order_sym)
    # Set expected strain tensor
    strain_sol = \
        torch.tensor([[strain_vfm[0], 0.5*strain_vfm[3], 0.5*strain_vfm[5]],
                      [0.5*strain_vfm[3], strain_vfm[1], 0.5*strain_vfm[4]],
                      [0.5*strain_vfm[5], 0.5*strain_vfm[4], strain_vfm[2]]])
    # Check recovered strain tensor
    if not torch.allclose(strain, strain_sol):
        errors.append('3D strain tensor was not properly recovered from '
                      'the corresponding Voigt matricial form.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_stress_vmf():
    """Test stress tensor Voigt matricial form."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set 2D problem parameters
    n_dim = 2
    comp_order_sym = ('11', '22', '12')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random 2D stress tensor
    random_tensor = torch.rand((2, 2))
    stress = 0.5*(random_tensor + random_tensor.T)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get stress tensor Voigt matricial form
    stress_vmf = get_stress_vmf(stress, n_dim, comp_order_sym)
    # Set expected stress tensor Voigt matricial form
    stress_vmf_sol = torch.tensor([stress[0, 0], stress[1, 1], stress[0, 1]])
    # Check stress tensor
    if not torch.allclose(stress_vmf, stress_vmf_sol):
        errors.append('2D stress tensor was not properly stored in Voigt '
                      'matricial form.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set 3D problem parameters
    n_dim = 3
    comp_order_sym = ('11', '22', '33', '12', '23', '13')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random 3D stress tensor
    random_tensor = torch.rand((3, 3))
    stress = 0.5*(random_tensor + random_tensor.T)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get stress tensor Voigt matricial form
    stress_vmf = get_stress_vmf(stress, n_dim, comp_order_sym)
    # Set expected stress tensor Voigt matricial form
    stress_vmf_sol = torch.tensor([stress[0, 0], stress[1, 1], stress[2, 2],
                                   stress[0, 1], stress[1, 2], stress[0, 2]])
    # Check stress tensor
    if not torch.allclose(stress_vmf, stress_vmf_sol):
        errors.append('3D stress tensor was not properly stored in Voigt '
                      'matricial form.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))