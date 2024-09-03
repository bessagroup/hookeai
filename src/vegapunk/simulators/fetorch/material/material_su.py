"""FETorch: Material state update.

Functions
---------
material_state_update
    Material state update for any given constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def material_state_update(strain_formulation, problem_type,
                          constitutive_model, inc_strain, state_variables_old,
                          def_gradient_old=None):
    """Material state update for any given constitutive model.

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    constitutive_model : ConstitutiveModel
        Material constitutive model.
    inc_strain : torch.Tensor(2d)
        Incremental strain second-order tensor.
    state_variables_old : dict
        Last converged material constitutive model state variables.
    def_gradient_old : torch.Tensor(2d)
        Last converged deformation gradient.

    Returns
    -------
    state_variables : dict
        Material constitutive model state variables.
    consistent_tangent_mf : torch.Tensor(2d)
        Material constitutive model material consistent tangent modulus stored
        in matricial form.
    """
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type)
    # Get material phase constitutive model strain type
    strain_type = constitutive_model.get_strain_type()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute incremental spatial logarithmic strain tensor
    if strain_formulation == 'finite' and strain_type == 'finite-kinext':
        raise RuntimeError('Not implemented.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform state update and compute material consistent tangent modulus
    state_variables, consistent_tangent_mf = \
        constitutive_model.state_update(inc_strain, state_variables_old)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Cauchy stress tensor and material consistent tangent modulus
    if (strain_formulation == 'finite' and strain_type == 'finite-kinext'):
        raise RuntimeError('Not implemented.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return state_variables, consistent_tangent_mf