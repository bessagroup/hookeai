"""FETorch: Finite Element Internal Forces.

Functions
---------
compute_element_internal_forces
    Compute finite element internal forces.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.utils.gradients import eval_shapefun_deriv, \
    build_discrete_sym_gradient, build_discrete_gradient
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def compute_element_internal_forces(strain_formulation, element, nodes_coords,
                                    nodes_disps, nodes_inc_disps):
    """Compute finite element internal forces.
    
    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    element : Element
        FETorch finite element.
    nodes_coords : torch.Tensor(2d)
        Nodes coordinates stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    nodes_disps : torch.Tensor(2d)
        Nodes displacements stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    nodes_inc_disps : torch.Tensor(2d)
        Nodes incremental displacements stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
        
    Returns
    -------
    internal_forces : torch.Tensor(1d)
        Element internal forces.
    """
    # Get element number of degrees of freedom per node
    n_dof_node = element.get_n_dof_node()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get element number of Gauss quadrature integration points
    n_gauss = element.get_n_gauss()
    # Get element Gauss quadrature integration points local coordinates and
    # weights
    gp_coords, gp_weights = element.get_gauss_integration_points()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize element internal forces
    internal_forces = torch.zeros((n_dof_node))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over Gauss integration points
    for i in range(n_gauss):
        # Get Gauss integration point local coordinates and weight
        local_coords = gp_coords[str(i + 1)]
        weight = gp_weights[str(i + 1)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate shape functions derivates and Jacobian
        shape_fun_deriv, _, jacobian_det = \
            eval_shapefun_deriv(element, nodes_coords, local_coords)
        # Build discrete symmetric gradient operator
        grad_operator_sym = build_discrete_sym_gradient(shape_fun_deriv)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental strains
        if strain_formulation == 'infinitesimal':
            # Compute incremental infinitesimal strain tensor (Voigt matricial
            # form)
            inc_strain_vmf = compute_infinitesimal_inc_strain(
                grad_operator_sym, nodes_inc_disps)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Material state update
        stress_vmf = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add Gauss integration point contribution to element internal forces
        internal_forces += weight*grad_operator_sym.T*stress_vmf*jacobian_det
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return internal_forces
# =============================================================================
def compute_infinitesimal_inc_strain(grad_operator_sym, nodes_inc_disps):
    """Compute incremental infinitesimal strain tensor.
    
    Strain components order is set by discrete symmetric gradient operator.
    
    Parameters
    ----------
    grad_operator_sym : torch.Tensor(2d)
        Discrete symmetric gradient operator evaluated at given local
        coordinates.
    nodes_inc_disps : torch.Tensor(2d)
        Nodes incremental displacements stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
        
    Returns
    -------
    inc_strain_vmf : torch.Tensor(1d)
        Incremental infinitesimal strain tensor (Voigt matricial form).
    """
    # Compute incremental infinitesimal strain tensor (Voigt matricial form)
    inc_strain_vmf = torch.matmul(grad_operator_sym, nodes_inc_disps.flatten())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return inc_strain_vmf