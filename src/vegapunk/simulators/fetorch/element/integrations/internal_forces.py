"""FETorch: Finite Element Internal Forces.

Functions
---------
compute_element_internal_forces
    Compute finite element internal forces.
compute_infinitesimal_inc_strain
    Compute incremental infinitesimal strain tensor.
compute_infinitesimal_strain
    Compute infinitesimal strain tensor.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import torch
# Local
from simulators.fetorch.element.derivatives.gradients import \
    eval_shapefun_deriv, vbuild_discrete_sym_gradient
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_from_mf
from simulators.fetorch.math.voigt_notation import vget_strain_from_vmf, \
    vget_stress_vmf
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def compute_element_internal_forces(strain_formulation, problem_type,
                                    element_type, element_material,
                                    element_state_old, nodes_coords,
                                    nodes_disps, nodes_inc_disps):
    """Compute finite element internal forces.
    
    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    element_type : Element
        FETorch finite element.
    element_material : ConstitutiveModel
        FETorch material constitutive model.
    element_state_old : dict
        Last converged material constitutive model state variables (item, dict)
        for each Gauss integration point (key, str[int]).
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
    element_state : dict
        Material constitutive model state variables (item, dict) for each Gauss
        integration point (key, str[int]).
    """
    # Get problem type parameters
    n_dim, comp_order_sym, _ = \
        get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get element number of nodes
    n_node = element_type.get_n_node()
    # Get element number of degrees of freedom per node
    n_dof_node = element_type.get_n_dof_node()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get element number of Gauss quadrature integration points
    n_gauss = element_type.get_n_gauss()
    # Get element Gauss quadrature integration points local coordinates and
    # weights
    gp_coords, gp_weights = element_type.get_gauss_integration_points()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize element internal forces
    internal_forces = torch.zeros((n_node*n_dof_node))
    # Initialize element material constitutive model state variables
    element_state = {key: None for key in element_state_old.keys()}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over Gauss integration points
    for i in range(n_gauss):
        # Get Gauss integration point local coordinates and weight
        local_coords = gp_coords[str(i + 1)]
        weight = gp_weights[str(i + 1)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate shape functions derivates and Jacobian
        shape_fun_deriv, _, jacobian_det = \
            eval_shapefun_deriv(element_type, nodes_coords, local_coords)
        # Build discrete symmetric gradient operator
        grad_operator_sym = \
            vbuild_discrete_sym_gradient(shape_fun_deriv, comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental strain tensor
        if strain_formulation == 'infinitesimal':
            # Compute incremental infinitesimal strain tensor (Voigt matricial
            # form)
            inc_strain_vmf = compute_infinitesimal_inc_strain(
                grad_operator_sym, nodes_inc_disps)
            # Get incremental strain tensor
            inc_strain = vget_strain_from_vmf(
                inc_strain_vmf, n_dim, comp_order_sym)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Gauss integration point last converged material constitutive
        # model state variables
        state_variables_old = copy.deepcopy(element_state_old[str(i + 1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Material state update
        state_variables, _ = material_state_update(
            strain_formulation, problem_type, element_material, inc_strain,
            state_variables_old, def_gradient_old=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store Gauss integration point material constitutive model state
        # variables
        element_state[str(i + 1)] = state_variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get stress tensor
        if strain_formulation == 'infinitesimal':
            # Get Cauchy stress tensor
            stress = vget_tensor_from_mf(state_variables['stress_mf'],
                                         n_dim, comp_order_sym)
            # Get Cauchy stress tensor (Voigt matricial form)
            stress_vmf = vget_stress_vmf(stress, n_dim, comp_order_sym)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add Gauss integration point contribution to element internal forces
        internal_forces += \
            weight*torch.matmul(grad_operator_sym.T, stress_vmf)*jacobian_det
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return internal_forces, element_state
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
# =============================================================================
def compute_infinitesimal_strain(grad_operator_sym, nodes_disps):
    """Compute infinitesimal strain tensor.
    
    Strain components order is set by discrete symmetric gradient operator.
    
    Parameters
    ----------
    grad_operator_sym : torch.Tensor(2d)
        Discrete symmetric gradient operator evaluated at given local
        coordinates.
    nodes_disps : torch.Tensor(2d)
        Nodes displacements stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
        
    Returns
    -------
    strain_vmf : torch.Tensor(1d)
        Infinitesimal strain tensor (Voigt matricial form).
    """
    # Compute infinitesimal strain tensor (Voigt matricial form)
    strain_vmf = torch.matmul(grad_operator_sym, nodes_disps.flatten())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_vmf