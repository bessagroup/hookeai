"""FETorch: Finite Element Jacobian.

Functions
---------
eval_jacobian
    Evaluate finite element Jacobian and determinant at given coordinates.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def eval_jacobian(element, nodes_coords, local_coords):
    """Evaluate finite element Jacobian and determinant at given coordinates.
    
    Parameters
    ----------
    element : Element
        FETorch finite element.
    nodes_coords : torch.Tensor(2d)
        Nodes coordinates stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    local_coords : torch.Tensor(1d)
        Local coordinates of point where Jacobian is evaluated.
    
    Returns
    -------
    jacobian : torch.Tensor(2d)
        Element Jacobian.
    jacobian_det : float
        Determinant of element jacobian.
    shape_fun_local_deriv : torch.Tensor(2d)
        Shape functions local derivatives evaluated at given local
        coordinates, sorted according with element nodes. Derivative of the
        i-th shape function with respect to the j-th local coordinate is
        stored in shape_fun_local_deriv[i, j].
    """
    # Get element number of degrees of freedom per node
    n_dof_node = element.get_n_dof_node()
    # Evaluate element shape functions local derivatives
    shape_fun_local_deriv = element.eval_shapefun_local_deriv(local_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Jacobian
    jacobian = torch.zeros((n_dof_node, n_dof_node))
    # Loop over dimensions
    for i in range(jacobian.shape[0]):
        # Loop over dimensions
        for j in range(jacobian.shape[1]):
            # Compute Jacobian
            jacobian[i, j] = \
                torch.inner(shape_fun_local_deriv[:, i], nodes_coords[:, j])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Jacobian determinant
    jacobian_det = torch.det(jacobian)
    # Check Jacobian determinant
    if jacobian_det <= 0.0:
        raise RuntimeError('Non-positive element Jacobian determinant signals '
                           'an invalid element configuration.\n\n'
                           f'det(J) = {jacobian_det:.8e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return jacobian, jacobian_det, shape_fun_local_deriv
