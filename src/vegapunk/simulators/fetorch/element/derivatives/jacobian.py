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
def eval_jacobian(element_type, nodes_coords, local_coords,
                  is_check_det=False):
    """Evaluate finite element Jacobian and determinant at given coordinates.
    
    Parameters
    ----------
    element_type : Element
        FETorch finite element.
    nodes_coords : torch.Tensor(2d)
        Nodes coordinates stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    local_coords : torch.Tensor(1d)
        Local coordinates of point where Jacobian is evaluated.
    is_check_det : bool, default=False
        If True, then check and raise error if Jacobian determinant is
        non-positive.

    Returns
    -------
    jacobian : torch.Tensor(2d)
        Element Jacobian evaluated at given local coordinates.
    jacobian_det : torch.Tensor(0d)
        Determinant of element Jacobian evaluated at given local coordinates.
    shape_fun_local_deriv : torch.Tensor(2d)
        Shape functions local derivatives evaluated at given local
        coordinates, sorted according with element nodes. Derivative of the
        i-th shape function with respect to the j-th local coordinate is
        stored in shape_fun_local_deriv[i, j].
    """
    # Evaluate element shape functions local derivatives
    shape_fun_local_deriv = \
        element_type.eval_shapefun_local_deriv(local_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Jacobian
    jacobian = torch.matmul(shape_fun_local_deriv.t(), nodes_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Jacobian determinant
    jacobian_det = torch.det(jacobian)
    # Check Jacobian determinant
    if is_check_det and jacobian_det <= 0.0:
        raise RuntimeError('Non-positive element Jacobian determinant signals '
                           'an invalid element configuration.\n\n'
                           f'det(J) = {jacobian_det:.8e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return jacobian, jacobian_det, shape_fun_local_deriv