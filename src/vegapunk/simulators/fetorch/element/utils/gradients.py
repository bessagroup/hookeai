"""FETorch: Finite Element discrete gradient operators.

Functions
---------
eval_shapefun_deriv
    Evaluate shape functions derivates at given coordinates.
build_discrete_sym_gradient
    Build discrete symmetric gradient operator.
build_discrete_gradient
    Build discrete gradient operator.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.utils.jacobian import eval_jacobian
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def eval_shapefun_deriv(element_type, nodes_coords, local_coords):
    """Evaluate shape functions derivates at given coordinates.
    
    Parameters
    ----------
    element_type : Element
        FETorch finite element.
    nodes_coords : torch.Tensor(2d)
        Nodes coordinates stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    local_coords : torch.Tensor(1d)
        Local coordinates of point where Jacobian is evaluated.
    
    Returns
    -------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    jacobian : torch.Tensor(2d)
        Element Jacobian.
    jacobian_det : float
        Determinant of element jacobian.
    """
    # Get element number of nodes
    n_node = element_type.get_n_node()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluate element shape functions local derivatives and Jacobian
    jacobian, jacobian_det, shape_fun_local_deriv = \
        eval_jacobian(element_type, nodes_coords, local_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize element shape functions derivatives
    shape_fun_deriv = torch.zeros_like(shape_fun_local_deriv)
    # Compute element shape functions derivatives
    for i in range(n_node):
        shape_fun_deriv[i, :] = \
            torch.matmul(torch.inverse(jacobian), shape_fun_local_deriv[i, :])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return shape_fun_deriv, jacobian, jacobian_det
# =============================================================================
def build_discrete_sym_gradient(shape_fun_deriv, comp_order_sym=None):
    """Build discrete symmetric gradient operator.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_sym : tuple, default=None
        Strain/Stress components symmetric order. If None, then the following
        order is assumed by default:
        
        2D : ('11', '22', '12')
        
        3D : ('11', '22', '33', '12', '23', '13')
        
    Returns
    -------
    grad_operator_sym : torch.Tensor(2d)
        Discrete symmetric gradient operator evaluated at given local
        coordinates.
    """
    # Infere element number of nodes and degrees of freedom from shape
    # functions derivatives
    n_node, n_dof_node = shape_fun_deriv.shape
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain/stress components order
    if comp_order_sym is None:
        if n_dof_node == 2:
            comp_order_sym = ('11', '22', '12')
        else:
            comp_order_sym = ('11', '22', '33', '12', '23', '13')
    # Get number of strain/stress components
    n_comps = len(comp_order_sym)
    # Check number of strain/stress components
    if n_comps != 0.5*n_dof_node*(n_dof_node + 1):
        raise RuntimeError('Invalid number of strain/stress components.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize discrete symmetric gradient operator
    grad_operator_sym = torch.zeros((n_comps, n_node*n_dof_node))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for j in range(n_node):
        # Get node initial assembly column
        node_index = j*n_dof_node
        # Loop over components
        for i, comp in enumerate(comp_order_sym):
            # Get component index
            comp_index_1, comp_index_2 = [int(comp[k]) - 1 for k in range(2)]
            # Assemble shape functions derivatives
            if comp_index_1 == comp_index_2:
                # Diagonal component
                grad_operator_sym[i, node_index + comp_index_1] = \
                    shape_fun_deriv[j, comp_index_1]
            else:
                # Off-diagonal component
                grad_operator_sym[i, node_index + comp_index_2] = \
                    shape_fun_deriv[j, comp_index_1]
                grad_operator_sym[i, node_index + comp_index_1] = \
                    shape_fun_deriv[j, comp_index_2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator_sym
# =============================================================================
def build_discrete_gradient(shape_fun_deriv, comp_order_nsym=None):
    """Build discrete gradient operator.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_nsym : tuple, default=None
        Strain/Stress components nonsymmetric order. If None, then the
        following order is assumed by default:

        2D : ('11', '21', '12', '22')

        3D : ('11', '21', '31', '12', '22', '32', '13', '23', '33')

    Returns
    -------
    grad_operator : torch.Tensor(2d)
        Discrete gradient operator evaluated at given local coordinates.
    """
    # Infere element number of nodes and degrees of freedom from shape
    # functions derivatives
    n_node, n_dof_node = shape_fun_deriv.shape
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain/stress components order
    if comp_order_nsym is None:
        if n_dof_node == 2:
            comp_order_nsym = ('11', '21', '12', '22')
        else:
            comp_order_nsym = \
                ('11', '21', '31', '12', '22', '32', '13', '23', '33')
    # Get number of strain/stress components
    n_comps = len(comp_order_nsym)
    # Check number of strain/stress components
    if n_comps != n_dof_node**2:
        raise RuntimeError('Invalid number of strain/stress components.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize discrete gradient operator
    grad_operator = torch.zeros((n_comps, n_node*n_dof_node))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for j in range(n_node):
        # Get node initial assembly column
        node_index = j*n_dof_node
        # Loop over components
        for i, comp in enumerate(comp_order_nsym):
            # Get component index
            comp_index_1, comp_index_2 = [int(comp[k]) - 1 for k in range(2)]
            # Assemble shape functions derivatives
            grad_operator[i, node_index + comp_index_1] = \
                shape_fun_deriv[j, comp_index_2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator