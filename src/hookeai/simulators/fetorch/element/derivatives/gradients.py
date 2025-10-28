"""FETorch: Finite Element discrete gradient operators.

Functions
---------
eval_shapefun_deriv
    Evaluate shape functions derivates at given coordinates.
build_discrete_sym_gradient
    Build discrete symmetric gradient operator.
vbuild_discrete_sym_gradient
    Build discrete symmetric gradient operator.
build_discrete_gradient
    Build discrete gradient operator.
vbuild_discrete_gradient
    Build discrete gradient operator.
vexpand_grad_operator_sym_2d_to_3d
    Expand 2D discrete symmetric gradient operator to 3D.
vreduce_grad_operator_sym_3d_to_2d
    Reduce 3D discrete symmetric gradient operator to 2D.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.derivatives.jacobian import eval_jacobian
from simulators.fetorch.math.matrixops import get_problem_type_parameters
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
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
    # Evaluate element shape functions local derivatives and Jacobian
    jacobian, jacobian_det, shape_fun_local_deriv = \
        eval_jacobian(element_type, nodes_coords, local_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Compute Jacobian inverse
    jacobian_inv = torch.inverse(jacobian)
    # Compute element shape functions derivatives
    shape_fun_deriv = torch.matmul(jacobian_inv, shape_fun_local_deriv.t()).t()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return shape_fun_deriv, jacobian, jacobian_det
# =============================================================================
def build_discrete_sym_gradient(shape_fun_deriv, comp_order_sym):
    """Build discrete symmetric gradient operator.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
        
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
    # Get number of strain/stress components
    n_comps = len(comp_order_sym)
    # Check number of strain/stress components
    if n_comps != 0.5*n_dof_node*(n_dof_node + 1):
        raise RuntimeError('Invalid number of strain/stress components.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize discrete symmetric gradient operator
    grad_operator_sym = torch.zeros((n_comps, n_node*n_dof_node),
                                    device=shape_fun_deriv.device)
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
def vbuild_discrete_sym_gradient(shape_fun_deriv, comp_order_sym):
    """Build discrete symmetric gradient operator.
    
    Compatible with vectorized mapping.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
        
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
    # Get number of strain/stress components
    n_comps = len(comp_order_sym)
    # Check number of strain/stress components
    if n_comps != 0.5*n_dof_node*(n_dof_node + 1):
        raise RuntimeError('Invalid number of strain/stress components.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize mapping index        
    index_mapping = []
    # Loop over dimensions
    for i in range(1, n_dof_node + 1):
        # Loop over components
        for comp in comp_order_sym:
            # Get component index
            cindex = [int(x) for x in comp]
            # Set mapping index
            if i in cindex:
                if cindex[0] == cindex[1]:
                    index = i
                else:
                    index = int([x for x in cindex if x != i][0])
            else:
                index = 0                
            # Assemble index
            index_mapping.append(index)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add zero element to shape functions derivatives
    mapping_values = \
        torch.cat((torch.zeros((n_node, 1), device=shape_fun_deriv.device),
                   shape_fun_deriv), dim=1)
    # Build discrete symmetric gradient operator
    grad_operator_sym = torch.cat(
        [mapping_values[i, :][index_mapping].reshape(-1, n_comps).t()
         for i in range(n_node)], dim=1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator_sym
# =============================================================================
def build_discrete_gradient(shape_fun_deriv, comp_order_nsym):
    """Build discrete gradient operator.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_nsym : tuple
        Strain/Stress components nonsymmetric order.

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
    grad_operator = torch.zeros((n_comps, n_node*n_dof_node),
                                device=shape_fun_deriv.device)
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
# =============================================================================
def vbuild_discrete_gradient(shape_fun_deriv, comp_order_nsym):
    """Build discrete gradient operator.
    
    Compatible with vectorized mapping.
    
    Parameters
    ----------
    shape_fun_deriv : torch.Tensor(2d)
        Shape functions derivatives evaluated at given local coordinates,
        sorted according with element nodes. Derivative of the i-th shape
        function with respect to the j-th local coordinate is stored in
        shape_fun_deriv[i, j].
    comp_order_nsym : tuple
        Strain/Stress components nonsymmetric order.

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
    # Initialize mapping index        
    index_mapping = []
    # Loop over dimensions
    for i in range(1, n_dof_node + 1):
        # Loop over components
        for comp in comp_order_nsym:
            # Get component index
            cindex = [int(x) for x in comp]
            # Set mapping index
            if i == cindex[0]:
                index = cindex[1]
            else:
                index = 0
            # Assemble index
            index_mapping.append(index)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add zero element to shape functions derivatives
    mapping_values = \
        torch.cat((torch.zeros((n_node, 1), device=shape_fun_deriv.device),
                   shape_fun_deriv), dim=1)
    # Build discrete symmetric gradient operator
    grad_operator = torch.cat(
        [mapping_values[i, :][index_mapping].reshape(-1, n_comps).t()
         for i in range(n_node)], dim=1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator
# =============================================================================
def vexpand_grad_operator_sym_2d_to_3d(grad_operator_sym_2d):
    """Expand 2D discrete symmetric gradient operator to 3D.
    
    Compatible with vectorized mapping.
    
    Parameters
    ----------
    grad_operator_sym_2d : torch.Tensor(2d)
        Discrete symmetric gradient operator (2D).
        
    Returns
    -------
    grad_operator_sym_3d : torch.Tensor(2d)
        Discrete symmetric gradient operator (3D).
    """
    # Get device
    device = grad_operator_sym_2d.device
    # Get float type
    dtype = grad_operator_sym_2d.dtype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get 2D strain/stress components order
    _, comp_order_sym_2d, _ = get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order
    _, comp_order_sym_3d, _ = get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes
    n_node = grad_operator_sym_2d.shape[1]//2
    # Get number of components
    n_comp_2d = grad_operator_sym_2d.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reshape 2D gradient operator to split nodes and degrees of freedom
    grad_operator_reshape = grad_operator_sym_2d.view(n_comp_2d, n_node, 2)
    # Initialize out-of-plane zero tensor
    oop_zeros = torch.zeros((n_comp_2d, n_node, 1), device=device, dtype=dtype)
    # Expand 2D gradient operator to include out-of-plane degree of freedom
    grad_operator_reshape = \
        torch.cat((grad_operator_reshape, oop_zeros), dim=2).reshape(
            n_comp_2d, 3*n_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect 3D gradient operator components
    grad_comps = [
        grad_operator_reshape[comp_order_sym_2d.index(comp), :].unsqueeze(0)
        if comp in comp_order_sym_2d
        else torch.zeros((1, 3*n_node), device=device, dtype=dtype)
        for comp in comp_order_sym_3d]
    # Assemble 3D gradient operator
    grad_operator_sym_3d = torch.cat(grad_comps, dim=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator_sym_3d
# =============================================================================
def vreduce_grad_operator_sym_3d_to_2d(grad_operator_sym_3d):
    """Reduce 3D discrete symmetric gradient operator to 2D.
    
    Compatible with vectorized mapping.
    
    Parameters
    ----------
    grad_operator_sym_3d : torch.Tensor(2d)
        Discrete symmetric gradient operator (3D).
        
    Returns
    -------
    grad_operator_sym_2d : torch.Tensor(2d)
        Discrete symmetric gradient operator (2D).
    """
    # Get device
    device = grad_operator_sym_3d.device
    # Get float type
    dtype = grad_operator_sym_3d.dtype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get 2D strain/stress components order
    _, comp_order_sym_2d, _ = get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order
    _, comp_order_sym_3d, _ = get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes
    n_node = grad_operator_sym_3d.shape[1]//3
    # Get number of components
    n_comp_3d = grad_operator_sym_3d.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reshape 3D gradient operator to split nodes and degrees of freedom
    grad_operator_reshape = grad_operator_sym_3d.view(n_comp_3d, n_node, 3)
    # Reduce 3D gradient operator to remove out-of-plane degree of freedom
    grad_operator_reshape = grad_operator_reshape[:, :, :2].reshape(
        n_comp_3d, 2*n_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect 2D gradient operator components
    grad_comps = [
        grad_operator_reshape[comp_order_sym_3d.index(comp), :].unsqueeze(0)
        if comp in comp_order_sym_2d
        else torch.zeros((1, 2*n_node), device=device, dtype=dtype)
        for comp in comp_order_sym_2d]
    # Assemble 2D gradient operator
    grad_operator_sym_2d = torch.cat(grad_comps, dim=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return grad_operator_sym_2d