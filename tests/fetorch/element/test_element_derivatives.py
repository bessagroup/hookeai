"""Test element derivatives (gradient operators, jacobian, derivatives)."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.vegapunk.simulators.fetorch.element.type.quad4 import FEQuad4
from src.vegapunk.simulators.fetorch.element.type.hexa8 import FEHexa8
from src.vegapunk.simulators.fetorch.element.derivatives.gradients import \
    build_discrete_sym_gradient, build_discrete_gradient, eval_shapefun_deriv
from src.vegapunk.simulators.fetorch.element.derivatives.jacobian import \
    eval_jacobian
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
def test_build_discrete_sym_gradient():
    """Test building of discrete symmetric gradient operator."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain/stress components symmetric order
    comp_order_sym_2d = ('11', '22', '12')
    comp_order_sym_3d = ('11', '22', '33', '12', '23', '13')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dummy shape functions derivatives (2D, 4 nodes)
    shape_fun_deriv = torch.arange(1, 9).reshape((-1, 2))
    # Build discrete symmetric gradient operator
    grad_operator_sym = build_discrete_sym_gradient(shape_fun_deriv,
                                                    comp_order_sym_2d)
    # Set expected discrete symmetric gradient operator
    grad_operator_sym_sol = \
        torch.tensor([[1, 0, 3, 0, 5, 0, 7, 0],
                      [0, 2, 0, 4, 0, 6, 0, 8],
                      [2, 1, 4, 3, 6, 5, 8, 7]], dtype=torch.float)
    # Check discrete symmetric gradient operator
    if not torch.allclose(grad_operator_sym, grad_operator_sym_sol):
        errors.append('Discrete symmetric gradient operator was not properly '
                      'assembled for 2D element with 4 nodes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dummy shape functions derivatives (3D, 8 nodes)
    shape_fun_deriv = torch.arange(1, 25).reshape((-1, 3))
    # Build discrete symmetric gradient operator
    grad_operator_sym = build_discrete_sym_gradient(shape_fun_deriv,
                                                    comp_order_sym_3d)
    # Set expected discrete symmetric gradient operator
    grad_operator_sym_sol = \
        torch.tensor([[1, 0, 0, 4, 0, 0, 7, 0, 0, 10, 0, 0, 13, 0, 0,
                       16, 0, 0, 19, 0, 0, 22, 0, 0],
                      [0, 2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0,
                       0, 17, 0, 0, 20, 0, 0, 23, 0],
                      [0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 12, 0, 0, 15,
                       0, 0, 18, 0, 0, 21, 0, 0, 24],
                      [2, 1, 0, 5, 4, 0, 8, 7, 0, 11, 10, 0, 14, 13, 0,
                       17, 16, 0, 20, 19, 0, 23, 22, 0],
                      [0, 3, 2, 0, 6, 5, 0, 9, 8,  0, 12, 11, 0, 15, 14,
                       0, 18, 17, 0, 21, 20, 0, 24, 23],
                      [3, 0, 1, 6, 0, 4, 9, 0, 7, 12, 0, 10, 15, 0, 13,
                       18, 0, 16, 21, 0, 19, 24, 0, 22]], dtype=torch.float)
    # Check discrete symmetric gradient operator
    if not torch.allclose(grad_operator_sym, grad_operator_sym_sol):
        errors.append('Discrete symmetric gradient operator was not properly '
                      'assembled for 3D element with 8 nodes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_build_discrete_gradient():
    """Test building of discrete gradient operator."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain/stress components symmetric order
    comp_order_nsym_2d = ('11', '21', '12', '22')
    comp_order_nsym_3d = ('11', '21', '31', '12', '22', '32', '13', '23', '33')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dummy shape functions derivatives (2D, 4 nodes)
    shape_fun_deriv = torch.arange(1, 9).reshape((-1, 2))
    # Build discrete symmetric gradient operator
    grad_operator = build_discrete_gradient(shape_fun_deriv,
                                            comp_order_nsym_2d)
    # Set expected discrete symmetric gradient operator
    grad_operator_sol = \
        torch.tensor([[1, 0, 3, 0, 5, 0, 7, 0],
                      [0, 1, 0, 3, 0, 5, 0, 7],
                      [2, 0, 4, 0, 6, 0, 8, 0],
                      [0, 2, 0, 4, 0, 6, 0, 8],], dtype=torch.float)
    # Check discrete symmetric gradient operator
    if not torch.allclose(grad_operator, grad_operator_sol):
        errors.append('Discrete gradient operator was not properly '
                      'assembled for 2D element with 4 nodes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dummy shape functions derivatives (3D, 8 nodes)
    shape_fun_deriv = torch.arange(1, 25).reshape((-1, 3))
    # Build discrete gradient operator
    grad_operator = build_discrete_gradient(shape_fun_deriv,
                                            comp_order_nsym_3d)
    # Set expected discrete gradient operator
    grad_operator_sol = \
        torch.tensor([[1, 0, 0, 4, 0, 0, 7, 0, 0, 10, 0, 0, 13, 0, 0,
                       16, 0, 0, 19, 0, 0, 22, 0, 0],
                      [0, 1, 0, 0, 4, 0, 0, 7, 0, 0, 10, 0, 0, 13, 0,
                       0, 16, 0, 0, 19, 0, 0, 22, 0],
                      [0, 0, 1, 0, 0, 4, 0, 0, 7, 0, 0, 10, 0, 0, 13,
                       0, 0, 16, 0, 0, 19, 0, 0, 22],
                      [2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0, 0,
                       17, 0, 0, 20, 0, 0, 23, 0, 0],
                      [0, 2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0,
                       0, 17, 0, 0, 20, 0, 0, 23, 0],
                      [0, 0, 2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14,
                       0, 0, 17, 0, 0, 20, 0, 0, 23],
                      [3, 0, 0, 6, 0, 0, 9, 0, 0, 12, 0, 0, 15, 0, 0,
                       18, 0, 0, 21, 0, 0, 24, 0, 0],
                      [0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 12, 0, 0, 15, 0,
                       0, 18, 0, 0, 21, 0, 0, 24, 0],
                      [0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 12, 0, 0, 15,
                       0, 0, 18, 0, 0, 21, 0, 0, 24]], dtype=torch.float)
    # Check discrete gradient operator
    if not torch.allclose(grad_operator, grad_operator_sol):
        errors.append('Discrete gradient operator was not properly '
                      'assembled for 3D element with 8 nodes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_eval_jacobian():
    """Test evaluation of finite element Jacobian and determinant."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize 4-Node Quadrilateral element
    element = FEQuad4()
    # Set nodes coordinates
    nodes_coords = torch.tensor([[0.0, 0.0],
                                 [1.0, 0.0],
                                 [2.0, 1.0],
                                 [0.0, 2.0]])
    # Set local coordinates
    local_coords = torch.tensor([0.2, 0.5])
    # Evaluate finite element Jacobian and determinant
    jacobian, jacobian_det, _ = \
        eval_jacobian(element, nodes_coords, local_coords)    
    # Set expected Jacobian and determinant
    jacobian_sol = torch.tensor([[0.875, -0.375],
                                 [0.300, 0.700]])
    jacobian_det_sol = torch.det(jacobian_sol)
    # Check Jacobian and determinant
    if not torch.allclose(jacobian, jacobian_sol):
        errors.append('Jacobian of 4-Node Quadrilateral element was not '
                      'computed correctly.')
    elif not torch.isclose(jacobian_det, jacobian_det_sol):
        errors.append('Jacobian determinant of 4-Node Quadrilateral element '
                      'was not computed correctly.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize 8-Node Hexahedral element
    element = FEHexa8()
    # Set nodes coordinates
    nodes_coords = torch.tensor([[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 2.0, 1.0],
                                 [1.0, 2.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [1.0, 0.0, 2.0],
                                 [2.0, 2.0, 2.0],
                                 [1.0, 2.0, 1.0],])
    # Set local coordinates
    local_coords = torch.tensor([0.2, 0.5, 0.1])
    # Evaluate finite element Jacobian and determinant
    jacobian, jacobian_det, _ = \
        eval_jacobian(element, nodes_coords, local_coords)
    # Set expected Jacobian and determinant
    jacobian_sol = torch.tensor([[0.50000, 0.00000, 0.44375],
                                 [0.50000, 1.00000, 0.13500],
                                 [0.00000, 0.00000, 0.57500]])
    jacobian_det_sol = torch.det(jacobian_sol)
    # Check Jacobian and determinant
    if not torch.allclose(jacobian, jacobian_sol, atol=1e-07):
        errors.append('Jacobian of 8-Node Hexahedral element was not computed '
                      'correctly.')
    elif not torch.isclose(jacobian_det, jacobian_det_sol):
        errors.append('Jacobian determinant of 8-Node Hexahedral element was'
                      'not computed correctly.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_eval_shapefun_deriv():
    """Test evaluation of shape functions derivatives."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize 4-Node Quadrilateral element
    element = FEQuad4()
    # Set nodes coordinates
    nodes_coords = torch.tensor([[0.0, 0.0],
                                 [1.0, 0.0],
                                 [2.0, 1.0],
                                 [0.0, 2.0]])
    # Set local coordinates
    local_coords = torch.tensor([0.2, 0.5])
    # Evaluate shape functions derivatives
    shape_fun_deriv, _, _ = \
        eval_shapefun_deriv(element, nodes_coords, local_coords)
    # Set expected shape functions derivatives
    shape_fun_deriv_sol = torch.tensor([[-0.22413793103, -0.18965517241],
                                        [-0.03448275862, -0.41379310345],
                                        [0.51724137931, 0.20689655172],
                                        [-0.25862068966, 0.39655172414]])
    # Check shape functions derivatives
    if not torch.allclose(shape_fun_deriv, shape_fun_deriv_sol):
        errors.append('Shape functions derivatives of 4-Node Quadrilateral '
                      'element were not computed correctly.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize 8-Node Hexahedral element
    element = FEHexa8()
    # Set nodes coordinates
    nodes_coords = torch.tensor([[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [2.0, 2.0, 1.0],
                                 [1.0, 2.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [1.0, 0.0, 2.0],
                                 [2.0, 2.0, 2.0],
                                 [1.0, 2.0, 1.0]])
    # Set local coordinates
    local_coords = torch.tensor([0.2, 0.5, 0.1])
    # Evaluate shape functions derivatives
    shape_fun_deriv, _, _ = \
        eval_shapefun_deriv(element, nodes_coords, local_coords)
    # Set expected shape functions derivatives
    shape_fun_deriv_sol = torch.tensor(
        [[-0.035326087, -0.060597826, -0.086956522],
         [0.228260870, -0.231521739, -0.130434783],
         [0.684782609, -0.154565217, -0.391304348],
         [-0.105978261, 0.178206522, -0.260869565],
         [-0.214673913, -0.014402174, 0.086956522],
         [0.021739130, -0.193478261, 0.130434783],
         [0.065217391, 0.079565217, 0.391304348],
         [-0.644021739, 0.396793478, 0.260869565]])
    # Check shape functions derivatives
    if not torch.allclose(shape_fun_deriv, shape_fun_deriv_sol):
        errors.append('Shape functions derivatives of 8-Node Hexahedral '
                      'element were not computed correctly.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))