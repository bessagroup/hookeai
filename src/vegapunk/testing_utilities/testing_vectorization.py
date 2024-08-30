# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import time
import itertools
# Third-party
import torch
# Local
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_mf, vget_tensor_mf, get_tensor_from_mf, vget_tensor_from_mf, \
    get_state_3Dmf_from_2Dmf, vget_state_3Dmf_from_2Dmf, \
    get_state_2Dmf_from_3Dmf, vget_state_2Dmf_from_3Dmf
from simulators.fetorch.math.voigt_notation import get_stress_vmf, \
    vget_stress_vmf, get_strain_from_vmf, vget_strain_from_vmf
from simulators.fetorch.element.derivatives.jacobian import eval_jacobian
from simulators.fetorch.element.derivatives.gradients import \
    build_discrete_sym_gradient, vbuild_discrete_sym_gradient, \
    build_discrete_gradient, vbuild_discrete_gradient
from simulators.fetorch.element.type.quad4 import FEQuad4
from simulators.fetorch.element.type.hexa8 import FEHexa8
# =============================================================================
# Summary: Testing vectorization and out-of-place operations
# =============================================================================
def vbuild_tensor_from_comps(n_dim, comps, comps_array, device=None):
    """Build strain/stress tensor from given components (vectorized).
    
    Parameters
    ----------
    n_dim : int
        Problem number of spatial dimensions.
    comps : tuple[str]
        Strain/Stress components order.
    comps_array : torch.Tensor(1d)
        Strain/Stress components array.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
    
    Returns
    -------
    tensor : torch.Tensor(2d)
        Strain/Stress tensor.
    """
    # Get device from input tensor
    if device is None:
        device = comps_array.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set row major components order       
    row_major_order = tuple(f'{i + 1}{j + 1}' for i, j
                            in itertools.product(range(n_dim), repeat=2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = [comps.index(x) if x in comps
                 else comps.index(x[::-1]) for x in row_major_order]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build tensor
    tensor = comps_array[index_map].view(n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor
# =============================================================================
def vstore_tensor_comps(comps, tensor, device=None):
    """Store strain/stress tensor components in array (vectorized).
    
    Parameters
    ----------
    comps : tuple[str]
        Strain/Stress components order.
    tensor : torch.Tensor(2d)
        Strain/Stress tensor.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
    
    Returns
    -------
    comps_array : torch.Tensor(1d)
        Strain/Stress components array.
    """
    # Get device from input tensor
    if device is None:
        device = tensor.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = tuple([int(x[i]) - 1 for x in comps] for i in range(2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build tensor components array
    comps_array = tensor[index_map]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return comps_array
# =============================================================================
def function_timer(function, args, n_calls=1):
    # Initialize total execution time
    total_time = 0
    # Loop over number of function calls
    for i in range(n_calls):
        # Set initial call time
        t0 = time.time()
        # Call function
        function(*args)
        # Add to total execution time
        total_time += time.time() - t0
    # Compute average time per function call
    avg_time_call = total_time/n_calls
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return avg_time_call
# =============================================================================
def testing_get_tensor_mf(device='cpu'):
    # Get 3D problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second order tensor
    tensor = torch.arange(0, 9, device=device).reshape(3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER TENSOR:\n')
    print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Second order
    #
    # Loop over symmetric and nonsymmetric cases
    for comp_order in (comp_order_sym, comp_order_nsym):
        print('\n' + 40*'-')
        print(f'\nSECOND ORDER CASE: {comp_order}')
        # Original
        o_tensor_mf = get_tensor_mf(tensor, n_dim, comp_order)
        o_avg_time_call = function_timer(get_tensor_mf,
                                        (tensor, n_dim, comp_order),
                                        n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_tensor_mf}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_tensor_mf = vget_tensor_mf(tensor, n_dim, comp_order)
        v_avg_time_call = function_timer(vget_tensor_mf,
                                        (tensor, n_dim, comp_order),
                                        n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_tensor_mf}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_tensor_mf, v_tensor_mf):
            RuntimeError('Original and vectorized results do not match!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create fourth order tensor
    tensor = torch.arange(0, 3**4, device=device).reshape(3, 3, 3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nFOURTH ORDER TENSOR:\n')
    #print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Fourth order
    #
    # Loop over symmetric and nonsymmetric cases
    for comp_order in (comp_order_sym, comp_order_nsym):
        print('\n' + 40*'-')
        print(f'\nFOURTH ORDER CASE: {comp_order}')
        # Original
        o_tensor_mf = get_tensor_mf(tensor, n_dim, comp_order)
        o_avg_time_call = function_timer(get_tensor_mf,
                                        (tensor, n_dim, comp_order),
                                        n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_tensor_mf}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_tensor_mf = vget_tensor_mf(tensor, n_dim, comp_order)
        v_avg_time_call = function_timer(vget_tensor_mf,
                                        (tensor, n_dim, comp_order),
                                        n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_tensor_mf}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_tensor_mf, v_tensor_mf):
            RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_get_tensor_from_mf(device='cpu'):
    # Get 3D problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second order tensor
    tensor = torch.arange(0, 9, device=device).reshape(3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER TENSOR:\n')
    print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Second order
    #
    # Loop over symmetric and nonsymmetric cases
    for comp_order in (comp_order_sym, comp_order_nsym):
        print('\n' + 40*'-')
        print(f'\nSECOND ORDER CASE: {comp_order}')
        # Get tensor matricial form
        tensor_mf = vget_tensor_mf(tensor, n_dim, comp_order)
        # Original
        o_tensor = get_tensor_from_mf(tensor_mf, n_dim, comp_order)
        o_avg_time_call = function_timer(get_tensor_from_mf,
                                         (tensor_mf, n_dim, comp_order),
                                         n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_tensor}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_tensor = vget_tensor_from_mf(tensor_mf, n_dim, comp_order)
        v_avg_time_call = function_timer(vget_tensor_from_mf,
                                         (tensor_mf, n_dim, comp_order),
                                         n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_tensor}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_tensor, v_tensor):
            RuntimeError('Original and vectorized results do not match!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create fourth order tensor
    tensor = torch.arange(0, 3**4, device=device).reshape(3, 3, 3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nFOURTH ORDER TENSOR:\n')
    #print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Fourth order
    #
    # Loop over symmetric and nonsymmetric cases
    for comp_order in (comp_order_sym, comp_order_nsym):
        print('\n' + 40*'-')
        print(f'\nFOURTH ORDER CASE: {comp_order}')
        # Get tensor matricial form
        tensor_mf = vget_tensor_mf(tensor, n_dim, comp_order)
        # Original
        o_tensor = get_tensor_from_mf(tensor_mf, n_dim, comp_order)
        o_avg_time_call = function_timer(get_tensor_from_mf,
                                         (tensor_mf, n_dim, comp_order),
                                         n_calls=1000)
        print(f'\nMatricial form (original):\n')
        #print(f' {o_tensor}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_tensor = vget_tensor_from_mf(tensor_mf, n_dim, comp_order)
        v_avg_time_call = function_timer(vget_tensor_from_mf,
                                         (tensor_mf, n_dim, comp_order),
                                         n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        #print(f' {v_tensor}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_tensor, v_tensor):
            RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_get_state_3Dmf_from_2Dmf(device='cpu'):
    # Create 2D tensor matricial form (symmetric)
    mf_2d = torch.arange(0, 3, device=device)
    # Set out-of-plane component
    comp_33 = 3.0
    # Display tensor
    print('\n' + 40*'-')
    print(f'\n2D MATRICIAL FORM:\n')
    print(f' {mf_2d} and {comp_33}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Symmetric
    #
    print('\n' + 40*'-')
    print(f'\nSYMMETRIC CASE:')
    # Original
    o_mf_3d = get_state_3Dmf_from_2Dmf(2, mf_2d, comp_33)
    o_avg_time_call = function_timer(get_state_3Dmf_from_2Dmf,
                                     (2, mf_2d, comp_33), n_calls=1000)
    print(f'\nMatricial form (original):\n')
    print(f' {o_mf_3d}')
    print(f'\n avg. time per call = {o_avg_time_call:.4e}')
    # Vectorized
    v_mf_3d = vget_state_3Dmf_from_2Dmf(mf_2d, comp_33)
    v_avg_time_call = function_timer(vget_state_3Dmf_from_2Dmf,
                                     (mf_2d, comp_33), n_calls=1000)
    print(f'\nMatricial form (vectorized):\n')
    print(f' {v_mf_3d}')
    print(f'\n avg. time per call = {v_avg_time_call:.4e}')
    # Check results
    if not torch.allclose(o_mf_3d, v_mf_3d):
        RuntimeError('Original and vectorized results do not match!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create 2D tensor matricial form (nonsymmetric)
    mf_2d = torch.arange(0, 4, device=device)
    # Set out-of-plane component
    comp_33 = 4.0
    # Display tensor
    print('\n' + 40*'-')
    print(f'\n2D MATRICIAL FORM:\n')
    print(f' {mf_2d} and {comp_33}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Symmetric
    #
    print('\n' + 40*'-')
    print(f'\nNONSYMMETRIC CASE:')
    # Original
    o_mf_3d = get_state_3Dmf_from_2Dmf(2, mf_2d, comp_33)
    o_avg_time_call = function_timer(get_state_3Dmf_from_2Dmf,
                                     (2, mf_2d, comp_33), n_calls=1000)
    print(f'\nMatricial form (original):\n')
    print(f' {o_mf_3d}')
    print(f'\n avg. time per call = {o_avg_time_call:.4e}')
    # Vectorized
    v_mf_3d = vget_state_3Dmf_from_2Dmf(mf_2d, comp_33)
    v_avg_time_call = function_timer(vget_state_3Dmf_from_2Dmf,
                                     (mf_2d, comp_33), n_calls=1000)
    print(f'\nMatricial form (vectorized):\n')
    print(f' {v_mf_3d}')
    print(f'\n avg. time per call = {v_avg_time_call:.4e}')
    # Check results
    if not torch.allclose(o_mf_3d, v_mf_3d):
        RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_get_state_2Dmf_from_3Dmf(device='cpu'):
    print('\n' + 40*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Second order
    #
    for case in ('symmetric', 'nonsymmetric'):
        # Create 3D second order tensor matricial form
        if case == 'symmetric':
            mf_3d = torch.arange(0, 6, dtype=torch.float, device=device)
        else:
            mf_3d = torch.arange(0, 9, dtype=torch.float, device=device)
        print(f'\n2D SECOND ORDER MATRICIAL FORM:\n')
        print(f' {mf_3d}')
        # Original
        o_mf_2d = get_state_2Dmf_from_3Dmf(2, mf_3d)
        o_avg_time_call = function_timer(get_state_2Dmf_from_3Dmf,
                                         (2, mf_3d), n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_mf_2d}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_mf_2d = vget_state_2Dmf_from_3Dmf(mf_3d)
        v_avg_time_call = function_timer(vget_state_2Dmf_from_3Dmf,
                                         (mf_3d,), n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_mf_2d}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_mf_2d, v_mf_2d):
            RuntimeError('Original and vectorized results do not match!')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 40*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Fourth order
    #
    for case in ('symmetric', 'nonsymmetric'):
        # Create 3D second order tensor matricial form
        if case == 'symmetric':
            mf_3d = torch.arange(0, 36, dtype=torch.float,
                                 device=device).reshape(6, 6)
        else:
            mf_3d = torch.arange(0, 81, dtype=torch.float,
                                 device=device).reshape(9, 9)
        print(f'\n2D FOURTH ORDER MATRICIAL FORM:\n')
        print(f' {mf_3d}')
        # Original
        o_mf_2d = get_state_2Dmf_from_3Dmf(2, mf_3d)
        o_avg_time_call = function_timer(get_state_2Dmf_from_3Dmf,
                                        (2, mf_3d), n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_mf_2d}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_mf_2d = vget_state_2Dmf_from_3Dmf(mf_3d)
        v_avg_time_call = function_timer(vget_state_2Dmf_from_3Dmf,
                                        (mf_3d,), n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_mf_2d}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_mf_2d, v_mf_2d):
            RuntimeError('Original and vectorized results do not match!')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 40*'-')
# =============================================================================
def testing_build_tensor_from_comps(device='cpu'):
    # Get 3D problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display tensor
    print('\n' + 40*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized
    #
    for case in ('symmetric', 'nonsymmetric'):
        # Create 3D second order tensor components
        if case == 'symmetric':
            comps_array = torch.arange(0, 6, dtype=torch.float, device=device)
            comps = comp_order_sym
            is_symmetric = True
        else:
            comps_array = torch.arange(0, 9, dtype=torch.float, device=device)
            comps = comp_order_nsym
            is_symmetric = False
        print(f'\n2D SECOND ORDER TENSOR COMPONENTS:\n')
        print(f' {comps_array}')
        # Original
        build_tensor_from_comps = \
            RecurrentConstitutiveModel.build_tensor_from_comps
        o_tensor = build_tensor_from_comps(n_dim, comps, comps_array,
                                           is_symmetric=is_symmetric)
        o_avg_time_call = function_timer(build_tensor_from_comps, 
                                         (n_dim, comps, comps_array,
                                          is_symmetric),
                                         n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_tensor}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        v_tensor = vbuild_tensor_from_comps(n_dim, comps, comps_array)
        v_avg_time_call = function_timer(vbuild_tensor_from_comps, 
                                         (n_dim, comps, comps_array),
                                         n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_tensor}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_tensor, v_tensor):
            RuntimeError('Original and vectorized results do not match!')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 40*'-')
# =============================================================================
def testing_store_tensor_comps(device='cpu'):
    # Get 3D problem type parameters
    _, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second order tensor
    tensor = torch.arange(0, 9, dtype=torch.float, device=device).reshape(3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER TENSOR:\n')
    print(f' {tensor}')
    print('\n' + 40*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized
    #
    for case in ('symmetric', 'nonsymmetric'):
        print(f'\nSYMMETRY CASE: {case}')
        # Create 3D second order tensor
        if case == 'symmetric':
            comps = comp_order_sym
        else:
            comps = comp_order_nsym
        # Original
        store_tensor_comps = RecurrentConstitutiveModel.store_tensor_comps
        o_comps_array = store_tensor_comps(comps, tensor)
        o_avg_time_call = function_timer(store_tensor_comps, 
                                         (comps, tensor), n_calls=1000)
        print(f'\nMatricial form (original):\n')
        print(f' {o_comps_array}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # Vectorized
        store_tensor_comps = RecurrentConstitutiveModel.store_tensor_comps
        v_comps_array = vstore_tensor_comps(comps, tensor)
        v_avg_time_call = function_timer(vstore_tensor_comps, 
                                         (comps, tensor), n_calls=1000)
        print(f'\nMatricial form (vectorized):\n')
        print(f' {v_comps_array}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # Check results
        if not torch.allclose(o_comps_array, v_comps_array):
            RuntimeError('Original and vectorized results do not match!')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 40*'-')
# =============================================================================
def testing_get_stress_vmf(device='cpu'):
    # Get 3D problem type parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second order tensor
    tensor = torch.arange(0, 9, dtype=torch.float, device=device).reshape(3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER TENSOR:\n')
    print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized:
    print('\n' + 40*'-')
    # Original
    o_tensor_mf = get_stress_vmf(tensor, n_dim, comp_order_sym)
    o_avg_time_call = function_timer(get_stress_vmf,
                                     (tensor, n_dim, comp_order_sym),
                                     n_calls=1000)
    print(f'\nMatricial form (original):\n')
    print(f' {o_tensor_mf}')
    print(f'\n avg. time per call = {o_avg_time_call:.4e}')
    # Vectorized
    v_tensor_mf = vget_stress_vmf(tensor, n_dim, comp_order_sym)
    v_avg_time_call = function_timer(vget_stress_vmf,
                                     (tensor, n_dim, comp_order_sym),
                                     n_calls=1000)
    print(f'\nMatricial form (vectorized):\n')
    print(f' {v_tensor_mf}')
    print(f'\n avg. time per call = {v_avg_time_call:.4e}')
    # Check results
    if not torch.allclose(o_tensor_mf, v_tensor_mf):
        RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_get_strain_from_vmf(device='cpu'):
    # Get 3D problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second order tensor
    tensor = torch.arange(0, 9, device=device).reshape(3, 3)
    # Display tensor
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER TENSOR:\n')
    print(f' {tensor}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Original vs Vectorized: Second order
    #
    print('\n' + 40*'-')
    print(f'\nSECOND ORDER CASE: {comp_order_sym}')
    # Get tensor matricial form
    tensor_mf = vget_stress_vmf(tensor, n_dim, comp_order_sym)
    # Original
    o_tensor = get_strain_from_vmf(tensor_mf, n_dim, comp_order_sym)
    o_avg_time_call = function_timer(get_strain_from_vmf,
                                     (tensor_mf, n_dim, comp_order_sym),
                                     n_calls=1000)
    print(f'\nMatricial form (original):\n')
    print(f' {o_tensor}')
    print(f'\n avg. time per call = {o_avg_time_call:.4e}')
    # Vectorized
    v_tensor = vget_strain_from_vmf(tensor_mf, n_dim, comp_order_sym)
    v_avg_time_call = function_timer(vget_strain_from_vmf,
                                     (tensor_mf, n_dim, comp_order_sym),
                                     n_calls=1000)
    print(f'\nMatricial form (vectorized):\n')
    print(f' {v_tensor}')
    print(f'\n avg. time per call = {v_avg_time_call:.4e}')
    # Check results
    if not torch.allclose(o_tensor, v_tensor):
        RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_eval_jacobian(device='cpu'):
    # Loop over number of spatial dimensions
    for n_dim in (2, ):
        print('\n' + 40*'-')
        print(f'\nNUMBER OF DIMENSIONS: {n_dim}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing case
        if n_dim == 2:
            # Set element type
            element_type = FEQuad4(n_gauss=4, device_type=device)
            # Get element type parameters
            n_node = element_type.get_n_node()
            n_dof_node = element_type.get_n_dof_node()
            # Set element node coordinates
            nodes_coords = torch.zeros((n_node, n_dof_node),
                                    dtype=torch.float, device=device)
            nodes_coords[0, :] = torch.tensor((-1.0, -2.0))
            nodes_coords[1, :] = torch.tensor((0.5, -1.0))
            nodes_coords[2, :] = torch.tensor((2.0, 1.0))
            nodes_coords[3, :] = torch.tensor((-1.0, 1.5))
            # Set Gauss point local coordinates
            local_coords = torch.tensor((-0.45, 0.25),
                                        dtype=torch.float, device=device)
        else:
            # Set element type
            element_type = FEHexa8(n_gauss=8, device_type=device)
            # Get element type parameters
            n_node = element_type.get_n_node()
            n_dof_node = element_type.get_n_dof_node()
            # Set element node coordinates
            nodes_coords = torch.zeros((n_node, n_dof_node),
                                    dtype=torch.float, device=device)
            nodes_coords[0, :] = torch.tensor((-2.0, -1.0, -1.5))
            nodes_coords[1, :] = torch.tensor((1.0, -1.5, -1.0))
            nodes_coords[2, :] = torch.tensor((1.0, 1.0, -1.0))
            nodes_coords[3, :] = torch.tensor((-1.5, 1.0, -1.0))
            nodes_coords[4, :] = torch.tensor((-1.0, -1.5, 1.0))
            nodes_coords[5, :] = torch.tensor((1.0, -1.0, 1.0))
            nodes_coords[6, :] = torch.tensor((2.0, 1.0, 2.0))
            nodes_coords[7, :] = torch.tensor((-1.0, 1.5, 1.0))
            # Set Gauss point local coordinates
            local_coords = torch.tensor((-0.45, 0.25, -0.10),
                                        dtype=torch.float, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Original: Function call
        #o_jacobian, o_jacobian_det, o_shape_fun_local_deriv = \
        #        eval_jacobian(element_type, nodes_coords, local_coords)
        ## Original: Average function execution time
        #o_avg_time_call = \
        #    function_timer(eval_jacobian,
        #                (element_type, nodes_coords, local_coords),
        #                n_calls=1000)
        ## Original: Display
        #print(f'\nResults & Time (original):')
        #print(f'\n {o_jacobian}')
        #print(f'\n {o_jacobian_det}')
        #print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized: Function call
        v_jacobian, v_jacobian_det, v_shape_fun_local_deriv = \
                eval_jacobian(element_type, nodes_coords, local_coords)
        # Vectorized: Average function execution time
        v_avg_time_call = \
            function_timer(eval_jacobian,
                        (element_type, nodes_coords, local_coords),
                        n_calls=1000)
        # Vectorized: Display
        print(f'\nResults & Time (vectorized):')
        print(f'\n {v_jacobian}')
        print(f'\n {v_jacobian_det}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Check results
        #if not torch.allclose(o_jacobian, v_jacobian):
        #    RuntimeError('Original and vectorized results do not match!')
        #if not torch.allclose(o_jacobian_det, v_jacobian_det):
        #    RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_build_discrete_sym_gradient(device='cpu'):
    # Loop over number of spatial dimensions
    for n_dim in (2, 3):
        print('\n' + 40*'-')
        print(f'\nNUMBER OF DIMENSIONS: {n_dim}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing case
        if n_dim == 2:
            # Get problem type parameters
            n_dim, comp_order_sym, _ = \
                get_problem_type_parameters(problem_type=1)
            # Set number of element nodes
            n_node = 4
            # Create tensor of shape functions derivatives
            shape_fun_deriv = torch.arange(0, n_node*n_dim, dtype=torch.float,
                                           device=device).reshape(n_node, -1)
        else:
            # Get problem type parameters
            n_dim, comp_order_sym, _ = \
                get_problem_type_parameters(problem_type=4)
            # Set number of element nodes
            n_node = 8
            # Create tensor of shape functions derivatives
            shape_fun_deriv = torch.arange(0, n_node*n_dim, dtype=torch.float,
                                           device=device).reshape(n_node, -1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f'\nShape functions derivatives:')
        print(f'\n {shape_fun_deriv}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Original: Function call
        o_grad_operator_sym = \
            build_discrete_sym_gradient(shape_fun_deriv, comp_order_sym)
        # Original: Average function execution time
        o_avg_time_call = \
            function_timer(build_discrete_sym_gradient,
                           (shape_fun_deriv, comp_order_sym),
                           n_calls=1000)
        # Original: Display
        print(f'\nResults & Time (original):')
        print(f'\n {o_grad_operator_sym}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized: Function call
        v_grad_operator_sym = \
            vbuild_discrete_sym_gradient(shape_fun_deriv, comp_order_sym)
        # Vectorized: Average function execution time
        v_avg_time_call = \
            function_timer(vbuild_discrete_sym_gradient,
                           (shape_fun_deriv, comp_order_sym),
                           n_calls=1000)
        # Vectorized: Display
        print(f'\nResults & Time (vectorized):')
        print(f'\n {v_grad_operator_sym}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Check results
        if not torch.allclose(o_grad_operator_sym, v_grad_operator_sym):
            RuntimeError('Original and vectorized results do not match!')
# =============================================================================
def testing_build_discrete_gradient(device='cpu'):
    # Loop over number of spatial dimensions
    for n_dim in (2, 3):
        print('\n' + 40*'-')
        print(f'\nNUMBER OF DIMENSIONS: {n_dim}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing case
        if n_dim == 2:
            # Get problem type parameters
            n_dim, _, comp_order_nsym = \
                get_problem_type_parameters(problem_type=1)
            # Set number of element nodes
            n_node = 4
            # Create tensor of shape functions derivatives
            shape_fun_deriv = torch.arange(0, n_node*n_dim, dtype=torch.float,
                                           device=device).reshape(n_node, -1)
        else:
            # Get problem type parameters
            n_dim, _, comp_order_nsym = \
                get_problem_type_parameters(problem_type=4)
            # Set number of element nodes
            n_node = 8
            # Create tensor of shape functions derivatives
            shape_fun_deriv = torch.arange(0, n_node*n_dim, dtype=torch.float,
                                           device=device).reshape(n_node, -1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f'\nShape functions derivatives:')
        print(f'\n {shape_fun_deriv}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Original: Function call
        o_grad_operator = \
            build_discrete_gradient(shape_fun_deriv, comp_order_nsym)
        # Original: Average function execution time
        o_avg_time_call = \
            function_timer(build_discrete_gradient,
                           (shape_fun_deriv, comp_order_nsym),
                           n_calls=1000)
        # Original: Display
        print(f'\nResults & Time (original):')
        print(f'\n {o_grad_operator}')
        print(f'\n avg. time per call = {o_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized: Function call
        v_grad_operator = \
            vbuild_discrete_gradient(shape_fun_deriv, comp_order_nsym)
        # Vectorized: Average function execution time
        v_avg_time_call = \
            function_timer(vbuild_discrete_gradient,
                           (shape_fun_deriv, comp_order_nsym),
                           n_calls=1000)
        # Vectorized: Display
        print(f'\nResults & Time (vectorized):')
        print(f'\n {v_grad_operator}')
        print(f'\n avg. time per call = {v_avg_time_call:.4e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Check results
        if not torch.allclose(o_grad_operator, v_grad_operator):
            RuntimeError('Original and vectorized results do not match!')
# =============================================================================
if __name__ == '__main__':
    # Set testing device
    testing_device = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set testing options
    is_testing_get_tensor_mf = False
    is_testing_get_tensor_from_mf = False
    is_testing_get_state_3Dmf_from_2Dmf = False
    is_testing_get_state_2Dmf_from_3Dmf = False
    is_testing_build_tensor_from_comps = False
    is_testing_store_tensor_comps = False
    is_testing_get_stress_vmf = False
    is_testing_get_strain_from_vmf = False
    is_testing_eval_jacobian = False
    is_testing_build_discrete_sym_gradient = False
    is_testing_build_discrete_gradient = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform tests
    if is_testing_get_tensor_mf:
        testing_get_tensor_mf(device=testing_device)
    if is_testing_get_tensor_from_mf:
        testing_get_tensor_from_mf(device=testing_device)
    if is_testing_get_state_3Dmf_from_2Dmf:
        testing_get_state_3Dmf_from_2Dmf(device=testing_device)
    if is_testing_get_state_2Dmf_from_3Dmf:
        testing_get_state_2Dmf_from_3Dmf(device=testing_device)
    if is_testing_build_tensor_from_comps:
        testing_build_tensor_from_comps(device=testing_device)
    if is_testing_store_tensor_comps:
        testing_store_tensor_comps(device=testing_device)
    if is_testing_get_stress_vmf:
        testing_get_stress_vmf(device=testing_device)
    if is_testing_get_strain_from_vmf:
        testing_get_strain_from_vmf(device=testing_device)
    if is_testing_eval_jacobian:
        testing_eval_jacobian(device=testing_device)
    if is_testing_build_discrete_sym_gradient:
        testing_build_discrete_sym_gradient(device=testing_device)
    if is_testing_build_discrete_gradient:
        testing_build_discrete_gradient(device=testing_device)