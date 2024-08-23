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
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_mf, vget_tensor_mf, get_tensor_from_mf, vget_tensor_from_mf, \
    get_state_3Dmf_from_2Dmf, vget_state_3Dmf_from_2Dmf, \
    get_state_2Dmf_from_3Dmf, vget_state_2Dmf_from_3Dmf
# =============================================================================
# Summary: Testing vectorization and out-of-place operations
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
    # Display tensor
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
if __name__ == '__main__':
    # Set testing device
    testing_device = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set testing options
    is_testing_get_tensor_mf = False
    is_testing_get_tensor_from_mf = False
    is_testing_get_state_3Dmf_from_2Dmf = False
    is_testing_get_state_2Dmf_from_3Dmf = False
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