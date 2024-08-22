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
# Third-party
import torch
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_mf, vget_tensor_mf, get_tensor_from_mf, vget_tensor_from_mf
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
if __name__ == '__main__':
    # Set testing device
    testing_device = 'cuda'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set testing options
    is_testing_get_tensor_mf = True
    is_testing_get_tensor_from_mf = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform tests
    if is_testing_get_tensor_mf:
        testing_get_tensor_mf(device=testing_device)
    if is_testing_get_tensor_from_mf:
        testing_get_tensor_from_mf(device=testing_device)