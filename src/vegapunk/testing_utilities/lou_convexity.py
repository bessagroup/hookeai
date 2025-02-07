# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pickle
# Third-party
import torch
# Local
from simulators.fetorch.material.models.standard.lou import LouZhangYoon
from ioput.iostandard import make_directory
# =============================================================================
# Summary: Pruning procedure of time series data set 
# =============================================================================
if __name__ == '__main__':
    # Compute convexity domain boundary
    convex_boundary = LouZhangYoon.compute_convex_boundary()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize parameters paths
    parameters_paths = {}
    # Set trial yield parameters
    parameters_trials = ((-2.5, 1.0), (-3.0, -0.25), (2.5, -1.0), (1.5, 1.5),
                         (1.0, 0.5), (-1.0, 0.0))
    # Loop over trial yield parameters
    for i, trial in enumerate(parameters_trials):
        # Set trial yield parameters
        yield_c_trial = torch.tensor(trial[0])
        yield_d_trial = torch.tensor(trial[1])
        # Perform convexity return-mapping
        is_convex, yield_c, yield_d = \
            LouZhangYoon.convexity_return_mapping(yield_c_trial, yield_d_trial)
        # Store convexity return-mapping
        parameters_paths[f'return-mapping-{i}'] = \
            torch.tensor(([[yield_c_trial, yield_d_trial],
                           [yield_c, yield_d]]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model parameters history record file path
    parameters_record_path = ('/home/bernardoferreira/Documents/brown/'
                              'projects/darpa_project/9_local_lou_rc_training/'
                              '0_standard_training/n20/3_model/'
                              'parameters_history_record.pkl')
    # Load model parameters history record
    with open(parameters_record_path, 'rb') as parameters_record_file:
        parameters_history_record = pickle.load(parameters_record_file)
    # Get model parameters history
    model_parameters_history = \
        parameters_history_record['model_parameters_history']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize parameters paths
    parameters_paths = {}
    # Build parameters path
    parameters_path = torch.tensor([model_parameters_history['yield_c_s0'],
                                    model_parameters_history['yield_d_s0']]).T
    # Store parameters path
    parameters_paths['optimization'] = parameters_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set rectangular search domain boundary
    rect_search_domain = ((-1.5, 1.5), (-0.5, 0.5))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/9_local_lou_rc_training/0_standard_training/'
                 'plots')
    # Create plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot convexity domain boundary
    LouZhangYoon.plot_convexity_boundary(
        convex_boundary, parameters_paths=parameters_paths,
        is_path_arrows=True, rect_search_domain=rect_search_domain,
        is_plot_legend=False, save_dir=plots_dir, is_save_fig=True,
        is_stdout_display=True, is_latex=True)