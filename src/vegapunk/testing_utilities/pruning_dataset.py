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
import re
# Third-party
import numpy as np
# Local
from rnn_base_model.data.time_dataset import load_dataset
from ioput.iostandard import make_directory, find_unique_file_with_regex
# =============================================================================
# Summary: Time series data set pruning procedure
# =============================================================================
def prune_time_series_dataset(pruning_dir, testing_types, pruning_params=None):
    """Prune time series data set.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    testing_types : tuple[str]
        Types of testing data sets used to assess the performance of the model
        trained on the pruned training data sets. Available testing types
        include: 'in_distribution', 'out_distribution' and 'unused_data'.
    """
    # Setup main pruning process directories
    base_datasets_dir, prun_datasets_dir, full_dataset_dir, \
        test_dataset_dirs = setup_pruning_dirs(pruning_dir, testing_types) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load full data set
    full_dataset = load_full_dataset(full_dataset_dir)
    # Get full data set size
    n_full = len(full_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get default pruning parameters
    if not isinstance(pruning_params, dict):
        pruning_params = set_default_pruning_parameters()
    # Get number of pruning steps
    n_prun_step = pruning_params['n_prun_step']
    # Get minimum data set size ratio
    min_size_ratio = pruning_params['min_size_ratio']
    # Compute number of pruned samples (per pruning step)
    n_prun_sample = \
        int(np.floor((1.0 - min_size_ratio)*(n_full/n_prun_step) + 1e-10))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize summary file
    write_summary_file(pruning_dir, pruning_params, mode='init',
                       mode_data={'n_full': n_full})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute number of iterations
    n_iter = n_prun_step + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over iterations
    for iter in range(n_iter):
        
        
        
        
        
        
        
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add iteration data to summary file
        testing_types_loss = {'in_distribution': 0.0,
                              'unused_data': 0.0}
        iter_data = {'n_full': n_full,
                     'iter': iter,
                     'n_dev': 0, 'n_train': 0, 'n_valid': 0, 'n_unused': 0,
                     'testing_types_loss': testing_types_loss}
        write_summary_file(pruning_dir, pruning_params, mode='iter',
                           mode_data=iter_data)
# =============================================================================
def setup_pruning_dirs(pruning_dir, testing_types):
    """Setup main pruning process directories.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    testing_types : tuple[str]
        Types of testing data sets used to assess the performance of the model
        trained on the pruned training data sets. Available testing types
        include: 'in_distribution', 'out_distribution' and 'unused_data'.
        
    Returns
    -------
    base_datasets_dir : str
        Base data sets directory.
    prun_datasets_dir : str
        Pruned data sets directory.
    full_dataset_dir : str
        Full data set directory.
    test_dataset_dirs : dict
        Testing data set directory (item, str) for each testing type
        (key, str).
    """
    # Check pruning main directory
    if not os.path.isdir(pruning_dir):
        raise RuntimeError('The pruning main directory has not been found:'
                           '\n\n' + pruning_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set base data sets directory
    base_datasets_dir = os.path.join(os.path.normpath(pruning_dir),
                                     'base_datasets') 
    # Check base data sets directory
    if not os.path.isdir(base_datasets_dir):
        raise RuntimeError('The base data sets directory has not been found:'
                           '\n\n' + base_datasets_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set full data set directory
    full_dataset_dir = os.path.join(os.path.normpath(base_datasets_dir),
                                    '1_training_dataset')
    # Check full data set directory
    if not os.path.isdir(full_dataset_dir):
        raise RuntimeError('The full data set directory has not been found:'
                           '\n\n' + full_dataset_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize testing data sets directories
    test_dataset_dirs = {}
    # Loop over testing types
    for testing_type in testing_types:
        # Set testing data set directory
        if testing_type == 'in_distribution':
            test_dataset_dir = os.path.join(
                os.path.normpath(base_datasets_dir), '5_testing_id_dataset')
        elif testing_type == 'out_distribution':
            test_dataset_dir = os.path.join(
                os.path.normpath(base_datasets_dir), '6_testing_od_dataset')
        elif testing_type == 'unused_data':
            test_dataset_dir = None
        else:
            raise RuntimeError('Unknown testing data set type.'
                               f'\n\nAvailable testing types: {testing_types}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check testing data set directory
        if (test_dataset_dir is not None
                and not os.path.isdir(test_dataset_dir)):
            raise RuntimeError(
                f'The {testing_type} testing data set directory has not been '
                f'found:\n\n' + test_dataset_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store testing data set directory
        test_dataset_dirs[testing_type] = test_dataset_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruned data sets directory
    prun_datasets_dir = os.path.join(os.path.normpath(pruning_dir),
                                     'pruned_datasets')
    # Create model predictions directory
    make_directory(prun_datasets_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return base_datasets_dir, prun_datasets_dir, full_dataset_dir, \
        test_dataset_dirs
# =============================================================================
def set_default_pruning_parameters():
    """Set default pruning parameters.
    
    Returns
    -------
    pruning_params : dict
        Pruning parameters.
    """
    # Set number of pruning steps
    n_prun_step = 5
    # Set minimum data set size ratio
    min_size_ratio = 0.05
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build pruning parameters
    pruning_params = {'n_prun_step': n_prun_step,
                      'min_size_ratio': min_size_ratio}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return pruning_params
# =============================================================================
def write_summary_file(pruning_dir, pruning_params, mode='init', mode_data={}):
    """Write summary file.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    pruning_params : dict
        Pruning parameters.
    mode : {'init', 'iter'}, default='init'
        Summary data mode.
    """
    # Set summary file path
    summary_file_path = os.path.join(os.path.normpath(pruning_dir),
                                     'summary.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize summary content
    summary = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if mode == 'init':
        # Set summary file header
        string = 'Summary - Data set pruning process'
        sep = len(string)*'-'
        summary += [f'\n{string}\n{sep}\n',]
        # Assemble full data set size
        n_full = mode_data['n_full']
        summary += [f'\n> Full data set size: {n_full}\n',]
        # Assemble pruning parameters
        n_prun_step = pruning_params['n_prun_step']
        summary += [f'\n> Number of pruning steps: {n_prun_step}\n',]
        min_size_ratio = pruning_params['min_size_ratio']
        summary += [f'\n> Minimum data set size ratio: {min_size_ratio}\n',]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble pruning iterations header
        summary += [f'\n> Data set pruning iterations ({n_prun_step + 1}):\n',]
        size_sep = 59*'-'
        perf_sep = 35*'-'
        summary += [f'\n{"Data sets":>44s} {"Testing performance":>60s}',
                    f'\n{size_sep:>71s} {perf_sep:>41s}',
                    f'\n{"iter":>6s} '
                    f'{"dev":>8s} {"%full":>6s} '
                    f'{"train":>8s} {"%full":>6s} '
                    f'{"valid":>8s} {"%full":>6s} '
                    f'{"unused":>9s} {"%full":>6s} '
                    f'{"in_dist":>13s} {"out_dist":>13s} {"unused":>13s}\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write summary file
        open(summary_file_path, 'w').writelines(summary)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mode == 'iter':
        # Get full data set size
        n_full = mode_data['n_full']
        # Gather iteration
        iter = mode_data['iter']
        # Gather iteration data sets data
        n_dev = mode_data['n_dev']
        ratio_dev = (n_dev/n_full)*100
        n_train = mode_data['n_train']
        ratio_train = (n_train/n_full)*100
        n_valid = mode_data['n_valid']
        ratio_valid = (n_valid/n_full)*100
        n_unused = mode_data['n_unused']
        ratio_unused = (n_unused/n_full)*100
        # Gather iteration testing performance data
        testing_types_loss = mode_data['testing_types_loss']
        testing_losses = []
        for testing_type in ('in_distribution', 'out_distribution',
                             'unused_data'):
            if testing_type in testing_types_loss.keys():
                testing_losses.append(testing_types_loss[testing_type])
            else:
                testing_losses.append(None)
        # Format iteration testing performance data
        testing_losses = [f'{x:>13.4e}' if x is not None else f'{"":>13s}'
                          for x in testing_losses]
        # Assemble iteration data
        summary += [f'{iter:>6d} '
                    f'{n_dev:>8d} {ratio_dev:>6.1f} '
                    f'{n_train:>8d} {ratio_train:>6.1f} '
                    f'{n_valid:>8d} {ratio_valid:>6.1f} '
                    f'{n_unused:>9d} {ratio_unused:>6.1f} ',
                    ' '.join(testing_losses),
                    '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write summary file
        open(summary_file_path, 'a').writelines(summary)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown summary data mode.')
# =============================================================================
def load_full_dataset(full_dataset_dir):
    """Load full data set.
    
    Parameters
    ----------
    full_dataset_dir : str
        Full data set directory.
        
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Get data set file path
    regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
    is_file_found, full_dataset_file_path = \
        find_unique_file_with_regex(full_dataset_dir, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Full data set file has not been found '
                           f'in data set directory:\n\n'
                           f'{full_dataset_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(full_dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset   
# =============================================================================
if __name__ == "__main__":
    # Set pruning main directory
    pruning_dir = \
        '/home/bernardoferreira/Documents/brown/projects/test_pruning'
    # Set types of testing data sets
    testing_types = ('in_distribution',)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prune time series data set
    prune_time_series_dataset(pruning_dir, testing_types)