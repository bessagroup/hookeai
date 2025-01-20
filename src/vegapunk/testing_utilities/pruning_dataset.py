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
import shutil
import re
import pickle
import time
import datetime
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from rnn_base_model.data.time_dataset import save_dataset, load_dataset, \
    split_dataset, get_parent_dataset_indices, TimeSeriesDatasetInMemory
from projects.darpa_metals.rnn_material_model.user_scripts.train_model import \
    perform_model_standard_training
from projects.darpa_metals.rnn_material_model.user_scripts.predict import \
    perform_model_prediction
from ioput.plots import plot_xy_data, save_figure
from ioput.iostandard import make_directory, find_unique_file_with_regex
# =============================================================================
# Summary: Pruning procedure of time series data set 
# =============================================================================
def prune_time_series_dataset(pruning_dir, testing_types, pruning_params=None,
                              is_remove_pruning_models=False,
                              device_type='cpu', is_verbose=False):
    """Prune time series data set.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    testing_types : tuple[str]
        Types of testing data sets used to assess the performance of the model
        trained on the pruned training data sets. Available testing types
        include: 'in_distribution', 'out_distribution' and 'unused_data'.
    pruning_params : dict, default=None
        Pruning parameters. If None, then a default set of pruning parameters
        is adopted.
    is_remove_pruning_models : bool, default=False
        If True, then remove pruning iteration models when pruning iteration is
        complete.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.

    Returns
    -------
    pruning_params : dict
        Pruning parameters.
    pruning_iterative_data : dict
        Pruning iterative data (item, dict) for each pruning iteration
        (key, str).
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nTime series data set pruning'
              '\n----------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Setup main pruning process directories
    _, prun_datasets_dir, full_dataset_dir, \
        test_dataset_dirs = setup_pruning_dirs(pruning_dir, testing_types) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Loading full data set...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load full data set
    full_dataset = TimeSeriesDatasetInMemory.from_dataset(
        load_full_dataset(full_dataset_dir))
    # Get full data set size
    n_full = len(full_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get default pruning parameters
    if not isinstance(pruning_params, dict):
        pruning_params = set_default_pruning_parameters()
    # Get maximum number of pruning iterations
    n_iter_max = pruning_params['n_iter_max']
    # Get pruning iteration data set split sizes
    pruning_iter_sizes = pruning_params['pruning_iter_sizes']
    # Add full data set size to pruning parameters
    pruning_params['n_full'] = n_full
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize summary file
    write_summary_file(pruning_dir, pruning_params, mode='init')
    # Initialize pruning iterative data
    pruning_iterative_data = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Initializing pruning iterative loop...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize pruning iteration counter
    iter = 0
    # Initialize pruning termination flag
    is_keep_pruning = True
    # Loop over iterations
    while is_keep_pruning:
        if is_verbose:
            print(f'\n> Pruning iteration {iter}:')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build development data set
        if iter == 0:
            # Initialize development data set
            dev_dataset = full_dataset
            # Initialize unused data set
            unused_dataset = TimeSeriesDatasetInMemory([])
        else:
            if is_verbose:
                print(f'\n  > Performing pruning step...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform pruning step
            is_valid_pruning_step, step_status, dev_dataset, unused_dataset = \
                perform_pruning_step(prun_datasets_dir, pruning_params,
                                     dev_dataset, unused_dataset,
                                     device_type=device_type,
                                     is_verbose=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check pruning step admissibility
            if not is_valid_pruning_step:
                # Set termination status
                termination_status = step_status
                # Terminate pruning
                is_keep_pruning = False
                continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n  > Development data set size ratio: '
                  f'{(len(dev_dataset)/n_full)*100:>6.1f}%')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Randomly split development data set
        dev_dataset_split = split_dataset(dev_dataset, pruning_iter_sizes)
        # Get pruning iteration data sets
        train_dataset = dev_dataset_split['training']
        val_dataset = dev_dataset_split['validation']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set pruning iteration directory
        pruning_iter_dir, model_directory, train_dataset_file_path, \
            val_dataset_file_path, test_dataset_file_paths, \
                test_prediction_dirs = \
                    set_pruning_iter_dir(prun_datasets_dir, train_dataset,
                                         val_dataset,
                                         test_dataset_dirs=test_dataset_dirs,
                                         unused_dataset=unused_dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n  > Performing pruning iteration model training...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform pruning iteration model standard training
        perform_model_standard_training(
            train_dataset_file_path, model_directory,
            val_dataset_file_path=val_dataset_file_path,
            device_type=device_type, is_verbose=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize pruning iteration model testing loss
        testing_types_loss = {}
        # Loop over testing types
        for testing_type in testing_types:
            # Get testing type prediction directory
            prediction_subdir = test_prediction_dirs[testing_type]
            # Get testing type data set file path
            test_dataset_file_path = test_dataset_file_paths[testing_type]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform pruning iteration model prediction
            if testing_type == 'unused_data' and len(unused_dataset) == 0:
                # Set testing loss to None if testing data set is null
                avg_predict_loss = None
            else:
                if is_verbose:
                    print(f'\n  > Performing pruning iteration model testing '
                          f'({testing_type})...')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform pruning iteration model prediction
                predict_subdir, avg_predict_loss = perform_model_prediction(
                    prediction_subdir, test_dataset_file_path, model_directory,
                    is_remove_sample_prediction=True,
                    device_type=device_type, is_verbose=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store testing loss
            testing_types_loss[testing_type] = avg_predict_loss
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove pruning iteration model
        if is_remove_pruning_models:
            shutil.rmtree(model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build pruning iteration data
        iter_data = {'n_full': n_full,
                     'iter': iter,
                     'n_dev': len(dev_dataset),
                     'n_train': len(train_dataset),
                     'n_valid': len(val_dataset),
                     'n_unused': len(unused_dataset),
                     'testing_types_loss': testing_types_loss}     
        # Add pruning iteration data to summary file
        write_summary_file(pruning_dir, pruning_params, mode='iter',
                           mode_data=iter_data)
        # Store pruning iteration data
        pruning_iterative_data[str(iter)] = iter_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            total_time_sec = time.time() - start_time_sec
            print(f'\n  > Total elapsed time (s): '
                  f'{str(datetime.timedelta(seconds=int(total_time_sec)))}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check pruning iterations
        if iter >= n_iter_max:
            # Set termination status
            termination_status = \
                'Maximum number of pruning iterations reached.'
            # Terminate pruning
            is_keep_pruning = False
        else:
            # Update pruning iteration counter
            iter += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Termination status: {termination_status}')
        print('\n> Pruning iterative loop finished successfully!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add termination status to summary file
    write_summary_file(pruning_dir, pruning_params, mode='end',
                       mode_data={'termination_status': termination_status})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build pruning data
    pruning_data = {'pruning_params': pruning_params,
                    'pruning_iterative_data': pruning_iterative_data}
    # Set pruning data file path
    pruning_data_file_path = os.path.join(os.path.normpath(pruning_dir),
                                          'pruning_data.pkl')
    # Save pruning data
    with open(pruning_data_file_path, 'wb') as pruning_file:
        pickle.dump(pruning_data, pruning_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return pruning_params, pruning_iterative_data
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
    # Set pruning plots directory
    prun_plots_dir = os.path.join(os.path.normpath(pruning_dir), 'plots')
    # Create pruning plots directory
    make_directory(prun_plots_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return base_datasets_dir, prun_datasets_dir, full_dataset_dir, \
        test_dataset_dirs
# =============================================================================
def set_pruning_iter_dir(prun_datasets_dir, train_dataset,
                         val_dataset, unused_dataset,
                         test_dataset_dirs={},
                         dataset_basename='ss_paths_dataset'):
    """Setup pruning iteration directory.
    
    Parameters
    ----------
    prun_datasets_dir : str
        Pruned data sets directory.
    train_dataset : torch.utils.data.Dataset
        Time series training data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    val_dataset : torch.utils.data.Dataset
        Time series validation data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    unused_dataset : torch.utils.data.Dataset
        Time series unused data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    test_dataset_dirs : dict, default={}
        Testing data set directory (item, str) for each testing type
        (key, str).
    dataset_basename : str, defaut='ss_paths_dataset'
        Data set file base name.
        
    Returns
    -------
    pruning_iter_dir : str
        Pruning iteration directory.
    model_directory : str
        Directory where model is stored.
    train_dataset_file_path : str
        Training data set file path.
    val_dataset_file_path : str
        Validation data set file path.
    test_dataset_file_paths : dict
        Testing data set file path (item, str) for each testing type
        (key, str).
    test_prediction_dirs : dict
        Directory (item, str) where samples predictions results files are
        stored for each testing type (key, str).
    """
    # Set pruning iteration directory
    pruning_iter_dir = os.path.join(os.path.normpath(prun_datasets_dir),
                                    f'n{len(train_dataset)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create pruning iteration directory
    make_directory(pruning_iter_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning training data set directory
    train_dataset_dir = os.path.join(os.path.normpath(pruning_iter_dir),
                                     '1_training_dataset') 
    # Create pruning training data set directory
    make_directory(train_dataset_dir, is_overwrite=True)
    # Save pruning training data set
    train_dataset_file_path = \
        save_dataset(train_dataset, dataset_basename, train_dataset_dir,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning validation data set directory
    val_dataset_dir = os.path.join(os.path.normpath(pruning_iter_dir),
                                   '2_validation_dataset') 
    # Create pruning validation data set directory
    make_directory(val_dataset_dir, is_overwrite=True)
    # Save pruning validation data set
    val_dataset_file_path = \
        save_dataset(val_dataset, dataset_basename, val_dataset_dir,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(os.path.normpath(pruning_iter_dir),
                                   '3_model')
    # Create model directory
    make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize testing data sets file paths
    test_dataset_file_paths = {}
    # Loop over testing data set directories
    for testing_type, src_dataset_dir in test_dataset_dirs.items():
        # Set testing data set file
        if testing_type in ('in_distribution', 'out_distribution'):
            # Build pruning testing data set directory
            dest_dataset_dir = os.path.join(os.path.normpath(pruning_iter_dir),
                                            os.path.basename(src_dataset_dir))
            # Copy pruning testing data set directory
            shutil.copytree(src_dataset_dir, dest_dataset_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get testing data set file path
            regex = (r'^' + dataset_basename + r'_n[0-9]+.pkl$',)
            is_file_found, test_dataset_file_path = \
                find_unique_file_with_regex(dest_dataset_dir, regex)
            # Check testing data set file
            if not is_file_found:
                raise RuntimeError(f'Testing data set file has not been found '
                                   f'in data set directory:\n\n'
                                   f'{dest_dataset_dir}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove all content except testing data set file
            for name in os.listdir(dest_dataset_dir):
                # Get item path
                item_path = os.path.join(dest_dataset_dir, name)
                # Remove item
                if item_path != test_dataset_file_path:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif testing_type == 'unused_data':
            # Check unused data set
            if unused_dataset is None:
                raise RuntimeError('The unused data set must be provided to '
                                   'set the respective pruning iteration '
                                   'testing data set directory.')
            # Set pruning testing data set directory
            test_dataset_dir = os.path.join(os.path.normpath(pruning_iter_dir),
                                            '8_testing_unused_dataset') 
            # Create pruning testing data set directory
            make_directory(test_dataset_dir, is_overwrite=True)
            # Save pruning testing data set
            test_dataset_file_path = \
                save_dataset(unused_dataset, dataset_basename,
                             test_dataset_dir, is_append_n_sample=True)
        else:
            raise RuntimeError(f'Unknown handling of \'{testing_type}\' '
                               f'testing data set directory.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store testing data set file path
        test_dataset_file_paths[testing_type] = test_dataset_file_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning predictions directory
    prediction_dir = os.path.join(os.path.normpath(pruning_iter_dir),
                                  '7_prediction')
    # Create pruning predictions directory
    make_directory(prediction_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize testing prediction directories
    test_prediction_dirs = {}
    # Loop over testing types
    for testing_type in test_dataset_dirs.keys():
        # Set testing predictions subdirectory
        prediction_subdir = os.path.join(
            os.path.normpath(prediction_dir), testing_type)
        # Create prediction subdirectory
        make_directory(prediction_subdir, is_overwrite=True)
        # Store testing predictions subdirectory
        test_prediction_dirs[testing_type] = prediction_subdir  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return pruning_iter_dir, model_directory, train_dataset_file_path, \
        val_dataset_file_path, test_dataset_file_paths, test_prediction_dirs
# =============================================================================
def set_default_pruning_parameters():
    """Set default pruning parameters.
    
    Returns
    -------
    pruning_params : dict
        Pruning parameters.
    """
    # Set pruning scheduler
    prun_scheduler_type = ('constant', 'proportional')[1]
    # Set pruning scheduler parameters
    if prun_scheduler_type == 'proportional':
        prun_scheduler_params = {'prun_ratio': 0.05}
    else:
        prun_scheduler_params = {'n_prun_sample': 10}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set maximum number of pruning iterations
    n_iter_max = 100
    # Set minimum development data set size
    n_dev_min = 10
    # Set minimum development data set size ratio
    ratio_dev_min = 0.05
    # Set minimum pruning testing data set size
    n_prun_test_min = 10
    # Set maximum pruned samples testing ratio
    prun_test_ratio_max = 0.5
    # Set pruning step data sets split sizes
    pruning_step_sizes = {'training': 0.7, 'validation': 0.2, 'testing': 0.1}
    # Set pruning iteration data sets split sizes
    pruning_iter_sizes = {'training': 0.8, 'validation': 0.2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build pruning parameters
    pruning_params = {'n_iter_max': n_iter_max,
                      'prun_scheduler_type': prun_scheduler_type,
                      'prun_scheduler_params': prun_scheduler_params,
                      'n_dev_min': n_dev_min,
                      'ratio_dev_min': ratio_dev_min,
                      'n_prun_test_min': n_prun_test_min,
                      'prun_test_ratio_max': prun_test_ratio_max,
                      'pruning_step_sizes': pruning_step_sizes,
                      'pruning_iter_sizes': pruning_iter_sizes}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return pruning_params
# =============================================================================
def get_n_prun_sample(prun_scheduler_type, prun_scheduler_params, dev_dataset):
    """Get number of pruned samples.
    
    Parameters
    ----------
    prun_scheduler_type : {'constant', 'proportional'}
        Pruning scheduler type defining the number of pruned samples in each
        pruning iteration. 'constant' sets a constant number of pruned samples,
        while 'proportional' sets the number of pruned samples as a ratio of
        the current development data set size.
    prun_scheduler_params : dict
        Pruning scheduler type parameters.
    dev_dataset : torch.utils.data.Dataset
        Time series development data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    
    Returns
    -------
    n_prun_sample : int
        Number of pruned samples.
    """
    # Get number of pruned samples
    if prun_scheduler_type == 'constant':
        # Get number of pruned samples
        n_prun_sample = prun_scheduler_params['n_prun_sample']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif prun_scheduler_type == 'proportional':
        # Get pruning ratio w.r.t. development data set size
        prun_ratio = prun_scheduler_params['prun_ratio']
        # Get development data set size
        n_dev = len(dev_dataset)
        # Get number of pruned samples
        n_prun_sample = int(prun_ratio*n_dev)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown pruning scheduler.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_prun_sample
# =============================================================================
def perform_pruning_step(prun_datasets_dir, pruning_params, dev_dataset,
                         unused_dataset, device_type='cpu', is_verbose=False):
    """Perform pruning step.
    
    Parameters
    ----------
    prun_datasets_dir : str
        Pruned data sets directory.
    pruning_params : dict
        Pruning parameters.
    dev_dataset : torch.utils.data.Dataset
        Time series development data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    unused_dataset : torch.utils.data.Dataset
        Time series unused data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    is_valid_pruning_step : bool
        If True, then pruning step is admissible and is performed. If False,
        then pruning step is non-admissible and data sets are unchanged.
    step_status : str
        Pruning step status.
    dev_dataset : torch.utils.data.Dataset
        Time series development data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    unused_dataset : torch.utils.data.Dataset
        Time series unused data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Get full data set size
    n_full = pruning_params['n_full']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get pruning scheduler type
    prun_scheduler_type = pruning_params['prun_scheduler_type']
    # Get pruning scheduler type parameters
    prun_scheduler_params = pruning_params['prun_scheduler_params']
    # Get number of pruned samples
    n_prun_sample = get_n_prun_sample(
        prun_scheduler_type, prun_scheduler_params, dev_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get minimum development data set size
    n_dev_min = pruning_params['n_dev_min']
    # Get minimum development data set size ratio
    ratio_dev_min = pruning_params['ratio_dev_min']
    # Get minimum pruning testing data set size
    n_prun_test_min = pruning_params['n_prun_test_min']
    # Get maximum pruned samples testing ratio
    prun_test_ratio_max = pruning_params['prun_test_ratio_max']
    # Get pruning step data set split sizes
    pruning_step_sizes = pruning_params['pruning_step_sizes']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get development data set size
    n_dev = len(dev_dataset)
    # Randomly split development data set
    dev_dataset_split = split_dataset(dev_dataset, pruning_step_sizes)
    # Get pruning step data sets
    train_dataset = dev_dataset_split['training']
    val_dataset = dev_dataset_split['validation']
    test_dataset = dev_dataset_split['testing']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize pruning step status
    step_status = None
    # Check pruning step admissibility
    if n_dev <= n_dev_min:
        step_status = 'Minimum development data set size reached.'
    elif n_dev/n_full < ratio_dev_min:
        step_status = 'Minimum pruning testing data set size ratio reached.'
    elif n_dev < n_prun_test_min:
        step_status = 'Minimum pruning testing data set size reached.'
    elif n_prun_sample/len(test_dataset) > prun_test_ratio_max:
        step_status = 'Maximum pruned samples testing ratio reached.'
    # Set pruning step admissibility
    if step_status is None:
        is_valid_pruning_step = True
    else:
        is_valid_pruning_step = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform pruning step
    if is_valid_pruning_step:
        # Set pruning step directory
        pruning_step_dir, model_directory, train_dataset_file_path, \
            val_dataset_file_path, test_dataset_file_path, \
                    prediction_subdir = set_pruning_step_dir(
                        prun_datasets_dir, train_dataset, val_dataset,
                        test_dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform pruning step model standard training
        perform_model_standard_training(
            train_dataset_file_path, model_directory,
            val_dataset_file_path=val_dataset_file_path,
            device_type=device_type, is_verbose=is_verbose)
        # Perform pruning step model prediction
        predict_subdir, _ = perform_model_prediction(
            prediction_subdir, test_dataset_file_path, model_directory,
            device_type=device_type, is_verbose=is_verbose)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect samples prediction loss
        _, samples_loss = read_samples_loss_from_dir(predict_subdir)
        # Get sorted indices based on samples prediction loss
        sorted_indices = sorted(range(len(samples_loss)),
                                key=lambda i: samples_loss[i])
        # Extract pruned samples testing indices
        prune_samples_test_indices = sorted_indices[:n_prun_sample]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get pruned samples development indices
        prune_samples_dev_indices = get_parent_dataset_indices(
            test_dataset, prune_samples_test_indices)
        # Get pruned samples data
        prune_samples = [dev_dataset[i] for i in prune_samples_dev_indices]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prune development data set
        dev_dataset.remove_dataset_samples(prune_samples_dev_indices)
        # Update unused data set
        unused_dataset.add_dataset_samples(prune_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove pruning step directory
        shutil.rmtree(pruning_step_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set step status
        step_status = 'Pruning step performed successfully.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_valid_pruning_step, step_status, dev_dataset, unused_dataset
# =============================================================================
def set_pruning_step_dir(prun_datasets_dir, train_dataset, val_dataset,
                         test_dataset, dataset_basename='ss_paths_dataset'):
    """Setup pruning iteration directory.
    
    Parameters
    ----------
    prun_datasets_dir : str
        Pruned data sets directory.
    train_dataset : torch.utils.data.Dataset
        Time series training data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    val_dataset : torch.utils.data.Dataset
        Time series validation data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    test_dataset : torch.utils.data.Dataset
        Time series testing data set. Each sample is stored as a dictionary
        where each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    dataset_basename : str, defaut='ss_paths_dataset'
        Data set file base name.
        
    Returns
    -------
    pruning_step_dir : str
        Pruning step directory.
    model_directory : str
        Directory where model is stored.
    train_dataset_file_path : str
        Training data set file path.
    val_dataset_file_path : str
        Validation data set file path.
    test_dataset_file_path : str
        Testing data set file path.
    prediction_subdir : str
        Directory where samples predictions results files are stored.
    """
    # Set pruning step directory
    pruning_step_dir = os.path.join(os.path.normpath(prun_datasets_dir),
                                    f'pruning_step')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create pruning step directory
    make_directory(pruning_step_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning training data set directory
    train_dataset_dir = os.path.join(os.path.normpath(pruning_step_dir),
                                     '1_training_dataset') 
    # Create pruning training data set directory
    make_directory(train_dataset_dir, is_overwrite=True)
    # Save pruning training data set
    train_dataset_file_path = \
        save_dataset(train_dataset, dataset_basename, train_dataset_dir,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning validation data set directory
    val_dataset_dir = os.path.join(os.path.normpath(pruning_step_dir),
                                   '2_validation_dataset') 
    # Create pruning validation data set directory
    make_directory(val_dataset_dir, is_overwrite=True)
    # Save pruning validation data set
    val_dataset_file_path = \
        save_dataset(val_dataset, dataset_basename, val_dataset_dir,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(os.path.normpath(pruning_step_dir),
                                   '3_model')
    # Create model directory
    make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning testing data set directory
    test_dataset_dir = os.path.join(os.path.normpath(pruning_step_dir),
                                    '5_testing_id_dataset') 
    # Create pruning testing data set directory
    make_directory(test_dataset_dir, is_overwrite=True)
    # Save pruning testing data set
    test_dataset_file_path = \
        save_dataset(test_dataset, dataset_basename, val_dataset_dir,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning predictions directory
    prediction_dir = os.path.join(os.path.normpath(pruning_step_dir),
                                        '7_prediction')
    # Create pruning predictions directory
    make_directory(prediction_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create pruning predictions subdirectory
    prediction_subdir = os.path.join(
        os.path.normpath(prediction_dir), 'in_distribution')
    # Create prediction subdirectory
    make_directory(prediction_subdir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return pruning_step_dir, model_directory, train_dataset_file_path, \
        val_dataset_file_path, test_dataset_file_path, prediction_subdir
# =============================================================================
def write_summary_file(pruning_dir, pruning_params, mode='init', mode_data={}):
    """Write summary file.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    pruning_params : dict
        Pruning parameters.
    mode : {'init', 'iter', 'end'}, default='init'
        Summary mode.
    mode_data : dict, default={}
        Summary mode data.
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
        n_full = pruning_params['n_full']
        summary += [f'\n> Full data set size: {n_full}\n',]
        # Assemble pruning parameters
        n_iter_max = pruning_params['n_iter_max']
        summary += [f'\n> Maximum number of pruning iterations: '
                    f'{n_iter_max}\n',]
        n_dev_min = pruning_params['n_dev_min']
        summary += [f'\n> Minimum development data set size: '
                    f'{n_dev_min}\n',]
        ratio_dev_min = pruning_params['ratio_dev_min']
        summary += [f'\n> Minimum development data set size ratio: '
                    f'{ratio_dev_min}\n',]
        n_prun_test_min = pruning_params['n_prun_test_min']
        summary += [f'\n> Minimum pruning testing data set size: '
                    f'{n_prun_test_min}\n',]
        prun_test_ratio_max = pruning_params['prun_test_ratio_max']
        summary += [f'\n> Maximum pruned samples testing ratio: '
                    f'{prun_test_ratio_max}\n',]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble pruning iterations header
        summary += [f'\n> Data set pruning iterations:\n',]
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
        ratio_dev_pc = (n_dev/n_full)*100
        n_train = mode_data['n_train']
        ratio_train_pc = (n_train/n_full)*100
        n_valid = mode_data['n_valid']
        ratio_valid_pc = (n_valid/n_full)*100
        n_unused = mode_data['n_unused']
        ratio_unused_pc = (n_unused/n_full)*100
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
                    f'{n_dev:>8d} {ratio_dev_pc:>6.1f} '
                    f'{n_train:>8d} {ratio_train_pc:>6.1f} '
                    f'{n_valid:>8d} {ratio_valid_pc:>6.1f} '
                    f'{n_unused:>9d} {ratio_unused_pc:>6.1f} ',
                    ' '.join(testing_losses),
                    '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write summary file
        open(summary_file_path, 'a').writelines(summary)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mode == 'end':
        # Get termination status
        termination_status = mode_data['termination_status']
        # Assemble termination status
        summary += [f'\n> Termination status: {termination_status}\n\n',]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write summary file
        open(summary_file_path, 'a').writelines(summary)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown summary mode.')
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
def read_samples_loss_from_dir(predictions_dir):
    """Read loss samples from prediction directory.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
        
    Returns
    -------
    samples_id : list[int]
        Samples IDs.
    samples_loss : list[float]
        Samples prediction loss.
    """
    # Check sample predictions directory
    if not os.path.isdir(predictions_dir):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize samples predictions files
    prediction_files = []
    prediction_files_id = []
    # Get samples prediction files
    for filename in directory_list:
        # Check if file is sample prediction file
        id = re.search(r'^prediction_sample_([0-9]+).pkl$', filename)
        # Store sample prediction file and ID
        if id is not None:
            # Store sample file path
            prediction_files.append(
                os.path.join(os.path.normpath(predictions_dir), filename))
            # Store sample ID
            prediction_files_id.append(int(id.groups()[0]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check prediction files
    if not prediction_files:
        raise RuntimeError('No sample results files have been found in '
                           'directory where samples predictions results files '
                           'are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get sorted indices based on samples prediction files
    sorted_indices = sorted(
        range(len(prediction_files)),
        key=lambda i: int(re.search(r'(\d+)\D*$',
                                    prediction_files[i]).groups()[-1]))
    # Sort samples prediction files
    prediction_files = [prediction_files[i] for i in sorted_indices]
    prediction_files_id = [prediction_files_id[i] for i in sorted_indices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set samples IDs
    samples_id = prediction_files_id
    # Initialize samples prediction loss
    samples_loss = []
    # Loop over samples prediction files
    for sample_file_path in prediction_files:
        # Load sample predictions results
        with open(sample_file_path, 'rb') as sample_file:
            sample_results = pickle.load(sample_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get sample prediction loss
        if 'prediction_loss_data' in sample_results.keys():
            # Get sample prediction loss data
            prediction_loss_data = sample_results['prediction_loss_data']
            # Gather sample prediction loss
            loss = prediction_loss_data[2]
            # Store sample prediction loss
            samples_loss.append(loss)
        else:
            raise RuntimeError(f'Sample prediction loss data is not available '
                               f'in sample prediction file:\n\n'
                               f'{sample_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return samples_id, samples_loss
# =============================================================================
def plot_pruning_iterative_data(
        pruning_dir, pruning_params, testing_types, pruning_iterative_data,
        save_dir=None, is_save_fig=False, is_stdout_display=False,
        is_latex=True):
    """Plot pruning iterative data.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    pruning_params : dict
        Pruning parameters.
    testing_types : tuple[str]
        Types of testing data sets used to assess the performance of the model
        trained on the pruned training data sets.
    pruning_iterative_data : dict
        Pruning iterative data (item, dict) for each pruning iteration
        (key, str).
    save_dir : str, default=None
        Directory where data set plots are saved. If None, then plots are
        saved in current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Get full data set size
    n_full = pruning_params['n_full']
    # Get number of pruning iterations
    n_iter = len(pruning_iterative_data.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.zeros((n_iter, 8))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over pruning iterations
    for iter in range(n_iter):
        # Get pruning iteration data
        iter_data = pruning_iterative_data[str(iter)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get iteration development data set size
        n_dev = iter_data['n_dev']
        ratio_dev_pc = (n_dev/n_full)*100
        # Compute training data set size
        n_train = iter_data['n_train']
        ratio_train_pc = (n_train/n_full)*100
        # Compute validation data set size
        n_valid = iter_data['n_valid']
        ratio_valid_pc = (n_valid/n_full)*100
        # Compute unused data set size
        n_unused = iter_data['n_unused']
        ratio_unused_pc = (n_unused/n_full)*100
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble iteration data
        data_xy[iter, :] = \
            (iter, ratio_dev_pc, iter, ratio_train_pc, iter, ratio_valid_pc,
             iter, ratio_unused_pc)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        data_labels = ('Development', 'Training', 'Validation', 'Unused')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Pruning iterations'
        y_label = '\% of full data set'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot data
        figure, _ = plot_xy_data(
            data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
            marker='o', is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set filename
        filename = f'pruning_iterations_dataset_sizes'
        # Save figure
        if is_save_fig:
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy_all = np.full((n_iter, 0), fill_value=None)
    # Loop over testing types
    for testing_type in testing_types:
        # Initialize data array
        data_xy = np.full((n_iter, 2), fill_value=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over pruning iterations
        for iter in range(n_iter):
            # Get pruning iteration data
            iter_data = pruning_iterative_data[str(iter)]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get iteration development data set size
            n_dev = iter_data['n_dev']
            ratio_dev_pc = (n_dev/n_full)*100
            # Get iteration testing loss
            avg_predict_loss = iter_data['testing_types_loss'][testing_type]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble iteration data
            data_xy[iter, 0] = ratio_dev_pc
            data_xy[iter, 1] = avg_predict_loss
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set axes labels
            x_label = 'Development size (\% of full data set)'
            y_label = 'Avg. prediction loss'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot data
            figure, _ = plot_xy_data(
                data_xy, x_label=x_label, y_label=y_label, x_scale='linear',
                y_scale='linear', marker='o', is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set filename
            filename = f'pruning_testing_convergence_{testing_type}'
            # Save figure
            if is_save_fig:
                save_figure(figure, filename, format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display figure
            if is_stdout_display:
                plt.show()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Close plot
            plt.close('all')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store testing type data
        data_xy_all = np.concatenate((data_xy_all, data_xy), axis=1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels mapping
    data_labels_map = {'in_distribution': 'In-distribution',
                       'out_distribution': 'Out-of-distribution',
                       'unused_data': 'Unused'}
    # Set data labels
    data_labels = tuple([data_labels_map[x] for x in testing_types])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Development size (\% of full data set)'
    y_label = 'Avg. prediction loss'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, _ = plot_xy_data(
        data_xy_all, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_scale='linear', y_scale='linear', marker='o', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set filename
    filename = f'pruning_testing_convergence'
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def preview_pruning_iterations(pruning_dir, pruning_params=None):
    """Preview pruning iterations.
    
    Parameters
    ----------
    pruning_dir : str
        Pruning main directory.
    pruning_params : dict, default=None
        Pruning parameters. If None, then a default set of pruning parameters
        is adopted.
    """
    print('\nPreview - Time series data set pruning'
          '\n--------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set full data set directory
    full_dataset_dir = os.path.join(os.path.normpath(pruning_dir),
                                    'base_datasets', '1_training_dataset')
    # Check full data set directory
    if not os.path.isdir(full_dataset_dir):
        raise RuntimeError('The full data set directory has not been found:'
                           '\n\n' + full_dataset_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load full data set
    full_dataset = load_full_dataset(full_dataset_dir)
    # Get full data set size
    n_full = len(full_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get default pruning parameters
    if not isinstance(pruning_params, dict):
        pruning_params = set_default_pruning_parameters()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get pruning scheduler type
    prun_scheduler_type = pruning_params['prun_scheduler_type']
    # Get pruning scheduler type parameters
    prun_scheduler_params = pruning_params['prun_scheduler_params']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get maximum number of pruning iterations
    n_iter_max = pruning_params['n_iter_max']
    # Get pruning iteration data set split sizes
    pruning_iter_sizes = pruning_params['pruning_iter_sizes']
    # Get minimum development data set size
    n_dev_min = pruning_params['n_dev_min']
    # Get minimum development data set size ratio
    ratio_dev_min = pruning_params['ratio_dev_min']
    # Get minimum pruning testing data set size
    n_prun_test_min = pruning_params['n_prun_test_min']
    # Get maximum pruned samples testing ratio
    prun_test_ratio_max = pruning_params['prun_test_ratio_max']
    # Get pruning step data set split sizes
    pruning_step_sizes = pruning_params['pruning_step_sizes']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display
    print(f'\n> Full data set size: {n_full}')
    print(f'\n> Maximum number of pruning iterations: {n_iter_max}')
    print(f'\n> Minimum development data set size: {n_dev_min}')
    print(f'\n> Minimum development data set size ratio: {ratio_dev_min}')
    print(f'\n> Minimum pruning testing data set size: {n_prun_test_min}')
    print(f'\n> Maximum pruned samples testing ratio: {prun_test_ratio_max}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display
    print(f'\n> Data set pruning iterations:')
    size_sep = 59*'-'
    print(f'\n{"Data sets":>44s}',
          f'\n{size_sep:>71s}',
          f'\n{"iter":>6s} '
          f'{"dev":>8s} {"%full":>6s} '
          f'{"train":>8s} {"%full":>6s} '
          f'{"valid":>8s} {"%full":>6s} '
          f'{"unused":>9s} {"%full":>6s}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize pruning iteration counter
    iter = 0
    # Initialize pruning termination flag
    is_keep_pruning = True
    # Loop over iterations
    while is_keep_pruning:
        # Compute development data set size
        if iter == 0:
            n_dev = n_full
            ratio_dev_pc = (n_dev/n_full)*100
            n_unused = 0
            ratio_unused_pc = (n_unused/n_full)*100
        else:
            # Create dummy development dataset
            dev_dataset = [x for x in range(n_dev)]
            # Get number of pruned samples
            n_prun_sample = get_n_prun_sample(
                prun_scheduler_type, prun_scheduler_params, dev_dataset)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute pruning training data set size
            n_prun_train = \
                int(np.floor(pruning_step_sizes['training']*n_dev))
            # Compute pruning validation data set size
            n_prun_valid = \
                int(np.floor(pruning_step_sizes['validation']*n_dev))
            # Compute pruning testing data set size
            n_prun_test = n_dev - (n_prun_train + n_prun_valid)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display pruning step
            if iter > 0:
                print(17*' ' + f'(pruning step: '
                      f'n_prun = {n_prun_sample}, '
                      f'T{n_prun_train}|V{n_prun_valid}|T{n_prun_test})')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize pruning step status
            step_status = None
            # Check pruning step admissibility
            if n_dev <= n_dev_min:
                step_status = 'Minimum development data set size reached.'
            elif n_dev/n_full <= ratio_dev_min:
                step_status = \
                    'Minimum development data set size ratio reached.'
            elif n_prun_test < n_prun_test_min:
                step_status = 'Minimum pruning testing data set size reached.'
            elif n_prun_sample/n_prun_test > prun_test_ratio_max:
                step_status = 'Maximum pruned samples testing ratio reached.'
            # Set pruning step admissibility
            if step_status is None:
                is_valid_pruning_step = True
            else:
                is_valid_pruning_step = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check pruning step admissibility
            if not is_valid_pruning_step:
                # Set termination status
                termination_status = step_status
                # Terminate pruning
                is_keep_pruning = False
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update development data set size
            n_dev = n_dev - n_prun_sample
            ratio_dev_pc = (n_dev/n_full)*100
            # Update unused data set size
            n_unused += n_prun_sample
            ratio_unused_pc = (n_unused/n_full)*100
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute training data set size
        n_train = int(np.floor(pruning_iter_sizes['training']*n_dev))
        ratio_train_pc = (n_train/n_full)*100
        # Compute validation data set size
        n_valid = n_dev - n_train
        ratio_valid_pc = (n_valid/n_full)*100
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display pruning iteration
        print(f'{iter:>6d} '
              f'{n_dev:>8d} {ratio_dev_pc:>6.1f} '
              f'{n_train:>8d} {ratio_train_pc:>6.1f} '
              f'{n_valid:>8d} {ratio_valid_pc:>6.1f} '
              f'{n_unused:>9d} {ratio_unused_pc:>6.1f} ')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check pruning iterations
        if iter >= n_iter_max:
            # Set termination status
            termination_status = \
                'Maximum number of pruning iterations reached.'
            # Terminate pruning
            is_keep_pruning = False
        else:
            # Update pruning iteration counter
            iter += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'\n> Termination status: {termination_status}\n')
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_pruning_preview = False
    is_dataset_pruning = True
    is_plot_pruning = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set pruning main directory
    pruning_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                   'test_pruning/polynomial')
    # Set types of testing data sets
    testing_types = ('in_distribution', 'unused_data')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pruning preview
    if is_pruning_preview:
        preview_pruning_iterations(pruning_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data set pruning
    if is_dataset_pruning:
        # Prune time series data set
        pruning_params, pruning_iterative_data = prune_time_series_dataset(
            pruning_dir, testing_types, is_remove_pruning_models=True,
            device_type=device_type, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pruning plot
    if is_plot_pruning:
        # Set pruning data file path
        pruning_data_file_path = os.path.join(os.path.normpath(pruning_dir),
                                              'pruning_data.pkl')
        # Load pruning data
        if os.path.isfile(pruning_data_file_path):
            # Load pruning data
            with open(pruning_data_file_path, 'rb') as pruning_file:
                pruning_data = pickle.load(pruning_file)
            # Collect pruning data
            pruning_params = pruning_data['pruning_params']
            pruning_iterative_data = pruning_data['pruning_iterative_data']
        else:
            raise RuntimeError(f'Pruning data file has not been found:\n\n'
                               f'{pruning_data_file_path}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set pruning plots directory
        prun_plots_dir = os.path.join(os.path.normpath(pruning_dir), 'plots')
        # Create pruning plots directory
        if not os.path.isdir(prun_plots_dir):
            make_directory(prun_plots_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot pruning iterative data
        plot_pruning_iterative_data(
            pruning_dir, pruning_params, testing_types, pruning_iterative_data,
            save_dir=prun_plots_dir, is_save_fig=True, is_stdout_display=False,
            is_latex=True)
    

    
    