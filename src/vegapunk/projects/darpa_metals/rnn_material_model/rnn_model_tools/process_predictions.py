"""Post-process of RNN-based model predictions.

Functions
---------
build_prediction_data_arrays
    Build samples predictions data arrays with predictions and ground-truth.
build_time_series_predictions_data
    Build times series prediction and ground-truth data arrays.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
# Third-party
import numpy as np
# Local
from rnn_base_model.data.time_dataset import load_dataset
from gnn_base_model.predict.prediction import load_sample_predictions
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def build_prediction_data_arrays(predictions_dir, prediction_type,
                                 samples_ids='all'):
    """Build samples predictions data arrays with predictions and ground-truth.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    prediction_type : {'stress_comps', 'acc_p_strain'}
        Type of prediction data arrays:
        
        'stress_comps' : Stress components paths
        
        'acc_p_strain' : Accumulated plastic strain

    samples_ids : {'all', list[int]}, default='all'
        Samples IDs whose prediction results are collated in each prediction
        data array.
    
    Returns
    -------
    prediction_data_arrays : list[numpy.ndarray(2d)]
        Prediction components data arrays. Each data array collates data from
        all specified samples and is stored as a numpy.ndarray(2d) of shape
        (n_points, 2), where data_array[:, 0] stores the ground-truth
        and data_array[:, 1] stores the predictions.
    """
    # Check sample predictions directory
    if not os.path.isdir(predictions_dir):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # Check samples IDs
    if samples_ids != 'all' and not isinstance(samples_ids, list):
        raise RuntimeError('Samples IDs must be specified as "all" or as '
                           'list[int].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get prediction files samples IDs
    prediction_files_ids = []
    for filename in directory_list:
        # Check if file is sample results file
        id = re.search(r'^prediction_sample_([0-9]+).pkl$', filename)
        # Assemble sample ID
        if id is not None:
            prediction_files_ids.append(int(id.groups()[0]))
    # Check prediction files
    if not prediction_files_ids:
        raise RuntimeError('No sample results files have been found in '
                           'directory where samples predictions results files '
                           'are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set all available samples
    if samples_ids == 'all':
        samples_ids = prediction_files_ids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of stress components
    n_stress_comps = 6
    # Set number of prediction data arrays
    if prediction_type == 'stress_comps':  
        # Set number of prediction data arrays
        n_data_arrays = n_stress_comps
    elif prediction_type == 'acc_p_strain':   
        # Set number of prediction data arrays
        n_data_arrays = 1
    else:
        raise RuntimeError('Unknown prediction data array type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction data arrays
    prediction_data_arrays = n_data_arrays*[np.empty((0, 2)),]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample_id in samples_ids:
        # Check if sample ID prediction results file is available
        if sample_id not in prediction_files_ids:
            raise RuntimeError(f'The prediction results file for sample '
                               f'{sample_id} has not been found in directory: '
                               f'\n\n{predictions_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample predictions file path
        sample_prediction_path = \
            os.path.join(os.path.normpath(predictions_dir),
                         f'prediction_sample_{sample_id}.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = load_sample_predictions(sample_prediction_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction data arrays
        for i in range(n_data_arrays):
            # Build sample data array
            if prediction_type == 'stress_comps':
                # Get stress components predictions
                stress_path = \
                    sample_results['features_out'][:, :n_stress_comps]
                # Get stress components ground-truth
                stress_path_target = \
                    sample_results['targets'][:, :n_stress_comps]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check availability of ground-truth
                if stress_path_target is None:
                    raise RuntimeError(f'Stress components path ground-truth '
                                       f'is not available for sample '
                                       f'{sample_id}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build sample data array
                data_array = np.concatenate(
                    (stress_path_target[:, i].reshape((-1, 1)),
                     stress_path[:, i].reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif prediction_type == 'acc_p_strain':
                # Get accumulated plastic strain prediction
                acc_p_strain_path = \
                    sample_results['features_out'][:, n_stress_comps]
                # Get accumulated plastic strain ground-truth
                acc_p_strain_path_target = \
                    sample_results['targets'][:, n_stress_comps]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check availability of ground-truth
                if acc_p_strain_path_target is None:
                    raise RuntimeError(f'Accumulated plastic strain path '
                                       f'ground-truth is not available for '
                                       f'sample {sample_id}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build sample data array
                data_array = np.concatenate(
                    (acc_p_strain_path_target.reshape((-1, 1)),
                     acc_p_strain_path.reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample prediction data
            prediction_data_arrays[i] = \
                np.append(prediction_data_arrays[i], data_array, axis=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_data_arrays
# =============================================================================
def build_time_series_predictions_data(dataset_file_path, predictions_dir,
                                       prediction_type, samples_ids='all',
                                       is_uncertainty_quantification=False):
    """Build times series prediction and ground-truth data arrays.
    
    Parameters
    ----------
    dataset_file_path : str
        Time series testing data set file path.
    predictions_dir : str
        Directory where samples predictions results files are stored.
    prediction_type : {'stress_comps', 'acc_p_strain'}
        Type of prediction data arrays:
        
        'stress_comps' : Stress components paths
        
        'acc_p_strain' : Accumulated plastic strain

    samples_ids : {'all', list[int]}, default='all'
        Samples for which the data arrays with the time series prediction
        and ground-truth are built.
    is_uncertainty_quantification : bool, default=False
        If True, then build the prediction data arrays for each sample
        accounting for one or more model samples. Each model sample prediction
        directory is inferred from the provided prediction directory (assumed
        existing in base model directory). Uncertainty quantification data
        accounting for different model samples predictions is required.
    
    Returns
    -------
    prediction_data_arrays : list[dict]
        Prediction components data arrays for each sample. Each prediction
        component is stored as a dictionary, where the data array
        (item, np.ndarray(2d)) of each sample (key, str) is stored as a
        numpy.ndarray(2d) of shape (sequence_length, 2 + n_predictions), where
        data_array[:, 0] stores the time series discrete time, data_array[:, 1]
        stores the time series ground-truth and data_array[:, 2:] stores the
        time series predictions.
    """
    # Get model base directory
    model_base_dir = os.path.dirname(os.path.dirname(dataset_file_path))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model samples to process uncertainty quantification
    if is_uncertainty_quantification:
        # Set model uncertainty quantification directory
        uq_directory = os.path.join(os.path.normpath(model_base_dir),
                                    'uncertainty_quantification')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model samples directories
        model_sample_dirs = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get files and directories in uncertainty quantification directory
        directory_list = os.listdir(uq_directory)
        # Loop over files and directories
        for dirname in directory_list:
            # Check if model sample directory
            is_sample_model= \
                bool(re.search(r'^' + 'model' + r'_[0-9]+', dirname))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Append model sample directory
            if is_sample_model:
                model_sample_dirs.append(
                    os.path.join(os.path.normpath(uq_directory), dirname))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort model samples directories
        model_sample_dirs = sorted(model_sample_dirs)
        # Get number of model samples
        n_model_sample = len(model_sample_dirs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract testing type from prediction subdirectory
        testing_type = os.path.basename(os.path.dirname(predictions_dir))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load testing data set
    test_dataset = load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check sample predictions directory
    if (not is_uncertainty_quantification
            and not os.path.isdir(predictions_dir)):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # Check samples IDs
    if samples_ids != 'all' and not isinstance(samples_ids, list):
        raise RuntimeError('Samples IDs must be specified as "all" or as '
                           'list[int].')
    elif (isinstance(samples_ids, list)
          and max(samples_ids) >= len(test_dataset)):
        raise RuntimeError(f'Sample ID ({max(samples_ids)}) is outside of the '
                           f'data set of size {len(test_dataset)}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in samples predictions results directory
    if is_uncertainty_quantification:
        # Probe first model sample directory
        directory_list = os.listdir(
            os.path.join(os.path.normpath(model_sample_dirs[0]),
                         '7_prediction', testing_type))
    else:
        directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get prediction files samples IDs
    prediction_files_ids = []
    for filename in directory_list:
        # Check if file is sample results file
        file_id = re.search(r'^prediction_sample_([0-9]+).pkl$', filename)
        # Assemble sample ID
        if file_id is not None:
            prediction_files_ids.append(int(file_id.groups()[0]))
    # Check prediction files
    if not prediction_files_ids:
        raise RuntimeError('No sample results files have been found in '
                           'directory where samples predictions results files '
                           'are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set all available samples
    if samples_ids == 'all':
        samples_ids = prediction_files_ids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of stress components
    n_stress_comps = 6 
    # Set number of prediction components
    if prediction_type == 'stress_comps':   
        # Set number of prediction components
        n_pred_comps = n_stress_comps
    elif prediction_type == 'acc_p_strain':   
        # Set number of prediction components
        n_pred_comps = 1
    else:
        raise RuntimeError('Unknown prediction data array type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction components data
    prediction_data_arrays = [{} for _ in range(n_pred_comps)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample_id in samples_ids:
        # Get time series discrete time
        time_path = test_dataset[sample_id]['time_hist']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if sample prediction results file is available
        if sample_id not in prediction_files_ids:
            raise RuntimeError(f'The prediction results file for sample '
                               f'{sample_id} has not been found in directory: '
                               f'\n\n{predictions_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model samples prediction results
        models_sample_results = []
        # Get sample predictions
        if is_uncertainty_quantification:
            # Loop over model samples
            for i in range(n_model_sample):
                # Get model sample directory
                sample_dir = model_sample_dirs[i]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set model sample prediction directory
                sample_pred_dir = os.path.join(os.path.normpath(sample_dir),
                                               '7_prediction', testing_type)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set sample predictions file path
                sample_prediction_path = \
                    os.path.join(os.path.normpath(sample_pred_dir),
                                 f'prediction_sample_{sample_id}.pkl')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Load sample predictions
                sample_results = \
                    load_sample_predictions(sample_prediction_path)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble sample predictions
                models_sample_results.append(sample_results)
        else:
            # Set sample predictions file path
            sample_prediction_path = \
                os.path.join(os.path.normpath(predictions_dir),
                             f'prediction_sample_{sample_id}.pkl')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load sample predictions
            sample_results = load_sample_predictions(sample_prediction_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample predictions
            models_sample_results.append(sample_results)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction components
        for i in range(n_pred_comps):
            # Initialize sample data array
            data_array = time_path.reshape((-1, 1))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build sample data array
            if prediction_type == 'stress_comps':
                # Loop over sample predictions
                for j, sample_results in enumerate(models_sample_results):
                    # Get stress component predictions
                    stress_path = \
                        sample_results['features_out'][:, :n_stress_comps]
                    # Get stress components ground-truth
                    stress_path_target = \
                        sample_results['targets'][:, :n_stress_comps]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble sample ground-truth data
                    if j == 0:
                        if stress_path_target is None:
                            raise RuntimeError(
                                f'Stress component path ground-truth is not '
                                f'available for sample {sample_id}.')
                        else:
                            data_array = np.concatenate(
                                (data_array,
                                 stress_path_target[:, i].reshape((-1, 1))),
                                axis=1)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Concatenate sample prediction data
                    data_array = np.concatenate(
                        (data_array,
                         stress_path[:, i].reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif prediction_type == 'acc_p_strain':
                # Loop over sample predictions
                for j, sample_results in enumerate(models_sample_results):
                    # Get accumulated plastic strain prediction
                    acc_p_strain_path = \
                        sample_results['features_out'][:, n_stress_comps]
                    # Get accumulated plastic strain ground-truth
                    acc_p_strain_path_target = \
                        sample_results['targets'][:, n_stress_comps]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check availability of ground-truth
                    if j == 0:
                        if acc_p_strain_path_target is None:
                            raise RuntimeError(
                                f'Accumulated plastic strain path '
                                f'ground-truth is not available for sample '
                                f'{sample_id}.')
                        else:
                            data_array = np.concatenate(
                                (data_array,
                                 acc_p_strain_path_target.reshape((-1, 1))),
                                axis=1)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Build sample data array
                    data_array = np.concatenate(
                        (data_array,
                         acc_p_strain_path.reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble prediction component sample data
            prediction_data_arrays[i][str(sample_id)] = data_array
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_data_arrays