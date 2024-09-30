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
import pickle
import shutil
# Third-party
import torch
import numpy as np
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory
from rnn_base_model.data.time_dataset import save_dataset
from projects.darpa_metals.rnn_material_model.user_scripts \
    .uncertainty_quantification import perform_model_uq, gen_model_uq_plots
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    convergence_plots import plot_prediction_loss_convergence_uq
from ioput.iostandard import make_directory, find_unique_file_with_regex
# =============================================================================
# Summary: Training GRU from Shunyu's local strain-stress data sets
# =============================================================================
def process_datasets_type(src_dir):
    """Convert source data sets to TimeSeriesDatasetInMemory.
    
    Parameters
    ----------
    src_dir : str
        Data sets source directory.
    """
    # Initialize source directory data sets directories
    src_dataset_dirs = []
    dataset_dir_names = []
    # Get files and directories in source directory
    src_dir_list = os.listdir(src_dir)
    # Loop over files and directories
    for dir_name in src_dir_list:
        # Check if data set directory
        is_dataset_dir= bool(re.search(r'^n[0-9]+$', dir_name))
        # Store data set directory
        if is_dataset_dir:
            src_dataset_dirs.append(
                os.path.join(os.path.normpath(src_dir), dir_name))
            dataset_dir_names.append(dir_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file basename
    dataset_basename = 'ss_paths_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets directories
    for dataset_dir in src_dataset_dirs:
        # Loop over training, validation and testing data sets
        for dataset_type in ('training', 'validation', 'testing_id',
                             'testing_od'):
            # Set data set type basename
            if dataset_type == 'training':
                dataset_type_basename = '1_training_dataset'
            elif dataset_type == 'validation':
                dataset_type_basename = '2_validation_dataset'
            elif dataset_type == 'testing_id':
                dataset_type_basename = '5_testing_id_dataset'
            elif dataset_type == 'testing_od':
                dataset_type_basename = '6_testing_od_dataset'
            else:
                raise RuntimeError('Unknown data set type.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set type directory
            dataset_type_dir = os.path.join(dataset_dir, dataset_type_basename)
            # Check data set type directory
            if not os.path.isdir(dataset_type_dir):
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get files and directories in data set directory
            dataset_dir_list = os.listdir(dataset_type_dir)
            # Loop over files and directories
            for filename in dataset_dir_list:
                # Check if data set file
                is_dataset_file= bool(re.search(r'^' + f'{dataset_basename}'
                                                + r'_n[0-9]+.pkl$', filename))
                # Set data set path
                if is_dataset_file:
                    dataset_path = os.path.join(dataset_type_dir, filename)
                    break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load data set
            with open(dataset_path, 'rb') as dataset_file:
                dataset = pickle.load(dataset_file)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           
            # Check data set type
            if isinstance(dataset, TimeSeriesDatasetInMemory):
                pass
            elif isinstance(dataset, list):
                # Save original data set (renamed)
                save_dataset(dataset, dataset_basename + '_list',
                             dataset_type_dir)
                # Convert data set to time series data set (in-memory)
                dataset = TimeSeriesDatasetInMemory(dataset)
                # Save data set (overwrite)
                save_dataset(dataset, dataset_basename, dataset_type_dir)
            else:
                raise RuntimeError('Invalid data set type stored in file:'
                                   '\n\n' + dataset_path)
# =============================================================================
def gru_training_and_predictions(src_dir):
    """Training and prediction of local GRU with uncertainty quantification.
    
    Parameters
    ----------
    src_dir : str
        Data sets source directory.
        
    Returns
    -------
    output_data : dict
        GRU performance output data.
    """
    # Set number of model samples for uncertainty quantification
    n_model_sample = 3
    # Set computation processes
    is_model_training = True
    # Set testing type
    testing_type = ('training', 'validation', 'in_distribution',
                    'out_distribution')[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize source directory data sets directories
    src_dataset_dirs = []
    dataset_dir_names = []
    # Get files and directories in source directory
    src_dir_list = os.listdir(src_dir)
    # Loop over files and directories
    for dir_name in src_dir_list:
        # Check if data set directory
        is_dataset_dir= bool(re.search(r'^n[0-9]+$', dir_name))
        # Store data set directory
        if is_dataset_dir:
            src_dataset_dirs.append(
                os.path.join(os.path.normpath(src_dir), dir_name))
            dataset_dir_names.append(dir_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort model samples directories
    src_dataset_dirs = \
        sorted(src_dataset_dirs,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
    dataset_dir_names = \
        sorted(dataset_dir_names,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize models directories
    training_dirs = []
    testing_dirs = []
    predictions_dirs = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize average prediction loss of each model
    models_avg_prediction_loss = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets directories
    for i, case_study_dir in enumerate(src_dataset_dirs):
        # Check case study directory
        if not os.path.isdir(case_study_dir):
            raise RuntimeError('The case study directory has not been found:'
                               '\n\n' + case_study_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '1_training_dataset')
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(training_dataset_dir, regex)
        # Check data set file
        if is_file_found:
            training_dirs.append(training_dataset_dir)
        else:
            raise RuntimeError(f'Training data set file has not been found '
                               f'in data set directory:\n\n'
                               f'{training_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = \
            os.path.join(os.path.normpath(case_study_dir), '3_model')
        # Create model directory (overwrite)
        make_directory(model_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set validation data set directory
        val_dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '2_validation_dataset')
        # Get validation data set file path
        val_dataset_file_path = None
        if os.path.isdir(val_dataset_directory):
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, val_dataset_file_path = \
                find_unique_file_with_regex(val_dataset_directory, regex)
            # Check data set file
            if not is_file_found:
                raise RuntimeError(f'Validation data set file has not been '
                                   f'found in data set directory:\n\n'
                                   f'{val_dataset_directory}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing data set directory
        if testing_type == 'training':
            # Set testing data set directory (training data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '1_training_dataset')
        elif testing_type == 'validation':
            # Set testing data set directory (validation data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '2_validation_dataset')
        elif testing_type == 'in_distribution':
            # Set testing data set directory (in-distribution testing data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '5_testing_id_dataset')
        elif testing_type == 'out_distribution':
            # Set testing data set directory (out-of-distribution testing
            # data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '6_testing_od_dataset')
        else:
            raise RuntimeError('Unknown testing type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get testing data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, test_dataset_file_path = \
            find_unique_file_with_regex(testing_dataset_dir, regex)
        # Check data set file
        if is_file_found:
            testing_dirs.append(testing_dataset_dir)
        else:
            raise RuntimeError(f'Testing data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{testing_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model predictions directory
        prediction_directory = os.path.join(os.path.normpath(case_study_dir),
                                            '7_prediction')
        # Create model predictions directory
        if not os.path.isdir(prediction_directory):
            make_directory(prediction_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create model predictions subdirectory
        prediction_subdir = os.path.join(
            os.path.normpath(prediction_directory), testing_type)
        # Create prediction subdirectory
        if not os.path.isdir(prediction_subdir):
            make_directory(prediction_subdir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set uncertainty quantification directory
        uq_directory = os.path.join(os.path.normpath(case_study_dir),
                                    'uncertainty_quantification')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform model uncertainty quantification
        perform_model_uq(uq_directory, n_model_sample, train_dataset_file_path,
                         model_directory, prediction_subdir,
                         test_dataset_file_path,
                         is_model_training=is_model_training,
                         val_dataset_file_path=val_dataset_file_path,
                         device_type=device_type, is_verbose=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate plots of model uncertainty quantification
        gen_model_uq_plots(uq_directory, n_model_sample, testing_dataset_dir,
                           testing_type, is_save_fig=True,
                           is_stdout_display=False, is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model samples average prediction loss
        samples_avg_prediction_loss = \
            get_prediction_loss_uq(uq_directory, n_model_sample,
                                   testing_dataset_dir, testing_type)
        # Store model samples average prediction loss
        models_avg_prediction_loss[dataset_dir_names[i]] = \
            samples_avg_prediction_loss
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set prediction data set directory
        prediction_dir = os.path.join(os.path.normpath(case_study_dir),
                                      f'7_prediction/{testing_type}/'
                                      'prediction_set_0')
        # Store prediction directory
        if os.path.isdir(prediction_dir):
            predictions_dirs.append(prediction_dir)
        else:
            raise RuntimeError('The prediction directory has not been '
                               'found:\n\n' + prediction_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove model directory
        if os.path.isdir(model_directory):
            shutil.rmtree(model_directory)
        # Remove model predictions directory
        if os.path.isdir(prediction_directory):
            shutil.rmtree(prediction_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence analysis plots directory
    if testing_type == 'in_distribution':
        plots_dir = os.path.join(os.path.normpath(src_dir),
                                 'plots_id_testing')
    elif testing_type == 'out_distribution':
        plots_dir = os.path.join(os.path.normpath(src_dir),
                                 'plots_od_testing')
    # Create convergence analysis plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot average prediction loss versus training data set size
    plot_prediction_loss_convergence_uq(
        src_dataset_dirs, training_dirs, predictions_dirs,
        filename='testing_loss_convergence_uq',
        save_dir=plots_dir, is_save_fig=True,
        is_stdout_display=False,
        is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set output data
    output_data = {}
    output_data['avg. prediction loss'] = models_avg_prediction_loss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return output_data
# =============================================================================
def get_prediction_loss_uq(uq_directory, n_model_sample, testing_dataset_dir,
                           testing_type):
    """Get model average prediction loss with uncertainty quantification.
    
    Parameters
    ----------
    uq_directory : str
        Directory where the uncertainty quantification models samples
        directories are stored.
    n_model_sample : int
        Number of model samples.
    testing_dataset_dir : str
        Directory where the testing data set is stored (common to all models
        samples).
    testing_type : {'training', 'validation', 'in_distribution',
                    'out_distribution'}
        Testing type according with the testing data set.

    Returns
    -------
    samples_avg_prediction_loss : list
        Average prediction loss of each model sample.
    """    
    # Initialize model samples average prediction loss
    samples_avg_prediction_loss = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check uncertainty quantification directory
    if not os.path.isdir(uq_directory):
        raise RuntimeError('The uncertainty quantification directory has not '
                           'been found:\n\n' + uq_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model samples directories
    model_sample_dirs = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files and directories in uncertainty quantification directory
    directory_list = os.listdir(uq_directory)
    # Loop over files and directories
    for dirname in directory_list:
        # Check if model sample directory
        is_sample_model= bool(re.search(r'^' + 'model' + r'_[0-9]+', dirname))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append model sample directory
        if is_sample_model:
            model_sample_dirs.append(
                os.path.join(os.path.normpath(uq_directory), dirname))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model samples directories
    if len(model_sample_dirs) < 1:
        raise RuntimeError('Model samples have not been found in the '
                           'uncertainty quantification directory:'
                           '\n\n' + uq_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort model samples directories
    model_sample_dirs = \
        sorted(model_sample_dirs,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model samples testing and predictions directories
    sample_testing_dirs = []
    sample_prediction_dirs = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model samples
    for i in range(n_model_sample):
        # Get model sample directory
        sample_dir = model_sample_dirs[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append model sample testing directory (shared)
        sample_testing_dirs.append(testing_dataset_dir)
        # Append model sample prediction directory
        sample_prediction_dirs.append(os.path.join(
            os.path.normpath(sample_dir), f'7_prediction/{testing_type}'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model samples
    for j in range(n_model_sample):
        # Get model sample prediction directory
        sample_pred_dir = sample_prediction_dirs[j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get prediction summary file
        regex = (r'^summary.dat$',)
        is_file_found, summary_file_path = \
            find_unique_file_with_regex(sample_pred_dir, regex) 
        # Check prediction summary file
        if not is_file_found:
            raise RuntimeError(f'Prediction summary file has not been '
                               f'found in directory:\n\n{sample_pred_dir}')
        # Open prediction summary file
        summary_file = open(summary_file_path, 'r')
        summary_file.seek(0)
        # Look for average prediction loss
        avg_predict_loss = None
        line_number = 0
        for line in summary_file:
            line_number = line_number + 1
            if 'Avg. prediction loss per sample' in line:
                avg_predict_loss = float(line.split()[-1])
                break
        # Store average prediction loss
        if avg_predict_loss is None:
            raise RuntimeError('Average prediction loss has not been '
                               'found in prediction summary file:\n\n'
                               f'{summary_file_path}')
        else:
            samples_avg_prediction_loss.append(avg_predict_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return samples_avg_prediction_loss
# =============================================================================
def write_output_file(src_dir, **kwargs):
    """Write output file with GRU performance output data.
    
    Parameters
    ----------
    src_dir : str
        Data sets source directory.
    kwargs: dict
        Keyword-based data to be written in output data file.
    """
    # Set output directory
    output_dir = os.path.join(os.path.normpath(src_dir), 'output_data')
    # Create output directory
    if not os.path.isdir(output_dir):
        make_directory(output_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set output file path
    output_file_path = os.path.join(output_dir, 'output_file.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open output data file
    output_file = open(output_file_path, 'w')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize output file content
    output_data = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over summary parameters
    for key1, val1 in kwargs.items():
        # Append parameter
        if isinstance(val1, dict):
            output_data.append(f'\n"{str(key1)}":\n')
            for key2, val2 in val1.items():
                if isinstance(val2, dict):
                    output_data.append(f'\n  "{str(key2)}":\n')
                    for key3, val3 in val2.items():
                        output_data.append(f'    "{str(key3)}": {str(val3)}\n')
                else:
                    output_data.append(f'  "{str(key2)}": {str(val2)}\n')
        elif isinstance(val1, np.ndarray):
            output_data.append(f'\n"{str(key1)}":\n  {str(val1)}\n')
        elif isinstance(val1, str) and '\n' in val1:
            output_data.append(f'\n"{str(key1)}":\n\n{str(val1)}\n')
        else:
            output_data.append(f'\n"{str(key1)}":  {str(val1)}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write output data file
    output_file.writelines(output_data)
    # Close output data file
    output_file.close()
# =============================================================================
def get_model_avg_prediction_loss(src_dir, n_model_sample, testing_type):
    """Get model average prediction loss.
    
    Parameters
    ----------
    src_dir : str
        Data sets source directory.
    n_model_sample : int
        Number of model samples.
    testing_type : {'training', 'validation', 'in_distribution',
                    'out_distribution'}
        Testing type according with the testing data set.
    """
    # Initialize source directory data sets directories
    src_dataset_dirs = []
    dataset_dir_names = []
    # Get files and directories in source directory
    src_dir_list = os.listdir(src_dir)
    # Loop over files and directories
    for dir_name in src_dir_list:
        # Check if data set directory
        is_dataset_dir= bool(re.search(r'^n[0-9]+$', dir_name))
        # Store data set directory
        if is_dataset_dir:
            src_dataset_dirs.append(
                os.path.join(os.path.normpath(src_dir), dir_name))
            dataset_dir_names.append(dir_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort model samples directories
    src_dataset_dirs = \
        sorted(src_dataset_dirs,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
    dataset_dir_names = \
        sorted(dataset_dir_names,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize average prediction loss of each model
    models_avg_prediction_loss = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets directories
    for i, case_study_dir in enumerate(src_dataset_dirs):
        # Check case study directory
        if not os.path.isdir(case_study_dir):
            raise RuntimeError('The case study directory has not been found:'
                               '\n\n' + case_study_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '1_training_dataset')
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, _ = \
            find_unique_file_with_regex(training_dataset_dir, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found '
                               f'in data set directory:\n\n'
                               f'{training_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing data set directory
        if testing_type == 'training':
            # Set testing data set directory (training data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '1_training_dataset')
        elif testing_type == 'validation':
            # Set testing data set directory (validation data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '2_validation_dataset')
        elif testing_type == 'in_distribution':
            # Set testing data set directory (in-distribution testing data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '5_testing_id_dataset')
        elif testing_type == 'out_distribution':
            # Set testing data set directory (out-of-distribution testing
            # data set)
            testing_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '6_testing_od_dataset')
        else:
            raise RuntimeError('Unknown testing type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get testing data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, _ = \
            find_unique_file_with_regex(testing_dataset_dir, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Testing data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{testing_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set uncertainty quantification directory
        uq_directory = os.path.join(os.path.normpath(case_study_dir),
                                    'uncertainty_quantification')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model samples average prediction loss
        samples_avg_prediction_loss = \
            get_prediction_loss_uq(uq_directory, n_model_sample,
                                   testing_dataset_dir, testing_type)
        # Store model samples average prediction loss
        models_avg_prediction_loss[dataset_dir_names[i]] = \
            samples_avg_prediction_loss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display data
    print('\nExtracting Model Avg. Prediction Loss',
          '\n-------------------------------------')
    print(f'\nSource directory: {src_dir}')
    print(f'\nNumber of model samples: {n_model_sample}')
    print(f'\nTesting type: {testing_type}')
    print('\nAvg. prediction loss:\n')
    for key, val in models_avg_prediction_loss.items():
        val_frmt = [f'{x:11.4e}' for x in val]
        print(f'  "{str(key):5s}":  {str(val_frmt)}')
    print('\nPlot data - Avg. prediction loss (mean):\n')
    x_data = [int(x.strip('n')) for x in models_avg_prediction_loss.keys()]
    y_data = [np.mean(x) for x in models_avg_prediction_loss.values()]
    print('  if all(x is None for x in data_labels):')
    print(f'      x_data = {x_data}')
    print(f'      y_data = {y_data}')
    print("      axes.plot(x_data, y_data, color='#EE6677', marker='v', "
          "markersize=3)")
    print('')
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_compute_gru_performance = True
    is_get_reference_loss = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute GRU performance data
    if is_compute_gru_performance:
        # Set data sets directory
        src_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                   'colaboration_shunyu/1_testing')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process data sets directory (convert to TimeSeriesDatasetInMemory)
        process_datasets_type(src_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GRU training and prediction (with uncertainty quantification)
        output_data = gru_training_and_predictions(src_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write GRU performance output data file
        write_output_file(src_dir, **output_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get reference model average prediction loss
    if is_get_reference_loss:
        # Set data sets directory
        src_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                   'colaboration_shunyu/0_base_random_datasets')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of model samples for uncertainty quantification
        n_model_sample = 5
        # Set testing type
        testing_type = ('training', 'validation', 'in_distribution',
                        'out_distribution')[2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get reference model average prediction loss
        get_model_avg_prediction_loss(src_dir, n_model_sample, testing_type)