"""DARPA METALS PROJECT: Uncertainty quantification of RNN material model.

Functions
---------
perform_model_uq
    Perform model uncertainty quantification.
gen_model_uq_plots
    Generate plots of model uncertainty quantification.
plot_prediction_loss_uq
    Plot model average prediction loss uncertainty quantification.
plot_time_series_uq
    Plot time series model uncertainty quantification.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import shutil
import re
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from time_series_data.time_dataset import load_dataset
from rnn_base_model.predict.prediction_plots import plot_time_series_prediction
from projects.darpa_metals.rnn_material_model.user_scripts.train_model import \
    perform_model_standard_training
from projects.darpa_metals.rnn_material_model.user_scripts.predict import \
    perform_model_prediction
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    process_predictions import build_time_series_predictions_data
from ioput.iostandard import make_directory, find_unique_file_with_regex
from ioput.plots import plot_boxplots, save_figure

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_model_uq(uq_directory, n_model_sample, train_dataset_file_path,
                     model_directory, predict_directory,
                     test_dataset_file_path, is_model_training=True,
                     val_dataset_file_path=None,
                     device_type='cpu', is_verbose=False):
    """Perform model uncertainty quantification.
    
    Parameters
    ----------
    uq_directory : str
        Model uncertainty quantification directory.
    n_model_sample : int
        Number of model samples.
    train_dataset_file_path : str
        Training data set file path.
    model_directory : str
        Directory where model is stored.
    predict_directory : str
        Directory where model predictions results are stored.
    test_dataset_file_path : str
        Testing data set file path.
    is_model_training : bool, default=True
        If True, then overwrite the whole uncertainty quantification directory
        and perform the training and prediction for each model sample.
        If False, then perform predictions for each existing model sample.
    val_dataset_file_path : str, default=None
        Validation data set file path.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """      
    # Create uncertainty quantification directory
    if is_model_training:
        # Overwrite existing directory if training model samples
        make_directory(uq_directory, is_overwrite=True)
    else:
        # Preserve existing directory if predicting with existing model samples
        if not os.path.isdir(uq_directory):
            make_directory(uq_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model samples
    for i in range(n_model_sample):
        # Set model sample directory
        sample_dir = os.path.join(os.path.normpath(uq_directory), f'model_{i}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model sample model directory
        sample_model_dir = \
            os.path.join(os.path.normpath(sample_dir), '3_model')
        # Set model sample prediction directory
        sample_prediction_dir = \
            os.path.join(os.path.normpath(sample_dir), '7_prediction')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform model sample training
        if is_model_training:
            # Create model sample directory (overwrite)
            make_directory(sample_dir, is_overwrite=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform model standard training
            perform_model_standard_training(
                train_dataset_file_path, model_directory,
                val_dataset_file_path=val_dataset_file_path,
                device_type=device_type, is_verbose=is_verbose)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save model sample model directory
            shutil.copytree(model_directory, sample_model_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check model sample model directory
        if os.path.isdir(sample_model_dir):
            is_model_prediction = True
        else:
            is_model_prediction = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform model sample prediction
        if is_model_prediction:
            # Create model sample prediction directory
            if not os.path.isdir(sample_prediction_dir):
                make_directory(sample_prediction_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create model prediction subdirectory (overwrite)
            make_directory(predict_directory, is_overwrite=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform model prediction
            predict_subdir, _ = perform_model_prediction(
                predict_directory, test_dataset_file_path, sample_model_dir,
                device_type=device_type, is_verbose=is_verbose)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model sample prediction subdirectory
            sample_prediction_subdir = os.path.join(
                os.path.normpath(sample_prediction_dir),
                os.path.basename(os.path.normpath(predict_directory)))
            # Remove existing prediction subdirectory
            if os.path.isdir(sample_prediction_subdir):
                shutil.rmtree(sample_prediction_subdir)
            # Save model sample prediction
            shutil.copytree(predict_subdir, sample_prediction_subdir)
# =============================================================================
def gen_model_uq_plots(uq_directory, n_model_sample, testing_dataset_dir,
                       testing_type, filename='model_uq', is_save_fig=False,
                       is_stdout_display=False, is_latex=True):
    """Generate plots of model uncertainty quantification.
    
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
    filename : str, default='model_uq'
        Figure name.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check uncertainty quantification directory
    if not os.path.isdir(uq_directory):
        raise RuntimeError('The uncertainty quantification directory has not '
                           'been found:\n\n' + uq_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create uncertainty quantification plots directory
    plots_dir = os.path.join(os.path.normpath(uq_directory), 'plots')
    # Create uncertainty quantification plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # Create uncertainty quantification plots data subdirectory
    plot_data_dir = os.path.join(os.path.normpath(plots_dir), 'data')
    # Create uncertainty quantification data directory
    if not os.path.isdir(plot_data_dir):
        make_directory(plot_data_dir)
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
    # Set prediction types and corresponding labels
    prediction_types = {}
    prediction_types['stress_comps'] = ('stress_11', 'stress_22', 'stress_33',
                                        'stress_12', 'stress_23', 'stress_13')
    #prediction_types['acc_p_strain'] = ('acc_p_strain',)
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
    # Plot model average prediction loss uncertainty quantification
    plot_prediction_loss_uq(model_sample_dirs, sample_prediction_dirs,
                            filename=filename + '_testing_loss',
                            save_dir=plots_dir, is_save_fig=is_save_fig,
                            is_stdout_display=is_stdout_display,
                            is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set samples for which time series data is plotted
    samples_ids = list(np.arange(5, dtype=int))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot time series model uncertainty quantification
    plot_time_series_uq(model_sample_dirs, sample_testing_dirs,
                        sample_prediction_dirs, prediction_types,
                        samples_ids=samples_ids,
                        filename=filename + '_prediction',
                        save_dir=plots_dir, is_save_fig=is_save_fig,
                        is_stdout_display=is_stdout_display,
                        is_latex=is_latex)
# =============================================================================
def plot_prediction_loss_uq(model_sample_dirs, predictions_dirs,
                            filename='testing_loss_uq',
                            save_dir=None, is_save_fig=False,
                            is_stdout_display=False, is_latex=True):
    """Plot model average prediction loss uncertainty quantification.
    
    Parameters
    ----------
    model_sample_dirs : tuple[str]
        Directory of each model sample.
    predictions_dirs : tuple[str]
        Directory where each model sample predictions results files are
        stored.
    filename : str, default='testing_loss_uq'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Initialize model samples average prediction loss
    samples_avg_prediction_loss = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of model samples
    n_model_sample = len(model_sample_dirs)
    # Loop over model samples
    for j in range(n_model_sample):
        # Get model sample prediction directory
        sample_pred_dir = predictions_dirs[j]
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
    # Build model samples average prediction loss data
    samples_loss_data = np.array(samples_avg_prediction_loss).reshape(-1, 1)
    # Set data labels
    data_labels = (f'Number of realizations: {n_model_sample}',)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model samples average prediction loss box plot
    figure, _ = plot_boxplots(samples_loss_data,
                              data_labels=data_labels,
                              y_label='Avg. prediction loss',
                              is_mean_line=True, is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir,
                    is_tight_layout=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def plot_time_series_uq(model_sample_dirs, testing_dirs, predictions_dirs,
                        prediction_types, samples_ids='all',
                        filename='time_series_uq',
                        save_dir=None, is_save_fig=False,
                        is_stdout_display=False, is_latex=True):
    """Plot time series model uncertainty quantification.
    
    Parameters
    ----------
    model_sample_dirs : tuple[str]
        Directory of each model sample.
    testing_dirs : tuple[str]
        Directory where each model testing data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model sample predictions results files are
        stored.
    samples_ids : {'all', list[int]}, default='all'
        Samples IDs whose prediction data is plotted.
    filename : str, default='prediction_path_convergence'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Get number of model samples
    n_model_sample = len(model_sample_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prediction types
    for prediction_type, prediction_labels in prediction_types.items():
        # Initialize models time series predictions data
        models_prediction_data = []
        # Initialize testing data set sizes
        testing_sizes = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        # Loop over testing directories
        for i, test_dir in enumerate(testing_dirs):
            # Get testing data set file path
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, test_dataset_file_path = \
                find_unique_file_with_regex(test_dir, regex)
            # Check testing data set file
            if not is_file_found:
                raise RuntimeError(f'Testing data set file has not been found '
                                   f'in data set directory:\n\n'
                                   f'{test_dir}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get testing data set size
            testing_sizes.append(len(load_dataset(test_dataset_file_path)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get corresponding prediction directory
            pred_dir = predictions_dirs[i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build prediction components data arrays for each sample
            prediction_data_arrays = build_time_series_predictions_data(
                test_dataset_file_path, pred_dir,
                prediction_type, prediction_labels,
                samples_ids=samples_ids)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store model time series prediction components data arrays
            models_prediction_data.append(prediction_data_arrays)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check testing data sets
        if len(set(testing_sizes)) != 1:
            raise RuntimeError('Expecting the same testing data set but got '
                               'testing data sets with different sizes.')
        # Get testing data set size
        testing_size = testing_sizes[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction type components
        for j, prediction_comp in enumerate(prediction_labels):
            # Plot model times series prediction component
            # Set all available samples
            if samples_ids == 'all':
                samples_ids = range(testing_size)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over samples
            for sample_id in samples_ids:
                # Initialize prediction processes
                prediction_sets = {}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble discrete time path and ground-truth path by
                # probing the first model sample data                    
                prediction_sets['Ground-Truth'] = \
                    models_prediction_data[0][j][str(sample_id)][:, [0, 1]]
                # Get time series and corresponding sequence length
                time_hist = prediction_sets['Ground-Truth'][:, 0]
                n_time = len(time_hist)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize model samples prediction data
                prediction_sets['Realizations'] = \
                    np.zeros((n_time, 1 + n_model_sample))
                # Assemble time series time (shared)
                prediction_sets['Realizations'][:, 0] = time_hist
                # Loop over model samples
                for i in range(n_model_sample):
                    # Get model sample time series predictions data
                    model_prediction_data = models_prediction_data[i]
                    # Check model sample time series time
                    sample_time_hist = \
                        model_prediction_data[j][str(sample_id)][:, 0]
                    if not np.allclose(sample_time_hist, time_hist):
                        raise RuntimeWarning(f'Model sample {i} predictions '
                                             'does not share the same time '
                                             'series time as the '
                                             'ground-truth.')
                    # Assemble model sample predictions data
                    prediction_sets['Realizations'][:, i + 1] = \
                        model_prediction_data[j][str(sample_id)][:, 2]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set prediction type label
                if prediction_type == 'stress_comps':
                    y_label = 'Stress (MPa)'
                elif prediction_type == 'acc_p_strain':
                    y_label = 'Accumulated plastic strain'
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot model times series predictions
                plot_time_series_prediction(
                    prediction_sets, is_reference_data=True,
                    x_label='Time', y_label=y_label,
                    is_normalize_data=False,
                    is_uncertainty_quantification=True,
                    range_type='min-max',
                    filename=(filename + f'_{prediction_comp}'
                              + f'_path_sample_{sample_id}'),
                    save_dir=save_dir, is_save_fig=is_save_fig,
                    is_stdout_display=is_stdout_display, is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set number of model samples for uncertainty quantification
    n_model_sample = 3
    # Set computation processes
    is_model_training = True
    is_convergence_analysis = False
    # Set testing type
    testing_type = ('training', 'validation', 'in_distribution',
                    'out_distribution')[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'colaboration_shunyu/0_base_random_datasets_mre')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize case study directories
    case_study_dirs = []
    # Set case study directories
    if is_convergence_analysis:
        # Set training data set sizes
        training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        # Set case study directories
        case_study_dirs += [os.path.join(os.path.normpath(base_dir), f'n{n}/')
                            for n in training_sizes]
    else:
        case_study_dirs += [base_dir,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over case study directories
    for case_study_dir in case_study_dirs:
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
        if not is_file_found:
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
            # Set testing data set directory (out-of-distribution testing data
            # set)
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
        if not is_file_found:
            raise RuntimeError(f'Testing data set file has not been found '
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
        # Remove model directory
        if os.path.isdir(model_directory):
            shutil.rmtree(model_directory)
        # Remove model predictions directory
        if os.path.isdir(prediction_directory):
            shutil.rmtree(prediction_directory)