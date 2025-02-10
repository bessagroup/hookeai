"""Plots for convergence analysis based on RNN-based models.

Functions
---------
plot_prediction_loss_convergence
    Plot average prediction loss against training data set size.
plot_best_parameters_convergence
    Plot best state parameters versus training data set size.
plot_time_series_convergence
    Plot time series predictions versus ground-truth.
plot_prediction_loss_convergence_uq
    Plot average prediction loss versus training data set size.
plot_best_parameters_convergence_uq
    Plot best state parameters versus training data set size.
plot_time_series_convergence_uq
    Plot time series predictions versus ground-truth.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
import pickle
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from time_series_data.time_dataset import load_dataset
from rnn_base_model.predict.prediction_plots import plot_time_series_prediction
from rc_base_model.train.training import read_best_parameters_from_file
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    process_predictions import build_prediction_data_arrays, \
        build_time_series_predictions_data
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
from ioput.plots import scatter_xy_data, plot_boxplots, save_figure
from ioput.iostandard import find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def plot_prediction_loss_convergence(models_base_dirs, training_dirs,
                                     predictions_dirs,
                                     filename='testing_loss_convergence',
                                     save_dir=None, is_save_fig=False,
                                     is_stdout_display=False, is_latex=True):
    """Plot average prediction loss versus training data set size.
    
    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    filename : str, default='testing_loss_convergence'
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
    # Get number of models
    n_models = len(models_base_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize average prediction losses per sample
    avg_predict_losses = []
    # Loop over prediction directories
    for pred_dir in predictions_dirs:
        # Get prediction summary file
        regex = (r'^summary.dat$',)
        is_file_found, summary_file_path = \
            find_unique_file_with_regex(pred_dir, regex) 
        # Check prediction summary file
        if not is_file_found:
            raise RuntimeError(f'Prediction summary file has not been found  '
                               f'in directory:\n\n'
                               f'{pred_dir}')
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
            raise RuntimeError('Average prediction loss has not been found in '
                               'prediction summary file:\n\n'
                               f'{summary_file_path}')
        else:
            avg_predict_losses.append(avg_predict_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_models, 2), fill_value=None)
    # Loop over models
    for i in range(n_models):
        # Assemble model training data set size and average prediction loss
        data_xy[i, 0] = training_sizes[i]
        data_xy[i, 1] = avg_predict_losses[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, _ = scatter_xy_data(
        data_xy, x_label=x_label, y_label=y_label, x_scale='log',
        y_scale='log', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
def plot_best_parameters_convergence(
    models_base_dirs, training_dirs, filename='best_parameters_convergence',
    save_dir=None, is_save_fig=False, is_stdout_display=False, is_latex=True):
    """Plot best state parameters versus training data set size.
    
    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    filename : str, default='best_parameters_convergence_uq'
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
    # Get number of models
    n_models = len(training_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize models parameters data
    models_parameters_data = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over models directories
    for i, model_base_dir in enumerate(models_base_dirs):
        # Set model directory
        model_dir = os.path.join(os.path.normpath(model_base_dir), '3_model')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model parameters data file path
        regex = (r'^parameters_best.pkl$',)
        is_file_found, parameters_file_path = \
            find_unique_file_with_regex(model_dir, regex)
        # Check model parameters data file
        if not is_file_found:
            raise RuntimeError(f'Model parameters data file has not '
                               f'been found in data set directory:\n\n'
                               f'{model_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model parameters
        best_model_parameters, _ = \
            read_best_parameters_from_file(parameters_file_path)
        # Store model parameters data
        models_parameters_data.append(best_model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get parameters names (shared between different models)
        if i == 0:
            # Get model parameters names
            model_parameters_names = tuple(best_model_parameters.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters
    for j, param in enumerate(model_parameters_names):
        # Initialize data array
        data_xy = np.full((n_models, 2), fill_value=None)
        # Loop over models
        for i in range(n_models):
            # Assemble model training data set size and parameter
            data_xy[i, 0] = training_sizes[i]
            data_xy[i, 1] = models_parameters_data[i][param]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Training data set size'
        y_label = param
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot data
        figure, _ = scatter_xy_data(
            data_xy, x_label=x_label, y_label=y_label, x_scale='log',
            is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, filename + f'_{param}', format='pdf',
                        save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
# =============================================================================
def plot_time_series_convergence(models_base_dirs, training_dirs, testing_dirs,
                                 predictions_dirs, prediction_types,
                                 plot_type='time_series_scatter',
                                 samples_ids='all',
                                 filename='time_series_convergence',
                                 save_dir=None, is_save_fig=False,
                                 is_stdout_display=False, is_latex=True):
    """Plot time series predictions versus ground-truth.
    
    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    testing_dirs : tuple[str]
        Directory where each model testing data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    plot_type : {'time_series_scatter', 'time_series_path'}, \
                default='time_series_scatter'
        Time series plot type.
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
    # Check time series plot type
    if plot_type not in ('time_series_scatter', 'time_series_path'):
        raise RuntimeError('Unknown time series plot type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of models
    n_models = len(models_base_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
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
            # Build times series predictions components data arrays
            if plot_type == 'time_series_scatter':
                # Build prediction components data arrays
                prediction_data_arrays = build_prediction_data_arrays(
                    pred_dir, prediction_type, prediction_labels,
                    samples_ids=samples_ids)
            elif plot_type == 'time_series_path':
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
            if plot_type == 'time_series_scatter':
                # Initialize prediction processes
                prediction_sets = {}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over models
                for i in range(n_models):
                    # Get model time series predictions data
                    model_prediction_data = models_prediction_data[i]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set prediction label
                    label = 'N = ' + f'{training_sizes[i]}'
                    # Assemble prediction data
                    prediction_sets[label] = model_prediction_data[j]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                 
                # Plot model times series predictions
                plot_truth_vs_prediction(
                    prediction_sets,
                    error_bound=0.1,
                    is_normalize_data=False,
                    filename=(filename + f'_{prediction_comp}_scatter'),
                    save_dir=save_dir, is_save_fig=is_save_fig,
                    is_stdout_display=is_stdout_display, is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif plot_type == 'time_series_path':
                # Set all available samples
                if samples_ids == 'all':
                    samples_ids = range(testing_size)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over samples
                for sample_id in samples_ids:
                    # Initialize prediction processes
                    prediction_sets = {}
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble discrete time path and ground-truth path by
                    # probing the first model data                    
                    prediction_sets['Ground-Truth'] = \
                        models_prediction_data[0][j][str(sample_id)][:, [0, 1]]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over models
                    for i in range(n_models):
                        # Get model time series predictions data
                        model_prediction_data = models_prediction_data[i]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Set prediction label
                        label = 'N = ' + f'{training_sizes[i]}'
                        # Assemble sample prediction data
                        prediction_sets[label] = model_prediction_data[j] \
                            [str(sample_id)][:, [0, 2]]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set prediction type label
                    if prediction_type == 'stress_comps':
                        y_label = 'Stress (MPa)'
                    elif prediction_type == 'acc_p_strain':
                        y_label = 'Accumulated plastic strain'
                    elif prediction_type == 'p_strain_comps':
                        y_label = 'Plastic strain'
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Plot model times series predictions
                    plot_time_series_prediction(
                        prediction_sets, is_reference_data=True,
                        x_label='Time', y_label=y_label,
                        is_normalize_data=False,
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
def plot_prediction_loss_convergence_uq(models_base_dirs, training_dirs,
                                        predictions_dirs,
                                        filename='testing_loss_convergence_uq',
                                        save_dir=None, is_save_fig=False,
                                        is_stdout_display=False,
                                        is_latex=True):
    """Plot average prediction loss versus training data set size.
    
    Uncertainty quantification data accounting for different model samples
    predictions for each training data set size is required. The corresponding
    directory named 'uncertainty_quantification' should exist in each model
    base directory.
    
    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    filename : str, default='testing_loss_convergence'
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
    # Get number of models
    n_models = len(models_base_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize average prediction losses per sample
    avg_predict_losses = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over models directories
    for i, model_base_dir in enumerate(models_base_dirs):
        # Extract testing type from prediction subdirectory
        testing_type = os.path.basename(os.path.dirname(predictions_dirs[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        model_sample_dirs = \
            sorted(model_sample_dirs,
                   key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model samples average prediction loss
        samples_avg_prediction_loss = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of model samples
        n_model_sample = len(model_sample_dirs)
        # Loop over model samples
        for j in range(n_model_sample):
            # Get model sample directory
            sample_dir = model_sample_dirs[j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model sample prediction directory
            sample_pred_dir = os.path.join(os.path.normpath(sample_dir),
                                           '7_prediction', testing_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model samples average prediction loss
        avg_predict_losses.append(samples_avg_prediction_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_models, 2*n_model_sample), fill_value=None)
    # Loop over models
    for i in range(n_models):
        # Assemble model training data set size and average prediction loss
        data_xy[i, 0::2] = n_model_sample*[training_sizes[i]]
        data_xy[i, 1::2] = avg_predict_losses[i][:]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (None, None)
    y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, _ = scatter_xy_data(
        data_xy, is_error_bar=True, range_type='min-max', x_lims=x_lims,
        y_lims=y_lims, x_label=x_label, y_label=y_label, x_scale='log',
        y_scale='log', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
def plot_best_parameters_convergence_uq(
    models_base_dirs, training_dirs, filename='best_parameters_convergence_uq',
    save_dir=None, is_save_fig=False, is_stdout_display=False, is_latex=True):
    """Plot best state parameters versus training data set size.
    
    Uncertainty quantification data accounting for different model samples
    for each training data set size is required. The corresponding directory
    named 'uncertainty_quantification' should exist in each model base
    directory.
    
    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    filename : str, default='best_parameters_convergence_uq'
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
    # Get number of models
    n_models = len(training_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize models parameters data
    models_parameters_data = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over models directories
    for i, model_base_dir in enumerate(models_base_dirs):
        # Set model uncertainty quantification directory
        uq_directory = os.path.join(os.path.normpath(model_base_dir),
                                    'uncertainty_quantification')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model samples parameters data file directory
        data_dir = os.path.join(uq_directory, 'plots', 'data')
        # Get model samples parameters data file path
        regex = (r'.*_best_parameters_data.pkl$',)
        is_file_found, parameters_file_path = \
            find_unique_file_with_regex(data_dir, regex)
        # Check model samples parameters data file
        if not is_file_found:
            raise RuntimeError(f'Model samples parameters data file has not '
                               f'been found in data set directory:\n\n'
                               f'{data_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model samples parameters
        with open(parameters_file_path, 'rb') as parameters_file:
            parameters_record = pickle.load(parameters_file)
        # Get model samples parameters data
        samples_parameters_data = parameters_record['samples_parameters_data']
        # Store model samples parameters data
        models_parameters_data.append(samples_parameters_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of model samples and parameters names (shared between
        # different models)
        if i == 0:
            # Get number of model samples
            n_model_sample = samples_parameters_data.shape[0]
            # Get model parameters names
            model_parameters_names = \
                parameters_record['model_parameters_names']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters
    for j, param in enumerate(model_parameters_names):
        # Initialize data array
        data_boxplots = np.zeros((n_model_sample, n_models))
        # Loop over models
        for k in range(n_models):
            # Assemble model samples parameter data
            data_boxplots[:, k] = models_parameters_data[k][:, j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        data_labels = training_sizes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Training data set size'
        y_label = param
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot model parameters box plot
        figure, _ = plot_boxplots(data_boxplots, data_labels, x_label=x_label,
                                  y_label=y_label, is_mean_line=True,
                                  is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, filename + f'_{param}', format='pdf',
                        save_dir=save_dir, is_tight_layout=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
# =============================================================================
def plot_time_series_convergence_uq(models_base_dirs, training_dirs,
                                    testing_dirs, predictions_dirs,
                                    prediction_types,
                                    plot_type='time_series_scatter',
                                    samples_ids='all',
                                    filename='time_series_convergence',
                                    save_dir=None, is_save_fig=False,
                                    is_stdout_display=False, is_latex=True):
    """Plot time series predictions versus ground-truth.

    Uncertainty quantification data accounting for different model samples
    predictions for each training data set size is required. The corresponding
    directory named 'uncertainty_quantification' should exist in each model
    base directory.

    Parameters
    ----------
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    testing_dirs : tuple[str]
        Directory where each model testing data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    plot_type : {'time_series_scatter', 'time_series_path'}, \
                default='time_series_scatter'
        Time series plot type.
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
    # Check time series plot type
    if plot_type not in ('time_series_scatter', 'time_series_path'):
        raise RuntimeError('Unknown time series plot type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of models
    n_models = len(models_base_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
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
            # Build times series predictions components data arrays
            if plot_type == 'time_series_scatter':
                # Build prediction components data arrays
                raise RuntimeError('Not implemented.')
            elif plot_type == 'time_series_path':
                # Build prediction components data arrays for each sample
                prediction_data_arrays = build_time_series_predictions_data(
                    test_dataset_file_path, pred_dir,
                    prediction_type, prediction_labels,
                    samples_ids=samples_ids,
                    is_uncertainty_quantification=True)
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
            if plot_type == 'time_series_scatter':
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif plot_type == 'time_series_path':
                # Set all available samples
                if samples_ids == 'all':
                    samples_ids = range(testing_size)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over samples
                for sample_id in samples_ids:
                    # Initialize prediction processes
                    prediction_sets = {}
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble discrete time path and ground-truth path by
                    # probing the first model data                    
                    prediction_sets['Ground-Truth'] = \
                        models_prediction_data[0][j][str(sample_id)][:, [0, 1]]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over models
                    for i in range(n_models):
                        # Get model time series predictions data
                        model_prediction_data = models_prediction_data[i]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Set prediction label
                        label = 'N = ' + f'{training_sizes[i]}'
                        # Assemble sample prediction data (remove ground-truth)
                        prediction_sets[label] = np.delete(
                            model_prediction_data[j][str(sample_id)], 1,
                            axis=1)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set prediction type label
                    if prediction_type == 'stress_comps':
                        y_label = 'Stress (MPa)'
                    elif prediction_type == 'acc_p_strain':
                        y_label = 'Accumulated plastic strain'
                    elif prediction_type == 'p_strain_comps':
                        y_label = 'Plastic strain'
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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