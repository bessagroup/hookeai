"""Plots for convergence analysis based on RNN-based models.

Functions
---------
plot_prediction_loss_convergence
    Plot average prediction loss against training data set size.
plot_time_series_convergence
    Plot multiple models time series predictions versus ground-truth.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from rnn_base_model.data.time_dataset import load_dataset
from rnn_base_model.predict.prediction_plots import plot_time_series_prediction
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    process_predictions import build_prediction_data_arrays, \
        build_time_series_predictions_data
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
from ioput.plots import scatter_xy_data, save_figure
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
def plot_prediction_loss_convergence(training_dirs, predictions_dirs,
                                     filename='testing_loss_convergence',
                                     save_dir=None, is_save_fig=False,
                                     is_stdout_display=False, is_latex=True):
    """Plot average prediction loss versus training data set size.
    
    Parameters
    ----------
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
    # Set axes limits and scale
    x_lims = (None, None)
    y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ground-truth versus predictions
    figure, _ = scatter_xy_data(
        data_xy, x_lims=x_lims, y_lims=y_lims, x_label=x_label,
        y_label=y_label, x_scale='log', y_scale='log', is_latex=is_latex)
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
def plot_time_series_convergence(training_dirs, testing_dirs,
                                 predictions_dirs, prediction_types,
                                 plot_type='time_series_scatter',
                                 samples_ids='all',
                                 filename='time_series_convergence',
                                 save_dir=None, is_save_fig=False,
                                 is_stdout_display=False, is_latex=True):
    """Plot multiple models time series predictions versus ground-truth.
    
    Parameters
    ----------
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
    # Loop over prediction types
    for prediction_type, prediction_comps in prediction_types.items():
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
                    pred_dir, prediction_type=prediction_type,
                    samples_ids=samples_ids)
            elif plot_type == 'time_series_path':
                # Build prediction components data arrays for each sample
                prediction_data_arrays = build_time_series_predictions_data(
                    test_dataset_file_path, pred_dir,
                    prediction_type=prediction_type,
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
        for j, prediction_comp in enumerate(prediction_comps):
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