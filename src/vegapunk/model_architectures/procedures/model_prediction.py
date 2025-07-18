"""Procedures associated to model prediction.

Functions
---------
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_predictions
    Save model prediction results for given sample.
load_sample_predictions
    Load model prediction results for given sample.
write_prediction_summary_file
    Write summary data file for model prediction process.
plot_time_series_prediction
    Plot time series predictions.
plot_truth_vs_prediction
    Plot ground-truth versus predictions.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import datetime
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.iostandard import make_directory, write_summary_file
from ioput.plots import plot_xy_data, plot_xny_data, scatter_xy_data, \
    save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def make_predictions_subdir(predict_directory):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.

    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_directory, 'prediction_set_0')
    while os.path.exists(predict_subdir):
        predict_subdir = os.path.join(
            predict_directory,
            'prediction_set_' + str(int(predict_subdir.split('_')[-1]) + 1))
    # Create model predictions subdirectory
    predict_subdir = make_directory(predict_subdir, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
# =============================================================================
def save_sample_predictions(predictions_dir, sample_id, sample_results):
    """Save model prediction results for given sample.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where sample prediction results are stored.
    sample_id : int
        Sample ID. Sample ID is appended to sample prediction results file
        name.
    sample_results : dict
        Sample prediction results.
    """
    # Check prediction results directory
    if not os.path.exists(predictions_dir):
        raise RuntimeError('The prediction results directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set sample prediction results file path
    sample_path = os.path.join(predictions_dir,
                               'prediction_sample_'+ str(sample_id) + '.pkl')
    # Save sample prediction results
    with open(sample_path, 'wb') as sample_file:
        pickle.dump(sample_results, sample_file)
# =============================================================================
def load_sample_predictions(sample_prediction_path):
    """Load model prediction results for given sample.
    
    Parameters
    ----------
    sample_prediction_path : str
        Sample prediction results file path.
        
    Returns
    -------
    sample_results : dict
        Sample prediction results.
    """
    # Check sample prediction results file
    if not os.path.isfile(sample_prediction_path):
        raise RuntimeError('Sample prediction results file has not been '
                           'found:\n\n' + sample_prediction_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sample prediction results
    with open(sample_prediction_path, 'rb') as sample_prediction_file:
        sample_results = pickle.load(sample_prediction_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_results
# =============================================================================
def write_prediction_summary_file(
    predict_subdir, device_type, seed, model_directory, model_load_state,
    loss_type, loss_kwargs, is_normalized_loss, dataset_file_path, dataset,
    avg_predict_loss, total_time_sec, avg_time_sample):
    """Write summary data file for model prediction process.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where model is stored.
    model_load_state : {'default', 'init', int, 'best', 'last'}
        Available model state to be loaded from the model directory.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    dataset_file_path : str
        Data set file path if such file exists. Only used for output purposes.
    dataset : torch.utils.data.Dataset
        Data set.
    avg_predict_loss : float
        Average prediction loss per sample.
    total_time_sec : int
        Total prediction time in seconds.
    avg_time_sample : float
        Average prediction time per sample.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['model_load_state'] = model_load_state
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['is_normalized_loss'] = is_normalized_loss
    summary_data['Prediction data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Prediction data set size'] = len(dataset)
    summary_data['Avg. prediction loss per sample'] = \
        f'{avg_predict_loss:.8e}' if avg_predict_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per sample'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=predict_subdir,
        summary_title='Summary: Model prediction',
        **summary_data)
# =============================================================================
def plot_time_series_prediction(prediction_sets, is_reference_data=False,
                                x_label='Time', y_label='Value',
                                is_normalize_data=False,
                                is_uncertainty_quantification=False,
                                range_type='min-max',
                                filename='time_series_prediction',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Plot time series predictions.
    
    Parameters
    ----------
    prediction_sets : dict
        One or more time series prediction processes, where each process
        (key, str) is stored as a data array (item, numpy.ndarray(2d)) of shape
        (sequence_length, 1 + n_predictions) as follows: data_array[:, 0]
        stores the time series time, data_array[:, 1:] stores the time series
        predictions.
    is_reference_data : bool, default=False
        If True, then the first prediction process is assumed to be the
        reference and is formatted independently (black, dashed, on top).
    x_label : str, default='Time'
        x-axis label.
    y_label : str, default='Value'
        y-axis label.
    is_normalize_data : bool, default=False
        Normalize time and predictions data to the range [0, 1].
    is_uncertainty_quantification: bool, default=False
        If True, then plot the prediction processes range of time series
        predictions for each time. Assumes the same time series time for all
        the different prediction processes.
    range_type : {'min-max', 'mean-std', None}, default='min-max'
        Type of range used to plot the range of time series predictions for
        each time. If None, only the mean is plotted. Only effective if
        is_uncertainty_quantification is set to True.
    filename : str, default='time_series_prediction'
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
    # Check time series predictions data
    if not isinstance(prediction_sets, dict):
        raise RuntimeError('Prediction processes are not provided as a dict.')
    elif not all([isinstance(x, np.ndarray)
                  for x in prediction_sets.values()]):
        raise RuntimeError('Prediction processes must be provided as a dict '
                           'where each process (key, str) is stored as a '
                           'numpy.ndarray of shape '
                           '(sequence_length, 1 + n_predictions).')
    elif not all([x.shape[1] >= 2 for x in prediction_sets.values()]):
        raise RuntimeError('Prediction processes must be provided as a dict '
                           'where each process (key, str) is stored as a '
                           'numpy.ndarray of shape '
                           '(sequence_length, 1 + n_predictions).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build prediction plot data
    if is_uncertainty_quantification:        
        # Initialize prediction data list and data labels
        data_xy_list = []
        data_labels = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction processes
        for i, (key, val) in enumerate(prediction_sets.items()):
            # Assemble prediction process data
            data_xy_list.append(val)
            # Assemble prediction process label
            data_labels.append(key)       
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize prediction data
        if is_normalize_data:
            raise RuntimeError('Not implemented.')
    else:
        # Get number of prediction processes
        n_processes = len(prediction_sets.keys())
        # Get maximum sequence length
        max_sequence_len = max([x.shape[0] for x in prediction_sets.values()])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data array and data labels
        data_xy = np.full((max_sequence_len, 2*n_processes), fill_value=None)
        data_labels = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction processes
        for i, (key, val) in enumerate(prediction_sets.items()):
            # Get prediction process sequence length
            n_time = val.shape[0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble prediction process data
            data_xy[:n_time, 2*i] = val[:, 0]
            data_xy[:n_time, 2*i + 1] = val[:, 1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble prediction process label
            data_labels.append(key)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize prediction data
        if is_normalize_data:
            # Get minimum and maximum prediction values
            min_pred = np.min(data_xy[:, 1::2])
            max_pred = np.max(data_xy[:, 1::2])
            # Normalize data array
            for i in range(n_processes):
                # Get prediction process sequence length
                n_time = val.shape[0]
                # Normalize time
                data_xy[:n_time, 2*i] = np.linspace(0, 1.0, num=n_time)
                # Normalize predictions
                data_xy[:n_time, 2*i+1] = \
                    (data_xy[:, 2*i+1] - min_pred)/(max_pred - min_pred)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    if is_normalize_data:
        x_lims = (0, 1)
        y_lims = (0, 1)
    else:
        x_lims = (None, None)
        y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    if is_normalize_data:
        x_label += ' (Normalized)'
        y_label += ' (Normalized)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot time series predictions
    if is_uncertainty_quantification:
        figure, _ = plot_xny_data(data_xy_list, range_type=range_type,
                                  is_error_bar=False,
                                  is_error_shading=True,
                                  data_labels=data_labels,
                                  is_reference_data=is_reference_data,
                                  x_lims=x_lims, y_lims=y_lims,
                                  x_label=x_label, y_label=y_label,
                                  is_latex=is_latex)
    else:
        figure, _ = plot_xy_data(data_xy, data_labels=data_labels,
                                 is_reference_data=is_reference_data,
                                 x_lims=x_lims, y_lims=y_lims, x_label=x_label,
                                 y_label=y_label, is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_truth_vs_prediction(prediction_sets, error_bound=None,
                             is_r2_coefficient=False,
                             is_direct_loss_estimator=False,
                             is_normalize_data=False,
                             filename='prediction_vs_groundtruth',
                             save_dir=None, is_save_fig=False,
                             is_stdout_display=False, is_latex=False):
    """Plot ground-truth versus predictions.
    
    Parameters
    ----------
    prediction_sets : dict
        One or more prediction processes, where each process (key, str) is
        stored as a data array (item, numpy.ndarray(2d)) as follows: the i-th
        row is associated with i-th prediction point, data_array[i, 0] holds
        the ground-truth and data_array[i, 1] holds the prediction. Dictionary
        keys are taken as labels for the corresponding prediction processes.
    error_bound : float, default=None
        Relative error between ground-truth and prediction that defines an
        symmetric error-based shaded area with respect to the identity line.
    is_r2_coefficient : bool, default=False
        Plot coefficient of determination. Only effective if plotting a single
        prediction process.
    is_direct_loss_estimator : bool, default=False
        Plot Direct Loss Estimator (DLE) based on Linear Regression model.
        Only effective if plotting a single prediction process.
    is_normalize_data : bool, default=False
        Normalize predictions and ground-truth data to the range [0, 1] for
        each prediction process.
    filename : str, default='prediction_vs_groundtruth'
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
    # Check loss history
    if not isinstance(prediction_sets, dict):
        raise RuntimeError('Prediction processes are not provided as a dict.')
    elif not all([isinstance(x, np.ndarray)
                  for x in prediction_sets.values()]):
        raise RuntimeError('Prediction processes must be provided as a dict '
                           'where each process (key, str) is stored as a '
                           'numpy.ndarray of shape (n_points, 2).')
    elif not all([x.shape[1] == 2 for x in prediction_sets.values()]):
        raise RuntimeError('Prediction processes must be provided as a dict '
                           'where each process (key, str) is stored as a '
                           'numpy.ndarray of shape (n_points, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of prediction processes
    n_processes = len(prediction_sets.keys())
    # Get maximum number of prediction points
    max_n_points = max([x.shape[0] for x in prediction_sets.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_points, 2*n_processes), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prediction processes
    for i, (key, val) in enumerate(prediction_sets.items()):
        # Normalize prediction process data
        if is_normalize_data:
            val = (val - val.min())/(val.max() - val.min())
        # Assemble prediction process
        data_xy[:val.shape[0], 2*i] = val[:val.shape[0], 0]
        data_xy[:len(val), 2*i + 1] = val[:val.shape[0], 1]
        # Assemble data label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set relative error parameters
    if error_bound is not None:
        is_identity_line = True
        identity_error = float(error_bound)
    else:
        is_identity_line=None
        identity_error=None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    if is_normalize_data:
        x_lims = (0, 1)
        y_lims = (0, 1)
    else:
        x_lims = (None, None)
        y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Ground-truth'
    y_label = 'Prediction'
    if is_normalize_data:
        x_label += ' (Normalized)'
        y_label += ' (Normalized)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ground-truth versus predictions
    figure, _ = scatter_xy_data(
        data_xy, data_labels=data_labels, is_identity_line=is_identity_line,
        identity_error=identity_error, is_r2_coefficient=is_r2_coefficient,
        is_direct_loss_estimator=is_direct_loss_estimator,
        x_lims=x_lims, y_lims=y_lims,
        x_label=x_label, y_label=y_label, is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='png', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)