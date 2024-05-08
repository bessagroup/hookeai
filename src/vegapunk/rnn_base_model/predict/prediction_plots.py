"""Plots to assess model predictions.

Functions
---------
plot_time_series_prediction
    Plot time series predictions.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, plot_xny_data, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def plot_time_series_prediction(prediction_sets, is_reference_data=False,
                                x_label='Time', y_label='Value',
                                is_normalize_data=False,
                                is_uncertainty_quantification=False,
                                filename='time_series_prediction',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Plot time series predictions.
    
    Parameters
    ----------
    prediction_sets : dict
        One or more time series prediction processes, where each process
        (key, str) is stored as a data array (item, numpy.ndarray(2d)) of shape
        (sequence_length, 2) as follows: data_array[:, 0] stores the time
        series time, data_array[:, 1] stores the time series prediction.
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
        predictions for each time. Assumes the same time sequence length and
        concatenates all the different prediction processes. Data labels are
        suppressed.
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
                           'numpy.ndarray of shape (sequence_length, 2).')
    elif not all([x.shape[1] == 2
                  for x in prediction_sets.values()]):
        raise RuntimeError('Prediction processes must be provided as a dict '
                           'where each process (key, str) is stored as a '
                           'numpy.ndarray of shape (sequence_length, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of prediction processes
    n_processes = len(prediction_sets.keys())
    # Get maximum sequence length
    max_sequence_len = max([x.shape[0] for x in prediction_sets.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_sequence_len, 2*n_processes), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prediction processes
    for i, (key, val) in enumerate(prediction_sets.items()):
        # Get prediction process sequence length
        n_time = val.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble prediction process data
        data_xy[:n_time, 2*i] = val[:, 0]
        data_xy[:n_time, 2*i + 1] = val[:, 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble prediction process label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # Restructure prediction data (uncertainty quantification)
    if is_uncertainty_quantification:
        # Check if all processes share the same sequence length
        if len(set([x.shape[0] for x in prediction_sets.values()])) != 1:
            raise RuntimeError('All prediction processes must share the same '
                               'sequence length for uncertainy quantification '
                               'plot.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize prediction data list
        data_xy_list = []
        # Get time sequence from the first prediction process
        x_data = data_xy[:, 0].reshape(-1, 1)
        # Assemble reference data
        if is_reference_data:
            # Get reference prediction data
            y_data = data_xy[:, 1].reshape(-1, 1)
            # Assemble reference prediction data
            data_xy_list.append(np.concatenate((x_data, y_data), axis=1))
        # Concatenate all prediction processes time series predictions
        if is_reference_data:
            y_data = np.delete(data_xy[:, 2:], np.s_[0::2], axis=1)
        else:
            y_data = np.delete(data_xy[:, 0:], np.s_[0::2], axis=1)
        # Assemble prediction data
        data_xy_list.append(np.concatenate((x_data, y_data), axis=1))
        # Suppress data labels
        data_labels = None
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
        figure, _ = plot_xny_data(data_xy_list, range_type='mean-std',
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