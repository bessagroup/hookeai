"""Plots to assess model predictions.

Functions
---------
plot_time_series_prediction
    Plot time series ground-truth and prediction.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def plot_time_series_prediction(time_series_data_array,
                                x_label='Time', y_label='Value',
                                is_normalize_data=False,
                                filename='time_series_prediction',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Plot time series ground-truth and prediction.
    
    Parameters
    ----------
    time_series_data_array : np.ndarray(2d)
        Time series prediction data array stored as a numpy.ndarray(2d) of
        shape (n_nodes, 3), where data_array[i, 0] stores the time series time,
        data_array[i, 0] stores the time series ground-truth and
        data_array[i, 1] stores the time series prediction prediction.
    x_label : str, default='Time'
        x-axis label.
    y_label : str, default='Value'
        y-axis label.
    is_normalize_data : bool, default=False
        Normalize time, predictions and ground-truth data to the range [0, 1].
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
    # Check time series data array
    if not isinstance(time_series_data_array, np.ndarray):
        raise RuntimeError('Time series predition data array must be a '
                           'numpy.ndarray of shape (sequence_length, 3).')
    elif (len(time_series_data_array.shape) != 2
          or time_series_data_array.shape[1] != 3):
        raise RuntimeError('Time series predition data array must be a '
                           'numpy.ndarray of shape (sequence_length, 3).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time series sequence length
    n_time = time_series_data_array.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_time, 4), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data array
    if is_normalize_data:
        # Get minimum and maximum values
        min_val = np.min(time_series_data_array[:, 1:])
        max_val = np.max(time_series_data_array[:, 1:])
        range_val = max_val - min_val
        # Set normalized data array
        data_xy[:, 0] = np.linspace(0, 1.0, num=n_time)
        data_xy[:, 1] = (time_series_data_array[:, 1] - min_val)/range_val
        data_xy[:, 2] = np.linspace(0, 1.0, num=n_time)
        data_xy[:, 3] = (time_series_data_array[:, 2] - min_val)/range_val
    else:
        data_xy[:, 0] = time_series_data_array[:, 0]
        data_xy[:, 1] = time_series_data_array[:, 1]
        data_xy[:, 2] = time_series_data_array[:, 0]
        data_xy[:, 3] = time_series_data_array[:, 2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = ['Ground-truth', 'Prediction']
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
    if is_normalize_data:
        x_label += ' (Normalized)'
        y_label += ' (Normalized)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot time series ground-truth and prediction
    figure, _ = plot_xy_data(data_xy, data_labels=data_labels, x_lims=x_lims,
                             y_lims=y_lims, x_label=x_label, y_label=y_label,
                             is_latex=is_latex)
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