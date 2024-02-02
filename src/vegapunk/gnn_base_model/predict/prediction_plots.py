"""Plots to assess prediction of Graph Neural Network model.

Functions
---------
plot_truth_vs_prediction
    Plot ground-truth against predictions.
plot_prediction_loss_history
    Plot model prediction process loss history.
plot_dle_error_histogram
    Plot Direct Loss Estimator (DLE) absolute error histogram.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
# Local
from ioput.plots import plot_xy_data, scatter_xy_data, plot_histogram, \
    save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
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
    # Set title
    title = 'Prediction vs Ground-truth'
    if is_normalize_data:
        title += ' (Normalized)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot ground-truth versus predictions
    figure, _ = scatter_xy_data(
        data_xy, data_labels=data_labels, is_identity_line=is_identity_line,
        identity_error=identity_error, is_r2_coefficient=is_r2_coefficient,
        is_direct_loss_estimator=is_direct_loss_estimator,
        x_lims=x_lims, y_lims=y_lims, title=title,
        x_label=x_label, y_label=y_label, is_latex=is_latex)
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
def plot_prediction_loss_history(loss_history, loss_type=None,
                                 is_log_loss=False, loss_scale='linear',
                                 filename='prediction_loss_history',
                                 save_dir=None, is_save_fig=False,
                                 is_stdout_display=False, is_latex=False):
    """Plot model prediction process loss history.
    
    Parameters
    ----------
    loss_history : dict
        One or more prediction processes loss histories, where each loss
        history (key, str) is stored as a list of prediction steps loss values
        (item, list). Dictionary keys are taken as labels for the corresponding
        prediction processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='prediction_loss_history'
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
    if not isinstance(loss_history, dict):
        raise RuntimeError('Loss history is not a dict.')
    elif not all([isinstance(x, list) for x in loss_history.values()]):
        raise RuntimeError('Data must be provided as a dict where each loss '
                           'history (key, str) is stored as a list[float] '
                           '(item, list).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of prediction processes
    n_loss_history = len(loss_history.keys())
    # Get maximum number of prediction steps
    max_n_predict_steps = max([len(x) for x in loss_history.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_predict_steps, 2*n_loss_history), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prediction processes
    for i, (key, val) in enumerate(loss_history.items()):
        # Assemble loss history
        data_xy[:len(val), 2*i] = tuple([*range(0, len(val))])
        if is_log_loss:
            data_xy[:len(val), 2*i + 1] = tuple(np.log(val))
        else:
            data_xy[:len(val), 2*i + 1] = tuple(val)
        # Assemble data label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, max_n_predict_steps)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Prediction steps'
    if loss_type is None:
        if is_log_loss:
            y_label = 'log(Loss)'
        else:
            y_label = 'Loss'
    else:
        if is_log_loss:
            y_label = f'log(Loss) ({loss_type})'
        else:
            y_label = f'Loss ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Prediction loss history'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xy_data(data_xy, data_labels=data_labels, x_lims=x_lims,
                             y_lims=y_lims, title=title, x_label=x_label,
                             y_label=y_label, y_scale=y_scale,
                             x_tick_format='int', is_latex=is_latex)
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
def plot_dle_error_histogram(prediction_data_array, is_normalize_data=False,
                             filename='dle_lr_error_histogram',
                             save_dir=None, is_save_fig=False,
                             is_stdout_display=False, is_latex=False):
    """Plot Direct Loss Estimator (DLE) absolute error histogram.
    
    Direct Loss Estimator (DLE) is based on Linear Regression model.
    
    Parameters
    ----------
    is_normalize_data : bool, default=False
        Normalize predictions and ground-truth data to the range [0, 1]
        before estimating the error.
    filename : str, default='dle_lr_error_histogram'
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
    # Initialize data array
    data_xy = prediction_data_array
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize prediction data
    if is_normalize_data:
        data_xy = (data_xy - data_xy.min())/(data_xy.max() - data_xy.min())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get reference and predicted data
    ref_data = data_xy[:, 0].reshape(-1, 1)
    pred_data = data_xy[:, 1].reshape(-1, 1)
    # Fit linear regression model (monitored model)
    monitored_model = sklearn.linear_model.LinearRegression()
    monitored_model.fit(ref_data, pred_data)
    # Get monitored model predicted data
    pred_data_monitored = monitored_model.predict(ref_data)
    # Compute absolute error between predicted data and monitored model
    # predictions
    abs_error = abs(pred_data - pred_data_monitored).reshape(-1)
    # Compute absolute mean error
    mean_abs_error = np.mean(abs_error)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set histogram data sets
    hist_data = (abs_error,)
    # Set probability density flag
    density = True
    # Set title
    title = 'DLE-LR error distribution'
    if is_normalize_data:
        title += ' (normalized data)'
    # Set histogram axes labels
    x_label = 'DLE-LR absolute error'
    if density:
        y_label = 'Probability density'
    else:
        y_label = 'Frequency'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Direct Loss Estimator (DLE) absolute error histogram
    figure, axes = plot_histogram(hist_data, bins=100, density=density,
                                  title=title, x_label=x_label,
                                  y_label=y_label, x_lims=(0.0, None),
                                  is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set mean absolute error color
    error_color = 'black'
    # Set mean absolute error string
    if is_latex:
        mean_str = r'$\mu_{\epsilon}=' + f'{mean_abs_error:.2e}' + '$'
    else:
        mean_str = r'Avg. error =' + f'{mean_abs_error:.2e}'
    # Set mean absolute text box properties
    text_box_props = dict(boxstyle='round', facecolor='#ffffff',
                          edgecolor=error_color, alpha=1.0)
    # Plot mean absolut error
    axes.axvline(mean_abs_error, color=error_color, ls='--', zorder=20)
    axes.text(0.97, 0.97, mean_str, fontsize=10, ha='right', va='top',
              transform=axes.transAxes, bbox=text_box_props, zorder=20)
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