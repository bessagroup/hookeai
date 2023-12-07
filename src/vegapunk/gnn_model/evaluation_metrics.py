"""Metrics to assess performance of GNN-based material patch model.

Functions
---------
plot_training_loss_history
    Plot model training process loss history.
plot_training_loss_and_lr_history
    Plot model training process loss and learning rate histories.
plot_loss_convergence_test
    Plot testing and training loss for different training data set sizes.
plot_truth_vs_prediction
    Plot ground-truth against predictions.
plot_xy_data
    Plot data in xy axes.
plot_xy2_data
    Plot data in xy axes with two y axes.
plot_xny_data
    Plot data in xy axes with given range of y-values for each x-value.
scatter_xy_data
    Scatter data in xy axes.
plot_boxplots
    Plot set of box plots.
save_figure
    Save Matplotlib figure.
tex_str
    Format string according with LaTeX rendering option.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cycler
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def plot_training_loss_history(loss_history, loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='training_loss_history',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=False):
    """Plot model training process loss history.
    
    Parameters
    ----------
    loss_history : dict
        One or more training processes loss histories, where each loss history
        (key, str) is stored as a list of epochs loss values (item, list).
        Dictionary keys are taken as labels for the corresponding training
        processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='training_loss_history'
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
    # Get number of training processes
    n_loss_history = len(loss_history.keys())
    # Get maximum number of training epochs
    max_n_train_epochs = max([len(x) for x in loss_history.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_train_epochs, 2*n_loss_history), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training processes
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
    x_lims = (0, max_n_train_epochs)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
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
    title = 'Training loss history'
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
def plot_training_loss_and_lr_history(loss_history, lr_history, loss_type=None,
                                      is_log_loss=False, loss_scale='linear',
                                      lr_type=None,
                                      filename='training_loss_and_lr_history',
                                      save_dir=None, is_save_fig=False,
                                      is_stdout_display=False, is_latex=False):
    """Plot model training process loss and learning rate histories.
    
    Parameters
    ----------
    loss_history : list[float]
        Training process loss history stored as a list of training epochs
        loss values.
    lr_history : list[float]
        Training process learning rate history stored as a list of training
        epochs learning rate values.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    lr_type : str, default=None
        Learning rate scheduler type. If provided, then learning rate scheduler
        type is added to the y-axis label.    
    filename : str, default='training_loss_history'
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
    if not isinstance(loss_history, list):
        raise RuntimeError('Loss history is not a list[float].')
    # Check learning rate history
    if not isinstance(lr_history, list):
        raise RuntimeError('Learning rate history is not a list[float].')
    elif len(lr_history) != len(loss_history):
        raise RuntimeError('Number of epochs of learning rate history is not '
                           'consistent with loss history.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data arrays
    x = tuple([*range(0, len(loss_history))])
    if is_log_loss:
        data_xy1 = np.column_stack((x, tuple(np.log(loss_history))))
    else:
        data_xy1 = np.column_stack((x, tuple(loss_history)))
    data_xy2 = np.column_stack((x, tuple(lr_history)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, len(loss_history))
    y1_lims = (None, None)
    y2_lims = (None, None)
    y1_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
    if loss_type is None:
        if is_log_loss:
            y1_label = 'log(Loss)'
        else:
            y1_label = 'Loss'
    else:
        if is_log_loss:
            y1_label = f'log(Loss) ({loss_type})'
        else:
            y1_label = f'Loss ({loss_type})'
    if lr_type is None:
        y2_label = 'Learning rate'
    else:
        y2_label = f'Learning rate ({lr_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Training loss and learning rate history'    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss and learning rate history
    figure, _ = plot_xy2_data(data_xy1, data_xy2, x_lims=x_lims,
                              y1_lims=y1_lims, y2_lims=y2_lims, title=title,
                              x_label=x_label, y1_label=y1_label,
                              y2_label=y2_label, y1_scale=y1_scale,
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
def plot_loss_convergence_test(testing_loss, training_loss=None,
                               loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='loss_convergence_test',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=False):
    """Plot testing and training loss for different training data set sizes.
    
    Parameters
    ----------
    testing_loss : numpyp.ndarray(2d)
        Testing loss data array where i-th row is associated with the i-th
        testing process for a given training data set size and the
        corresponding data is stored as folows: testing_loss[i, 0] is the size
        of the dataset used to train the model, testing_loss[i, 1:] is the
        testing loss for each trained model (e.g., different training data sets
        in k-fold cross-validation). Missing loss values should be stored as
        None.
    training_loss : numpy.ndarray(2d), default=None
        Training loss data array where i-th row is associated with the i-th
        training process for given training data set size and the corresponding
        data is stored as folows: training_loss[i, 0] is the size of the
        data set used to train the model, training_loss[i, 1:] is the training
        loss for each trained model (e.g., different training data sets in
        k-fold cross-validation). Missing loss values should be stored as None.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='training_loss_history'
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
    # Check testing loss data array
    if not isinstance(testing_loss, np.ndarray):
        raise RuntimeError('Testing loss data array is not a np.ndarray.')
    elif len(testing_loss.shape) != 2:
        raise RuntimeError('Testing loss data array is not a np.ndarray '
                           'of shape (n_training_sizes, n_testing_loss).')
    # Check training loss data array
    if training_loss is not None:
        if not isinstance(training_loss, np.ndarray):
            raise RuntimeError('Training loss data array is not a np.ndarray.')
        elif len(training_loss.shape) != 2:
            raise RuntimeError('Training loss data array is not a np.ndarray '
                               'of shape (n_training_sizes, n_training_loss).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply logarithm to loss
    if is_log_loss:
        testing_loss[:, 1:] = np.log(testing_loss[:, 1:])
        if training_loss is not None:
            training_loss[:, 1:] = np.log(training_loss[:, 1:])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Set list of data array
    if training_loss is not None:
        data_xy_list = [training_loss, testing_loss]
    else:
        data_xy_list = [testing_loss,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = None
    if training_loss is not None:
        data_labels = ['Training', 'Testing']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, None)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
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
    title = 'Data set size convergence test'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xny_data(data_xy_list, range_type='mean-std',
                              data_labels=data_labels, x_lims=x_lims,
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
def plot_truth_vs_prediction(prediction_sets, error_bound=None,
                             is_normalize_data=False,
                             filename='prediction_vs_groundtruth',
                             save_dir=None, is_save_fig=False,
                             is_stdout_display=False, is_latex=False):
    """Plot ground-truth against predictions.
    
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
            val = val/val.max()
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
    # Plot loss history
    figure, _ = scatter_xy_data(data_xy, data_labels=data_labels,
                                is_identity_line=is_identity_line,
                                identity_error=identity_error,
                                x_lims=x_lims, y_lims=y_lims, title=title,
                                x_label=x_label, y_label=y_label,
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
# =============================================================================
def plot_kfold_cross_validation(k_fold_loss_array, loss_type=None,
                                loss_scale='linear',
                                filename='kfold_cross_validation',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Plot k-fold cross-validation results.
    
    Parameters
    ----------
    k_fold_loss_array : numpy.ndarray(2d)
        k-fold cross-validation loss array. For the i-th fold,
        data_array[i, 0] stores the best training loss and data_array[i, 1]
        stores the average prediction loss per sample.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
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
    if not isinstance(k_fold_loss_array, np.ndarray):
        raise RuntimeError('k-fold cross-validation loss array must be '
                           'numpy.ndarray(2d) of shape (n_fold, 2).')
    elif k_fold_loss_array.shape[1] != 2:
        raise RuntimeError('k-fold cross-validation loss array must be '
                           'numpy.ndarray(2d) of shape (n_fold, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of cross-validation folds
    n_fold = k_fold_loss_array.shape[0]
    # Set folds labels
    folds_labels = tuple([f'Fold {x + 1}' for x in range(n_fold)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build folds training and validation loss data
    folds_data = {}
    folds_data['Training'] = tuple(k_fold_loss_array[:, 0])
    folds_data['Validation'] = tuple(k_fold_loss_array[:, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = None
    y_label = 'Loss'
    if loss_type is not None:
        y_label += f' ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'k-Fold Cross-Validation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = grouped_bar_chart(groups_labels=folds_labels,
                                  groups_data=folds_data,
                                  is_avg_hline=True,
                                  title=title, x_label=x_label,
                                  y_label=y_label, y_scale=loss_scale,
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
# =============================================================================
def plot_xy_data(data_xy, data_labels=None, x_lims=(None, None),
                 y_lims=(None, None), title=None, x_label=None, y_label=None,
                 x_scale='linear', y_scale='linear', x_tick_format=None,
                 y_tick_format=None, is_latex=False):
    """Plot data in xy axes.

    Parameters
    ----------
    data_xy : numpy.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        data set (x_i, y_i) is stored in columns (2*i, 2*i + 1), respectively.
    data_labels : list, default=None
        Labels of data sets (x_i, y_i) provided in data_xy and sorted
        accordingly. If None, then no labels are displayed.
    x_lims : tuple, default=(None, None)
        x-axis limits in data coordinates.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    x_scale : str {'linear', 'log'}, default='linear'
        x-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any x-axis ticks formatting.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    x_tick_format : {'int', 'float', 'exp'}, default=None
        x-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y_tick_format : {'int', 'float', 'exp'}, default=None
        y-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data
    if data_xy.shape[1] % 2 != 0:
        raise RuntimeError('Data array must have an even number of columns, '
                           'two for each dataset (x_i, y_i).')
    else:
        # Get number of data sets
        n_datasets = int(data_xy.shape[1]/2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check datasets labels
    if data_labels is not None:
        if len(data_labels) != n_datasets:
            raise RuntimeError('Number of data set labels is not consistent '
                               'with number of data sets.')
    else:
        data_labels = n_datasets*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1, 1, 1)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set axes tick formats
    if x_scale != 'log' and x_tick_format in ('int', 'float', 'exp'):
        axes.xaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[x_tick_format]))
    if y_scale != 'log' and y_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=-20)
    # Configure grid minor lines
    axis_option = {'log-log': 'both', 'log-linear': 'x', 'linear-log': 'y'}
    xy_scale = f'{x_scale}-{y_scale}'
    if xy_scale in axis_option.keys():
        axes.grid(which='minor', axis=axis_option[xy_scale], linestyle='-',
                  linewidth=0.5, color='0.5', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets
    for i in range(n_datasets):
        # Plot dataset
        axes.plot(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                  label=tex_str(data_labels[i], is_latex))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        axes.legend(loc='upper left', frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    if x_lims != (None, None):
        axes.set_xlim(x_lims)
    if y_lims != (None, None):
        axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def plot_xy2_data(data_xy1, data_xy2, x_lims=(None, None),
                  y1_lims=(None, None), y2_lims=(None, None), title=None,
                  x_label=None, y1_label=None, y2_label=None, x_scale='linear',
                  y1_scale='linear', y2_scale='linear', x_tick_format=None,
                  y1_tick_format=None, y2_tick_format=None, is_latex=False):
    """Plot data in xy axes with two y axes.

    Parameters
    ----------
    data_xy1 : numpy.ndarray(2d)
        Data array containing the plot data associated with the first y-axis
        stored columnwise as (x_i, y_i).
    data_xy2 : numpy.ndarray(2d)
        Data array containing the plot data associated with the second y-axis
        stored columnwise as (x_i, y_i).
    x_lims : tuple, default=(None, None)
        x-axis limits in data coordinates.
    y1_lims : tuple, default=(None, None)
        y1-axis limits in data coordinates.
    y2_lims : tuple, default=(None, None)
        y2-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y1_label : str, default=None
        y1-axis label.
    y2_label : str, default=None
        y2-axis label.
    x_scale : str {'linear', 'log'}, default='linear'
        x-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any x-axis ticks formatting.
    y1_scale : str {'linear', 'log'}, default='linear'
        y1-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y1-axis ticks formatting.
    y2_scale : str {'linear', 'log'}, default='linear'
        y2-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y2-axis ticks formatting.
    x_tick_format : {'int', 'float', 'exp'}, default=None
        x-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y1_tick_format : {'int', 'float', 'exp'}, default=None
        y1-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y2_tick_format : {'int', 'float', 'exp'}, default=None
        y2-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data
    if data_xy1.shape[1] != 2:
        raise RuntimeError('Data array for first y-axis dataset must have '
                           'shape (n_points, 2).')
    if data_xy2.shape[1] != 2:
        raise RuntimeError('Data array for second y-axis dataset must have '
                           'shape (n_points, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create first axes
    axes = figure.add_subplot(1, 1, 1)
    # Set first axes color
    color_y1 = '#4477AA'
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set first axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y1_label, is_latex), fontsize=12, labelpad=10,
                    color=color_y1)
    # Set first axes color
    axes.spines["right"].set_visible(False)
    axes.spines['left'].set(color=color_y1, linewidth=1.1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second axes (sharing the x axis)
    axes2 = axes.twinx()
    # Set second axes color
    color_y2 = '#EE6677'
    # Configure second axes label
    axes2.set_ylabel(tex_str(y2_label, is_latex), fontsize=12, labelpad=10,
                     color=color_y2)
    # Set second axes color
    axes2.spines["left"].set_visible(False)
    axes2.spines['right'].set(color=color_y2, linewidth=1.1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set first axes color
    axes.tick_params(which='both', axis='y', color=color_y1,
                     labelcolor=color_y1)
    # Set first axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y1_scale in ('linear', 'log'):
        axes.set_yscale(y1_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set axes tick formats
    if x_scale != 'log' and x_tick_format in ('int', 'float', 'exp'):
        axes.xaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[x_tick_format]))
    if y1_scale != 'log' and y1_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y1_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set second axes color
    axes2.tick_params(which='both', axis='y', color=color_y2,
                      labelcolor=color_y2)
    # Set second axes scales
    if y2_scale in ('linear', 'log'):
        axes2.set_yscale(y2_scale)
    if y2_scale != 'log' and y2_tick_format in ('int', 'float', 'exp'):
        axes2.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y2_tick_format]))    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=-20)
    # Configure grid minor lines
    axis_option = {'log-log': 'both', 'log-linear': 'x', 'linear-log': 'y'}
    xy_scale = f'{x_scale}-{y1_scale}'
    if xy_scale in axis_option.keys():
        axes.grid(which='minor', axis=axis_option[xy_scale], linestyle='-',
                  linewidth=0.5, color='0.5', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot dataset in first axes
    axes.plot(data_xy1[:, 0], data_xy1[:, 1], color=color_y1)
    # Plot dataset in second axes
    axes2.plot(data_xy2[:, 0], data_xy2[:, 1], color=color_y2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set first axes limits
    if x_lims != (None, None):
        axes.set_xlim(x_lims)
    if y1_lims != (None, None):
        axes.set_ylim(y1_lims)
    # Set second axes limits
    if y2_lims != (None, None):
        axes2.set_ylim(y2_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def plot_xny_data(data_xy_list, range_type='min-max', data_labels=None,
                  x_lims=(None, None), y_lims=(None, None), title=None,
                  x_label=None, y_label=None, x_scale='linear',
                  y_scale='linear', x_tick_format=None, y_tick_format=None,
                  is_latex=False):
    """Plot data in xy axes with given range of y-values for each x-value.

    Parameters
    ----------
    data_xy_list : list[np.ndarray(2d)]
        List of data arrays. Each data array contains plot data stored
        columnwise such that data_array[:, 0] holds the x-axis data and
        data_array[:, 1:] holds the y-axis data. For each x-value, the mean and
        variance of the y-values are plotted.
    range_type : {'min-max', 'mean-std'}, default='min-max'
        Type of range of y-values to be plotted for each x-value.
    data_labels : list, default=None
        Labels of data arrays provided in data_xy_list and sorted accordingly.
        If None, then no labels are displayed.
    x_lims : tuple, default=(None, None)
        x-axis limits in data coordinates.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    x_scale : str {'linear', 'log'}, default='linear'
        x-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any x-axis ticks formatting.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    x_tick_format : {'int', 'float', 'exp'}, default=None
        x-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y_tick_format : {'int', 'float', 'exp'}, default=None
        y-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data
    if not isinstance(data_xy_list, list):
        raise RuntimeError('Data must be provided as a list of data arrays '
                           'stored as numpy.ndarray(2d).')
    elif not all([isinstance(x, np.ndarray) for x in data_xy_list]):
        raise RuntimeError('Data must be provided as a list of data arrays '
                           'stored as numpy.ndarray(2d).')
    elif not all([len(x.shape) == 2 for x in data_xy_list]):
        raise RuntimeError('Data must be provided as a list of data arrays '
                           'stored as numpy.ndarray(2d).')
    # Get number of data sets
    n_datasets = len(data_xy_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check datasets labels
    if data_labels is not None:
        if len(data_labels) != n_datasets:
            raise RuntimeError('Number of data set labels is not consistent '
                               'with list of data arrays.')
    else:
        data_labels = n_datasets*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1, 1, 1)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set axes tick formats
    if x_scale != 'log' and x_tick_format in ('int', 'float', 'exp'):
        axes.xaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[x_tick_format]))
    if y_scale != 'log' and y_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=-20)
    # Configure grid minor lines
    axis_option = {'log-log': 'both', 'log-linear': 'x', 'linear-log': 'y'}
    xy_scale = f'{x_scale}-{y_scale}'
    if xy_scale in axis_option.keys():
        axes.grid(which='minor', axis=axis_option[xy_scale], linestyle='-',
                  linewidth=0.5, color='0.5', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets
    for i in range(n_datasets):
        # Get data array
        data_xy = data_xy_list[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get x-values
        x = data_xy[:, 0]
        # Get y-values range
        if range_type == 'min-max':
           y_mean = np.mean(data_xy[:, 1:], axis=1)
           y_err = np.concatenate(
               (np.min(data_xy[:, 1:], axis=1).reshape(1, -1),
                np.max(data_xy[:, 1:], axis=1).reshape(1, -1)), axis=0)
        elif range_type == 'mean-std':
           y_mean = np.mean(data_xy[:, 1:], axis=1)
           y_err = np.concatenate(
               (1.96*np.std(data_xy[:, 1:], axis=1).reshape(1, -1),
                1.96*np.std(data_xy[:, 1:], axis=1).reshape(1, -1)), axis=0)
        else:
            raise RuntimeError('Unknown type of range.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot data set
        axes.errorbar(x, y_mean, yerr=y_err, capsize=3,
                      label=tex_str(data_labels[i], is_latex))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        axes.legend(loc='upper left', frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    if x_lims != (None, None):
        axes.set_xlim(x_lims)
    if y_lims != (None, None):
        axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def scatter_xy_data(data_xy, data_labels=None, is_identity_line=False,
                    identity_error=None, x_lims=(None, None),
                    y_lims=(None, None), title=None, x_label=None,
                    y_label=None, x_scale='linear', y_scale='linear',
                    x_tick_format=None, y_tick_format=None, is_latex=False):
    """Scatter data in xy axes.

    Parameters
    ----------
    data_xy : numpy.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        data set (x_i, y_i) is stored in columns (2*i, 2*i + 1), respectively.
    data_labels : list, default=None
        Labels of data sets (x_i, y_i) provided in data_xy and sorted
        accordingly. If None, then no labels are displayed.
    is_identity_line : bool, default=False
        Plot identity line.
    identity_error : float, default=None
        Relative error between x-data and y-data defining a symmetric
        error-based shaded area with respect to the identity line.
    x_lims : tuple, default=(None, None)
        x-axis limits in data coordinates.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    x_scale : str {'linear', 'log'}, default='linear'
        x-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any x-axis ticks formatting.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    x_tick_format : {'int', 'float', 'exp'}, default=None
        x-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y_tick_format : {'int', 'float', 'exp'}, default=None
        y-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data
    if data_xy.shape[1] % 2 != 0:
        raise RuntimeError('Data array must have an even number of columns, '
                           'two for each dataset (x_i, y_i).')
    else:
        # Get number of data sets
        n_datasets = int(data_xy.shape[1]/2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check datasets labels
    if data_labels is not None:
        if len(data_labels) != n_datasets:
            raise RuntimeError('Number of data set labels is not consistent '
                               'with number of data sets.')
    else:
        data_labels = n_datasets*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1, 1, 1)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set axes tick formats
    if x_scale != 'log' and x_tick_format in ('int', 'float', 'exp'):
        axes.xaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[x_tick_format]))
    if y_scale != 'log' and y_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=0)
    # Configure grid minor lines
    axis_option = {'log-log': 'both', 'log-linear': 'x', 'linear-log': 'y'}
    xy_scale = f'{x_scale}-{y_scale}'
    if xy_scale in axis_option.keys():
        axes.grid(which='minor', axis=axis_option[xy_scale], linestyle='-',
                  linewidth=0.5, color='0.5', zorder=-20)        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets
    for i in range(n_datasets):
        # Plot dataset
        axes.scatter(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                     s=10, facecolor='#4477AA', edgecolor='k', linewidth=0.5,
                     label=tex_str(data_labels[i], is_latex),
                     zorder=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_xlim(x_lims)
    axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot identity line and identity error bounds
    if is_identity_line:
        # Plot identity line
        axes.axline((0, 0), slope=1, color='k', linestyle='--',
                    label='Identity line', zorder=5)
        # Plot identity error bounds
        if identity_error is not None:            
            x = np.linspace(0.0, axes.axis()[1])
            axes.fill_between(x=x, y1=(1 + identity_error)*x,
                              y2=(1 - identity_error)*x,
                              color='#BBBBBB',
                              label=f'{identity_error*100:.0f}\\% error',
                              zorder=-15)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]) or identity_error is not None:
        axes.legend(loc='upper left', frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def grouped_bar_chart(groups_labels, groups_data, bar_width=None,
                      is_avg_hline=False, y_lims=(None, None), title=None,
                      x_label=None, y_label=None, y_scale='linear',
                      y_tick_format=None, is_latex=False):
    """Plot grouped bar chart.

    Parameters
    ----------
    groups_labels = tuple[str]
        Bar groups labels.
    groups_data = dict
        Store groups data (item, tuple) for data set (key, str). Dictionary
        keys are taken as labels for each data set.
    bar_width : float, default=None
        Width of bars.
    is_avg_hline, default=False
        Plot horizontal line for average value of each data set.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check bar groups labels
    if not isinstance(groups_labels, tuple):
        raise RuntimeError('Bar groups labels must be provided as a tuple of '
                           'strings.')
    else:
        # Get number of bar groups
        n_groups = len(groups_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check bar groups data
    if not isinstance(groups_data, dict):
        raise RuntimeError('Bar groups data must be provided as a dictionary '
                           'of tuples.')
    elif not all([isinstance(x, tuple) and len(x) == n_groups
                  for x in groups_data.values()]):
        raise RuntimeError('Bar groups data must be provided as a dictionary '
                           'of tuples with size (n_groups,).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set bar width
    if bar_width is None:
        bar_width = 1/(2 + len(groups_data.keys()))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1, 1, 1)
    axes.set_axisbelow(True)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set groups labels locations
    x = np.arange(n_groups)
    axes.set_xticks(x + bar_width, groups_labels)
    # Set axes tick formats
    if y_scale != 'log' and y_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='y', linestyle='-', linewidth=0.5,
              color='0.5')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize bar groups position counter
    group_id = 0
    # Loop over data sets
    for ds_label, ds_values in groups_data.items():
        # Set data set bar offset
        ds_offset = (0.5 + group_id)*bar_width
        # Plot data set bars
        dataset_bars = axes.bar(x + ds_offset, ds_values, width=bar_width,
                                label=tex_str(ds_label, is_latex),
                                align='center')
        # Plot data set average line
        if is_avg_hline:
            axes.axhline(np.mean(ds_values), linestyle='--', linewidth=1.0,
                         color=dataset_bars[-1].get_facecolor())
        # Increment position counter
        group_id += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add average line to legend (dummy plot)
    if is_avg_hline:
        axes.plot(np.mean(list(groups_data.values())[0]),
                  linestyle='--', color='k', label='Average')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    axes.legend(loc='best', frameon=True, fancybox=True,
                facecolor='inherit', edgecolor='inherit', fontsize=10,
                framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def plot_boxplots(data_boxplots, data_labels=None, is_mean_line=False,
                  y_lims=(None, None), title=None, x_label=None, y_label=None,
                  y_scale='linear', y_tick_format=None, is_latex=False):
    """Plot set of box plots.

    Parameters
    ----------
    data_boxplots : numpy.ndarray(2d)
        Data array where each box plot data is stored columnwise such that the
        i-th boxplot is stored in the i-th column.
    data_labels : list, default=None
        Labels of each box plot provided in data_boxplots and sorted
        accordingly. If None, then no labels are displayed.
    is_mean_line : bool, default=False
        If True, then plot mean line for each boxplot.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    y_tick_format : {'int', 'float', 'exp'}, default=None
        y-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Reset matplotlib internal default style
    plt.rcParams.update(plt.rcParamsDefault)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of boxplots
    n_boxplots = data_boxplots.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check datasets labels
    if data_labels is not None:
        if len(data_labels) != n_boxplots:
            raise RuntimeError('Number of labels is not consistent with '
                               'number of boxplots.')
    else:
        data_labels = n_boxplots*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#000000'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default cycler
    default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check LaTeX availability
    if not bool(shutil.which('latex')):
        is_latex = False
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1, 1, 1)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        frmt = tex_str('{:2d}', is_latex)
        return frmt.format(int(x))
    def floatTickFormat(x, pos):
        frmt = tex_str('{:3.1f}', is_latex)
        return frmt.format(x)
    def expTickFormat(x, pos):
        frmt = tex_str('{:7.2e}', is_latex)
        return frmt.format(x)
    tick_formats = {'int': intTickFormat, 'float': floatTickFormat,
                    'exp': expTickFormat}
    # Set axes tick formats
    if y_scale != 'log' and y_tick_format in ('int', 'float', 'exp'):
        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[y_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=0)     
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set median line properties
    medianprops = dict(linestyle='-', linewidth=1, color='k')
    # Set outliers properties
    flierprops = dict(marker='o', markersize=5)
    # Set mean line properties
    meanlineprops = None
    if is_mean_line:
        meanlineprops = dict(linestyle='-', linewidth=1.5, color='#d62728')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot set of box plots
    bp = axes.boxplot(data_boxplots, labels=data_labels,
                      flierprops=flierprops, medianprops=medianprops,
                      showmeans=is_mean_line, meanline=is_mean_line,
                      meanprops=meanlineprops, patch_artist=True,
                      zorder=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set box plots face color
    for patch in bp['boxes']:
        patch.set(facecolor='#4477AA')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot mean line annotation
    for i, line in enumerate(bp['means']):
        # Get mean line coordinates
        x0, _ = line.get_xydata()[0, :]
        x1, y1 = line.get_xydata()[1, :]
        # Set annotation text coordinates
        x_text = x1 + 0.1*(x1 - x0)
        y_text = y1
        # Set annotation text
        if is_latex:
            text = '$\mu$'
        else:
            text = 'μ'
        # Plot mean line annotation
        axes.annotate(text, xy=(x_text, y_text), color='#d62728', va='center',
                      size=12)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    if y_lims != (None, None):
        axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def save_figure(figure, filename, format='pdf', save_dir=None):
    """Save Matplotlib figure.
    
    Parameters
    ----------
    figure : Matplotlib Figure
        Figure.
    filename : str
        Figure name.
    format : {'pdf', 'png'}, default='pdf'
        File format.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    """
    # Set figure directory
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        if not os.path.isdir(save_dir):
            raise RuntimeError('The provided directory has not been found:'
                               '\n\n' + save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure path
    filepath = os.path.join(save_dir, f'{str(filename)}.{format}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure size (inches)
    figure.set_figheight(3.6, forward=False)
    figure.set_figwidth(3.6, forward=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    figure.savefig(filepath, transparent=False, dpi=300, bbox_inches='tight')
# =============================================================================
def tex_str(x, is_latex):
    """Format string conveniently according with LaTeX rendering option.
    
    Parameters
    ----------
    x : {str, None}
        String.
    is_latex : bool
        If False, then remove any leading or trailing dollar signs from string.
        If True, then keep string unchanged.
        
    Returns
    -------
    new_x : {str, None}
        Formatted string.
    """
    # Initialize formatted string
    new_x = x
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Format string according with LaTeX rendering option
    if new_x is None:
        # Keep string unchanged
        pass
    elif isinstance(x, str):
        if is_latex:
            # Keep string unchanged
            new_x = x
        else:
            # Remove any leading or trailing dollar signs
            new_x = new_x.strip('$')
    else:
        raise RuntimeError(f'Input must be str or None, not {type(x)}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return new_x
    