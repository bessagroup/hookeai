"""Metrics to assess performance of GNN-based material patch model.

Functions
---------
plot_training_loss_history
    Plot model training process loss history (loss vs training steps).
plot_training_loss_and_lr_history
    Plot model training process loss and learning rate histories.
plot_loss_convergence_test
    Plot testing and training loss for different training data set sizes.
plot_xy_data
    Plot data in xy axes.
plot_xy2_data
    Plot data in xy axes with two y axes.
plot_xny_data
    Plot data in xy axes with given range of y-values for each x-value.
save_figure
    Save Matplotlib figure.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
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
                               loss_scale='linear', total_n_train_steps=0, 
                               filename='training_loss_history',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False):
    """Plot model training process loss history.
    
    Parameters
    ----------
    loss_history : dict
        One or more training processes loss histories, where each loss history
        (key, str) is stored as a list of training steps loss values
        (item, list). Dictionary keys are taken as labels for the corresponding
        training processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    total_n_train_steps : int, default=0
        Total number of training steps prescribed for training process. If
        provided, then it sets the x-axis upper limit if greater than number
        of steps in loss history.
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
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
    # Get maximum number of training steps
    max_n_train_steps = max([len(x) for x in loss_history.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max(max_n_train_steps, total_n_train_steps),
                       2*n_loss_history), fill_value=None)
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
    x_lims = (0, max_n_train_steps)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training steps'
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
                             x_tick_format='int', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def plot_training_loss_and_lr_history(loss_history, lr_history, loss_type=None,
                                      is_log_loss=False, loss_scale='linear',
                                      lr_type=None, total_n_train_steps=0,
                                      filename='training_loss_and_lr_history',
                                      save_dir=None, is_save_fig=False,
                                      is_stdout_display=False):
    """Plot model training process loss and learning rate histories.
    
    Parameters
    ----------
    loss_history : list[float]
        Training process loss history stored as a list of training steps loss
        values.
    lr_history : list[float]
        Training process learning rate history stored as a list of training
        steps learning rate values.
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
    total_n_train_steps : int, default=0
        Total number of training steps prescribed for training process. If
        provided, then it sets the x-axis upper limit if greater than number
        of steps in loss history.
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Check loss history
    if not isinstance(loss_history, list):
        raise RuntimeError('Loss history is not a list[float].')
    # Check learning rate history
    if not isinstance(lr_history, list):
        raise RuntimeError('Learning rate history is not a list[float].')
    elif len(lr_history) != len(loss_history):
        raise RuntimeError('Number of training steps of learning rate history '
                           'is not consistent with loss history.')
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
    x_lims = (0, max(len(loss_history), total_n_train_steps))
    y1_lims = (None, None)
    y2_lims = (None, None)
    y1_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training steps'
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
                              x_tick_format='int', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def plot_loss_convergence_test(testing_loss, training_loss=None,
                               loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='loss_convergence_test',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False):
    """Plot testing and training loss for different training data set sizes.
    
    Parameters
    ----------
    testing_loss : np.ndarray(2d)
        Testing loss data array where i-th row is associated with the i-th
        testing process for a given training data set size and the
        corresponding data is stored as folows: testing_loss[i, 0] is the size
        of the dataset used to train the model, testing_loss[i, 1:] is the
        testing loss for each trained model (e.g., different training data sets
        in k-fold cross-validation). Missing loss values should be stored as
        None.
    training_loss : np.ndarray(2d), default=None
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
                              x_tick_format='int', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def plot_xy_data(data_xy, data_labels=None, x_lims=(None, None),
                 y_lims=(None, None), title=None, x_label=None, y_label=None,
                 x_scale='linear', y_scale='linear', x_tick_format=None,
                 y_tick_format=None, is_latex=False):
    """Plot data in xy axes.

    Parameters
    ----------
    data_xy : np.ndarray(2d)
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
        If True, then render all strings in LaTeX.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
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
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1,1,1)
    # Set title
    axes.set_title(title, fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    axes.set_ylabel(y_label, fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        return '${:2d}$'.format(int(x))
    def floatTickFormat(x, pos):
        return '${:3.1f}$'.format(x)
    def expTickFormat(x, pos):
        return '${:7.2e}$'.format(x)
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
                  label=data_labels[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        axes.legend(loc='upper left', frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_xlim(x_lims)
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
    data_xy1 : np.ndarray(2d)
        Data array containing the plot data associated with the first y-axis
        stored columnwise as (x_i, y_i).
    data_xy2 : np.ndarray(2d)
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
        If True, then render all strings in LaTeX.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
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
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create first axes
    axes = figure.add_subplot(1,1,1)
    # Set first axes color
    color_y1 = '#4477AA'
    # Set title
    axes.set_title(title, fontsize=12, pad=10)
    # Set first axes labels
    axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    axes.set_ylabel(y1_label, fontsize=12, labelpad=10, color=color_y1)
    # Set first axes color
    axes.spines["right"].set_visible(False)
    axes.spines['left'].set(color=color_y1, linewidth=1.1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create second axes (sharing the x axis)
    axes2 = axes.twinx()
    # Set second axes color
    color_y2 = '#EE6677'
    # Configure second axes label
    axes2.set_ylabel(y2_label, fontsize=12, labelpad=10, color=color_y2)
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
        return '${:2d}$'.format(int(x))
    def floatTickFormat(x, pos):
        return '${:3.1f}$'.format(x)
    def expTickFormat(x, pos):
        return '${:7.2e}$'.format(x)
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
    axes.set_xlim(x_lims)
    axes.set_ylim(y1_lims)
    # Set second axes limits
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
        If True, then render all strings in LaTeX.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
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
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1,1,1)
    # Set title
    axes.set_title(title, fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    axes.set_ylabel(y_label, fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    # Set tick formatting functions
    def intTickFormat(x, pos):
        return '${:2d}$'.format(int(x))
    def floatTickFormat(x, pos):
        return '${:3.1f}$'.format(x)
    def expTickFormat(x, pos):
        return '${:7.2e}$'.format(x)
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
               (np.std(data_xy[:, 1:], axis=1).reshape(1, -1),
                np.std(data_xy[:, 1:], axis=1).reshape(1, -1)), axis=0)
        else:
            raise RuntimeError('Unknown type of range.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot data set
        axes.errorbar(x, y_mean, yerr=y_err, capsize=3,
                      label=data_labels[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        axes.legend(loc='upper left', frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_xlim(x_lims)
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)