"""Metrics to assess performance of GNN-based material patch model.

Functions
---------
plot_training_loss_history
    Plot model training process loss history (loss vs training steps).
plot_xy_data
    Plot data in xy axes.
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
# Local

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def plot_training_loss_history(loss_history, loss_type=None,
                               total_n_train_steps=0,
                               filename='training_loss_history',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False):
    """Plot model training process loss history (loss vs training steps).
    
    Parameters
    ----------
    loss_history : dict
        One or more training processes loss histories, where each loss history
        (key, str) is stored as a list of loss values for each training step
        (item, list). Dictionary keys are taken as labels for the corresponding
        training processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
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
        True if displaying figure to standard output device, False
        otherwise.
    """
    # Check loss history
    if not isinstance(loss_history, dict):
        raise RuntimeError('Loss history is not a dict.')
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
        data_xy[:len(val), 2*i + 1] = tuple(val)
        # Assemble data label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    x_lims = (0, max_n_train_steps)
    y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training steps'
    if loss_type is None:
        y_label = 'Loss'
    else:
        y_label = f'Loss ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Training loss history'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xy_data(data_xy, data_labels=data_labels, x_lims=x_lims,
                             y_lims=y_lims, title=title, x_label=x_label,
                             y_label=y_label, y_scale='log',
                             x_tick_format='int', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        # Set figure directory
        if save_dir is None:
            save_dir = os.getcwd()
        else:
            if not os.path.isdir(save_dir):
                raise RuntimeError('The provided directory has not been found:'
                                   '\n\n' + save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set figure path
        filepath = os.path.join(save_dir, f'{str(filename)}.pdf')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set figure size (inches)
        figure.set_figheight(3.6, forward=False)
        figure.set_figwidth(3.6, forward=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        figure.savefig(filepath, transparent=False, dpi=300,
                       bbox_inches='tight')
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
    data_xy : np.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        dataset (x_i, y_i) is stored in columns (2*i, 2*i + 1), respectively.
    data_labels : list, default=None
        Labels of datasets (x_i, y_i) provided in data_xy and sorted
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
        # Get number of datasets
        n_datasets = int(data_xy.shape[1]/2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check datasets labels
    if data_labels is not None:
        if len(data_labels) != n_datasets:
            raise RuntimeError('Number of dataset labels is not consistent '
                               'with number of datasets.')
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
    # Loop over datasets
    for i in range(n_datasets):
        # Plot dataset
        axes.plot(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                  label=data_labels[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if data_labels is not None:
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
