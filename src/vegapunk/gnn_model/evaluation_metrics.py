"""Metrics to assess performance of GNN-based material patch model.

Functions
---------
plot_training_loss_history
    Plot model training process loss history (loss vs training steps).
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
    loss_history : list[float]
        Training process loss history.
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
    if not isinstance(loss_history, list):
        raise RuntimeError('Loss history is not a list[float].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data
    x = tuple([*range(0, len(loss_history))])
    y = tuple(loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    x_lims = (0, max(len(loss_history), total_n_train_steps))
    y_lims = (0, None)
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
    figure, _ = plot_1d_function(x, y, x_lims=x_lims, y_lims=y_lims,
                                 title=title, x_label=x_label, y_label=y_label,
                                 is_latex=True)
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
def plot_1d_function(x, y, x_lims=(None, None), y_lims=(None, None),
                     title=None, x_label=None, y_label=None, color='k',
                     linestyle='-', marker=None, label=None, vlines=None,
                     is_latex=False):
    """Plot 1d function.

    Parameters
    ----------
    x : array-like
        x-axis data.
    y : array-like
        y-axis data.
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
    color : str, default='k'
        Line color.
    linestyle : str, default='-'
        Line style.
    marker : str, default=None
        Marker style.
    label : str, default=None
        Data label.
    vlines : tuple, default=None
        Vertical lines to be plotted. Tuple is to be structured as
        (vl_coords, vl_colors, vl_alphas), where vl_coords, vl_colors and
        vl_alphas are tuples containing vertical lines' data coordinates,
        colors and transparencies, respectively. If vl_colors and vl_alphas are
        specified as a single value, then it is adopted for all vertical lines.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX.

    Returns
    -------
    figure : Matplotlib Figure
        Figure.
    axes : Matplotlib Axes
        Axes.
    """
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(6, forward=True)
    figure.set_figwidth(6, forward=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set LaTeX font
    if is_latex:
        # Default LaTeX Computer Modern Roman
        plt.rc('text',usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create axes
    axes = figure.add_subplot(1,1,1)
    # Set title
    axes.set_title(title, fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    axes.set_ylabel(y_label, fontsize=12, labelpad=10)
    # Configure grid
    axes.grid(linestyle='-', linewidth=0.5, color='0.5', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot function
    axes.plot(x, y, color=color, ls=linestyle, marker=marker, label=label)
    # Set legend
    if label != None:
        axes.legend(loc='upper left')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if vlines is not None:
        # Unzip vertical lines
        vl_coords, vl_colors, vl_alphas = vlines
        # Convert to tuples
        vl_coords = convert_to_tuple(vl_coords)
        vl_colors = convert_to_tuple(vl_colors, n_copy=len(vl_coords))
        vl_alphas = convert_to_tuple(vl_alphas, n_copy=len(vl_coords))
        # Loop over vertical lines
        for i, vline in enumerate(vl_coords):
            axes.axvline(x=vline, color=vl_colors[i], alpha=vl_alphas[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_xlim(x_lims)
    axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def convert_to_tuple(x, n_copy=1):
    """Convert object to tuple.

    Parameters
    ----------
    x : int, str, float, list, tuple, set
        Object to be converted to tuple.
    n_copy : int, default=1
        If x is int, str or float, then the tuple is populated with n_copy
        copies of x.

    Returns
    -------
    tuple_conversion : tuple
        Tuple conversion.
    """
    if isinstance(x, (int, float, str)):
        tuple_conversion = n_copy*(x,)
    elif isinstance(x, (list, tuple, set)):
        tuple_conversion = tuple(x)
    else:
        raise RuntimeError(f'A {type(x)} object cannnot be converted to '
                           'tuple.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tuple_conversion
