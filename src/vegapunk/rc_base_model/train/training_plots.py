"""Plots to assess model training.

Functions
---------
plot_model_parameters_history
    Plot model learnable parameters history.
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
def plot_model_parameters_history(model_parameters_history,
                                  model_parameters_bounds,
                                  filename='model_parameter_history',
                                  save_dir=None, is_save_fig=False,
                                  is_stdout_display=False, is_latex=False):
    """Plot model learnable parameters history.
    
    Parameters
    ----------
    model_parameters_history : dict
        Model learnable parameters history. For each model parameter
        (key, str), store the corresponding training history (item, list).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    filename : str, default='model_parameter_history'
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
    # Check model parameters history
    if not isinstance(model_parameters_history, dict):
        raise RuntimeError('Model parameters history is not a dict.')
    elif not all([isinstance(x, list)
                  for x in model_parameters_history.values()]):
        raise RuntimeError('Data must be provided as a dict where each '
                           'parameter history (key, str) is stored as a '
                           'list.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters
    for name, history in model_parameters_history.items():
        # Initialize data array
        data_xy = np.zeros((len(history), 2))
        # Build data array
        data_xy[:, 0] = np.arange(len(history))
        data_xy[:, 1] = history
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        x_lims = (0, len(history))
        y_lims = (None, None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Epochs'
        y_label = f'Parameter: {name}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot parameter history
        figure, axes = plot_xy_data(data_xy, x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    x_tick_format='int', marker='o',
                                    markersize=2, is_latex=is_latex)
        # Plot parameter bounds
        axes.hlines(model_parameters_bounds[name], 0, len(history),
                    colors='k', linestyles='dashed')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, f'{filename}_{name}',
                        format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')