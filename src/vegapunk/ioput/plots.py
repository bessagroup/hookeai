"""General plotting tools.

Functions
---------
plot_xy_data
    Plot data in xy axes.
plot_xy2_data
    Plot data in xy axes with two y axes.
plot_xny_data
    Plot data in xy axes with given range of y-values for each x-value.
plot_xyz_data
    Plot data in xyz axes.
scatter_xy_data
    Scatter data in xy axes.
grouped_bar_chart
    Plot grouped bar chart.
plot_boxplots
    Plot set of box plots.
plot_histogram
    Plot 1D histogram.
plot_histogram_2d
    Plot 2D histogram.
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
import sklearn.linear_model
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def plot_xy_data(data_xy, data_labels=None, is_reference_data=False,
                 x_lims=(None, None), y_lims=(None, None), title=None,
                 x_label=None, y_label=None, x_scale='linear',
                 y_scale='linear', x_tick_format=None, y_tick_format=None,
                 marker=None, markersize=None, is_latex=False):
    """Plot data in xy axes.

    Parameters
    ----------
    data_xy : numpy.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        data set (x_i, y_i) is stored in columns (2*i, 2*i + 1), respectively.
    data_labels : list, default=None
        Labels of data sets (x_i, y_i) provided in data_xy and sorted
        accordingly. If None, then no labels are displayed.
    is_reference_data : bool, default=False
        If True, then the first data set is assumed to be the reference and is
        formatted independently (black, dashed, on top).
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
    marker : str, default=None
        Marker type.
    markersize : float, default=None
        Marker size in points.
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
        # Plot reference data set
        if is_reference_data and i == 0:
            axes.plot(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                      label=tex_str(data_labels[i], is_latex),
                      marker=marker, markersize=markersize,
                      color='k', linestyle='--', zorder=20)
            continue
        # Plot data set
        axes.plot(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                  label=tex_str(data_labels[i], is_latex),
                  marker=marker, markersize=markersize)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'upper center'
            ncols = 2
        else:
            loc = 'upper left'
            ncols = 1
        # Plot legend
        legend = axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                             facecolor='inherit', edgecolor='inherit',
                             fontsize=8, framealpha=1.0)
        legend.set_zorder(50)
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
                                          '#BBBBBB'])
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'upper center'
            ncols = 2
        else:
            loc = 'upper left'
            ncols = 1
        # Plot legend
        axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=8,
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
def plot_xyz_data(data_xyz, data_labels=None, x_lims=(None, None),
                  y_lims=(None, None), z_lims=(None, None), title=None,
                  x_label=None, y_label=None, z_label=None,
                  x_scale='linear', y_scale='linear', z_scale='linear',
                  x_tick_format=None, y_tick_format=None, z_tick_format=None,
                  view_angles_deg=(30, 30, 0), marker=None, markersize=None,
                  is_latex=False):
    """Plot data in xyz axes.

    Parameters
    ----------
    data_xyz : numpy.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        data set (x_i, y_i, z_i) is stored in columns (3*i, 3*i + 1, 3*i + 2),
        respectively.
    data_labels : list, default=None
        Labels of data sets (x_i, y_i, z_i) provided in data_xyz and sorted
        accordingly. If None, then no labels are displayed.
    x_lims : tuple, default=(None, None)
        x-axis limits in data coordinates.
    y_lims : tuple, default=(None, None)
        y-axis limits in data coordinates.
    z_lims : tuple, default=(None, None)
        z-axis limits in data coordinates.
    title : str, default=None
        Plot title.
    x_label : str, default=None
        x-axis label.
    y_label : str, default=None
        y-axis label.
    z_label : str, default=None
        z-axis label.
    x_scale : str {'linear', 'log'}, default='linear'
        x-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any x-axis ticks formatting.
    y_scale : str {'linear', 'log'}, default='linear'
        y-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    z_scale : str {'linear', 'log'}, default='linear'
        z-axis scale. If None or invalid format, then default scale is set.
        Scale 'log' overrides any y-axis ticks formatting.
    x_tick_format : {'int', 'float', 'exp'}, default=None
        x-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    y_tick_format : {'int', 'float', 'exp'}, default=None
        y-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    z_tick_format : {'int', 'float', 'exp'}, default=None
        z-axis ticks formatting. If None or invalid format, then default
        formatting is set.
    view_angles_deg : tuple[float], default=None
        Elevation (0), aximuth (1) and roll angles (2) that define the view
        angle of the 3D plot (defined in degrees).
    marker : str, default=None
        Marker type.
    markersize : float, default=None
        Marker size in points.
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
    if data_xyz.shape[1] % 3 != 0:
        raise RuntimeError('Data array must have an even number of columns, '
                           'two for each dataset (x_i, y_i, z_i).')
    else:
        # Get number of data sets
        n_datasets = int(data_xyz.shape[1]/3)
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
    axes = figure.add_subplot(1, 1, 1, projection='3d')
    # Set view angle
    axes.view_init(elev=view_angles_deg[0], azim=view_angles_deg[1],
                   roll=view_angles_deg[2])
    # Set zoom
    axes.set_box_aspect(None, zoom=0.75)
    # Set title
    axes.set_title(tex_str(title, is_latex), fontsize=12, pad=10)
    # Set axes labels
    axes.set_xlabel(tex_str(x_label, is_latex), fontsize=12, labelpad=10)
    axes.set_ylabel(tex_str(y_label, is_latex), fontsize=12, labelpad=10)
    axes.set_zlabel(tex_str(z_label, is_latex), fontsize=12, labelpad=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes scales
    if x_scale in ('linear', 'log'):
        axes.set_xscale(x_scale)
    if y_scale in ('linear', 'log'):
        axes.set_yscale(y_scale)
    if z_scale in ('linear', 'log'):
        axes.set_zscale(z_scale)
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
    if z_scale != 'log' and z_tick_format in ('int', 'float', 'exp'):
        axes.zaxis.set_major_formatter(
            ticker.FuncFormatter(tick_formats[z_tick_format]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure grid major lines
    axes.grid(which='major', axis='both', linestyle='-', linewidth=0.5,
              color='0.5', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data sets
    for i in range(n_datasets):
        # Plot dataset
        axes.plot(data_xyz[:, 3*i], data_xyz[:, 3*i + 1], data_xyz[:, 3*i + 2],
                  label=tex_str(data_labels[i], is_latex),
                  marker=marker, markersize=markersize)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'upper center'
            ncols = 2
        else:
            loc = 'upper left'
            ncols = 1
        # Plot legend
        axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=8,
                    framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    if x_lims != (None, None):
        axes.set_xlim(x_lims)
    if y_lims != (None, None):
        axes.set_ylim(y_lims)
    if z_lims != (None, None):
        axes.set_zlim(z_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
def scatter_xy_data(data_xy, data_labels=None, is_identity_line=False,
                    identity_error=None, is_r2_coefficient=False,
                    is_direct_loss_estimator=False, is_marginal_dists=False,
                    x_lims=(None, None), y_lims=(None, None), title=None,
                    x_label=None, y_label=None, x_scale='linear',
                    y_scale='linear', x_tick_format=None, y_tick_format=None,
                    is_latex=False):
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
    is_r2_coefficient : bool, default=False
        Plot coefficient of determination. Only effective if plotting a single
        data set: reference data stored in data_xy[:, 0] and prediction data
        stored in data_xy[:, 1].
    is_direct_loss_estimator : bool, default=False
        Plot Direct Loss Estimator (DLE) based on Linear Regression model.
        Only effective if plotting a single data set: reference data stored in
        data_xy[:, 0] and prediction data stored in data_xy[:, 1].
    is_marginal_dists : bool, default=False
        Plot marginal distributions along both dimensions. Multiple data sets
        are concatenated along each dimension.
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
    if is_marginal_dists:
        # Create grid layout to place subplots
        gs = figure.add_gridspec(2, 2, 
                                 width_ratios=(4, 1), height_ratios=(1, 4),
                                 left=0.1, right=0.9, bottom=0.1, top=0.9,
                                 wspace=0.05, hspace=0.05)
        # Set main axes
        axes = figure.add_subplot(gs[1, 0])
        # Set marginal distributions axes
        axes_hist_x = figure.add_subplot(gs[0, 0], sharex=axes)
        axes_hist_y = figure.add_subplot(gs[1, 1], sharey=axes)
        # Supress marginal distributions tick labels
        axes_hist_x.tick_params(axis='both',
                                bottom=False, labelbottom=False,
                                left=False, labelleft=False)
        axes_hist_y.tick_params(axis='both',
                                left=False, labelleft=False,
                                bottom=False, labelbottom=False)
    else:
        # Set main axes
        axes = figure.add_subplot(1, 1, 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                     s=10, edgecolor='k', linewidth=0.5,
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
        axes.axline((0, 0), slope=1, color='k', linestyle='--', zorder=5)
        # Identity error bounds
        if identity_error is not None:
            # Set identity error bounds label
            if is_latex:
                label = f'{identity_error*100:.0f}\\% error'
            else:
                label = f'{identity_error*100:.0f}% error'
            # Plot identity error bounds
            if identity_error is not None:            
                x = np.linspace(axes.axis()[0], axes.axis()[1])
                axes.fill_between(x=x, y1=(1 + identity_error)*x,
                                  y2=(1 - identity_error)*x,
                                  color='#BBBBBB', label=label,
                                  zorder=-15)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute and plot coefficient of determination
    if is_r2_coefficient and n_datasets == 1:
        # Get reference and predicted data
        ref_data = data_xy[:, 0]
        pred_data = data_xy[:, 1]
        # Compute mean of reference data
        data_mean = np.mean(ref_data)
        # Compute sum of squares of residuals
        ssres = np.sum((ref_data - pred_data)**2)
        # Compute total sum of squares
        sstot = np.sum((ref_data - data_mean)**2)
        # Avoid division by zero
        tolerance = 1e-8
        if sstot < tolerance*ssres:
            sstot = 1.0
        # Compute coefficient of determination
        r2 = 1 - ssres/sstot
        # Get coefficient of determination string
        r2_str = tex_str(r'$R^2=' + f'{r2:.2f}' + '$', is_latex)
        # Set text box properties
        text_box_props = dict(boxstyle='round', facecolor='#ffffff',
                              edgecolor='#000000', alpha=1.0)
        # Plot coefficient of determination
        axes.text(0.97, 0.03, r2_str, fontsize=10, ha='right', va='bottom',
                  transform=axes.transAxes, bbox=text_box_props, zorder=20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_direct_loss_estimator and n_datasets == 1:
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
        abs_error = abs(pred_data - pred_data_monitored)
        # Build input features and targets to predict absolute error
        features_matrix = np.hstack((ref_data, pred_data_monitored))
        targets_matrix = abs_error.reshape(-1, 1)
        # Fit linear regression model (predict absolute error)
        abs_error_model = sklearn.linear_model.LinearRegression()
        abs_error_model.fit(features_matrix, targets_matrix)
        abs_error_estimate = abs_error_model.predict(features_matrix)
        # Build monitored model prediction bounds
        pred_data_monitored_lbound = pred_data_monitored - abs_error_estimate
        pred_data_monitored_ubound = pred_data_monitored + abs_error_estimate
        # Set label for monitored model predicted data and error bounds
        if is_latex:
            label = '$\mathrm{DLE: LR} \pm \epsilon_{\mathrm{LR}}$'
        else:
            label = 'DLE-LR ' + u'\u00B1' + ' error_{LR}'
        # Plot monitored model predicted data and corresponding bounds
        axes.plot(ref_data, pred_data_monitored, color='#EE7733',
                  label=label, zorder=40)
        axes.plot(ref_data, pred_data_monitored_lbound, color='#EE7733',
                  ls='--', zorder=40)
        axes.plot(ref_data, pred_data_monitored_ubound, color='#EE7733',
                  ls='--', zorder=40)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot marginal distributions
    if is_marginal_dists:
        # Set marginal distributions (concatenating data from all data sets)
        hist_x_data = \
            np.concatenate([data_xy[:, 2*i] for i in range(n_datasets)])
        hist_y_data = \
            np.concatenate([data_xy[:, 2*i + 1] for i in range(n_datasets)])
        # Plot marginal distributions
        axes_hist_x.hist(hist_x_data, bins=20, density=True, color='#EE7733')
        axes_hist_y.hist(hist_y_data, bins=20, density=True, color='#EE7733',
                         orientation='horizontal')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]) or identity_error is not None:
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'upper left'
            ncols = 2
        else:
            loc = 'upper left'
            ncols = 1
        # Plot legend
        axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=8,
                    framealpha=1.0).set_zorder(30)
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
                facecolor='inherit', edgecolor='inherit', fontsize=8,
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
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
    if y_scale in ('linear', 'log', 'symlog'):
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
def plot_histogram(data, data_labels=None, bins=None, bins_range=None,
                   density=False, x_lims=(None, None), y_lims=(None, None),
                   title=None, x_label=None, y_label=None,
                   x_scale='linear', y_scale='linear', x_tick_format=None,
                   y_tick_format=None, is_latex=False):
    """Plot 1D histogram.

    Parameters
    ----------
    data : tuple[numpy.ndarray(1d)]
        Histogram data set.
    data_labels : tuple[str], default=None
        Histogram data sets labels. If None, then no labels are displayed.
    bins : {int, tuple, str}, default=None
        Histogram bins, provided either as an integer (number of equal-width
        bins in the range) or as a sequence (bins edges). Some binning
        strategies are also available (e.g., 'auto', 'fd').
    bins_range : tuple, default=None
        The lower (bins_range[0]) and upper (bins_range[1]) range of the bins,
        ignoring outliers outside the provided range. Not effective if bins
        sequence is provided.
    density : bool, default=False
        If True, draw and return a probability density (area under the
        histogram integrates to 1).
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
    if not isinstance(data, tuple):
        raise RuntimeError('Histogram data sets must be provided as a tuple '
                           'of numpy 1d array.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of data sets
    n_datasets = len(data)
    # Check data sets
    for i in range(n_datasets):
        if not isinstance(data[i], np.ndarray) or len(data[i].shape) != 1:
            raise RuntimeError('Each histogram data set must be stored as '
                               'a numpy 1d array.')
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
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
    # Set default cycler
    default_cycler = cycler_color
    plt.rc('axes', prop_cycle = cycler_color)
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
    # Plot histogram
    bins_vals, bins_edges, _ = axes.hist(data, bins=bins, range=bins_range,
                                         density=density, zorder=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'upper center'
            ncols = 2
        else:
            loc = 'upper left'
            ncols = 1
        # Plot legend
        axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=8,
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
def plot_histogram_2d(data, bins=10, bins_range=None,
                      density=False, x_lims=(None, None), y_lims=(None, None),
                      title=None, x_label=None, y_label=None,
                      x_scale='linear', y_scale='linear', x_tick_format=None,
                      y_tick_format=None, is_latex=False):
    """Plot 2D histogram.

    Parameters
    ----------
    data : numpy.ndarray(2d)
        Histogram data set provided as a numpy.ndarray(2d) of shape
        (n_points, 2).
    bins : {int, [int, int]}, default=10
        Histogram bins for each dimension.
    bins_range : np.ndarray(2d), default=None
        The lower and upper range of the bins along each dimension, ignoring
        outliers outside the provided range. The bounds for the i-th dimension
        are stored as [min, max] in bins_range[i, :].
    density : bool, default=False
        If True, draw and return a probability density (area under the
        histogram integrates to 1).
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
    if not isinstance(data, np.ndarray):
        raise RuntimeError('Histogram data set must be provided as a '
                           'numpy.ndarray(2d) of shape (n_points, 2).')
    elif len(data.shape) != 2 or data.shape[1] != 2:
        raise RuntimeError('Histogram data set must be provided as a '
                           'numpy.ndarray(2d) of shape (n_points, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['#4477AA', '#EE6677', '#228833',
                                          '#CCBB44', '#66CCEE', '#AA3377',
                                          '#BBBBBB', '#EE7733', '#009988',
                                          '#CC3311', '#DDAA33', '#999933',
                                          '#DDCC77', '#882255'])
    # Set default cycler
    default_cycler = cycler_color
    plt.rc('axes', prop_cycle = cycler_color)
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
    figure.set_figwidth(7, forward=True)
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
              color='0.0', zorder=-20)
    # Configure grid minor lines
    axis_option = {'log-log': 'both', 'log-linear': 'x', 'linear-log': 'y'}
    xy_scale = f'{x_scale}-{y_scale}'
    if xy_scale in axis_option.keys():
        axes.grid(which='minor', axis=axis_option[xy_scale], linestyle='-',
                  linewidth=0.5, color='0.0', zorder=-20)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot 2D histogram
    _, x_bins_edges, y_bins_edges, image = \
        axes.hist2d(data[:, 0], data[:, 1], bins=bins, range=bins_range,
                    density=density, cmap='Blues')
    # Plot colorbar
    cb = figure.colorbar(image, ax=axes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set colobar label
    if density:
        colorbar_label = 'Probability density'
    else:
        colorbar_label = 'Frequency'
    # Plot colorbar label
    cb.set_label(label=colorbar_label, size=12)
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