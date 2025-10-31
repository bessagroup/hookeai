"""Plot material model convergence analysis (noisy data).

This module is adapted from 'plot_convergence_analysis' to handle data stemming
from different noise scenarios. It only includes the plotting of the average
prediction loss versus training data set size.

Functions
---------
generate_convergence_plots
    Generate plots of convergence analysis.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import shutil
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cycler
# Local
from time_series_data.time_dataset import load_dataset
from ioput.iostandard import make_directory
from ioput.plots import save_figure, tex_str
from ioput.iostandard import find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def generate_convergence_plots(n_noise_case, noise_case_labels,
                               models_base_dirs, training_dirs,
                               testing_dirs, predictions_dirs,
                               is_uncertainty_quantification=False,
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=True):
    """Generate plots of convergence analysis.
    
    Parameters
    ----------
    n_noise_case : int
        Number of noise cases.
    noise_case_labels : tuple[str]
        Noise cases labels.
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    testing_dirs : tuple[str]
        Directory where each model testing data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    is_uncertainty_quantification: bool, default=False
        If True, then account for multiple model samples for each training
        data set size.
    save_dir : str, default=None
        Directory where data set plots are saved. If None, then plots are
        saved in current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Plot average prediction loss versus training data set size
    if is_uncertainty_quantification:
        # Plot average prediction loss versus training data set size
        plot_prediction_loss_convergence_uq(
            n_noise_case, noise_case_labels, models_base_dirs, training_dirs,
            predictions_dirs, filename='testing_loss_convergence_uq',
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display,
            is_latex=is_latex)
    else:
        # Plot average prediction loss versus training data set size
        raise RuntimeError('Not implemented.')
# =============================================================================
def plot_prediction_loss_convergence_uq(n_noise_case, noise_case_labels,
                                        models_base_dirs,
                                        training_dirs, predictions_dirs,
                                        filename='testing_loss_convergence_uq',
                                        save_dir=None, is_save_fig=False,
                                        is_stdout_display=False,
                                        is_latex=True):
    """Plot average prediction loss versus training data set size.
    
    Uncertainty quantification data accounting for different model samples
    predictions for each training data set size is required. The corresponding
    directory named 'uncertainty_quantification' should exist in each model
    base directory.
    
    Parameters
    ----------
    n_noise_case : int
        Number of noise cases.
    noise_case_labels : tuple[str]
        Noise cases labels.
    models_base_dirs : tuple[str]
        Base directory where each model is stored.
    training_dirs : tuple[str]
        Directory where each model training data set is stored.
    predictions_dirs : tuple[str]
        Directory where each model samples predictions results files are
        stored.
    filename : str, default='testing_loss_convergence'
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
    # Get number of models
    n_models = len(models_base_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training data set sizes
    training_sizes = []
    # Loop over training data sets
    for train_dir in training_dirs:
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(train_dir, regex)
        # Check training data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{train_dir}')
        # Get training data set size
        training_sizes.append(len(load_dataset(train_dataset_file_path)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize average prediction losses per sample
    avg_predict_losses = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over models directories
    for i, model_base_dir in enumerate(models_base_dirs):
        # Extract testing type from prediction subdirectory
        testing_type = os.path.basename(os.path.dirname(predictions_dirs[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model uncertainty quantification directory
        uq_directory = os.path.join(os.path.normpath(model_base_dir),
                                    'uncertainty_quantification')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model samples directories
        model_sample_dirs = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get files and directories in uncertainty quantification directory
        directory_list = os.listdir(uq_directory)
        # Loop over files and directories
        for dirname in directory_list:
            # Check if model sample directory
            is_sample_model= \
                bool(re.search(r'^' + 'model' + r'_[0-9]+', dirname))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Append model sample directory
            if is_sample_model:
                model_sample_dirs.append(
                    os.path.join(os.path.normpath(uq_directory), dirname))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort model samples directories
        model_sample_dirs = \
            sorted(model_sample_dirs,
                   key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model samples average prediction loss
        samples_avg_prediction_loss = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of model samples
        n_model_sample = len(model_sample_dirs)
        # Loop over model samples
        for j in range(n_model_sample):
            # Get model sample directory
            sample_dir = model_sample_dirs[j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model sample prediction directory
            sample_pred_dir = os.path.join(os.path.normpath(sample_dir),
                                           '7_prediction', testing_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get prediction summary file
            regex = (r'^summary.dat$',)
            is_file_found, summary_file_path = \
                find_unique_file_with_regex(sample_pred_dir, regex) 
            # Check prediction summary file
            if not is_file_found:
                raise RuntimeError(f'Prediction summary file has not been '
                                   f'found in directory:\n\n{sample_pred_dir}')
            # Open prediction summary file
            summary_file = open(summary_file_path, 'r')
            summary_file.seek(0)
            # Look for average prediction loss
            avg_predict_loss = None
            line_number = 0
            for line in summary_file:
                line_number = line_number + 1
                if 'Avg. prediction loss per sample' in line:
                    avg_predict_loss = float(line.split()[-1])
                    break
            # Store average prediction loss
            if avg_predict_loss is None:
                raise RuntimeError('Average prediction loss has not been '
                                   'found in prediction summary file:\n\n'
                                   f'{summary_file_path}')
            else:
                samples_avg_prediction_loss.append(avg_predict_loss)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model samples average prediction loss
        avg_predict_losses.append(samples_avg_prediction_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_models, 2*n_model_sample), fill_value=None)
    # Loop over models
    for i in range(n_models):
        # Assemble model training data set size and average prediction loss
        data_xy[i, 0::2] = n_model_sample*[training_sizes[i]]
        data_xy[i, 1::2] = avg_predict_losses[i][:]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reshape data array (horizontal stacking of noise cases data)
    data_xy = np.hstack(np.split(data_xy, n_noise_case, axis=0))
    # Set data labels
    data_labels = noise_case_labels
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (8e0, 3e3)
    y_lims = (10e2, 1e6)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss (MSE)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, _ = scatter_xy_data_noise_cases(
        data_xy, data_labels=data_labels, n_noise_case=n_noise_case,
        n_model_sample=n_model_sample, is_error_bar=True, range_type='min-max',
        x_lims=x_lims, y_lims=y_lims, x_label=x_label, y_label=y_label,
        x_scale='log', y_scale='log', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def scatter_xy_data_noise_cases(data_xy, data_labels=None, n_noise_case=1,
                                n_model_sample=1, is_error_bar=False,
                                range_type='min-max', x_lims=(None, None),
                                y_lims=(None, None), title=None, x_label=None,
                                y_label=None, x_scale='linear',
                                y_scale='linear', x_tick_format=None,
                                y_tick_format=None, is_latex=False):
    """Scatter data in xy axes.

    Parameters
    ----------
    data_xy : numpy.ndarray(2d)
        Data array where the plot data is stored columnwise such that the i-th
        data set (x_i, y_i) is stored in columns (2*i, 2*i + 1), respectively.
    data_labels : list, default=None
        Labels of data sets (x_i, y_i) provided in data_xy and sorted
        accordingly. If None, then no labels are displayed.
    n_noise_case : int, default=1
        Number of noise cases.
    n_model_sample : int, default=1
        Number of model samples.
    is_error_bar : bool, default=False
        If True, then plot error bar according with range type. Multiple data
        sets are concatenated and the x values are assumed common between all
        data sets. Data labels are suppressed.
    range_type : {'min-max', 'mean-std', None}, default='min-max'
        Type of range of y-values to be plotted for each x-value around the
        mean. If None, only the mean is plotted. Only effective if is_error_bar
        is set to True.
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
        if is_error_bar and len(data_labels) != n_noise_case:
            raise RuntimeError('Number of data set labels is not consistent '
                               'with number of noise cases.')
        elif not is_error_bar and len(data_labels) != n_datasets:
            raise RuntimeError('Number of data set labels is not consistent '
                               'with number of data sets.')
    else:
        if is_error_bar:
            data_labels = n_noise_case*[None,]
        else:
            data_labels = n_datasets*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',
                                 ['#4477AA', '#EE6677', '#228833',
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
    # Plot data
    if is_error_bar:
        # Get x-values
        x = data_xy[:, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize noise case indexes
        j_init = 1
        j_end = j_init + 2*n_model_sample
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over noise cases
        for i in range(n_noise_case):
            # Get noise case y-values
            y_noise_case = data_xy[:, j_init:j_end:2]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get y-values mean
            y_mean = np.mean(y_noise_case, axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get y-values range
            if range_type == 'min-max':
                # Set lower and upper errors: [|mean - min|, |mean - max|]
                y_err = np.concatenate((np.absolute(
                    y_mean - np.min(y_noise_case, axis=1)).reshape(1, -1),
                                        np.absolute(
                    y_mean - np.max(y_noise_case, axis=1).reshape(1, -1))),
                                       axis=0)
            elif range_type == 'mean-std':
                # Set lower and upper errors: [1.96*std, 1.96*std]
                y_err = np.concatenate(
                    (1.96*np.std(y_noise_case.astype(float),
                                 axis=1).reshape(1, -1),
                     1.96*np.std(y_noise_case.astype(float),
                                 axis=1).reshape(1, -1)),
                    axis=0)
            else:
                # Skip range computation
                y_err = None            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot data (concatenating data from all data sets)
            error_bar_option = 1
            if error_bar_option == 1:
                axes.errorbar(x, y_mean, yerr=y_err, fmt='o', markersize=3,
                              markeredgecolor='k', markeredgewidth=0.5,
                              elinewidth=1.0, capsize=2, linestyle='-',
                              label=data_labels[i])
            elif error_bar_option == 2:
                axes.errorbar(x, y_mean, yerr=y_err, fmt='o', markersize=3,
                              markeredgecolor='k', markeredgewidth=0.5,
                              elinewidth=0.0, capsize=0, linestyle='-',
                              label=data_labels[i])
            else:
                axes.errorbar(x, y_mean, yerr=y_err, fmt='o', markersize=3,
                              markeredgecolor='k', markeredgewidth=0.5,
                              ecolor='k', elinewidth=1.0, capsize=2,
                              linestyle='-', label=data_labels[i])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update noise case indexes
            j_init += 2*n_model_sample
            j_end = j_init + 2*n_model_sample
    else:
        # Loop over data sets
        for i in range(n_datasets):
            # Plot data set
            axes.scatter(data_xy[:, 2*i], data_xy[:, 2*i + 1],
                         s=10, edgecolor='k', linewidth=0.5,
                         label=tex_str(data_labels[i], is_latex),
                         zorder=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    axes.set_xlim(x_lims)
    axes.set_ylim(y_lims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set legend
    if not all([x is None for x in data_labels]):
        # Set legend position and number of columns
        if len(data_labels) > 2:
            loc = 'lower left'
            ncols = 2
        else:
            loc = 'lower left'
            ncols = 1
        # Plot legend
        legend = axes.legend(loc=loc, ncols=ncols, frameon=True, fancybox=True,
                             facecolor='inherit', edgecolor='inherit',
                             fontsize=8, framealpha=1.0)
        legend.set_zorder(50)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return figure and axes handlers
    return figure, axes
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_uncertainty_quantification = True
    # Set testing type
    testing_type = ('in_distribution', 'out_distribution')[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence analyses base directory
    base_dir = ('/home/username/Documents/brown/projects/'
                'darpa_paper_examples/local/ml_models/polynomial/'
                'convergence_analysis_noise/'
                'convergence_analyses_heteroscedastic_spiked_gaussian')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set noise variability
    noise_variability = 'heteroscedastic'
    # Set noise variability label
    if noise_variability == 'heteroscedastic':
        noise_var_label = 'het'
    else:
        noise_var_label = 'hom'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set noise distribution type
    noise_distribution = 'spiked_gaussian'
    # Set noise cases and labels
    if noise_distribution == 'uniform':
        # Set noise cases
        noise_cases = ('noiseless',
                       f'{noise_var_label}uni_noise_4e-2',
                       f'{noise_var_label}uni_noise_1e-1',
                       f'{noise_var_label}uni_noise_2e-1',
                       f'{noise_var_label}uni_noise_4e-1')
        # Set noise cases labels
        noise_cases_labels = ('$\\tilde{\epsilon}=0.0$',
                              '$\\tilde{\epsilon}=0.04$',
                              '$\\tilde{\epsilon}=0.1$',
                              '$\\tilde{\epsilon}=0.2$',
                              '$\\tilde{\epsilon}=0.4$')
    elif noise_distribution == 'gaussian':
        # Set noise cases
        noise_cases = ('noiseless',
                       f'{noise_var_label}gau_noise_1e-2',
                       f'{noise_var_label}gau_noise_2d5e-2',
                       f'{noise_var_label}gau_noise_5e-2',
                       f'{noise_var_label}gau_noise_1e-1')
        # Set noise cases labels
        noise_cases_labels = ('$\\tilde{\epsilon}=0.0$',
                              '$\\tilde{\epsilon}=0.01$',
                              '$\\tilde{\epsilon}=0.025$',
                              '$\\tilde{\epsilon}=0.05$',
                              '$\\tilde{\epsilon}=0.1$')
    elif noise_distribution == 'spiked_gaussian':
        # Set noise cases
        noise_cases = ('noiseless',
                       f'{noise_var_label}sgau_noise_1e-2',
                       f'{noise_var_label}sgau_noise_2d5e-2',
                       f'{noise_var_label}sgau_noise_5e-2',
                       f'{noise_var_label}sgau_noise_1e-1')
        # Set noise cases labels
        noise_cases_labels = ('$\\tilde{\epsilon}_{s}=0.0$',
                              '$\\tilde{\epsilon}_{s}=0.01$',
                              '$\\tilde{\epsilon}_{s}=0.025$',
                              '$\\tilde{\epsilon}_{s}=0.05$',
                              '$\\tilde{\epsilon}_{s}=0.1$')
    else:
        raise RuntimeError('Unknown noise distribution.')
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute number of noises cases
    n_noise_case = len(noise_cases)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize convergence analyses models base directories
    models_base_dirs = []
    # Loop over noise cases
    for noise_case in noise_cases:
        # Loop over training data set sizes
        for n_path in training_sizes:
            # Set model base directory
            model_base_dir = os.path.join(os.path.normpath(base_dir),
                                          f'{noise_case}', f'n{n_path}')
            # Store model base directory
            models_base_dirs.append(model_base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize models directories
    training_dirs = []
    testing_dirs = []
    predictions_dirs = []
    # Loop over models
    for model_base_dir in models_base_dirs:
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(model_base_dir),
                                            '1_training_dataset')
        # Store training data set directory
        if os.path.isdir(training_dataset_dir):
            training_dirs.append(training_dataset_dir)
        else:
            raise RuntimeError('The training data set directory has not been '
                               'found:\n\n' + training_dataset_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing data set directory
        if testing_type == 'in_distribution':
            testing_dataset_dir = os.path.join(
                os.path.normpath(model_base_dir), '5_testing_id_dataset')
        elif testing_type == 'out_distribution':
            testing_dataset_dir = os.path.join(
                os.path.normpath(model_base_dir), '6_testing_od_dataset')
        # Store testing data set directory
        if os.path.isdir(testing_dataset_dir):
            testing_dirs.append(testing_dataset_dir)
        else:
            raise RuntimeError('The testing data set directory has not been '
                               'found:\n\n' + testing_dataset_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set prediction data set directory
        prediction_dir = os.path.join(os.path.normpath(model_base_dir),
                                      f'7_prediction/{testing_type}/'
                                      'prediction_set_0')
        # Store prediction directory
        if os.path.isdir(prediction_dir) or is_uncertainty_quantification:
            predictions_dirs.append(prediction_dir)
        else:
            raise RuntimeError('The prediction directory has not been '
                               'found:\n\n' + prediction_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence analysis plots directory
    if testing_type == 'in_distribution':
        plots_dir = os.path.join(os.path.normpath(base_dir),
                                 'plots_id_testing')
    elif testing_type == 'out_distribution':
        plots_dir = os.path.join(os.path.normpath(base_dir),
                                 'plots_od_testing')
    # Create convergence analysis plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of convergence analysis
    generate_convergence_plots(
        n_noise_case, noise_cases_labels,
        models_base_dirs, training_dirs, testing_dirs, predictions_dirs,
        is_uncertainty_quantification=is_uncertainty_quantification,
        save_dir=plots_dir, is_save_fig=True, is_stdout_display=False,
        is_latex=True)