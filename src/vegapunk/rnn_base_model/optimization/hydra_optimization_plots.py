"""Automatic plotting for Hydra hyperparameter optimization.

Functions
---------
plot_optimization_history
    Plot Hydra multi-run optimization process history.
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
def plot_optimization_history(optim_history, optim_metric, is_log_metric=False,
                              objective_scale='linear', is_data_labels=False,
                              filename=None,
                              save_dir=None, is_save_fig=False,
                              is_stdout_display=False, is_latex=False):
    """Plot Hydra multi-run optimization process history.
    
    Assumes that each Hydra multi-run optimization process generates a
    'job_summary.dat' file for each job (in hydra_cfg.runtime.output_dir) with
    data formatted as < optim_metric >: < value >.
    
    Parameters
    ----------
    optim_history : dict
        One or more multi-run optimization processes (key, str) jobs
        directories (item, str). The multi-run job directory is set in
        Hydra configuration file (hydra.sweep.dir). Dictionary keys are taken
        as labels in the corresponding optimization processes history plot.
    optim_metric : str
        The metric whose optimization process history is to be plotted.
        Must be available from all optimization processes jobs summary
        data files in the format < optim_metric >: < value >.
    is_log_metric : bool, default=False
        Applies logarithm to optimization metric values if True, keeps
        original metric values otherwise.
    objective_scale : {'linear', 'log'}, default='linear'
        Optimization metric values axis scale type.
    is_data_labels : bool, default=False
        If True, then plot data labels according with optimization processes
        dictionary keys.
    filename : str, default=None
        Figure name. If None, then figure name is set as
        optimization_history_{optim_metric}.
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
    # Get number of optimization processes
    n_optim_history = len(optim_history.keys())
    # Initialize maximum number of jobs (function evaluations)
    max_n_jobs = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize optimization processes data
    optim_data = {}
    # Initialize data labels
    data_labels=None
    if is_data_labels:
        data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over optimization processes
    for i, (label, optim_dir) in enumerate(optim_history.items()):
        # Check optimization process directory
        if not os.path.isdir(optim_dir):
            raise RuntimeError('The optimization jobs directory has not been '
                               'found:\n\n' + optim_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get job files in optimization process directory
        directory_list = sorted(os.listdir(optim_dir),
               key=lambda x: int(re.search(r'^(\d+)$', x).groups()[-1]))
        # Check directory
        if not directory_list:
            raise RuntimeError('No job files have been found in optimization '
                               'process directory:\n\n' + optim_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize optimization metric history
        metric_hist = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over optimization process jobs
        for job_dir in directory_list:
            # Check if optimization process job directory
            job_id = re.search(r'^([0-9]+)$', job_dir)
            # Extract history data from job summary file
            if job_id is not None:
                # Get job ID
                job_id = int(job_id.groups()[0])
                # Check job ID
                expected_dir = os.path.join(os.path.normpath(optim_dir),
                                            str(len(metric_hist)))
                if job_id != len(metric_hist):
                    raise RuntimeError(
                        f'Job ID {len(metric_hist)} directory has not been '
                        f'found:\n\n{expected_dir}')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get job summary file path
                job_summary_path = \
                    os.path.join(os.path.normpath(optim_dir),
                                 f'{job_id}/job_summary.dat')
                # Check job summary file path
                if not os.path.isfile(job_summary_path):
                    raise RuntimeError(f'The job summary file path has not '
                                       f'been found for job ID {job_id}:'
                                       f'\n\n {job_summary_path}')
                # Open job summary file
                job_summary_file = open(job_summary_path, 'r')
                job_summary_file.seek(0)
                # Look for optimization metric value
                metric = None
                line_number = 0
                for line in job_summary_file:
                    line_number = line_number + 1
                    if str(optim_metric) in line:
                        metric = float(line.split()[-1])
                        break
                # Append job optimization metric to history
                if metric is None:
                    raise RuntimeError(f'Optimization metric {optim_metric} '
                                       f'has not been found in job summary '
                                       f'file:\n\n{job_summary_path}')
                else:
                    metric_hist.append(metric)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update maximum number of jobs
        if len(metric_hist) > max_n_jobs:
            max_n_jobs = len(metric_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble optimization metric history
        optim_data[label] = metric_hist
        # Assemble data label
        if is_data_labels:
            data_labels.append(label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_jobs, 2*n_optim_history), fill_value=None)
    # Build data array
    for i, (label, data_hist) in enumerate(optim_data.items()):
        # Get optimization process number of jobs
        n_jobs = len(data_hist)
        # Assemble optimization process history to data array
        data_xy[:n_jobs, 2*i] = tuple([*range(0, n_jobs)])
        if is_log_metric:
            data_xy[:n_jobs, 2*i + 1] = tuple(np.log(metric_hist))
        else:
            data_xy[:n_jobs, 2*i + 1] = tuple(metric_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, max_n_jobs)
    y_lims = (None, None)
    y_scale = objective_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Jobs (Function Evaluations)'
    if is_log_metric:
        y_label = f'log({optim_metric})'
    else:
        y_label = f'{optim_metric.capitalize()}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Hyperparameter optimization history'
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
    # Set default figure name
    if filename is None:
        filename = f'optimization_history_{optim_metric}'
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
if __name__ == "__main__":
    # Set optimization processes
    optim_history = {}
    optim_history['label'] = \
        ('/home/bernardoferreira/Desktop/hyperparameter_opt/'
         'optimize_gru_material_model_von_mises/2024-10-10/09-16-06')
    # Set plot directory
    save_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/7_local_hybrid_training/'
                'case_erroneous_von_mises_properties/hyp_opt_datasets')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimization metric
    optim_metric = 'objective'
    # Set optimization metric scale
    objective_scale = 'linear'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot optimization process history.
    plot_optimization_history(optim_history, optim_metric,
                              objective_scale=objective_scale,
                              save_dir=save_dir, is_save_fig=True,
                              is_stdout_display=False, is_latex=True)
    