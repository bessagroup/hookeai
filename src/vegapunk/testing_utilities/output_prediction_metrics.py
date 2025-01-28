# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from utilities.prediction_metrics import compute_prediction_metrics
from ioput.iostandard import make_directory
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Output mean prediction metrics from prediction directories
# =============================================================================
def compute_processes_prediction_metrics(predictions_dirs,
                                         mean_prediction_metrics,
                                         save_dir=None, is_save_file=False,
                                         is_display_results=False):
    """Compute mean prediction metrics for multiple prediction processes.

    Parameters
    ----------
    predictions_dirs : dict
        For each prediction process (key, str), store the directory (item, str)
        where the corresponding samples predictions results files are stored.
    mean_prediction_metrics : list[str]
        Mean prediction metrics.
    save_dir : str, default=None
        Directory where file with mean prediction metrics is saved.
    is_save_file : bool, default=False
        If True, then save file with mean prediction metrics in predictions
        dedicated subdirectory.
    is_display_results : bool, default=False
        If True, then display mean prediction metrics to standard output
        device.
        
    Returns
    -------
    processes_results : dict
        For each prediction process (key, str), store the corresponding
        mean prediction metrics data (item, dict).
    """
    # Initialize display
    if is_display_results:
        print('\nMean prediction metrics'
              '\n-----------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction processes results
    processes_results = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over predictions processes directories
    for process_label, predictions_dir in predictions_dirs.items():
        # Compute mean prediction metrics
        n_sample, mean_metrics_results = \
            compute_directory_prediction_metrics(predictions_dir,
                                                 mean_prediction_metrics)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store prediction process results
        processes_results[process_label] = {
            'n_sample': n_sample,
            'mean_metrics_results': mean_metrics_results}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save file with mean prediction metrics
        if is_save_file:
            # Write prediction metrics file
            write_mean_metrics_results_file(save_dir, n_sample,
                                            mean_metrics_results,
                                            process_label=process_label,
                                            is_overwrite=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display mean prediction metrics
        if is_display_results:
            # Get formatted mean prediction metrics
            formatted_results = \
                format_mean_metrics_results(n_sample, mean_metrics_results,
                                            process_label=process_label)
            # Display
            sys.stdout.writelines(formatted_results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return processes_results
# =============================================================================
def compute_directory_prediction_metrics(
        predictions_dir, mean_prediction_metrics, process_label='',
        is_save_file=False, is_display_results=False):
    """Compute mean prediction metrics for given prediction directory.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    mean_prediction_metrics : list[str]
        Mean prediction metrics.
    process_label : str, default=''
        Prediction process label.
    is_save_file : bool, default=False
        If True, then save file with mean prediction metrics in predictions
        dedicated subdirectory.
    is_display_results : bool, default=False
        If True, then display mean prediction metrics to standard output
        device.

    Returns
    -------
    n_sample : int
        Number of samples.
    mean_metrics_results : dict
        Samples mean value (item, torch.Tensor) of each prediction metric
        (key, str).
    """
    # Get samples prediction files
    prediction_file_paths, _ = \
        get_samples_prediction_files(predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(prediction_file_paths)
    # Compute samples mean prediction metrics
    mean_metrics_results = \
        compute_mean_prediction_metrics(prediction_file_paths,
                                        mean_prediction_metrics)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save file with mean prediction metrics
    if is_save_file:
        # Set prediction metrics directory
        prediction_metrics_dir = set_prediction_metrics_dir(predictions_dir)
        # Write prediction metrics file
        write_mean_metrics_results_file(prediction_metrics_dir, n_sample,
                                        mean_metrics_results,
                                        process_label=process_label,
                                        is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display mean prediction metrics
    if is_display_results:
        # Get formatted mean prediction metrics
        formatted_results = \
            format_mean_metrics_results(n_sample, mean_metrics_results,
                                        process_label=process_label)
        # Display
        sys.stdout.writelines(formatted_results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_sample, mean_metrics_results
# =============================================================================
def set_prediction_metrics_dir(predictions_dir):
    """Set prediction metrics directory.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    
    Returns
    -------
    prediction_metrics_dir : str
        Prediction metrics directory.
    """
    # Check sample predictions directory
    if not os.path.isdir(predictions_dir):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set prediction metrics directory
    prediction_metrics_dir = os.path.join(os.path.normpath(predictions_dir),
                                          f'prediction_metrics')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create prediction metrics directory
    make_directory(prediction_metrics_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_metrics_dir
# =============================================================================
def get_samples_prediction_files(predictions_dir):
    """Get samples prediction files from prediction directory.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    
    Returns
    -------
    prediction_file_paths : list[str]
        Samples prediction files paths.
    prediction_files_ids : list[int]
        Samples IDs.
    """
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize samples prediction files paths and samples IDs
    prediction_file_paths = []
    prediction_files_ids = []
    # Loop over files
    for filename in directory_list:
        # Check if file is sample prediction file
        id = re.search(r'^prediction_sample_([0-9]+).pkl$', filename)
        # Store sample prediction file and ID
        if id is not None:
            # Store sample file path
            prediction_file_paths.append(
                os.path.join(os.path.normpath(predictions_dir), filename))
            # Store sample ID
            prediction_files_ids.append(int(id.groups()[0]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_file_paths, prediction_files_ids
# =============================================================================
def compute_mean_prediction_metrics(prediction_file_paths,
                                    mean_prediction_metrics):
    """Compute samples mean prediction metrics from prediction files.
    
    Parameters
    ----------
    prediction_file_paths : list[str]
        Samples prediction files paths.
    mean_prediction_metrics : list[str]
        Mean prediction metrics.

    Returns
    -------
    mean_metrics_results : dict
        Samples mean value (item, torch.Tensor) of each prediction metric
        (key, str).
    """
    # Initialize samples prediction metrics
    samples_metrics_results = {x: [] for x in mean_prediction_metrics}
    # Loop over samples prediction files
    for sample_prediction_path in prediction_file_paths:
        # Compute sample prediction metrics
        sample_metrics_results = \
            compute_prediction_metrics(sample_prediction_path,
                                       ['rmse', 'mav_gt'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get sample prediction metrics
        sample_rmse = sample_metrics_results['rmse']
        sample_mav_gt = sample_metrics_results['mav_gt']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over mean prediction metrics
        for metric in mean_prediction_metrics:
            if metric == 'rmse':
                # Collect sample Root Mean Squared Error (RMSE)
                samples_metrics_results[metric].append(sample_rmse)
            elif metric == 'nrmse':
                # Compute sample Normalized Root Mean Squared Error (NRMSE)
                samples_metrics_results[metric].append(
                    sample_rmse/sample_mav_gt)
            else:
                raise RuntimeError(f'Unknown mean prediction metric: '
                                   f'\'{metric}\'')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize mean prediction metrics
    mean_metrics_results = {}
    # Loop over mean prediction metrics
    for metric in mean_prediction_metrics:
        # Compute mean prediction metric
        mean_metrics_results[metric] = torch.mean(
            torch.stack(samples_metrics_results[metric], dim=0), dim=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mean_metrics_results
# =============================================================================
def write_mean_metrics_results_file(save_dir, n_sample, mean_metrics_results,
                                    process_label='',
                                    filename='mean_prediction_metrics',
                                    is_overwrite=False):
    """Write file with mean prediction metrics.
    
    Parameters
    ----------
    save_dir : str
        Directory where file with mean prediction metrics is saved.        
    n_sample : int
        Number of samples.
    mean_metrics_results : dict
        Samples mean value (item, torch.Tensor) of each prediction metric
        (key, str).
    process_label : str, default=''
        Prediction process label.
    filename : str, default='mean_prediction_metrics'
        File name.
    is_overwrite : bool, default=False
        If True, then overwrite existing file.
    """
    # Check saving directory
    if not os.path.exists(save_dir):
        raise RuntimeError(f'The saving directory has not been found:'
                           f'\n\n{save_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set file path
    file_path = os.path.join(os.path.normpath(save_dir), f'{filename}.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize file content
    file_content = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default opening mode
    open_mode = 'w'
    # Set appending mode
    if os.path.isfile(file_path) and not is_overwrite:
        open_mode = 'a'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set file header
    if open_mode == 'w':
        string = 'Mean prediction metrics'
        sep = len(string)*'-'
        file_content += [f'\n{string}\n{sep}\n',]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get formatted mean prediction metrics
    formatted_results = \
        format_mean_metrics_results(n_sample, mean_metrics_results,
                                    process_label=process_label)
    # Add formatted mean prediction metrics to file content
    file_content += formatted_results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write file
    open(file_path, open_mode).writelines(file_content)
# =============================================================================
def format_mean_metrics_results(n_sample, mean_metrics_results,
                                process_label=''):
    """Format samples mean prediction metrics.
    
    Parameters
    ----------
    n_sample : int
        Number of samples.
    mean_metrics_results : dict
        Samples mean value (item, torch.Tensor) of each prediction metric
        (key, str).
    process_label : str, default=''
        Prediction process label.
    
    Returns
    -------
    formatted_results : list[str]
        Formatted samples mean prediction metrics results.
    """
    # Initialize formatted results
    formatted_results = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set process label
    formatted_results += [f'\n> Process: {process_label}']
    # Add number of samples
    formatted_results += [f'\n    > n_sample = {n_sample}']
    # Loop over mean prediction metrics
    for metric, metric_results in mean_metrics_results.items():
        # Convert mean prediction metrics to list
        metric_results_list = \
            ', '.join([f'{x:15.8e}' for x in metric_results.tolist()])
        # Add mean prediction metric results
        formatted_results += [f'\n    > {metric}: [{metric_results_list}]']
    # Add blankline
    formatted_results += ['\n\n',]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return formatted_results
# =============================================================================
def plot_prediction_metrics_convergence(
        processes_results, mean_prediction_metrics,
        metric_features_labels=None, save_dir=None, is_save_fig=False,
        is_stdout_display=False, is_latex=False):
    """Plot mean prediction metrics convergence analysis.
    
    Only prediction processes named as 'nX', where X is a given training
    dataset size, are processed.
    
    Parameters
    ----------
    processes_results : dict
        For each prediction process (key, str), store the corresponding
        mean prediction metrics data (item, dict).
    mean_prediction_metrics : list[str]
        Mean prediction metrics.
    metric_features_labels : dict, default=None
        For each prediction metric (key, str), store the corresponding features
        labels (list, str).
    save_dir : str, default=None
        Directory where data set plots are saved.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Initialize training data set sizes
    training_sizes = []
    # Collect training data set sizes
    for process_label in processes_results.keys():
        # Check if process label is training data set size
        training_size = re.search(r'^n([0-9]+)$', process_label)
        # Store prediction process training data set size
        if id is not None:
            # Get training data set size
            training_size = int(training_size.groups()[0])
            # Store training data set size
            training_sizes.append(training_size)
    # Get number of training data set sizes
    n_size = len(training_sizes)
    # Sort training data set sizes
    training_sizes = sorted(training_sizes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over mean prediction metrics
    for metric in mean_prediction_metrics:
        # Initialize mean prediction metric convergence data
        metric_convergence_data = []
        # Loop over training data set sizes
        for training_size in training_sizes:
            # Get mean prediction metrics data
            mean_metrics_results = \
                processes_results[f'n{training_size}']['mean_metrics_results']
            # Get metric data
            metric_data = mean_metrics_results[metric]
            # Store metric data
            metric_convergence_data.append(metric_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build convergence metric tensor
        metric_convergence  = torch.vstack(metric_convergence_data).numpy()
        # Get metric number of dimensions
        n_metric_dim = metric_convergence.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data array
        data_xy = np.zeros((n_size, 2*n_metric_dim))
        # Loop over metric dimensions
        for i in range(n_metric_dim):
            # Assemble training data set size data
            data_xy[:, 2*i] = training_sizes
            # Assemble metric dimension data
            data_xy[:, 2*i+1] = metric_convergence[:, i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        if (isinstance(metric_features_labels, dict)
                and metric in metric_features_labels.keys()):
            # Get metric features labels
            data_labels = metric_features_labels[metric]
        else:
            # Set default metric features labels
            if n_metric_dim > 1:
                data_labels = [f'Feature {i}' for i in range(n_metric_dim)]
            else:
                data_labels = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Training data set size'
        y_label = metric.upper()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot data
        figure, _ = plot_xy_data(
            data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
            x_scale='log', y_scale='linear', marker='o', markersize=2,
            is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set filename
        filename = f'mean_{metric}_convergence'
        # Save figure
        if is_save_fig:
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_multiple_processes = False
    # Set mean predictions metrics to be plotted
    mean_prediction_metrics = ['rmse', 'nrmse',]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_multiple_processes:
        # Set testing type
        testing_type = ('in_distribution', 'out_distribution')[0]
        # Set training data set sizes
        training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        # Set convergence analysis base directory
        base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                    'test_output_metric/strain_to_stress')
        # Set saving directory
        save_dir = os.path.join(os.path.normpath(base_dir),
                                'prediction_metrics')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize prediction processes directories
        predictions_dirs = {}
        # Loop over training data set sizes
        for training_size in training_sizes:
            # Set model base directory
            model_base_dir = os.path.join(os.path.normpath(base_dir),
                                          f'n{training_size}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set prediction data set directory
            prediction_dir = os.path.join(os.path.normpath(model_base_dir),
                                          f'7_prediction/{testing_type}/'
                                          f'prediction_set_0')
            # Store prediction directory
            if os.path.isdir(prediction_dir):
                predictions_dirs[f'n{training_size}'] = prediction_dir
            else:
                raise RuntimeError('The prediction directory has not been '
                                   'found:\n\n' + prediction_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create saving directory (overwrite existing directory)
        make_directory(save_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute mean prediction metrics for multiple prediction processes
        processes_results = compute_processes_prediction_metrics(
            predictions_dirs, mean_prediction_metrics, save_dir=save_dir,
            is_save_file=True, is_display_results=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set mean prediction metrics features labels
        metric_features_labels = {
            metric: [f'Stress {x}'
                     for x in ('11', '22', '33', '12', '23', '13')]
            for metric in mean_prediction_metrics}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot mean prediction metrics convergence analysis
        plot_prediction_metrics_convergence(
            processes_results, mean_prediction_metrics,
            metric_features_labels=metric_features_labels, save_dir=save_dir,
            is_save_fig=True, is_stdout_display=True, is_latex=True)
    else:
        # Set predictions directory
        predictions_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                           'test_output_metric/prediction_set_0')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute mean prediction metrics for given prediction directory
        _, _ = compute_directory_prediction_metrics(
            predictions_dir, mean_prediction_metrics, is_save_file=True,
            is_display_results=True)