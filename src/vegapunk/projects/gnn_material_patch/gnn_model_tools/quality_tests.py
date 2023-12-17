"""Quality tests for GNN-based material patch model.

Functions
---------
perform_quality_tests
    Perform set of GNN-based material patch model quality tests.
qt_internal_forces_equilibrium
    Quality test: Equilibrium of node internal forces.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from gnn_base_model.predict.prediction import load_sample_predictions
from ioput.iostandard import make_directory
from ioput.plots import scatter_xy_data, plot_boxplots, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_quality_tests(predictions_dir, quality_tests='all',
                          is_save_fig=False, is_stdout_display=False):
    """Perform set of GNN-based material patch model quality tests.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    quality_tests : {'all', tuple[str]}
        Quality tests to be performed.
    is_save_fig : bool, default=False
        Save quality tests figures when available.
    is_stdout_display : bool, default=False
        True if displaying quality tests figures to standard output device,
        False otherwise.
        
    Returns
    -------
    quality_tests_results : dict
        Quality tests scores.
    """
    # Set available quality tests
    available_quality_tests = ('sum_internal_forces',)
    # Check if all quality tests are to be performed
    if quality_tests == 'all':
        quality_tests = available_quality_tests
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize summary of quality tests
    quality_tests_results = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over quality tests
    for qt in quality_tests:
        # Quality test: Node internal forces equilibrium.
        if qt == 'sum_internal_forces':
            # Compute quality test score
            quality_test_score = qt_internal_forces_equilibrium(
                predictions_dir, is_save_fig=is_save_fig,
                is_stdout_display=is_stdout_display)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Unknown quality test
        else:
            raise RuntimeError(f'Unknown quality test ({qt}).\n\n'
                               f'Available quality tests: '
                               f'{available_quality_tests}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store quality test score
        quality_tests_results[qt] = quality_test_score
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return quality_tests_results
# =============================================================================
def qt_internal_forces_equilibrium(predictions_dir, is_save_fig=False,
                                   is_stdout_display=False):
    """Quality test: Equilibrium of node internal forces.
    
    The sum of the internal forces over all nodes of given material patch
    must be equal to zero.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
        
    Returns
    -------
    quality_test_score : tuple
        Quality test score computed over all samples and stored as (mean, std).
    """
    # Initialize score
    quality_test_score = None
    # Initialize score (per sample)
    quality_test_score_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize internal forces global equilibrium per dimension
    int_forces_equilibrium_dim = {str(i + 1): [] for i in range(3)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Loop over files
    for filename in directory_list:
        # Check if file is sample results file
        is_sample_results_file = \
            bool(re.search(r'^prediction_sample_([0-9])+.pkl$', filename))
        # Ignore non-results file
        if not is_sample_results_file:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample predictions file path
        sample_prediction_path = \
            os.path.join(os.path.normpath(predictions_dir), filename)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = load_sample_predictions(sample_prediction_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node internal forces predictions
        int_forces = np.array(sample_results['node_features_out'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of dimensions
        n_dim = int_forces.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize sample quality test score
        score = 0
        # Initialize sample sum of normalized internal forces (all dimensions)
        norm_sum_int_forces = 0
        # Loop over dimensions
        for i in range(n_dim):
            # Get absolute sum of internal forces along dimension
            sum_int_forces = np.abs(np.sum(int_forces[:, i]))
            # Get maximum absolute internal force along dimension
            # (normalization factor)
            max_int_force = np.abs(np.max(int_forces[:, i]))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sum of normalized internal forces along dimension
            try:
                norm_sum_int_forces_dim = sum_int_forces/max_int_force
            except ZeroDivisionError:
                norm_sum_int_forces_dim = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Append sample sum of normalized internal forces along dimension
            int_forces_equilibrium_dim[str(i + 1)].append(
                norm_sum_int_forces_dim)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update sample sum of normalized internal forces (all dimensions)
            norm_sum_int_forces += norm_sum_int_forces_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute sample average sum of normalized internal forces
        avg_norm_sum_int_forces = norm_sum_int_forces/n_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store sample score
        quality_test_score_samples.append(avg_norm_sum_int_forces)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute score over all samples (mean and standard deviation)
    quality_test_score = (np.mean(quality_test_score_samples),
                          np.std(quality_test_score_samples))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    if is_save_fig:
        plot_dir = os.path.join(os.path.normpath(predictions_dir), 'plots')
        if not os.path.isdir(plot_dir):
            make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate quality test box plot
    if is_save_fig or is_stdout_display:
        # Get number of samples
        n_samples = len(quality_test_score_samples)
        # Build box plots data:
        # Initialize box plots data and labels
        data_boxplots = np.empty((n_samples, n_dim + 1))
        data_labels = []
        # Loop over dimensions
        for i in range(n_dim):
            # Get data along dimension and set corresponding label
            data_boxplots[:, i] = int_forces_equilibrium_dim[str(i + 1)]
            data_labels.append(f'dim {i + 1}')
        # Get average data and set corresponding label
        data_boxplots[:, -1] = quality_test_score_samples
        data_labels.append(f'avg (dim)')
        # Plot quality test box plots
        figure, _ = plot_boxplots(
            data_boxplots, data_labels, is_mean_line=True,
            title='Equilibrium of Internal Forces',
            y_label='Sum of normalized internal forces', y_scale='log',
            is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        # Set figure name
        filename = 'qt_int_forces_equilibrium_boxplot'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=plot_dir)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    if 'figure' in locals():
        plt.close(figure)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate quality test scatter plot
    if is_save_fig or is_stdout_display:
        
        # Build scatter plot data:
        # Initialize scatter plot data
        data_xy = np.empty((0, 2))
        # Loop over samples
        for i, score in enumerate(quality_test_score_samples):
            # Get sample data
            data_xy = np.append(
                data_xy, np.array((i, score)).reshape(-1, 2), axis=0)
        # Plot quality test scatter plot
        figure, _ = scatter_xy_data(
            data_xy, title='Equilibrium of Internal Forces', x_label='Samples',
            y_label='Average sum of normalized internal forces', y_scale='log',
            is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        # Set figure name
        filename = 'qt_int_forces_equilibrium_scatterplot'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=plot_dir)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    if 'figure' in locals():
        plt.close(figure)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return quality_test_score