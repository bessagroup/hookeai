"""DARPA METALS PROJECT: Plot convergence analysis of RNN material model.

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
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import os
# Third-party
import numpy as np
# Local
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    convergence_plots import plot_prediction_loss_convergence, \
        plot_time_series_convergence, plot_prediction_loss_convergence_uq, \
        plot_time_series_convergence_uq
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def generate_convergence_plots(models_base_dirs, training_dirs, testing_dirs,
                               predictions_dirs,
                               is_uncertainty_quantification=False,
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=True):
    """Generate plots of convergence analysis.
    
    Parameters
    ----------
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
            models_base_dirs, training_dirs, predictions_dirs,
            filename='testing_loss_convergence_uq',
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display,
            is_latex=is_latex)
    else:
        # Plot average prediction loss versus training data set size
        plot_prediction_loss_convergence(
            models_base_dirs, training_dirs, predictions_dirs,
            filename='testing_loss_convergence',
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display,
            is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set prediction types and components
    prediction_types = {}
    prediction_types['stress_comps'] = ('stress_11', 'stress_22', 'stress_33',
                                        'stress_12', 'stress_23', 'stress_13')
    #prediction_types['acc_p_strain'] = ('acc_p_strain',)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot models time series predictions versus ground-truth
    if is_uncertainty_quantification:
        # Plot models time series predictions versus ground-truth (path)
        plot_time_series_convergence_uq(
            models_base_dirs, training_dirs, testing_dirs, predictions_dirs,
            prediction_types, plot_type='time_series_path',
            samples_ids=list(np.arange(5, dtype=int)),
            filename='time_series_convergence_uq',
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display, is_latex=is_latex)
    else:
        # Plot models time series predictions versus ground-truth (scatter)
        plot_time_series_convergence(
            models_base_dirs, training_dirs, testing_dirs, predictions_dirs,
            prediction_types, plot_type='time_series_scatter',
            samples_ids=list(np.arange(5, dtype=int)),
            filename='time_series_convergence',
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display, is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot models time series predictions versus ground-truth (path)
        plot_time_series_convergence(
            models_base_dirs, training_dirs, testing_dirs, predictions_dirs,
            prediction_types, plot_type='time_series_path',
            samples_ids=list(np.arange(5, dtype=int)),
            filename='time_series_convergence', save_dir=save_dir,
            is_save_fig=is_save_fig, is_stdout_display=is_stdout_display,
            is_latex=is_latex)
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_uncertainty_quantification = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence analysis base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/6_local_rnn_training_noisy/von_mises/'
                'convergence_analyses_homoscedastic_gaussian/noiseless')
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # Set convergence analyses models base directories
    models_base_dirs = [os.path.join(os.path.normpath(base_dir), f'n{x}')
                        for x in training_sizes]
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
        testing_dataset_dir = os.path.join(os.path.normpath(model_base_dir),
                                           '5_testing_id_dataset')
        # Store testing data set directory
        if os.path.isdir(testing_dataset_dir):
            testing_dirs.append(testing_dataset_dir)
        else:
            raise RuntimeError('The testing data set directory has not been '
                               'found:\n\n' + testing_dataset_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set prediction data set directory
        prediction_dir = os.path.join(os.path.normpath(model_base_dir),
                                      '7_prediction/in_distribution/'
                                      'prediction_set_0')
        # Store prediction directory
        if os.path.isdir(prediction_dir) or is_uncertainty_quantification:
            predictions_dirs.append(prediction_dir)
        else:
            raise RuntimeError('The prediction directory has not been '
                               'found:\n\n' + prediction_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence analysis plots directory
    plots_dir = os.path.join(os.path.normpath(base_dir), 'plots')
    # Create convergence analysis plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of convergence analysis
    generate_convergence_plots(
        models_base_dirs, training_dirs, testing_dirs, predictions_dirs,
        is_uncertainty_quantification=is_uncertainty_quantification,
        save_dir=plots_dir, is_save_fig=True, is_stdout_display=False,
        is_latex=True)