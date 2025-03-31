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
import pickle
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.iostandard import make_directory, find_unique_file_with_regex
from ioput.plots import scatter_xny_data, save_figure
# =============================================================================
# Summary: Generate manual plots for model discovery paper
# =============================================================================
def plot_prediction_loss_convergence(filename='testing_loss_convergence',
                                     save_dir=None, is_save_fig=False,
                                     is_save_plot_data=False,
                                     is_stdout_display=False):
    """Plot average prediction loss of one or more models.


    To be included in scatter_xny_data() to add candidate models:

    # Set candidate models average prediction loss
    candidates_loss = \
        [None, 8.05744189e+03, 3.61672375e+05, 1.81742800e+06]
    # Get color of the last plotted data
    last_color = axes.lines[-1].get_color()
    # Plot constant value
    if candidates_loss[i] is not None:
        plt.axhline(y=candidates_loss[i], color=last_color,
                    linestyle='--', zorder=2)

    Parameters
    ----------
    filename : str, default='testing_loss_convergence'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_save_plot_data : bool, default=False
        Save plot data. Plot data is stored in a file with a single dictionary
        where each item corresponds to a relevant variable used to generate the
        plot. If the figure directory is provided, then plot data is saved in
        the same directory, otherwise is saved in the current working
        directory.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Set plot option
    option = ('hybrid', 'pretraining')[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if option == 'hybrid':
        # Set models convergence analysis base directories
        models_base_dirs = (
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/ml_models/polynomial/'
            'convergence_analysis',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/dp_plus_gru/'
            'dp_4d97_plus_gru',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/dp_plus_gru/'
            'dp_2d50_plus_gru',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/dp_plus_gru/'
            'dp_0d01_plus_gru',
            )
        # Set models labels
        models_labels = [os.path.basename(x) for x in models_base_dirs]
        models_labels = ['GRU',
                         'Hybrid: D-P ($\phi \\approx 4.97^{\circ}$)',
                         'Hybrid: D-P ($\phi=2.50^{\circ}$)',
                         'Hybrid: VM']
    elif option == 'pretraining':
        # Set models convergence analysis base directories
        models_base_dirs = (
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/ml_models/polynomial/'
            'convergence_analysis',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/pretraining_dp/'
            'pt_gru_dp4d97/pt_n_fixed_2560',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/pretraining_dp/'
            'pt_gru_dp2d50/pt_n_fixed_2560',
            '/home/bernardoferreira/Documents/brown/projects/'
            'darpa_paper_examples/local/hybrid_models/pretraining_dp/'
            'pt_gru_dp0d01/pt_n_fixed_2560',
            )
        # Set models labels
        models_labels = [os.path.basename(x) for x in models_base_dirs]
        models_labels = ['GRU',
                         'Pre-train: D-P ($\phi \\approx 4.97^{\circ}$)',
                         'Pre-train: D-P ($\phi=2.50^{\circ}$)',
                         'Pre-train: VM']
    else:
        raise RuntimeError('Invalid option.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize models loss convergence data file paths
    models_data_files_paths = []
    # Set models loss convergence data file paths
    for model_base_dir in models_base_dirs:
        search_dir = os.path.join(os.path.normpath(model_base_dir),
                                  'plots_id_testing/plot_data/')
        regex = (r'^testing_loss_convergence(_uq)?_data\.pkl$',)
        is_file_found, data_file_path = \
            find_unique_file_with_regex(search_dir, regex)
        # Check data file path
        if not is_file_found:
            raise RuntimeError('The model loss convergence data file has not '
                               'been found in the following directory:\n\n',
                               search_dir)
        # Store data file path
        models_data_files_paths.append(data_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize list of data arrays
    data_xy_list = []
    # Loop over models
    for i, data_file_path in enumerate(models_data_files_paths):
        # Load model data
        with open(data_file_path, 'rb') as dataset_file:
            model_data = pickle.load(dataset_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model loss convergence data
        data_xy = model_data['data_xy']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Restructure loss convergence data (single x-axis data)
        cols = [0, ] + [i for i in range(1, data_xy.shape[1], 2)]
        data_xy = data_xy[:, cols]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model loss convergence data
        data_xy_list.append(data_xy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = models_labels
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (8e0, 3e3)
    y_lims = (1e3, 4e6)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss (MSE)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, _ = scatter_xny_data(
        data_xy_list, data_labels=data_labels, is_error_bar=True,
        range_type='min-max', x_lims=x_lims,
        y_lims=y_lims, x_label=x_label, y_label=y_label, x_scale='log',
        y_scale='log', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save plot data
    if is_save_plot_data:
        # Set current working directory to save plot data
        if save_dir is None:
            save_dir = os.getcwd()
        # Set plot data subdirectory
        plot_data_dir = os.path.join(os.path.normpath(save_dir), 'plot_data')
        # Create plot data directory
        if not os.path.isdir(plot_data_dir):
            make_directory(plot_data_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build plot data
        plot_data = {}
        plot_data['data_xy'] = data_xy
        plot_data['x_label'] = x_label
        plot_data['y_label'] = y_label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plot data file path
        plot_data_file_path = os.path.join(
            plot_data_dir, filename + '_data' + '.pkl')
        # Save model samples best parameters data
        with open(plot_data_file_path, 'wb') as data_file:
            pickle.dump(plot_data, data_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set plot option
    plot_option = 'prediction_loss_convergence'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    if plot_option == 'prediction_loss_convergence':
        # Set save directory
        save_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                    'darpa_paper_examples/local/hybrid_models/'
                    'pretraining_dp/plots')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot average prediction loss
        plot_prediction_loss_convergence(filename='testing_loss_convergence',
                                         save_dir=save_dir, is_save_fig=True,
                                         is_save_plot_data=False,
                                         is_stdout_display=True)