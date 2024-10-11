# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, scatter_xy_data, save_figure
# =============================================================================
# Summary: Generate plot by providing data explicitly
# =============================================================================
def plot_avg_prediction_loss(save_dir, is_save_fig=False,
                             is_stdout_display=False):
    """Plot average prediction loss of multiple models.
    
    Parameters
    ----------
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # Set models labels
    models_labels = ('GRU model', 'Hybrid model')
    # Initialize models average prediction loss
    models_avg_predict_loss = {}
    # Set models average prediction loss
    models_avg_predict_loss['GRU model'] = \
        [481714.60020000004, 195728.23440000002, 150600.85940000002,
         70988.07029999999, 41977.939060000004, 27503.2418,
         19327.506260000002, 2476.8044440000003, 731.3099858]
    models_avg_predict_loss['Hybrid model'] = \
        [0.5*x for x in  models_avg_predict_loss['GRU model']]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get number of training data set sizes
    n_training_sizes = len(training_sizes)
    # Get number of models
    n_models = len(models_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_training_sizes, 2*n_models), fill_value=None)
    # Loop over models
    for i, model_label in enumerate(models_labels):
        # Assemble model training data set size and average prediction loss
        data_xy[:, 2*i] = training_sizes
        data_xy[:, 2*i+1] = models_avg_predict_loss[model_label]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = [x for x in models_avg_predict_loss.keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    y_label = 'Avg. prediction loss'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, axes = scatter_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_scale='log', y_scale='log', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Override plot legend (relocate)
    axes.legend(loc='upper right', ncols=1, frameon=True, fancybox=True,
                facecolor='inherit', edgecolor='inherit',
                fontsize=8, framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        filename = 'testing_loss_convergence'
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set plot processes
    is_plot_avg_prediction_loss = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = ('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot average prediction loss
    if is_plot_avg_prediction_loss:
        plot_avg_prediction_loss(save_dir=plots_dir, is_save_fig=False,
                                 is_stdout_display=True)