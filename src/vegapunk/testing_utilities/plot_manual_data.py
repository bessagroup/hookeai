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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set example
    example = 'learning_drucker_prager_pressure_dependency'
    # Set example data
    if example == 'erroneous_von_mises_properties':
        # Set models labels
        models_labels = ('GRU', 'Hybrid (worst cand.)',
                         'Hybrid (best cand.)', 'Candidate (worst)',
                         'Candidate (best)')
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [789840.062, 163945.734, 174982.516, 55882.1094, 33157.9766,
             23583.9043, 14188.707, 2268.23462, 731.69928]
        models_avg_predict_loss['Hybrid (worst cand.)'] = \
            [87211.1328, 46635.0664, 33203.5078, 18189.6543, 7997.9668,
             7598.12549, 3608.25098, 820.335022, 330.519562]
        models_avg_predict_loss['Hybrid (best cand.)'] = \
            [5289.27295, 3571.09009, 3482.17212, 1808.65771, 1354.48035,
             653.862671, 388.217255, 92.2604599, 39.0522423]
        models_avg_predict_loss['Candidate (worst)'] = \
            len(models_avg_predict_loss['GRU'])*[2.43422850e+06,]
        models_avg_predict_loss['Candidate (best)'] = \
            len(models_avg_predict_loss['GRU'])*[1.40105781e+05,]
        # Set axes limits
        y_lims = (None, 2*10**7)
    elif example == 'learning_von_mises_hardening':
        # Set models labels
        models_labels = ('GRU', 'Hybrid', 'Candidate')
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [344412.344, 213267.188, 111589.875, 53791.375, 32541.502,
             22198.9668, 17533.1738, 2447.47217, 599.129028]
        models_avg_predict_loss['Hybrid'] = \
            [29293.6641, 9592.91211, 5925.93701, 4881.25781, 2361.27759,
             1783.32178, 1379.46692, 210.375671, 76.2688141]
        models_avg_predict_loss['Candidate'] = \
            len(models_avg_predict_loss['Hybrid'])*[1.13914746e+04,]
        # Set axes limits
        y_lims = (None, None)
    elif example == 'learning_drucker_prager_pressure_dependency':
        # Set models labels
        models_labels = ('GRU', 'Hybrid', 'Candidate')
        # Initialize models average prediction loss
        models_avg_predict_loss = {}
        # Set models average prediction loss
        models_avg_predict_loss['GRU'] = \
            [12339631.0, 13428719.0, 3992861.0, 2326316.5, 1208200.12,
             318718.031, 167748.719, 78753.625, 18856.5293]
        models_avg_predict_loss['Hybrid'] = \
            [16871554.0, 8107942.0, 5108467.0, 1671295.88, 879273.062,
             490463.125, 137429.875, 100223.531, 27711.1582]
        models_avg_predict_loss['Candidate'] = \
            len(models_avg_predict_loss['Hybrid'])*[5.40929320e+07,]
        # Set axes limits
        y_lims = (None, None)
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
    figure, axes = plot_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        y_lims=y_lims,
        x_scale='log', y_scale='log', marker='o', markersize=3,
        markeredgecolor='k', markeredgewidth=0.5, is_latex=True)
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
        if any('Candidate' in s for s in models_labels):
            filename = 'testing_loss_convergence_with_candidate'
        else:
            filename = 'testing_loss_convergence'
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def plot_hardening_laws(save_dir, is_save_fig=False, is_stdout_display=False):
    """Plot hardening laws.
    
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
    # Set accumulated plastic strain bounds
    acc_p_strain_min = 0.0
    acc_p_strain_max = 1.0
    # Set yield stress bounds (plot limits)
    yield_stress_min = None
    yield_stress_max = 1800
    # Set number of discretization points
    n_point = 200
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hardening laws labels
    hardening_labels = ('Ground-truth', 'Candidate (worst)',
                        'Candidate (best)')
    # Set accumulated plastic discrete points
    acc_p_strain = np.linspace(acc_p_strain_min, acc_p_strain_max, n_point)
    # Initialize hardening laws
    hardening_laws = {}
    # Set hardening laws
    hardening_laws['Ground-truth'] = \
        900 + 700*((acc_p_strain + 1e-5)**0.5)
    hardening_laws['Candidate (worst)'] = \
        400 + 300*acc_p_strain
    hardening_laws['Candidate (best)'] = \
        700 + 600*((acc_p_strain + 1e-5)**0.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get number of hardening laws
    n_laws = len(hardening_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.full((n_point, 2*n_laws), fill_value=None)
    # Loop over models
    for i, hardening_label in enumerate(hardening_labels):
        # Assemble model training data set size and average prediction loss
        data_xy[:, 2*i] = acc_p_strain
        data_xy[:, 2*i+1] = hardening_laws[hardening_label]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = [x for x in hardening_laws.keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Accumulated plastic strain'
    y_label = 'Yield stress (MPa)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, axes = plot_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_lims=(acc_p_strain_min, acc_p_strain_max),
        y_lims=(yield_stress_min, yield_stress_max), x_scale='linear',
        y_scale='linear', is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Override plot legend (relocate)
    axes.legend(loc='upper left', ncols=1, frameon=True, fancybox=True,
                facecolor='inherit', edgecolor='inherit',
                fontsize=8, framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        filename = 'hardening_laws'
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == "__main__":
    # Set plot processes
    is_plot_avg_prediction_loss = True
    is_plot_hardening_laws = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/7_local_hybrid_training/'
                 'case_learning_drucker_prager_pressure_dependency/'
                 'z_case_plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot average prediction loss
    if is_plot_avg_prediction_loss:
        plot_avg_prediction_loss(save_dir=plots_dir, is_save_fig=True,
                                 is_stdout_display=True)
    # Plot hardening laws
    if is_plot_hardening_laws:
        plot_hardening_laws(save_dir=plots_dir, is_save_fig=True,
                            is_stdout_display=True)