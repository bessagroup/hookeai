"""User script: Plot Hydra multi-run optimization process history."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Local
from model_architectures.rnn_base_model.optimization.hydra_optimization_plots \
    import plot_optimization_history
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    # Set multi-run optimization process jobs directories
    optim_history = {}
    optim_history['Standard'] = \
        ('/home/username/Documents/brown/projects/'
         'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
         '2_DTP1U_V2_data/loss_dirichlet_sets/4_hyperparameter_optimization/'
         'optimize_von_mises_parameters/2025-09-19/18-31-54')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimization process history metric
    optim_metric = 'objective'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set directory where figure is saved
    save_dir = os.path.join(optim_history['Standard'], 'plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Hydra multi-run optimization process history    
    plot_optimization_history(optim_history, optim_metric, save_dir=save_dir,
                              is_save_fig=True, is_latex=True, is_verbose=True)