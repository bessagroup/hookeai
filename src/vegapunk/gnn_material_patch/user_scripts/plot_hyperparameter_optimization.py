"""User script: Plot Hydra multi-run optimization process history."""
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
# Third-party
import torch
# Local
from gnn_base_model.optimization.hydra_optimization_plots import \
    plot_optimization_history
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    # Set multi-run optimization process jobs directories
    optim_history = {}
    optim_history['Standard'] = \
        '/home/bernardoferreira/Desktop/hydra_basename/2023-12-13/17-48-00'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimization process history metric
    optim_metric = 'objective'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set directory where figure is saved
    save_dir = '/home/bernardoferreira/Desktop/hydra_basename/2023-12-13'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Hydra multi-run optimization process history    
    plot_optimization_history(optim_history, optim_metric, save_dir=save_dir,
                              is_save_fig=True, is_latex=True)