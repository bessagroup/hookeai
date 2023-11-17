"""Hydra hyperparameter optimization of GNN-based material patch model.

Execute (multi-run mode):

    $ python3 optimization_gnn_model.py -m

Functions
---------
hydra_optimize_gnn_model
    Hydra hyperparameter optimization of GNN-based material patch model.
"""
#
#                                                                       Modules
# =============================================================================
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
# Third-party
import hydra
# Local
from ioput.iostandard import write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
@hydra.main(version_base=None, config_path='configs',
            config_name='config_gnn_model')
def hydra_optimize_gnn_model(cfg):
    """Hydra hyperparameter optimization of GNN-based material patch model.
    
    Parameters
    ----------
    cgf : omegaconf.DictConfig
        Configuration dictionary of YAML based hierarchical configuration
        system.
        
    Returns
    -------
    objective : float
        Objective to minimize.
    """
    # Get Hydra configuration singleton
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'\nLaunching Hydra job: #{hydra_cfg.job.id}'
          '\n' + len(f'Launching Hydra job: #{hydra_cfg.job.id}')*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get Hydra sweeper
    sweeper = hydra_cfg['sweeper']['_target_'].split('.')[-1]
    # Display sweeper
    if sweeper == 'BasicSweeper':
        sweeper_optimizer = 'None'
        print(f'\nSweeper:')
        print(f'  > Name      : {sweeper}')
        print(f'  > Optimizer : {sweeper_optimizer}')
    elif sweeper == 'NevergradSweeper':
        sweeper_optimizer = hydra_cfg['sweeper']['optim']['optimizer']
        print(f'\nSweeper:')
        print(f'  > Name      : {sweeper}')
        print(f'  > Optimizer : {sweeper_optimizer}')
    elif sweeper == 'OptunaSweeper':
        sweeper_optimizer = hydra_cfg['sweeper']['sampler']['_target_']
        print(f'\nSweeper:')
        print(f'  > Name      : {sweeper}')
        print(f'  > Optimizer : {sweeper_optimizer}')
    else:
        sweeper_optimizer = 'None/Unknown'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get current working directory
    cwd = os.getcwd()
    # Get current job output directory
    job_dir = hydra_cfg.runtime.output_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display directories
    print('\nJob directories:')
    print(f'  > Working directory : {cwd}')
    print(f'  > Output directory  : {job_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TBD
    # ...
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute objective to be minimized
    objective = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display parameters
    print('\nParameters:')
    for key, val in cfg.items():
        print(f'  > {key:{max([len(x) for x in cfg.keys()])}} : {val}')
    # Display objective
    print(f'\nFunction evaluation:')
    print(f'  > Objective : {objective}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set summary data
    summary_data = {}
    summary_data['sweeper'] = sweeper
    summary_data['sweeper_optimizer'] = sweeper_optimizer
    for key, val in cfg.items():
        summary_data[key] = val
    summary_data['objective'] = objective
    # Write summary file
    write_summary_file(summary_directory=job_dir,
                       filename='job_summary',
                       summary_title='Hydra - Hyperparameter Optimization Job',
                       **summary_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return objective
# =============================================================================
if __name__ == "__main__":
    hydra_optimize_gnn_model()