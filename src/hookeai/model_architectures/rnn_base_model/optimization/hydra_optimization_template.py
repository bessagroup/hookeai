"""Illustrative benchmark of Hydra hyperparameter optimization.

Execute (multi-run mode):

    $ python3 optimization_template.py -m

Functions
---------
dummy_function
    Dummy function to be minimized.
display_hydra_job_header(hydra_cfg)
    Display Hydra hyperparameter optimization job header.
"""
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
# Third-party
import hydra
# Local
from ioput.iostandard import write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
@hydra.main(version_base=None, config_path='.',
            config_name='hydra_optimization_template')
def dummy_function(cfg):
    """Dummy function to be minimized.
    
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
    # Display Hydra hyperparameter optimization job header
    sweeper, sweeper_optimizer, job_dir = display_hydra_job_header(hydra_cfg)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get batch size
    batch_size = cfg.batch_size
    # Get learning rate
    lr = cfg.lr
    # Get learning rate scheduler
    lr_scheduler_type = cfg.lr_scheduler_type
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute objective to be minimized
    objective = abs(batch_size - 4) + abs(lr - 0.25) + len(lr_scheduler_type)
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
def display_hydra_job_header(hydra_cfg):
    """Display Hydra hyperparameter optimization job header.
    
    Parameters
    ----------
    hydra_cfg : hydra.core.singleton.Singleton
        Hydra configuration.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    sweeper : str
        Hydra hyperparameter optimization process sweeper.
    sweeper_optimizer : str
        Hydra hyperparameter optimization process optimization algorithm.
    job_dir : str
        Hydra hyperparameter optimization job output directory.
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
    return sweeper, sweeper_optimizer, job_dir
# =============================================================================
if __name__ == "__main__":
    dummy_function()