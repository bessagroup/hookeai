"""Hydra hyperparameter optimization.

Execute (multi-run mode):

    $ python3 hydra_optimization.py -m

Functions
---------
hydra_wrapper
    Wrapper of Hydra hyperparameter optimization main function.
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
import re
import shutil
# Third-party
import hydra
import torch
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from user_scripts.global_model_update.material_finder.gen_specimen_data \
    import gen_specimen_dataset, get_specimen_history_paths
from user_scripts.global_model_update.material_finder \
    .gen_specimen_local_paths import gen_specimen_local_dataset
from model_architectures.rnn_base_model.optimization.\
    hydra_optimization_template import display_hydra_job_header
from ioput.iostandard import make_directory, write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def hydra_wrapper(specimen_name, src_simulation_dir, device_type='cpu'):
    """Wrapper of Hydra hyperparameter optimization main function.
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    src_simulation_dir : str
        Source simulation data directory.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    """
    # Set Hydra main function
    @hydra.main(version_base=None, config_path='.', config_name='hydra_config')
    def hydra_optimizer(cfg):
        """Hydra hyperparameter optimization.
        
        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Configuration dictionary of YAML based hierarchical configuration
            system.
            
        Returns
        -------
        objective : float
            Objective to minimize.
        """
        # Get Hydra configuration singleton
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display Hydra hyperparameter optimization job header
        sweeper, sweeper_optimizer, job_dir = \
            display_hydra_job_header(hydra_cfg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set case study directory
        case_study_name = 'material_model_finder'
        case_study_dir = \
            os.path.join(os.path.normpath(job_dir), f'{case_study_name}')
        # Set specimen raw data directory
        specimen_raw_dir = \
            os.path.join(os.path.normpath(case_study_dir), '0_simulation')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Copy source simulation data directory
        shutil.copytree(src_simulation_dir, specimen_raw_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen mesh input file path (.inp file)
        specimen_inp_path = os.path.join(os.path.normpath(specimen_raw_dir),
                                         f'{specimen_name}.inp')
        # Set specimen history data directory
        specimen_history_dir = os.path.join(os.path.normpath(specimen_raw_dir),
                                            'specimen_history_data')
        # Get specimen history time step file paths
        specimen_history_paths = \
            get_specimen_history_paths(specimen_history_dir, specimen_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '1_training_dataset')
        # Create training data set directory
        if not os.path.isdir(training_dataset_dir):
            make_directory(training_dataset_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material model name and parameters
        strain_formulation, problem_type, n_dim, model_name, \
            model_parameters, model_kwargs = set_material_model_parameters(cfg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material models initialization subdirectory
        material_models_dir = os.path.join(
            os.path.normpath(training_dataset_dir), 'material_models_init')
        # Create material models initialization subdirectory
        make_directory(material_models_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material model directory
        material_model_dir = os.path.join(
            os.path.normpath(material_models_dir), 'model_1')
        # Create material model directory
        make_directory(material_model_dir, is_overwrite=True)
        # Assign material model directory
        model_kwargs['model_directory'] = material_model_dir
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate specimen data and material state files (training data set)
        specimen_data_path, specimen_material_state_path = \
            gen_specimen_dataset(
                specimen_name, specimen_raw_dir, specimen_inp_path,
                specimen_history_paths, strain_formulation, problem_type,
                n_dim, model_name, model_parameters, model_kwargs,
                training_dataset_dir, is_save_specimen_data=True,
                is_save_specimen_material_state=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = os.path.join(
            os.path.normpath(case_study_dir), '3_model')
        # Create model directory
        if not os.path.isdir(model_directory):
            make_directory(model_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate specimen local strain-stress paths data set
        _, force_equilibrium_hist_loss = gen_specimen_local_dataset(
            specimen_data_path, specimen_material_state_path, model_directory,
            is_plot_dataset=False, device_type=device_type, is_verbose=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove simulation data directory
        shutil.rmtree(specimen_raw_dir)
        # Remove training data set directory
        shutil.rmtree(training_dataset_dir)
        # Remove specimen local strain-stress paths data set directory
        shutil.rmtree(os.path.join(os.path.normpath(model_directory),
                                   'local_response_dataset'))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set objective as force equilibrium loss
        objective = force_equilibrium_hist_loss.item()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display parameters
        print('\nParameters:')
        for key, val in cfg.items():
            print(f'  > {key:{max([len(x) for x in cfg.keys()])}} : {val}')
        # Display objective
        print(f'\nFunction evaluation:')
        print(f'  > Objective : {objective:.8e}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set summary data
        summary_data = {}
        summary_data['sweeper'] = sweeper
        summary_data['sweeper_optimizer'] = sweeper_optimizer
        for key, val in cfg.items():
            summary_data[key] = val
        summary_data['objective'] = f'{objective:.8e}'
        # Write summary file
        write_summary_file(
            summary_directory=job_dir,
            filename='job_summary',
            summary_title='Hydra - Hyperparameter Optimization Job',
            **summary_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return objective
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_material_model_parameters(cfg):
        """Set material model parameters.

        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Configuration dictionary of YAML based hierarchical configuration
            system.

        Returns
        -------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        n_dim : int
            Number of spatial dimensions.
        model_name : str
            Material constitutive model name.
        model_parameters : dict
            Material constitutive model parameters.
        model_kwargs : dict, default={}
            Other parameters required to initialize constitutive model.
        """
        # Set strain formulation
        strain_formulation = 'infinitesimal'
        # Set problem type
        problem_type = 4
        # Get problem type parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model name
        model_name = 'rc_von_mises_vmap'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model name parameters
        if bool(re.search(r'^rc_.*$', model_name)):
            # Set constitutive model specific parameters
            if model_name == 'rc_elastic':
                # Set constitutive model parameters
                model_parameters = {
                    'elastic_symmetry': 'isotropic',
                    'E': cfg.E, 'v': cfg.v,
                    'euler_angles': (0.0, 0.0, 0.0)}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set learnable parameters
                learnable_parameters = {}
                # Set material constitutive model name
                material_model_name = 'elastic'
                # Set material constitutive state variables (prediction)
                state_features_out = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif model_name in ('rc_von_mises', 'rc_von_mises_vmap'):            
                # Set constitutive model parameters
                model_parameters = {
                    'elastic_symmetry': 'isotropic',
                    'E': cfg.E, 'v': cfg.v,
                    'euler_angles': (0.0, 0.0, 0.0),
                    'hardening_law': get_hardening_law('nadai_ludwik'),
                    'hardening_parameters':
                        {'s0': cfg.s0,
                         'a': cfg.a,
                         'b': cfg.b,
                         'ep0': 1e-5}}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set learnable parameters
                learnable_parameters = {}                
                # Set material constitutive model name
                if model_name == 'rc_von_mises_vmap':
                    material_model_name = 'von_mises_vmap'
                else:
                    material_model_name = 'von_mises'
                # Set material constitutive state variables (prediction)
                state_features_out = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown recurrent constitutive model.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set automatic synchronization of material model parameters
            if bool(re.search(r'_vmap$', model_name)):
                is_auto_sync_parameters = False
            else:
                is_auto_sync_parameters = True
            # Set state update failure checking flag
            is_check_su_fail = False
            # Set parameters normalization
            is_normalized_parameters = True
            # Set model input and output features normalization
            is_model_in_normalized = False
            is_model_out_normalized = False
            # Set device type
            if torch.cuda.is_available():
                device_type = 'cuda'
            else:
                device_type = 'cpu'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set other parameters required to initialize constitutive model
            model_kwargs = {
                'n_features_in': len(comp_order_sym),
                'n_features_out': len(comp_order_sym),
                'learnable_parameters': learnable_parameters,
                'strain_formulation': strain_formulation,
                'problem_type': problem_type,
                'material_model_name': material_model_name,
                'material_model_parameters': model_parameters,
                'state_features_out': state_features_out,
                'is_auto_sync_parameters': is_auto_sync_parameters,
                'is_check_su_fail': is_check_su_fail,
                'model_directory': None,
                'model_name': model_name,
                'is_normalized_parameters': is_normalized_parameters,
                'is_model_in_normalized': is_model_in_normalized,
                'is_model_out_normalized': is_model_out_normalized,
                'device_type': device_type}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_formulation, problem_type, n_dim, model_name, \
            model_parameters, model_kwargs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Call Hydra main function
    hydra_optimizer()
# =============================================================================
if __name__ == "__main__":
    # Set float precision
    is_double_precision = True
    if is_double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen name
    specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
    # Set source simulation data directory
    src_simulation_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                          'colaboration_antonios/dtp_validation/'
                          '3_dtp1_j2_rowan_data/2_DTP1U_V2_data/'
                          'loss_dirichlet_sets/4_hyperparameter_optimization/'
                          '0_simulation_src')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute Hydra hyperparameter optimization
    hydra_wrapper(specimen_name, src_simulation_dir, device_type=device_type)