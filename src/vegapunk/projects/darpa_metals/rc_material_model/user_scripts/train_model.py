"""DARPA METALS PROJECT: Local training of recurrent constitutive model.

Functions
---------
perform_model_standard_training
    Perform standard training of recurrent constitutive model.
generate_standard_training_plots
    Generate plots of standard training of model.
set_default_model_parameters
    Set default model initialization parameters.
set_default_training_options
    Set default model training options.
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import random
# Third-party
import torch
import numpy as np
# Local
from rnn_base_model.data.time_dataset import load_dataset, \
    concatenate_dataset_features
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
from rc_base_model.train.training import train_model, \
    read_parameters_history_from_file
from rc_base_model.train.training_plots import plot_model_parameters_history
from gnn_base_model.train.training import \
    read_loss_history_from_file, read_lr_history_from_file
from gnn_base_model.model.model_summary import get_model_summary
from gnn_base_model.train.training_plots import plot_training_loss_history, \
    plot_training_loss_and_lr_history
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from ioput.iostandard import make_directory, find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_model_standard_training(train_dataset_file_path, model_directory,
                                    val_dataset_file_path=None,
                                    device_type='cpu', is_verbose=False):
    """Perform standard training of recurrent constitutive model.
    
    Parameters
    ----------
    train_dataset_file_path : str
        Training data set file path.
    model_directory : str
        Directory where model is stored.
    val_dataset_file_path : str, default=None
        Validation data set file path.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """    
    # Get model default initialization parameters
    model_init_args = set_default_model_parameters(model_directory,
                                                   device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model for training
    material_model_name = 'von_mises'
    # Set constitutive model parameters
    if material_model_name == 'von_mises':
        # Set material constitutive model parameters
        material_model_parameters = \
            {'elastic_symmetry': 'isotropic',
             'E': 110e3, 'v': 0.33,
             'euler_angles': (0.0, 0.0, 0.0),
             'hardening_law': get_hardening_law('nadai_ludwik'),
             'hardening_parameters': {'s0': 900,
                                      'a': 700,
                                      'b': 0.5,
                                      'ep0': 1e-5}}
        # Set learnable parameters
        learnable_parameters = {}
        learnable_parameters['E'] = \
            {'initial_value': random.uniform(100e3, 120e3),
             'bounds': (100e3, 120e3)}
        learnable_parameters['v'] = \
            {'initial_value': random.uniform(0.3, 0.4),
             'bounds': (0.3, 0.4)}
        learnable_parameters['s0'] = \
            {'initial_value': random.uniform(800, 1000),
             'bounds': (800, 1000)}
        learnable_parameters['a'] = \
            {'initial_value': random.uniform(500, 900),
             'bounds': (500, 900)}
        learnable_parameters['b'] = \
            {'initial_value': random.uniform(0.3, 0.7),
             'bounds': (0.3, 0.7)}
        # Set material constitutive state variables (prediction)
        state_features_out = {'acc_p_strain': 1,}
    elif material_model_name == 'drucker_prager':
        # Set frictional angle
        friction_angle = np.deg2rad(10)
        # Set dilatancy angle
        dilatancy_angle = friction_angle
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute angle-related material parameters
        # (matching with Mohr-Coulomb under uniaxial tension and compression)
        # Set yield surface cohesion parameter
        yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
        # Set yield pressure parameter
        yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
        # Set plastic flow pressure parameter
        flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material constitutive model parameters
        material_model_parameters = \
            {'elastic_symmetry': 'isotropic',
             'E': 110e3, 'v': 0.33,
             'euler_angles': (0.0, 0.0, 0.0),
             'hardening_law': get_hardening_law('nadai_ludwik'),
             'hardening_parameters': {'s0': 900/yield_cohesion_parameter,
                                      'a': 700/yield_cohesion_parameter,
                                      'b': 0.5,
                                      'ep0': 1e-5},
             'yield_cohesion_parameter': yield_cohesion_parameter,
             'yield_pressure_parameter': yield_pressure_parameter,
             'flow_pressure_parameter': flow_pressure_parameter}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set learnable parameters
        learnable_parameters = {}
        learnable_parameters['E'] = \
            {'initial_value': random.uniform(100e3, 120e3),
             'bounds': (100e3, 120e3)}
        learnable_parameters['v'] = \
            {'initial_value': random.uniform(0.3, 0.4),
             'bounds': (0.3, 0.4)}
        learnable_parameters['s0'] = \
            {'initial_value': random.uniform(800/yield_cohesion_parameter,
                                             1000/yield_cohesion_parameter),
             'bounds': (800/yield_cohesion_parameter,
                        1000/yield_cohesion_parameter)}
        learnable_parameters['a'] = \
            {'initial_value': random.uniform(500/yield_cohesion_parameter,
                                             900/yield_cohesion_parameter),
             'bounds': (500/yield_cohesion_parameter,
                        900/yield_cohesion_parameter)}
        learnable_parameters['b'] = \
            {'initial_value': random.uniform(0.3, 0.7),
             'bounds': (0.3, 0.7)}
        # Set material constitutive state variables (prediction)
        state_features_out = {'acc_p_strain': 1,}
    else:
        # Set constitutive model parameters
        material_model_parameters = \
            {'elastic_symmetry': 'isotropic',
             'E': 110e3, 'v': 0.33,
             'euler_angles': (0.0, 0.0, 0.0)}
        # Set learnable parameters
        learnable_parameters = {}
        learnable_parameters['E'] = \
            {'initial_value': random.uniform(100e3, 120e3),
             'bounds': (100e3, 120e3)}
        learnable_parameters['v'] = \
            {'initial_value': random.uniform(0.3, 0.4),
             'bounds': (0.3, 0.4)}
        # Set constitutive state variables to include in data set
        state_features_out = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update initialization parameters
    model_init_args['material_model_name'] = material_model_name
    model_init_args['material_model_parameters'] = \
        material_model_parameters
    model_init_args['learnable_parameters'] = learnable_parameters
    model_init_args['state_features_out'] = state_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
            is_early_stopping, early_stopping_kwargs = \
                set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data features for training
    features_option = 'stress'
    if features_option == 'stress_acc_p_strain':
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path',)
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path', 'acc_p_strain')
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 7
    else:
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path',)
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path',)
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model training options:
    # Set number of epochs
    n_max_epochs = 200
    # Set batch size
    batch_size = 4
    # Set learning rate
    lr_init = 1.0e+01
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute exponential decay (learning rate scheduler)
    lr_end = 1.0e-1
    gamma = (lr_end/lr_init)**(1/n_max_epochs)
    # Set learning rate scheduler
    lr_scheduler_type = 'explr'
    lr_scheduler_kwargs = {'gamma': gamma}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set early stopping
    is_early_stopping = False
    # Set early stopping parameters
    if is_early_stopping:
        # Check validation data set file path
        if val_dataset_file_path is None:
            raise RuntimeError('The validation data set file path must be '
                               'provided to process early stopping criterion.')
        else:
            # Load validation data set
            val_dataset = load_dataset(val_dataset_file_path)
            # Set validation data set features
            val_dataset = concatenate_dataset_features(
                val_dataset, new_label_in, cat_features_in,
                is_remove_features=True)
            val_dataset = concatenate_dataset_features(
                val_dataset, new_label_out, cat_features_out,
                is_remove_features=True)
        # Set early stopping parameters
        early_stopping_kwargs = {'validation_dataset': val_dataset,
                                 'validation_frequency': 1,
                                 'trigger_tolerance': 20,
                                 'improvement_tolerance': 1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training data set
    train_dataset = load_dataset(train_dataset_file_path)
    # Set training data set features
    train_dataset = concatenate_dataset_features(train_dataset, new_label_in,
                                                 cat_features_in,
                                                 is_remove_features=True)
    train_dataset = concatenate_dataset_features(train_dataset, new_label_out,
                                                 cat_features_out,
                                                 is_remove_features=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Training of recurrent constitutive model
    model, _, _ = train_model(n_max_epochs, train_dataset, model_init_args,
                              lr_init, opt_algorithm=opt_algorithm,
                              lr_scheduler_type=lr_scheduler_type,
                              lr_scheduler_kwargs=lr_scheduler_kwargs,
                              loss_nature=loss_nature, loss_type=loss_type,
                              loss_kwargs=loss_kwargs,
                              is_loss_normalization=is_loss_normalization,
                              batch_size=batch_size,
                              is_sampler_shuffle=is_sampler_shuffle,
                              is_early_stopping=is_early_stopping,
                              early_stopping_kwargs=early_stopping_kwargs,
                              load_model_state=None, save_every=None,
                              dataset_file_path=train_dataset_file_path,
                              device_type=device_type, seed=None,
                              is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of model training process
    generate_standard_training_plots(model_directory)
    # Generate plots of model learnable parameters
    generate_model_parameters_plots(model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display summary of PyTorch model
    _ = get_model_summary(model, device_type=device_type,
                          is_verbose=is_verbose)
# =============================================================================
def generate_standard_training_plots(model_directory):
    """Generate plots of standard training of model.
    
    Parameters
    ----------
    model_directory : str
        Directory where model is stored.
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model_directory, 'loss_history_record.pkl')
    # Read training process training and validation loss history
    loss_nature, loss_type, training_loss_history, validation_loss_history = \
        read_loss_history_from_file(loss_record_path)
    # Build training process loss history
    loss_histories = {}
    loss_histories['Training'] = training_loss_history
    if validation_loss_history is not None:
        loss_histories['Validation'] = validation_loss_history
    # Read training process learning rate history
    lr_scheduler_type, lr_history_epochs = \
        read_lr_history_from_file(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(model_directory), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model training process loss history
    plot_training_loss_history(loss_histories, loss_type.upper(),
                               loss_scale='log', save_dir=plot_dir,
                               is_save_fig=True, is_stdout_display=False,
                               is_latex=True)
    # Plot model training process loss and learning rate histories
    plot_training_loss_and_lr_history(training_loss_history,
                                      lr_history_epochs, loss_type=None,
                                      is_log_loss=False, loss_scale='log',
                                      lr_type=lr_scheduler_type,
                                      save_dir=plot_dir, is_save_fig=True,
                                      is_stdout_display=False, is_latex=True)
# =============================================================================
def generate_model_parameters_plots(model_directory):
    """Generate plots of model learnable parameters.
    
    Parameters
    ----------
    model_directory : str
        Directory where model is stored.
    """
    # Set model parameters record file path
    parameters_record_path = os.path.join(model_directory,
                                          'parameters_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read model learnable parameters history
    model_parameters_history, model_parameters_bounds = \
        read_parameters_history_from_file(parameters_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(model_directory), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model learnable parameters history
    plot_model_parameters_history(model_parameters_history,
                                  model_parameters_bounds, save_dir=plot_dir,
                                  is_save_fig=True, is_stdout_display=False,
                                  is_latex=True)
# =============================================================================
def save_model_init_state(model_init_args):
    """Save model initial state to file.
    
    Parameters
    ----------
    model_init_args : dict
        Model class initialization parameters (check
        RecurrentConstitutiveModel).
    """
    # Initialize recurrent constitutive model
    model = RecurrentConstitutiveModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model initial state
    model.save_model_init_state()
# =============================================================================
def set_default_model_parameters(model_directory, device_type='cpu'):
    """Set default model initialization parameters.
    
    Parameters
    ----------
    model_directory : str
        Directory where model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Model class initialization parameters (check
        RecurrentConstitutiveModel).
    """
    # Set model name
    model_name = 'material_rc_model'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of input features
    n_features_in = 6
    # Set number of output features
    n_features_out = 6
    # Set learnable parameters
    learnable_parameters = {}
    learnable_parameters['E'] = {'initial_value': 100e3,
                                 'bounds': (100e3, 120e3)}
    learnable_parameters['v'] = {'initial_value': 0.3,
                                 'bounds': (0.3, 0.36)}
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Set material constitutive model name
    material_model_name = 'elastic'
    # Set material constitutive model parameters
    material_model_parameters = {'elastic_symmetry': 'isotropic',
                                 'E': 110e3, 'v': 0.33,
                                 'euler_angles': (0.0, 0.0, 0.0)}
    # Set material constitutive state variables (prediction)
    state_features_out = {}
    # Set parameters normalization
    is_normalized_parameters = True
    # Set model input and output features normalization
    is_model_in_normalized = False
    is_model_out_normalized = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'n_features_in': n_features_in,
                       'n_features_out': n_features_out,
                       'learnable_parameters': learnable_parameters,
                       'strain_formulation': strain_formulation,
                       'problem_type': problem_type,
                       'material_model_name': material_model_name,
                       'material_model_parameters': material_model_parameters,
                       'state_features_out': state_features_out,
                       'model_directory': model_directory,
                       'model_name': model_name,
                       'is_normalized_parameters': is_normalized_parameters,
                       'is_model_in_normalized': is_model_in_normalized,
                       'is_model_out_normalized': is_model_out_normalized,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
def set_default_training_options():
    """Set default model training options.
    
    Returns
    -------
    opt_algorithm : {'adam',}
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)

    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
        'explr'   : Exponential decay (torch.optim.lr_scheduler.ExponentialLR)
        
        'linlr'   : Linear decay (torch.optim.lr_scheduler.LinearLR)

    lr_scheduler_kwargs : dict
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    loss_nature : {'features_out',}, default='features_out'
        Loss nature:
        
        'features_out' : Based on output features

    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_sampler_shuffle : bool
        If True, shuffles data set samples at every epoch.
    is_early_stopping : bool
        If True, then training process is halted when early stopping criterion
        is triggered.
    early_stopping_kwargs : dict
        Early stopping criterion parameters (key, str, item, value).
    """
    opt_algorithm = 'adam'
    lr_init = 1.0e-04
    lr_scheduler_type = None
    lr_scheduler_kwargs = None
    loss_nature = 'features_out'
    loss_type = 'mse'
    loss_kwargs = {}
    is_sampler_shuffle = False
    is_early_stopping = True
    early_stopping_kwargs = {'validation_dataset': None,
                             'validation_frequency': 1,
                             'trigger_tolerance': 20,
                             'improvement_tolerance': 1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
        is_early_stopping, early_stopping_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_standard_training = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/7_local_hybrid_training/'
                'case_learning_drucker_prager_pressure_dependency/'
                'w_candidate_dp_model_1deg/'
                '2_candidate_rc_drucker_prager_model')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set training data set sizes
    training_sizes = (10,)
    # Loop over training data set sizes
    for n in training_sizes:
        # Set case study directory
        case_study_name = f'n{n}'
        case_study_dir = os.path.join(os.path.normpath(base_dir),
                                      f'{case_study_name}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check case study directory
        if not os.path.isdir(case_study_dir):
            raise RuntimeError('The case study directory has not been found:'
                               '\n\n' + case_study_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '1_training_dataset')
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(training_dataset_dir, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found '
                               f'in data set directory:\n\n'
                               f'{training_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = \
            os.path.join(os.path.normpath(case_study_dir), '3_model')
        # Create model directory
        if is_standard_training:
            # Create model directory (overwrite)
            make_directory(model_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set validation data set directory
        val_dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '2_validation_dataset')
        # Get validation data set file path
        val_dataset_file_path = None
        if os.path.isdir(val_dataset_directory):
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, val_dataset_file_path = \
                find_unique_file_with_regex(val_dataset_directory, regex)
            # Check data set file
            if not is_file_found:
                raise RuntimeError(f'Validation data set file has not been '
                                   f'found in data set directory:\n\n'
                                   f'{val_dataset_directory}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform standard training of model
        if is_standard_training:
            perform_model_standard_training(
                train_dataset_file_path, model_directory,
                val_dataset_file_path=val_dataset_file_path,
                device_type=device_type, is_verbose=True)