"""DARPA METALS PROJECT: Local training of hybrid material model.

Functions
---------
perform_model_standard_training
    Perform standard training of hybrid material model.
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
from hybrid_base_model.train.training import train_model
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
    """Perform standard training of hybrid constitutive material model.
    
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
    # Set default model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
            is_early_stopping, early_stopping_kwargs, is_loss_normalization = \
                set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model initialization parameters
    model_init_args = {'model_directory': model_directory,
                       'model_name': 'hybrid_material_model',
                       'is_data_normalization': True,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data features for training
    features_option = 'stress'
    if features_option == 'stress':
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path',)
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path',)
        # Set number of input and output features
        n_features_in = 6
        model_init_args['n_features_in'] = n_features_in
        n_features_out = 6
        model_init_args['n_features_out'] = n_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybridized material models names
    hyb_models_names = []
    # Initialize hybridized material models initialization attributes
    hyb_models_init_args = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set candidate constitutive model name
    candidate_model_name = 'rc_drucker_prager_vmap'
    # Get physics-based candidate constitutive model initialization attributes
    candidate_model_init_args = get_candidate_model_init_args(
        candidate_model_name, model_directory, n_features_in, n_features_out,
        device_type=device_type)
    # Store candidate constitutive model
    hyb_models_names.append(candidate_model_name)
    hyb_models_init_args[candidate_model_name] = candidate_model_init_args
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridization scheme
    hybridization_scheme = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridization scheme transfer-learning and residual models
    if hybridization_scheme == 0:
        # Build transfer-learning model parameters
        tl_models_names = {}
        tl_models_init_args = {}
        is_tl_residual_connection = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set residual model name
        residual_model_name = 'gru_material_model'
        # Get residual model initialization attributes
        residual_model_init_args = get_gru_model_init_args(
            residual_model_name, model_directory, n_features_in,
            n_features_out, device_type=device_type)
        # Store residual model
        hyb_models_names.append(residual_model_name)
        hyb_models_init_args[residual_model_name] = residual_model_init_args
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'additive'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif hybridization_scheme == 1:
        # Set transfer-learning model name
        tl_model_name = 'gru_transfer_learning_model'
        # Set transfer-learning model residual connection
        is_residual_connection = True
        # Set transfer-learning model number of features
        if is_residual_connection:
            tl_n_features_in = n_features_in + n_features_out
            tl_n_features_out = n_features_out
        else:
            tl_n_features_in = n_features_out
            tl_n_features_out = n_features_out
        # Get transfer-learning model initialization attributes
        tl_model_init_args = get_gru_model_init_args(
            tl_model_name, model_directory, tl_n_features_in,
            tl_n_features_out, device_type=device_type)
        # Build transfer-learning model parameters
        tl_models_names = {candidate_model_name: tl_model_name}
        tl_models_init_args = {tl_model_name: tl_model_init_args}
        is_tl_residual_connection = {tl_model_name: is_residual_connection}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set residual model name
        residual_model_name = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'identity'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown hybridization scheme.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store hybridized material models
    model_init_args['hyb_models_names'] = hyb_models_names
    model_init_args['hyb_models_init_args'] = hyb_models_init_args
    # Store transfer-learning models
    model_init_args['tl_models_names'] = tl_models_names
    model_init_args['tl_models_init_args'] = tl_models_init_args
    model_init_args['is_tl_residual_connection'] = is_tl_residual_connection
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model training options:
    # Set number of epochs
    n_max_epochs = 200
    # Set batch size
    batch_size = 32
    # Set learning rate
    lr_init = 1.0e-03
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute exponential decay (learning rate scheduler)
    lr_end = 1.0e-5
    gamma = (lr_end/lr_init)**(1/n_max_epochs)
    # Set learning rate scheduler
    lr_scheduler_type = 'explr'
    lr_scheduler_kwargs = {'gamma': gamma}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set early stopping
    is_early_stopping = True
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display summary of PyTorch model
    _ = get_model_summary(model, device_type=device_type,
                          is_verbose=is_verbose)
# =============================================================================
def get_candidate_model_init_args(model_name, model_directory, n_features_in,
                                  n_features_out, device_type='cpu'):
    """Set physics-based candidate constitutive model parameters.
    
    Parameters
    ----------
    model_name : str
        Model name.
    model_directory : str
        Directory where model is stored.
    n_features_in : int
        Number of input features.
    n_features_out : int
        Number of output features.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Model class initialization parameters (check
        RecurrentConstitutiveModel).
    """
    # Set learnable parameters
    learnable_parameters = {}
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numerical example
    example = 'learning_drucker_prager_pressure_dependency'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set example candidate model
    if example == 'erroneous_von_mises_properties':
        # Set material constitutive model name
        material_model_name = 'von_mises_vmap'
        # Set best or worst candidate model
        is_worst_candidate = True
        # Set material constitutive model parameters
        if is_worst_candidate:
            # Worst candidate model
            material_model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 80e3, 'v': 0.30,
                 'euler_angles': (0.0, 0.0, 0.0),
                 'hardening_law': get_hardening_law('linear'),
                 'hardening_parameters': {'s0': 400,
                                          'a': 300}}
        else:
            # Best candidate model
            material_model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 100e3, 'v': 0.33,
                 'euler_angles': (0.0, 0.0, 0.0),
                 'hardening_law': get_hardening_law('nadai_ludwik'),
                 'hardening_parameters': {'s0': 700,
                                          'a': 600,
                                          'b': 0.5,
                                          'ep0': 1e-5}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == 'learning_von_mises_hardening':
        # Set material constitutive model name
        material_model_name = 'von_mises_mixed_vmap'
        # Set material constitutive model parameters
        material_model_parameters = \
            {'elastic_symmetry': 'isotropic',
             'E': 110e3, 'v': 0.33,
             'euler_angles': (0.0, 0.0, 0.0),
             'hardening_law': get_hardening_law('linear'),
             'hardening_parameters': {'s0': 900,
                                      'a': 0},
             'kinematic_hardening_law': get_hardening_law('linear'),
             'kinematic_hardening_parameters': {'s0': 0,
                                                'a': 0}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif example == 'learning_drucker_prager_pressure_dependency':
        # Set material constitutive model name
        material_model_name = 'drucker_prager_vmap'
        # Set frictional angle
        friction_angle = np.deg2rad(5)
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown numerical example.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material constitutive state variables (prediction)
    state_features_out = {}
    # Set parameters normalization
    is_normalized_parameters = True
    # Set data normalization
    is_data_normalization = False
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
                       'is_data_normalization': is_data_normalization,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
def get_gru_model_init_args(model_name, model_directory, n_features_in,
                            n_features_out, device_type='cpu'):
    """Set GRU recurrent neural network model parameters.
    
    Parameters
    ----------
    model_name : str
        Model name.
    model_directory : str
        Directory where model is stored.
    n_features_in : int
        Number of input features.
    n_features_out : int
        Number of output features.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Model class initialization parameters (check GRURNNModel).
    """
    # Set hidden layer size
    hidden_layer_size = 481
    # Set number of recurrent layers (stacked RNN)
    n_recurrent_layers = 2
    # Set dropout probability
    dropout = 0
    # Set data normalization
    is_data_normalization = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'n_features_in': n_features_in,
                       'n_features_out': n_features_out,
                       'hidden_layer_size': hidden_layer_size,
                       'n_recurrent_layers': n_recurrent_layers,
                       'dropout': dropout,
                       'model_directory': model_directory,
                       'model_name': model_name,
                       'is_data_normalization': is_data_normalization,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
def get_poly_regressor_model_init_args(n_features_in, device_type='cpu'):
    """Set polynomial linear regression model parameters.
    
    Parameters
    ----------
    n_features_in : int
        Number of input features.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Model class initialization parameters
        (check PolynomialLinearRegressor).
    """
    # Set degree of polynomial linear regression model
    poly_degree = 1
    # Set bias
    is_bias = True
    # Set data normalization
    is_data_normalization = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'n_features_in': n_features_in,
                       'poly_degree': poly_degree,
                       'is_bias': is_bias,
                       'is_data_normalization': is_data_normalization,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
def generate_standard_training_plots(model_directory):
    """Generate plots of standard training of model.
    
    Parameters
    ----------
    model_directory : str
        Directory where material patch model is stored.
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
    is_loss_normalization : bool
        If True, then output features are normalized for loss computation,
        False otherwise. Ignored if model is_data_normalization is set to True.
        The model data scalers are fitted and employed to normalize the
        output features.
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
    is_loss_normalization = True
    is_sampler_shuffle = False
    is_early_stopping = True
    early_stopping_kwargs = {'validation_dataset': None,
                             'validation_frequency': 1,
                             'trigger_tolerance': 20,
                             'improvement_tolerance':1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
        is_early_stopping, early_stopping_kwargs, is_loss_normalization
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_standard_training = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/7_local_hybrid_training/'
                'case_erroneous_von_mises_properties')
    # Set case study directory
    case_study_name = '2_hybrid_rc_von_mises_plus_gru_model'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Set training data set directory
    training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                        '1_training_dataset')
    # Get training data set file path
    regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
    is_file_found, train_dataset_file_path = \
        find_unique_file_with_regex(training_dataset_dir, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Training data set file has not been found  '
                           f'in data set directory:\n\n'
                           f'{training_dataset_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir), '3_model')
    # Create model directory
    if is_standard_training:
        # Create model directory (overwrite)
        make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            raise RuntimeError(f'Validation data set file has not been found  '
                               f'in data set directory:\n\n'
                               f'{val_dataset_directory}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform standard training of model
    if is_standard_training:
        perform_model_standard_training(
            train_dataset_file_path, model_directory,
            val_dataset_file_path=val_dataset_file_path,
            device_type=device_type, is_verbose=True)