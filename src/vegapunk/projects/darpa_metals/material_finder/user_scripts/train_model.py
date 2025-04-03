"""DARPA METALS PROJECT: Learning procedure of material model finder.

Functions
---------
perform_model_standard_training
    Perform standard training of material model finder.
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
# Third-party
import torch
# Local
from material_model_finder.train.training import train_model
from rc_base_model.train.training import read_parameters_history_from_file
from rc_base_model.train.training_plots import plot_model_parameters_history
from gnn_base_model.train.training import \
    read_loss_history_from_file, read_lr_history_from_file
from gnn_base_model.model.model_summary import get_model_summary
from gnn_base_model.train.training_plots import plot_training_loss_history, \
    plot_training_loss_and_lr_history
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_model_standard_training(specimen_data_path,
                                    specimen_material_state_path,
                                    model_directory,
                                    device_type='cpu', is_verbose=False):
    """Perform standard training of material model finder.
    
    Parameters
    ----------
    specimen_data_path : str
        Path of file containing the specimen numerical data translated from
        experimental results.
    specimen_material_state_path : str
        Path of file containing the FETorch specimen material state.
    model_directory : str
        Directory where model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """    
    # Get model default initialization parameters
    model_init_args = set_default_model_parameters(model_directory,
                                                   device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        is_params_stopping, params_stopping_kwargs = \
            set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model training options:
    # Set number of epochs
    n_max_epochs = 300
    # Set learning rate
    lr_init = 1.0e+01
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute exponential decay (learning rate scheduler)
    lr_end = 1.0e-01
    # Set learning rate scheduler
    lr_scheduler_type = ('explr', 'linlr')[1]
    # Set learning rate scheduler parameters
    if lr_scheduler_type == 'explr':
        gamma = (lr_end/lr_init)**(1/n_max_epochs)
        lr_scheduler_kwargs = {'gamma': gamma}
    elif lr_scheduler_type == 'linlr':
        end_factor = (lr_end/lr_init)
        lr_scheduler_kwargs = {'start_factor': 1.0,
                               'end_factor': end_factor,
                               'total_iters': n_max_epochs}
    else:
        raise RuntimeError('Unknown learning rate scheduler.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load specimen numerical data
    specimen_data = torch.load(specimen_data_path)
    # Load specimen material state
    specimen_material_state = torch.load(specimen_material_state_path)
    # Update material models device
    specimen_material_state.update_material_models_device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss scaling flag
    is_loss_scaling_factor = True
    # Set loss scaling factor
    if is_loss_scaling_factor:
        # Set characteristic Young modulus (MPa)
        young_ref = 100e3
        # Set characteristic length (mm)
        length_ref = 1.0
        # Set characteristic time (s)
        time_ref = 1.0
        # Set loss scaling factor
        loss_scaling_factor = 1.0/((young_ref**2)*(length_ref**4)*time_ref)
    else:
        # Set loss scaling factor
        loss_scaling_factor = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss time weights flag
    is_loss_time_weights = False
    # Set loss time weights
    if is_loss_time_weights:
        loss_time_weights = None
    else:
        loss_time_weights = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Discovery of material model
    model, _, _ = train_model(n_max_epochs, specimen_data,
                              specimen_material_state, model_init_args,
                              lr_init, opt_algorithm=opt_algorithm,
                              lr_scheduler_type=lr_scheduler_type,
                              lr_scheduler_kwargs=lr_scheduler_kwargs,
                              is_explicit_model_parameters=True,
                              is_params_stopping=is_params_stopping,
                              params_stopping_kwargs=params_stopping_kwargs,
                              loss_scaling_factor=loss_scaling_factor,
                              loss_time_weights=loss_time_weights,
                              save_every=None, device_type=device_type,
                              seed=None, is_verbose=is_verbose)
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
def set_default_model_parameters(model_directory, device_type='cpu'):
    """Set default model initialization parameters.
    
    Parameters
    ----------
    model_directory : str
        Directory where material patch model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Material model finder class initialization parameters (check class
        MaterialModelFinder).
    """
    # Set model name
    model_name = 'material_model_finder'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'model_directory': model_directory,
                       'model_name': model_name,
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
    is_params_stopping : bool
        If True, then training process is halted when parameters convergence
        criterion is triggered.
    params_stopping_kwargs : dict
        Parameters convergence stopping criterion parameters.
    """
    opt_algorithm = 'adam'
    lr_init = 1.0e-04
    lr_scheduler_type = None
    lr_scheduler_kwargs = None
    is_params_stopping = True
    params_stopping_kwargs = {'convergence_tolerance': 0.01,
                              'trigger_tolerance': 5,
                              'min_hist_length': 50}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        is_params_stopping, params_stopping_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set float precision
    is_double_precision = True
    if is_double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case study base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/5_global_specimens/'
                'rowan_specimen_tension_bv_hexa8_rc_von_mises_vmap/'
                '2_learning_parameters/4_elastoplastic_properties_E_v_s0_a')
    # Set case study directory
    case_study_name = 'material_model_finder'
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
    # Set specimen numerical data file path
    specimen_data_path = os.path.join(os.path.normpath(training_dataset_dir),
                                      'specimen_data.pt')
    # Check specimen numerical data file
    if not os.path.isfile(specimen_data_path):
        raise RuntimeError(f'The specimen numerical data file ('
                           f'specimen_data.pt) has not been found in data set '
                           f'directory:\n\n'
                           f'{training_dataset_dir}')
    # Set specimen material state file path
    specimen_material_state_path = \
        os.path.join(os.path.normpath(training_dataset_dir),
                                      'specimen_material_state.pt')
    # Check specimen material state file
    if not os.path.isfile(specimen_material_state_path):
        raise RuntimeError(f'The specimen material state file ('
                           f'specimen_material_state.pt) has not been found '
                           f'in data set directory:\n\n'
                           f'{training_dataset_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir), '3_model')
    # Create model directory (overwrite)
    make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform standard training of model
    perform_model_standard_training(
        specimen_data_path, specimen_material_state_path, model_directory,
        device_type=device_type, is_verbose=True)