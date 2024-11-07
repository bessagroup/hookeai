"""DARPA METALS PROJECT: Local training of RNN material model.

Functions
---------
perform_model_standard_training
    Perform standard training of RNN-based model.
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
from rnn_base_model.data.time_dataset import load_dataset, \
    concatenate_dataset_features, sum_dataset_features, \
    add_dataset_feature_init
from rnn_base_model.train.training import train_model
from gnn_base_model.train.training import \
    read_loss_history_from_file, read_lr_history_from_file
from gnn_base_model.model.model_summary import get_model_summary
from gnn_base_model.train.training_plots import plot_training_loss_history, \
    plot_training_loss_and_lr_history
from ioput.iostandard import make_directory, find_unique_file_with_regex
from projects.darpa_metals.rnn_material_model.rnn_model_tools.strain_features \
    import add_strain_features
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
    """Perform standard training of RNN-based model.
    
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
    # Set default model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
            is_early_stopping, early_stopping_kwargs = \
                set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize new strain-based feature flags
    strain_features_labels = None
    # Initialize features concatenation/summing flags
    is_cat_features_in = False
    is_cat_features_out = False
    is_sum_features_in = False
    is_sum_features_out = False
    # Set data features for training
    features_option = 'strain_to_stress'
    if features_option == 'stress_acc_p_strain':
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path',)
        is_cat_features_in = True
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path', 'acc_p_strain')
        is_cat_features_out = True
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 7
    elif features_option == 'strain_vf_to_stress':
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path', 'vf_path')
        is_cat_features_in = True
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path',)
        is_cat_features_out = True
        # Set number of input and output features
        model_init_args['n_features_in'] = 7
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_to_p_strain':
        # Set input features
        new_label_in = 'features_in'
        sum_features_in = ('strain_path',)
        features_in_weights = {'strain_path': 1.0,}
        is_sum_features_in = True
        # Set output features
        new_label_out = 'features_out'
        sum_features_out = ('strain_path', 'e_strain_mf')
        features_out_weights = {'strain_path': 1.0, 'e_strain_mf': -1.0}
        is_sum_features_out = True
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_i1_i2_to_stress':
        # Set new strain-based features labels
        strain_features_labels = ('i1', 'i2')
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path', *strain_features_labels)
        is_cat_features_in = True
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path',)
        is_cat_features_out = True
        # Set number of input and output features
        model_init_args['n_features_in'] = 8
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_to_stress':
        # Set input features
        new_label_in = 'features_in'
        cat_features_in = ('strain_path',)
        is_cat_features_in = True
        # Set output features
        new_label_out = 'features_out'
        cat_features_out = ('stress_path',)
        is_cat_features_out = True
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 6
    else:
        raise RuntimeError('Unknown features option.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hidden state initialization
    hidden_features_in = torch.zeros((model_init_args['n_recurrent_layers'],
                                      model_init_args['hidden_layer_size']))
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
            # Compute new strain-based features
            if strain_features_labels is not None:
                # Loop over strain-based features
                for strain_feature_label in strain_features_labels:
                    # Add strain-based feature to data set
                    val_dataset = add_strain_features(
                        val_dataset, strain_feature_label)
            # Set validation data set features
            if is_cat_features_in:
                val_dataset = concatenate_dataset_features(
                    val_dataset, new_label_in, cat_features_in,
                    is_remove_features=False)
            elif is_sum_features_in:
                val_dataset = sum_dataset_features(
                    val_dataset, new_label_in, sum_features_in,
                    features_weights=features_in_weights,
                    is_remove_features=False)
            if is_cat_features_out:
                val_dataset = concatenate_dataset_features(
                    val_dataset, new_label_out, cat_features_out,
                    is_remove_features=False)
            elif is_sum_features_out:
                val_dataset = sum_dataset_features(
                    val_dataset, new_label_out, sum_features_out,
                    features_weights=features_out_weights,
                    is_remove_features=False)
            # Add hidden state initialization to data set
            val_dataset = add_dataset_feature_init(
                val_dataset, 'hidden_features_in', hidden_features_in)
        # Set early stopping parameters
        early_stopping_kwargs = {'validation_dataset': val_dataset,
                                 'validation_frequency': 1,
                                 'trigger_tolerance': 20,
                                 'improvement_tolerance':1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training data set
    train_dataset = load_dataset(train_dataset_file_path)
    # Compute new strain-based features
    if strain_features_labels is not None:
        # Loop over strain-based features
        for strain_feature_label in strain_features_labels:
            # Add strain-based feature to data set
            train_dataset = add_strain_features(
                train_dataset, strain_feature_label)
    # Set training data set features
    if is_cat_features_in:
        train_dataset = concatenate_dataset_features(
            train_dataset, new_label_in, cat_features_in,
            is_remove_features=False)
    elif is_sum_features_in:
        train_dataset = sum_dataset_features(
            train_dataset, new_label_in, sum_features_in,
            features_weights=features_in_weights, is_remove_features=False)
    if is_cat_features_out:
        train_dataset = concatenate_dataset_features(
            train_dataset, new_label_out, cat_features_out,
            is_remove_features=False)
    elif is_sum_features_out:
        train_dataset = sum_dataset_features(
            train_dataset, new_label_out, sum_features_out,
            features_weights=features_out_weights, is_remove_features=False)
    # Add hidden state initialization to data set
    train_dataset = add_dataset_feature_init(
        train_dataset, 'hidden_features_in', hidden_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss type
    loss_type = 'mre'
    # Set loss parameters
    loss_kwargs = {}
    # Set model state loading
    load_model_state = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Training of RNN-based model
    model, _, _ = train_model(n_max_epochs, train_dataset, model_init_args,
                              lr_init, opt_algorithm=opt_algorithm,
                              lr_scheduler_type=lr_scheduler_type,
                              lr_scheduler_kwargs=lr_scheduler_kwargs,
                              loss_nature=loss_nature, loss_type=loss_type,
                              loss_kwargs=loss_kwargs, batch_size=batch_size,
                              is_sampler_shuffle=is_sampler_shuffle,
                              is_early_stopping=is_early_stopping,
                              early_stopping_kwargs=early_stopping_kwargs,
                              load_model_state=load_model_state,
                              save_every=None,
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
        Model class initialization parameters (check GRURNNModel).
    """
    # Set model name
    model_name = 'gru_material_model'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of input features
    n_features_in = 6
    # Set number of output features
    n_features_out = 6
    # Set hidden layer size
    hidden_layer_size = 500
    # Set number of recurrent layers (stacked RNN)
    n_recurrent_layers = 2
    # Set dropout probability
    dropout = 0
    # Set model input and output features normalization
    is_model_in_normalized = True
    is_model_out_normalized = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'n_features_in': n_features_in,
                       'n_features_out': n_features_out,
                       'hidden_layer_size': hidden_layer_size,
                       'n_recurrent_layers': n_recurrent_layers,
                       'dropout': dropout,
                       'model_directory': model_directory,
                       'model_name': model_name,
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
                             'improvement_tolerance':1e-2}
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
                'case_learning_drucker_prager_pressure/2_vanilla_gru_model/'
                'strain_to_stress/mean_relative_error/training_random')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize case study directories
    case_study_dirs = []
    # Set case study directories
    if True:
        # Set training data set sizes
        training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        # Set case study directories
        case_study_dirs += [os.path.join(os.path.normpath(base_dir), f'n{n}/')
                            for n in training_sizes]
    elif True:
        case_study_dirs += [os.path.join(os.path.normpath(base_dir),
                                         f'n2560/0_pretraining/'),]
    else:
        case_study_dirs += [base_dir,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over case study directories
    for case_study_dir in case_study_dirs:
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
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, val_dataset_file_path = \
            find_unique_file_with_regex(val_dataset_directory, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Validation data set file has not been found '
                               f'in data set directory:\n\n'
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