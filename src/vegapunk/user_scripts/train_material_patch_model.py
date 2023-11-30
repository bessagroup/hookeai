"""User script: Train GNN-based material patch model."""
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
import torch
# Local
from gnn_model.gnn_patch_dataset import GNNMaterialPatchDataset
from gnn_model.training import train_model, read_loss_history_from_file, \
    read_lr_history_from_file
from gnn_model.cross_validation import kfold_cross_validation
from gnn_model.model_summary import get_model_summary
from gnn_model.evaluation_metrics import plot_training_loss_history, \
    plot_kfold_cross_validation, plot_training_loss_and_lr_history
from ioput.iostandard import make_directory, find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def perform_model_standard_training(case_study_name, dataset_file_path,
                                    model_directory, device_type='cpu',
                                    is_verbose=False):
    """Perform standard training of GNN-based material patch model.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    dataset_file_path : str
        GNN-based material patch training data set file path.        
    model_directory : str
        Directory where material patch model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """    
    # Get model initialization parameters
    model_init_args = set_case_study_model_parameters(case_study_name,
                                                      model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based material patch model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_type, loss_kwargs, is_sampler_shuffle, is_early_stopping, \
            early_stopping_kwargs = set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model training options
    if case_study_name == 'cs_2d_elastic':
        # Set number of training steps
        n_train_steps = 500
        # Set batch size
        batch_size = 16
        # Set learning rate
        lr_init = 1.0e-03
        # Set learning rate scheduler
        lr_scheduler_type = 'explr'
        lr_scheduler_kwargs = {'gamma': 0.995}
        # Set early stopping
        is_early_stopping = True
        early_stopping_kwargs = {'validation_size': 0.2,
                                 'validation_frequency': 0.01*n_train_steps,
                                 'trigger_tolerance': 10}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch training data set
    dataset = GNNMaterialPatchDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Training of GNN-based material patch model
    model, _, _ = train_model(n_train_steps, dataset, model_init_args, lr_init,
                              opt_algorithm=opt_algorithm,
                              lr_scheduler_type=lr_scheduler_type,
                              lr_scheduler_kwargs=lr_scheduler_kwargs,
                              loss_type=loss_type, loss_kwargs=loss_kwargs,
                              batch_size=batch_size,
                              is_sampler_shuffle=is_sampler_shuffle,
                              is_early_stopping=is_early_stopping,
                              early_stopping_kwargs=early_stopping_kwargs,
                              load_model_state=None, save_every=None,
                              dataset_file_path=dataset_file_path,
                              device_type=device_type, seed=None,
                              is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record.pkl')
    # Read training process training and validation loss history
    loss_type, training_loss_history, validation_loss_history = \
        read_loss_history_from_file(loss_record_path)
    # Build training process loss history
    loss_histories = {}
    loss_histories['Training'] = training_loss_history
    if validation_loss_history is not None:
        loss_histories['Validation'] = validation_loss_history
    # Read training process learning rate history
    lr_scheduler_type, lr_history = read_lr_history_from_file(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(model_directory), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model training process loss history
    plot_training_loss_history(loss_histories, loss_type.upper(),
                               loss_scale='linear', save_dir=plot_dir,
                               is_save_fig=True, is_stdout_display=False,
                               is_latex=True)
    # Plot model training process loss and learning rate histories
    plot_training_loss_and_lr_history(training_loss_history, lr_history,
                                      loss_type=None, is_log_loss=False,
                                      loss_scale='linear',
                                      lr_type=lr_scheduler_type,
                                      save_dir=plot_dir, is_save_fig=True,
                                      is_stdout_display=False, is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display summary of PyTorch model
    _ = get_model_summary(model, device_type=device_type,
                          is_verbose=is_verbose)
# =============================================================================
def perform_model_kfold_cross_validation(case_study_name, dataset_file_path,
                                         model_directory, cross_validation_dir,
                                         device_type='cpu', is_verbose=False):
    """Perform k-fold cross validation of GNN-based material patch model.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    dataset_file_path : str
        GNN-based material patch training data set file path.
    model_directory : str
        Directory where material patch model is stored.
    cross_validation_dir : dir
        Directory where cross-validation process data is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Get model initialization parameters
    model_init_args = set_case_study_model_parameters(
        case_study_name, model_directory, device_type=device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based material patch model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_type, loss_kwargs, is_sampler_shuffle, is_early_stopping, \
        early_stopping_kwargs = set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model training options
    if case_study_name == 'cs_2d_elastic':
        # Set number of training steps
        n_train_steps = 500
        # Set batch size
        batch_size = 16
        # Set learning rate
        lr_init = 1.0e-03
        # Set learning rate scheduler        
        lr_scheduler_type = 'explr'
        lr_scheduler_kwargs = {'gamma': 0.995}
        # Set early stopping
        is_early_stopping = True
        early_stopping_kwargs = {'validation_size': 0.2,
                                 'validation_frequency': 0.01*n_train_steps,
                                 'trigger_tolerance': 10}
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch training data set
    dataset = GNNMaterialPatchDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of folds
    n_fold = 4
    # Perform k-fold cross validation of GNN-based material patch model
    k_fold_loss_array = kfold_cross_validation(
        cross_validation_dir, n_fold, n_train_steps, dataset, model_init_args,
        lr_init, opt_algorithm=opt_algorithm,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs, loss_type=loss_type,
        loss_kwargs=loss_kwargs, batch_size=batch_size,
        is_sampler_shuffle=is_sampler_shuffle,
        is_early_stopping=is_early_stopping,
        early_stopping_kwargs=early_stopping_kwargs,
        dataset_file_path=dataset_file_path,
        device_type=device_type, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(cross_validation_dir), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Generate k-fold cross-validation bar plot
    plot_kfold_cross_validation(k_fold_loss_array, loss_type=loss_type.upper(),
                                loss_scale='log', save_dir=plot_dir,
                                is_save_fig=True, is_stdout_display=False,
                                is_latex=True)
# =============================================================================
def set_case_study_model_parameters(case_study_name, model_directory,
                                    device_type='cpu'):
    """Set default GNN-based material patch model initialization parameters.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    model_directory : str
        Directory where material patch model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        GNN-based material patch model class initialization parameters (check
        class GNNMaterialPatchModel).
    """
    if case_study_name in 'cs_2d_elastic':
        # Set GNN-based material patch model name
        model_name = 'material_patch_model'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of node input and output features
        n_node_in = 4
        n_node_out = 2
        # Set number of edge input features
        n_edge_in = 6
        # Set number of message-passing steps (number of processor layers)
        n_message_steps = 1
        # Set number of FNN hidden layers
        enc_n_hidden_layers = 2
        pro_n_hidden_layers = 2
        dec_n_hidden_layers = 2
        # Set hidden layer size
        hidden_layer_size = 128
        # Set (shared) hidden unit activation function
        hidden_activation = 'relu'
        # Set (shared) output unit activation function
        output_activation = 'identity'
        # Set data normalization
        is_data_normalization = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif case_study_name == '2d_elastic':
        raise RuntimeError('Set case-study parameters.')
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'n_node_in': n_node_in,
                       'n_node_out': n_node_out,
                       'n_edge_in': n_edge_in,
                       'n_message_steps': n_message_steps,
                       'enc_n_hidden_layers': enc_n_hidden_layers,
                       'pro_n_hidden_layers': pro_n_hidden_layers,
                       'dec_n_hidden_layers': dec_n_hidden_layers,
                       'hidden_layer_size': hidden_layer_size,
                       'model_directory': model_directory,
                       'model_name': model_name,
                       'is_data_normalization': is_data_normalization,
                       'enc_node_hidden_activ_type': hidden_activation,
                       'enc_node_output_activ_type': output_activation,
                       'enc_edge_hidden_activ_type': hidden_activation,
                       'enc_edge_output_activ_type': output_activation,
                       'pro_node_hidden_activ_type': hidden_activation,
                       'pro_node_output_activ_type': output_activation,
                       'pro_edge_hidden_activ_type': hidden_activation,
                       'pro_edge_output_activ_type': output_activation,
                       'dec_node_hidden_activ_type': hidden_activation,
                       'dec_node_output_activ_type': output_activation,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
def set_default_training_options():
    """Set default GNN-based material patch model training options.
    
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
    lr_init = 1.0e-05
    lr_scheduler_type = None
    lr_scheduler_kwargs = None
    loss_type = 'mse'
    loss_kwargs = {}
    is_sampler_shuffle = True
    is_early_stopping = True
    early_stopping_kwargs = {'validation_size': 0.2, 'validation_frequency': 1,
                             'trigger_tolerance': 1}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_type, loss_kwargs, is_sampler_shuffle, is_early_stopping, \
        early_stopping_kwargs 
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_standard_training = False
    is_cross_validation = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'gnn_material_patch/case_studies/')
    # Set case study directory
    case_study_name = 'cs_2d_elastic'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Set GNN-based material patch training data set directory
    dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                     '1_training_dataset')
    # Get GNN-based material patch training data set file path
    regex = (r'^material_patch_graph_dataset_training_n[0-9]+.pkl$',
             r'^material_patch_graph_dataset_n[0-9]+.pkl$')
    is_file_found, dataset_file_path = \
        find_unique_file_with_regex(dataset_directory, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Training data set file has not been found  '
                           f'in data set directory:\n\n'
                           f'{dataset_directory}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                    '2_model')
    # Create model directory
    if is_standard_training:
        # Create model directory (overwrite)
        make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform standard training of GNN-based material patch model
    if is_standard_training:
        perform_model_standard_training(
            case_study_name, dataset_file_path, model_directory,
            device_type=device_type, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create cross-validation directory
    if is_cross_validation:
        # Set cross-validation directory
        cross_validation_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '3_cross_validation')
        # Create cross-validation directory
        make_directory(cross_validation_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform k-fold cross validation of GNN-based material patch model
    if is_cross_validation:
        perform_model_kfold_cross_validation(
            case_study_name, dataset_file_path, model_directory,
            cross_validation_dir, device_type=device_type, is_verbose=True)