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
import re
# Local
from gnn_model.gnn_patch_dataset import GNNMaterialPatchDataset
from gnn_model.training import train_model, read_loss_history_from_file
from gnn_model.evaluation_metrics import plot_training_loss_history
from ioput.iostandard import make_directory
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based material patch model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_type, loss_kwargs, is_sampler_shuffle = \
            set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if case_study_name == '2d_elastic':
        # Set GNN-based material patch model name
        model_name = 'material_patch_model'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of training steps
        n_train_steps = 100
        # Set batch size
        batch_size = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model parameters:
        # Set number of node input and output features
        n_node_in = 4
        n_node_out = 2
        # Set number of edge input features
        n_edge_in = 6
        # Set number of message-passing steps (number of processor layers)
        n_message_steps = 10
        # Set number of FNN hidden layers
        n_hidden_layers = 2
        # Set hidden layer size
        hidden_layer_size = 128
        # Set data normalization
        is_data_normalization = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build model initialization parameters
        model_init_args = {'n_node_in': n_node_in,
                           'n_node_out': n_node_out,
                           'n_edge_in': n_edge_in,
                           'n_message_steps': n_message_steps,
                           'n_hidden_layers': n_hidden_layers,
                           'hidden_layer_size': hidden_layer_size,
                           'model_directory': model_directory,
                           'model_name': model_name,
                           'is_data_normalization': is_data_normalization,
                           'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch training data set
    dataset = GNNMaterialPatchDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Training of GNN-based material patch model
    model = train_model(n_train_steps, dataset, model_init_args, lr_init,
                        opt_algorithm=opt_algorithm,
                        lr_scheduler_type=lr_scheduler_type,
                        lr_scheduler_kwargs=lr_scheduler_kwargs,
                        loss_type=loss_type, loss_kwargs=loss_kwargs,
                        batch_size=batch_size,
                        is_sampler_shuffle=is_sampler_shuffle,
                        load_model_state=None, save_every=None,
                        device_type='cpu', seed=None, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record.pkl')
    # Read training process loss history
    loss_type, loss_history = read_loss_history_from_file(loss_record_path)
    loss_histories = {f'$n_s = {len(dataset)}$': loss_history,}
    
    # Plot model training process loss history
    plot_training_loss_history(loss_histories, loss_type.upper(),
                               save_dir=model_directory,
                               is_save_fig=True, is_stdout_display=False)
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
    lr_scheduler_type : {'steplr',}
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)

    lr_scheduler_kwargs : dict
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_sampler_shuffle : bool
        If True, shuffles data set samples at every epoch.    
    """
    opt_algorithm = 'adam'
    lr_init = 1.0e-04
    lr_scheduler_type = 'steplr'
    lr_scheduler_kwargs = {'step_size': 5.0e+06, 'gamma': 0.1}
    loss_type = 'mse'
    loss_kwargs = {}
    is_sampler_shuffle = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_type, loss_kwargs, is_sampler_shuffle        
# =============================================================================
if __name__ == "__main__":
    # Set case study name
    case_study_name = '2d_elastic'
    # Set case study directory
    case_study_base_dirs = {
        '2d_elastic': f'/home/bernardoferreira/Documents/temp',}
    case_study_dir = \
        os.path.join(os.path.normpath(case_study_base_dirs[case_study_name]),
                     f'cs_{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Set GNN-based material patch data set directory
    dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                     '1_dataset')
    # Get files in GNN-based material patch data set directory
    directory_list = os.listdir(dataset_directory)
    # Loop over files
    is_training_dataset = False
    for filename in directory_list:
        # Check if file is training data set file
        is_training_dataset = \
            bool(re.search(r'^material_patch_graph_dataset_training_n'
                           r'[0-9]+.pkl$', filename))
        # Leave searching loop when training data set file is found
        if is_training_dataset:
            break
    # Set GNN-based material patch training data set file path
    if is_training_dataset:
        dataset_file_path = os.path.join(os.path.normpath(dataset_directory),
                                         filename)
    else:
        raise RuntimeError(f'Training data set file has not been found in '
                           'dataset directory:\n\n{dataset_directory}')      
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                   '2_model')
    # Create model directory
    make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    device_type = 'cpu'
    # Perform standard training of GNN-based material patch model
    perform_model_standard_training(case_study_name, dataset_file_path,
                                    model_directory, device_type,
                                    is_verbose=True)