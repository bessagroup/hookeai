"""Procedures associated to model training.

Functions
---------
save_training_state
    Save model and optimizer states at given training epoch.
save_loss_history
    Save training process loss history record.
write_training_summary_file
    Write summary data file for model training process.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
import pickle
import datetime
# Third-party
import torch
import numpy as np
# Local
from model_architectures.procedures.model_state_files import save_model_state
from ioput.iostandard import write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def save_training_state(model, optimizer, state_type, epoch=None,
                        is_remove_posterior=True):
    """Save model and optimizer states at given training epoch.
    
    Material patch model state file is stored in model_directory under the
    name < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
    
    Material patch model state file corresponding to the best performance
    is stored in model_directory under the name < model_name >-best.pt or
    < model_name >-< epoch >-best.pt if epoch is known.
        
    Optimizer state file is stored in model_directory under the name
    < model_name >_optim-< epoch >.pt.
    
    Optimizer state file corresponding to the best performance is stored in
    model_directory under the name < model_name >_optim-best.pt or
    < model_name >_optim-< epoch >-best.pt if epoch is known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    state_type : {'default', 'init', 'epoch', 'best'}, default='default'
        Saved model state file type.
        Options:
        
        'default' : Model default state
        
        'init'    : Model initial state
    
        'epoch'   : Model state of given training epoch
        
        'best'    : Model state of best performance

    epoch : int, default=None
        Training epoch.
    is_remove_posterior : bool, default=True
        Remove material patch model and optimizer state files corresponding to
        training epochs posterior to the saved state file. Effective only if
        saved epoch is known.
    """
    # Save model state
    save_model_state(model, state_type=state_type, epoch=epoch,
                     is_remove_posterior=is_remove_posterior)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize optimizer state filename
    optimizer_state_file = model.model_name + '_optim'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state type
    if state_type == 'init':
        pass
    else:
        # Append epoch
        if isinstance(epoch, int):
            optimizer_state_file += '-' + str(epoch) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set particular optimizer states
        if state_type == 'best':
            # Set optimizer state corresponding to the best performance
            optimizer_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get files in model directory
            directory_list = os.listdir(model.model_directory)
            # Loop over files in model directory
            for filename in directory_list:
                # Check if file is optimizer epoch best state file
                is_best_state_file = \
                    bool(re.search(r'^' + model.model_name + r'_optim'
                                   + r'-?[0-9]*' + r'-best' + r'\.pt',
                                   filename))
                # Delete state file
                if is_best_state_file:
                    os.remove(os.path.join(model.model_directory, filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set optimizer state file path
        optimizer_path = os.path.join(model.model_directory,
                                      optimizer_state_file + '.pt')
        # Save optimizer state
        optimizer_state = dict(state=optimizer.state_dict(), epoch=epoch)
        torch.save(optimizer_state, optimizer_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete optimizer epoch state files posterior to saved epoch
        if isinstance(epoch, int) and is_remove_posterior:
            remove_posterior_optim_state_files(model, epoch)
# =============================================================================
def remove_posterior_optim_state_files(model, epoch):
    """Delete optimizer training epoch state files posterior to given epoch.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    epoch : int
        Training epoch.
    """
    # Get files in material patch model directory
    directory_list = os.listdir(model.model_directory)
    # Loop over files in material patch model directory
    for filename in directory_list:
        # Check if file is optimizer epoch state file
        is_state_file = bool(re.search(r'^' + model.model_name + r'_optim'
                             + r'-[0-9]+' + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete optimizer epoch state file posterior to given epoch
        if is_state_file:
            # Get optimizer state epoch
            file_epoch = int(os.path.splitext(filename)[0].split('-')[-1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete optimizer epoch state file
            if file_epoch > epoch:
                os.remove(os.path.join(model.model_directory, filename))
# =============================================================================
def save_loss_history(model, n_max_epochs, loss_nature, loss_type,
                      training_loss_history, lr_scheduler_type=None,
                      lr_history_epochs=None, validation_loss_history=None):
    """Save training process loss history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.

    Overwrites existing loss history record file.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    n_max_epochs : int
        Maximum number of epochs of training process.
    loss_nature : str
        Loss nature.
    loss_type : str
        Loss function type.
    training_loss_history : list[float]
        Training process training loss history (per epoch).
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}, default=None
        Type of learning rate scheduler.
    lr_history_epochs : list[float], default=None
        Training process learning rate history (per epoch).
    validation_loss_history : list[float], default=None
        Training process validation loss history (e.g., early stopping
        criterion).
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build training loss history record
    loss_history_record = {}
    loss_history_record['n_max_epochs'] = int(n_max_epochs)
    loss_history_record['loss_nature'] = str(loss_nature)
    loss_history_record['loss_type'] = str(loss_type)
    loss_history_record['training_loss_history'] = list(training_loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store learning rate history record
    if lr_scheduler_type is not None:
        loss_history_record['lr_scheduler_type'] = str(lr_scheduler_type)
    else:
        loss_history_record['lr_scheduler_type'] = None
    if lr_history_epochs is not None:
        loss_history_record['lr_history_epochs'] = list(lr_history_epochs)
    else:
        loss_history_record['lr_history_epochs'] = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store validation loss history
    if validation_loss_history is not None:
        loss_history_record['validation_loss_history'] = \
            list(validation_loss_history)
    else:
        loss_history_record['validation_loss_history'] = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save loss history record
    with open(loss_record_path, 'wb') as loss_record_file:
        pickle.dump(loss_history_record, loss_record_file)
# =============================================================================
def write_training_summary_file(
    device_type, seed, model_directory, load_model_state, n_max_epochs,
    is_model_in_normalized, is_model_out_normalized, batch_size,
    is_sampler_shuffle, loss_nature, loss_type, loss_kwargs, opt_algorithm,
    lr_init, lr_scheduler_type, lr_scheduler_kwargs, n_epochs,
    dataset_file_path, dataset, best_loss, best_training_epoch, total_time_sec,
    avg_time_epoch, best_model_parameters=None, torchinfo_summary=None):
    """Write summary data file for model training process.
    
    Parameters
    ----------
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where material patch model is stored.
    load_model_state : {'best', 'last', int, None}
        Load available Graph Neural Network model state from the model
        directory. Data scalers are also loaded from model initialization file.
    n_max_epochs : int
        Maximum number of training epochs.
    is_model_in_normalized : bool, default=False
        If True, then model input features are assumed to be normalized
        (normalized input data has been seen during model training).
    is_model_out_normalized : bool, default=False
        If True, then model output features are assumed to be normalized
        (normalized output data has been seen during model training).
    batch_size : int
        Number of samples loaded per batch.
    is_sampler_shuffle : bool
        If True, shuffles data set samples at every epoch.
    loss_nature : str
        Loss nature.
    loss_type : str
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    opt_algorithm : str
        Optimization algorithm.
    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    lr_scheduler_type : str
        Type of learning rate scheduler.
    lr_scheduler_kwargs : dict
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    n_epochs : int
        Number of completed epochs in the training process.
    dataset_file_path : str
        Graph Neural Network graph data set file path if such file exists. Only
        used for output purposes.
    dataset : torch.utils.data.Dataset
        Graph Neural Network graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    best_loss : float
        Best loss during training process.
    best_training_epoch : int
        Training epoch corresponding to best loss during training process.
    total_time_sec : int
        Total training time in seconds.
    avg_time_epoch : float
        Average training time per epoch.
    best_model_parameters : dict
        Model parameters corresponding to best model state.
    torchinfo_summary : str, default=None
        Torchinfo model architecture summary.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['load_model_state'] = \
        load_model_state if load_model_state else None
    summary_data['n_max_epochs'] = n_max_epochs
    summary_data['is_model_in_normalized'] = is_model_in_normalized
    summary_data['is_model_out_normalized'] = is_model_out_normalized
    summary_data['batch_size'] = batch_size
    summary_data['is_sampler_shuffle'] = is_sampler_shuffle
    summary_data['loss_nature'] = loss_nature
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['opt_algorithm'] = opt_algorithm
    summary_data['lr_init'] = lr_init
    summary_data['lr_scheduler_type'] = \
        lr_scheduler_type if lr_scheduler_type else None
    summary_data['lr_scheduler_kwargs'] = \
        lr_scheduler_kwargs if lr_scheduler_kwargs else None
    summary_data['Number of completed epochs'] = n_epochs
    summary_data['Training data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Training data set size'] = len(dataset)
    summary_data['Best loss: '] = \
        f'{best_loss:.8e} (training epoch {best_training_epoch})'
    summary_data['Total training time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. training time per epoch'] = \
        str(datetime.timedelta(seconds=int(avg_time_epoch)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set summary optional data
    if best_model_parameters is not None:
        summary_data['Model parameters (best state)'] = best_model_parameters
    if torchinfo_summary is not None:
        summary_data['torchinfo summary'] = torchinfo_summary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=model_directory,
        summary_title='Summary: Model training',
        **summary_data)