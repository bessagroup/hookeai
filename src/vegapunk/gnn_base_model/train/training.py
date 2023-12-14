"""Training of Graph Neural Network model.

Classes
-------
EarlyStopper
    Early stopping procedure (implicit regularizaton).

Functions
---------
train_model
    Training of Graph Neural Network model.
get_pytorch_optimizer
    Get PyTorch optimizer.
get_learning_rate_scheduler
    Get PyTorch optimizer learning rate scheduler.
save_training_state
    Save model and optimizer states at given training epoch.
load_training_state
    Load model and optimizer states from available training data.
remove_posterior_optim_state_files
    Delete optimizer training epoch state files posterior to given epoch.
save_loss_history
    Save training process loss history record.
load_loss_history
    Load training process loss history record.
load_lr_history
    Load training process learning rate history record.
seed_worker
    Set workers seed in PyTorch data loaders to preserve reproducibility.
read_loss_history_from_file
    Read training loss history from loss history record file.
read_lr_history_from_file(loss_record_path)
    Read training learning rate history from loss history record file.
write_training_summary_file    
    Write summary data file for model training process.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import random
import re
import time
import datetime
import copy
# Third-party
import torch
import torch_geometric.loader
import numpy as np
# Local
from gnn_base_model.data.graph_dataset import split_dataset
from gnn_base_model.model.gnn_model import GNNEPDBaseModel
from gnn_base_model.train.torch_loss import get_pytorch_loss
from gnn_base_model.predict.prediction import predict
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
def train_model(n_max_epochs, dataset, model_init_args, lr_init,
                opt_algorithm='adam', lr_scheduler_type=None,
                lr_scheduler_kwargs={}, loss_type='mse', loss_kwargs={},
                batch_size=1, is_sampler_shuffle=False,
                is_early_stopping=False, early_stopping_kwargs={},
                load_model_state=None, save_every=None, dataset_file_path=None,
                device_type='cpu', seed=None, is_verbose=False):
    """Training of Graph Neural Network model.
    
    Parameters
    ----------
    n_max_epochs : int
        Maximum number of training epochs.
    dataset : torch.utils.data.Dataset
        Graph Neural Network graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_init_args : dict
        Graph Neural Network model class initialization parameters (check
        class GNNEPDBaseModel).
    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    opt_algorithm : {'adam',}, default='adam'
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)
        
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}, default=None
        Type of learning rate scheduler:

        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
        'explr'   : Exponential decay (torch.optim.lr_scheduler.ExponentialLR)
        
        'linlr'   : Linear decay (torch.optim.lr_scheduler.LinearLR)

    lr_scheduler_kwargs : dict, default={}
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict, default={}
        Arguments of torch.nn._Loss initializer.
    batch_size : int, default=1
        Number of samples loaded per batch.
    is_sampler_shuffle : bool, default=False
        If True, shuffles data set samples at every epoch.
    is_early_stopping : bool, default=False
        If True, then training process is halted when early stopping criterion
        is triggered. By default, 20% of the training data set is allocated for
        the underlying validation procedures.
    early_stopping_kwargs : dict, default={}
        Early stopping criterion parameters (key, str, item, value).
    load_model_state : {'best', 'last', int, None}, default=None
        Load available GNN-based model state from the model
        directory. Data scalers are also loaded from model initialization file.
        Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training epoch
        
        int         : Model state corresponding to given training epoch
        
        None        : Model default state file

    save_every : int, default=None
        Save Graph Neural Network model every save_every epochs. If None, then
        saves only last epoch and best performance states.
    dataset_file_path : str, default=None
        Graph Neural Network graph data set file path if such file exists. Only
        used for output purposes.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    seed : int, default=None
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    model : torch.nn.Module
        Graph Neural Network model.
    best_loss : float
        Best loss during training process.
    best_training_epoch : int
        Training epoch corresponding to best loss during training process.
    """
    # Set random number generators initialization for reproducibility
    if isinstance(seed, int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time_sec = time.time()
    if is_verbose:
        print('\nGraph Neural Network model training'
              '\n-----------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Graph Neural Network model
    model = GNNEPDBaseModel(**model_init_args)    
    # Set model device
    model.set_device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Move model to device
    model.to(device=device)
    # Set model in training mode
    model.train()
    # Get model parameters
    model_parameters = model.parameters(recurse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model data normalization
    is_data_normalization = model.is_data_normalization
    # Fit model data scalers  
    if is_data_normalization and load_model_state is None:
        model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize learning rate
    learning_rate = lr_init
    # Set optimizer    
    if opt_algorithm == 'adam':
        # Initialize optimizer, specifying the model (and submodels) parameters
        # that should be optimized. By default, model parameters gradient flag
        # is set to True, meaning that gradients with respect to the parameters
        # are required (operations on the parameters are recorded for automatic
        # differentiation)
        optimizer = torch.optim.Adam(params=model_parameters, lr=learning_rate)
    else:
        raise RuntimeError('Unknown optimization algorithm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize learning rate scheduler
    is_lr_scheduler = False
    if lr_scheduler_type is not None:
        is_lr_scheduler = True
        lr_scheduler = get_learning_rate_scheduler(
            optimizer=optimizer, scheduler_type=lr_scheduler_type,
            **lr_scheduler_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize loss function
    loss_function = get_pytorch_loss(loss_type, **loss_kwargs)
    # Initialize loss and learning rate histories (per epoch)
    loss_history_epochs = []
    lr_history_epochs = []
    # Initialize loss and learning rate histories (per training step)
    loss_history_steps = []
    lr_history_steps = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training flag
    is_keep_training = True
    # Initialize number of training epochs
    epoch = 0
    # Initialize number of training steps
    step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Graph Neural Network model model state
    if load_model_state is not None:   
        # Initialize Graph Neural Network model model
        # (includes loading of data scalers)
        model = GNNEPDBaseModel.init_model_from_file(
            model_init_args['model_directory'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Move model to device
        model.to(device=device)
        # Set model in training mode
        model.train()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Loading model state...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load Graph Neural Network model model state
        loaded_epoch = load_training_state(model, opt_algorithm, optimizer,
                                           load_model_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load loss history
        loss_history_epochs = load_loss_history(model, loss_type,
                                                epoch=loaded_epoch)
        # Load learning rate history
        lr_history_epochs = load_lr_history(model, epoch=loaded_epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update training epoch counter
        epoch = int(loaded_epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize validation loss history
    validation_loss_history = None
    # Initialize early stopping criterion
    if is_early_stopping:
        if is_verbose:
            print('\n> Initializing early stopping criterion...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize early stopping criterion
        early_stopper = EarlyStopper(dataset, **early_stopping_kwargs)
        # Get available training data set (remainder is set aside for early
        # stopping criterion validation procedures)
        dataset = early_stopper.get_training_dataset()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize early stopping flag
        is_stop_training = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Training data set (effective) size: {len(dataset)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    if isinstance(seed, int):
        data_loader = torch_geometric.loader.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size, worker_init_fn=seed_worker,
            generator=generator)
    else:
        data_loader = torch_geometric.loader.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_sampler_shuffle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        normalization_str = 'Yes' if is_data_normalization else 'No'
        print(f'\n> Data normalization: {normalization_str}')
        print('\n\n> Starting training process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training iterations
    while is_keep_training:
        # Store epoch initial training step
        epoch_init_step = step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over graph batches. A graph batch is a data object describing a
        # batch of graphs as one large (disconnected) graph.
        for pyg_graph in data_loader:
            # Move graph sample to device
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node output features predictions (forward propagation).
            # During the foward pass, PyTorch creates a computation graph for
            # the tensors that require gradients (gradient flag set to True) to
            # keep track of the operations on these tensors, i.e., the model
            # parameters. In addition, PyTorch additionally stores the
            # corresponding 'gradient functions' (mathematical operator) of the
            # executed operations to the output tensor, stored in the .grad_fn
            # attribute of the corresponding tensors. Tensor.grad_fn is set to
            # None for tensors corresponding to leaf-nodes of the computation
            # graph or for tensors with the gradient flag set to False.
            node_features_out = model.predict_node_output_features(
                pyg_graph, is_normalized=is_data_normalization)
            # Get node output features ground-truth
            node_targets = model.get_output_features_from_graph(
                pyg_graph, is_normalized=is_data_normalization)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute loss
            loss = loss_function(node_features_out, node_targets)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize gradients (set to zero)
            optimizer.zero_grad()
            # Compute gradients with respect to model parameters (backward
            # propagation). PyTorch backpropagates recursively through the
            # computation graph of loss and computes the gradients with respect
            # to the model parameters. On each Tensor, PyTorch computes the
            # local gradients using the previously stored .grad_fn mathematical
            # operators and combines them with the incoming gradients to
            # compute the complete gradient (i.e., building the
            # differentiation chain rule). The backward propagation recursive
            # path stops when a leaf-node is reached (e.g., a model parameter),
            # where .grad_fn is set to None. Gradients are cumulatively stored
            # in the .grad attribute of the corresponding tensors
            loss.backward()
            # Perform optimization step. Gradients are stored in the .grad
            # attribute of model parameters
            optimizer.step()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update optimizer learning rate
            if is_lr_scheduler:
                lr_scheduler.step()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                total_time_sec = time.time() - start_time_sec
                print('> Epoch: {:{width}d}/{:d} | Training step: {:d} | '
                      'Loss: {:.8e} | Elapsed time (s): {:}'.format(
                    epoch, n_max_epochs, step, loss,
                    str(datetime.timedelta(seconds=int(total_time_sec))),
                    width=len(str(n_max_epochs))), end='\r')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save training step loss and learning rate
            loss_history_steps.append(loss.clone().detach().cpu())
            if is_lr_scheduler:
                lr_history_steps.append(lr_scheduler.get_last_lr())
            else:
                lr_history_steps.append(lr_init)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Increment training step counter
            step += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save training epoch loss (epoch average loss value)
        epoch_avg_loss = np.mean(loss_history_steps[epoch_init_step:])
        loss_history_epochs.append(epoch_avg_loss)
        # Save training epoch learning rate (epoch last value)
        if is_lr_scheduler:
            lr_history_epochs.append(lr_scheduler.get_last_lr())
        else:
            lr_history_epochs.append(lr_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model and optimizer current states
        if save_every is not None and epoch % save_every == 0:
            save_training_state(model=model, optimizer=optimizer, epoch=epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model and optimizer best performance state corresponding to
        # minimum training loss
        if epoch_avg_loss <= min(loss_history_epochs):
            save_training_state(model=model, optimizer=optimizer,
                                epoch=epoch, is_best_state=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check early stopping criterion
        if is_early_stopping:
            # Evaluate early stopping criterion
            if early_stopper.is_evaluate_criterion(epoch):
                is_stop_training = early_stopper.evaluate_criterion(
                    model, optimizer, epoch, loss_type=loss_type,
                    loss_kwargs=loss_kwargs, device_type=device_type)
            # If early stopping is triggered, save model and optimizer best
            # performance corresponding to early stopping criterion
            if is_stop_training:
                # Load best performance model and optimizer states
                best_epoch = early_stopper.load_best_performance_state(
                    model, optimizer)
                # Save model and optimizer best performance states
                save_training_state(model, optimizer, epoch=best_epoch,
                                    is_best_state=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check training process flow
        if epoch >= n_max_epochs:
            # Completed maximum number of epochs
            is_keep_training = False
            break
        elif is_early_stopping and is_stop_training:
            # Early stopping criterion triggered
            is_keep_training = False
            break
        else:
            # Increment epoch counter
            epoch += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        if is_early_stopping and is_stop_training:
            print('\n\n> Early stopping has been triggered!',
                  '\n\n> Finished training process!')
        else:
            print('\n\n> Finished training process!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get validation loss history
    if is_early_stopping:
        validation_loss_history = \
            early_stopper.get_validation_loss_history()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model and optimizer final states
    save_training_state(model=model, optimizer=optimizer, epoch=epoch)
    # Save loss and learning rate histories
    save_loss_history(model, n_max_epochs, loss_type, loss_history_epochs,
                      lr_scheduler_type=lr_scheduler_type,
                      lr_history_epochs=lr_history_epochs,
                      validation_loss_history=validation_loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get best loss and corresponding training epoch
    best_loss = float(min(loss_history_epochs))
    best_training_epoch = loss_history_epochs.index(best_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Minimum loss: {:.8e} | Epoch: {:d}'.format(
              best_loss, best_training_epoch))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total training time and average training time per epoch
    total_time_sec = time.time() - start_time_sec
    avg_time_epoch = total_time_sec/epoch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Model directory: {model.model_directory}')
        total_time_sec = time.time() - start_time_sec
        print(f'\n> Total training time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. training time per epoch: '
              f'{str(datetime.timedelta(seconds=int(avg_time_epoch)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for model training process
    write_training_summary_file(
        device_type, seed, model.model_directory, load_model_state,
        n_max_epochs, is_data_normalization, batch_size, is_sampler_shuffle,
        loss_type, loss_kwargs, opt_algorithm, lr_init,
        lr_scheduler_type, lr_scheduler_kwargs, epoch, dataset_file_path,
        dataset, best_loss, best_training_epoch, total_time_sec,
        avg_time_epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model, best_loss, best_training_epoch
# =============================================================================
def get_pytorch_optimizer(algorithm, params, **kwargs):
    """Get PyTorch optimizer.
   
    Parameters
    ----------
    algorithm : {'adam',}
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)
        
    params : list
        List of parameters (torch.Tensors) to optimize or list of dicts
        defining parameter groups.
    **kwargs
        Arguments of torch.optim.Optimizer initializer.
        
    Returns
    -------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    """
    if algorithm == 'adam':
        optimizer = torch.optim.Adam(params, **kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch optimizer.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return optimizer
# =============================================================================
def get_learning_rate_scheduler(optimizer, scheduler_type, **kwargs):
    """Get PyTorch optimizer learning rate scheduler.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}
        Type of learning rate scheduler:

        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
        'explr'   : Exponential decay (torch.optim.lr_scheduler.ExponentialLR)
        
        'linlr'   : Linear decay (torch.optim.lr_scheduler.LinearLR)

    **kwargs
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        PyTorch optimizer learning rate scheduler.
    """
    if scheduler_type == 'steplr':
        # Check scheduler mandatory parameters
        if 'step_size' not in kwargs.keys():
            raise RuntimeError('The parameter \'step_size\' needs to be '
                               'provided to initialize step-based decay '
                               'learning rate scheduler.')
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif scheduler_type == 'explr':
        # Check scheduler mandatory parameters
        if 'gamma' not in kwargs.keys():
            raise RuntimeError('The parameter \'gamma\' needs to be '
                               'provided to initialize exponential decay '
                               'learning rate scheduler.')
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif scheduler_type == 'linlr':
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown or unavailable PyTorch optimizer '
                           'learning rate scheduler.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return scheduler
# =============================================================================
def save_training_state(model, optimizer, epoch=None,
                        is_best_state=False, is_remove_posterior=True):
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
    epoch : int, default=None
        Training epoch.
    is_best_state : bool, default=False
        If True, save material patch model state file corresponding to the best
        performance instead of regular state file.
    is_remove_posterior : bool, default=True
        Remove material patch model and optimizer state files corresponding to
        training epochs posterior to the saved state file. Effective only if
        saved epoch is known.
    """
    # Save Graph Neural Network model
    model.save_model_state(epoch=epoch, is_best_state=is_best_state)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file
    optimizer_state_file = model.model_name + '_optim'
    # Append epoch
    if isinstance(epoch, int):
        optimizer_state_file += '-' + str(epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file corresponding to best performance
    if is_best_state:
        # Append best performance
        optimizer_state_file += '-' + 'best'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimizer state files in material patch model directory
        directory_list = os.listdir(model.model_directory)
        # Loop over files in material patch model directory
        for filename in directory_list:
            # Check if file is optimizer epoch best state file
            is_best_state_file = \
                bool(re.search(r'^' + model.model_name + r'_optim'
                               + r'-?[0-9]*' + r'-best' + r'\.pt', filename))
            # Delete state file
            if is_best_state_file:
                os.remove(os.path.join(model.model_directory, filename))      
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file path
    optimizer_path = os.path.join(model.model_directory,
                                  optimizer_state_file + '.pt')
    # Save optimizer state
    optimizer_state = dict(state=optimizer.state_dict(),
                           epoch=epoch)
    torch.save(optimizer_state, optimizer_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Delete model and optimizer epoch state files posterior to saved epoch
    if isinstance(epoch, int) and is_remove_posterior:
        remove_posterior_optim_state_files(model, epoch)
# =============================================================================
def load_training_state(model, opt_algorithm, optimizer,
                        load_model_state=None, is_remove_posterior=True):
    """Load model and optimizer states from available training data.
    
    Material patch model state file is stored in model_directory under the
    name < model_name >.pt, < model_name >-< epoch >.pt,
    < model_name >-best.pt or < model_name >-< epoch >-best.pt.

    Optimizer state file is stored in model_directory under the name
    < model_name >_optim.pt or < model_name >_optim-< epoch >.pt.
    
    Both model and optimizer are updated 'in-place' with loaded state data.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    opt_algorithm : {'adam',}, default='adam'
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)

    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    load_model_state : {'best', 'last', int, None}, default=None
        Load available Graph Neural Network model state from the model
        directory. Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training epoch
        
        int         : Model state corresponding to given training epoch
        
        None        : Model default state file

    is_remove_posterior : bool, default=True
        Remove material patch model state files corresponding to training
        epochs posterior to the loaded state file. Effective only if
        loaded training epoch is known.

    Returns
    -------
    loaded_epoch : int
        Training epoch corresponding to loaded state data. Defaults to 0 if
        training epoch is unknown.
    """
    # Load Graph Neural Network model model state        
    loaded_epoch = \
        model.load_model_state(load_model_state=load_model_state,
                               is_remove_posterior=is_remove_posterior)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file
    optimizer_state_file = model.model_name + '_optim'
    # Append epoch
    if isinstance(loaded_epoch, int):
        optimizer_state_file += '-' + str(loaded_epoch)
    # Append best performance
    if load_model_state == 'best':
        optimizer_state_file += '-' + 'best'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file path
    optimizer_path = os.path.join(model.model_directory,
                                  optimizer_state_file + '.pt')
    # Load optimizer state
    if not os.path.isfile(optimizer_path):
        raise RuntimeError('Optimizer state file has not been found:\n\n'
                           + optimizer_path)
    else:
        # Initialize optimizer
        if opt_algorithm == 'adam':
            optimizer = torch.optim.Adam(params=model.parameters(recurse=True))
        else:
            raise RuntimeError('Unknown optimization algorithm')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load optimizer state
        optimizer_state = torch.load(optimizer_path)
        # Set loaded optimizer state
        optimizer.load_state_dict(optimizer_state['state'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete optimizer epoch state files posterior to loaded epoch
        if isinstance(loaded_epoch, int) and is_remove_posterior:
            remove_posterior_optim_state_files(model, loaded_epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loaded epoch to 0 if unknown from state file
    if loaded_epoch is None:
        loaded_epoch = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loaded_epoch
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
def save_loss_history(model, n_max_epochs, loss_type, training_loss_history,
                      lr_scheduler_type=None, lr_history_epochs=None,
                      validation_loss_history=None):
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
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)

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
def load_loss_history(model, loss_type, epoch=None):
    """Load training process training loss history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    epoch : int, default=None
        Epoch to which loss history is loaded (included), with the first epoch
        being 0. If None, then loads the full loss history.

    Returns
    -------
    training_loss_history : list[float]
        Training process training loss history (per epoch).
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training process training loss history
    if os.path.isfile(loss_record_path):
        # Load loss history record
        with open(loss_record_path, 'rb') as loss_record_file:
            loss_history_record = pickle.load(loss_record_file)
        # Check consistency between loss history type and current training
        # process loss type
        history_loss_type = loss_history_record['loss_type']
        if history_loss_type != loss_type:
            raise RuntimeError('Loss history type (' + str(history_loss_type)
                               + ') is not consistent with current training '
                               'process loss type (' + str(loss_type) + ').')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check training loss history
        loss_record = loss_history_record['training_loss_history']
        if not isinstance(loss_record, list):
            raise RuntimeError('Loaded loss history is not a list[float].')
        # Load training loss history
        if epoch is None or epoch + 1 == len(loss_record):
            training_loss_history = loss_record
        else:
            if epoch + 1 > len(loss_record):
                raise RuntimeError('Target epoch is beyond available loss '
                                   'history.')
            else:
                training_loss_history = loss_record[:epoch + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Build training loss history with None entries if loss history record
        # file cannot be found
        if epoch is None:
            raise RuntimeError('Training process loss history file has not '
                               'been found and loaded epoch is unknown.')
        else:
            training_loss_history = (epoch + 1)*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return training_loss_history
# =============================================================================
def load_lr_history(model, epoch=None):
    """Load training process learning rate history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.        
    epoch : int, default=None
        Training epoch to which loss history is loaded (included), with the
        first training epoch being 0. If None, then loads the full loss
        history.

    Returns
    -------
    lr_history_epochs : list[float]
        Training process learning rate history (per epoch).
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training process learning history
    if os.path.isfile(loss_record_path):
        # Load loss history record
        with open(loss_record_path, 'rb') as loss_record_file:
            loss_history_record = pickle.load(loss_record_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check learning rate history
        lr_record = loss_history_record['lr_history_epochs']
        if not isinstance(lr_record, list) and lr_record is not None:
            raise RuntimeError('Loaded learning rate history is not a '
                               'list[float] or None.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        # Load learning rate history
        if lr_record is None and epoch is None:
            # Build learning rate history with None entries if learning rate
            # history is not available
            lr_history_epochs = len(
                loss_history_record['training_loss_history'])*[None,]
        elif lr_record is None and epoch is not None:
            # Build learning rate history with None entries if learning rate
            # history is not available
            lr_history_epochs = (epoch + 1)*[None,]
        else:
            if epoch is None or epoch + 1 == len(lr_record):
                lr_history_epochs = lr_record
            elif epoch + 1 > len(lr_record):
                    raise RuntimeError('Target epoch is beyond available '
                                       'learning rate history.')
            else:
                lr_history_epochs = lr_record[:epoch + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Build learning rate history with None entries if loss history record
        # file cannot be found
        if epoch is None:
            raise RuntimeError('Training process loss history file has not '
                               'been found and loaded epoch is unknown.')
        else:
            lr_history_epochs = (epoch + 1)*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return lr_history_epochs
# =============================================================================
def seed_worker(worker_id):
    """Set workers seed in PyTorch data loaders to preserve reproducibility.
    
    Taken from: https://pytorch.org/docs/stable/notes/randomness.html
    
    Parameters
    ----------
    worker_id : int
        Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# =============================================================================
def read_loss_history_from_file(loss_record_path):
    """Read training process loss history from loss history record file.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Detaches loss values from computation graph and moves them to CPU.
    
    Parameters
    ----------
    loss_record_path : str
        Loss history record file path.
    
    Returns
    -------
    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)

    training_loss_history : list[float]
        Training process training loss history (per epoch).
    validation_loss_history : {None, list[float]}
        Training process validation loss history. Set to None if not available.
    """
    # Check loss history record file
    if not os.path.isfile(loss_record_path):
        raise RuntimeError('Loss history record file has not been found:\n\n'
                           + loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load loss history record
    with open(loss_record_path, 'rb') as loss_record_file:
        loss_history_record = pickle.load(loss_record_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check loss history
    if 'loss_type' not in loss_history_record.keys():
        raise RuntimeError('Loss type is not available in loss history '
                           'record.')
    elif 'training_loss_history' not in loss_history_record.keys():
        raise RuntimeError('Loss history is not available in loss history '
                           'record.')
    elif not isinstance(loss_history_record['training_loss_history'],
                        list):
        raise RuntimeError('Loss history is not a list[float].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss type
    loss_type = str(loss_history_record['loss_type'])
    # Set training loss history
    training_loss_history = []
    for x in loss_history_record['training_loss_history']:
        if isinstance(x, torch.Tensor):
            training_loss_history.append(x.detach().cpu())
        else:
            training_loss_history.append(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set validation loss history
    if isinstance(loss_history_record['validation_loss_history'], list):
        validation_loss_history = []
        for x in loss_history_record['validation_loss_history']:
            if isinstance(x, torch.Tensor):
                validation_loss_history.append(x.detach().cpu())
            else:
                validation_loss_history.append(x)
    else:
        validation_loss_history = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_type, training_loss_history, validation_loss_history
# =============================================================================
def read_lr_history_from_file(loss_record_path):
    """Read training learning rate history from loss history record file.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Parameters
    ----------
    loss_record_path : str
        Loss history record file path.
    
    Returns
    -------
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}
        Type of learning rate scheduler:

        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
        'explr'   : Exponential decay (torch.optim.lr_scheduler.ExponentialLR)
        
        'linlr'   : Linear decay (torch.optim.lr_scheduler.LinearLR)

    lr_history_epochs : list[float]
        Training process learning rate history (per epoch).
    """
    # Check loss history record file
    if not os.path.isfile(loss_record_path):
        raise RuntimeError('Loss history record file has not been found:\n\n'
                           + loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load loss history record
    with open(loss_record_path, 'rb') as loss_record_file:
        loss_history_record = pickle.load(loss_record_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check learning rate history
    if 'lr_scheduler_type' not in loss_history_record.keys():
        raise RuntimeError('Learning rate scheduler type is not available in '
                           'loss history record.')
    elif 'lr_history_epochs' not in loss_history_record.keys():
        raise RuntimeError('Learning rate history is not available in loss '
                           'history record.')
    elif not isinstance(loss_history_record['lr_history_epochs'], list):
        raise RuntimeError('Learning rate history is not a list[float].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set learning rate scheduler type
    lr_scheduler_type = loss_history_record['lr_scheduler_type']
    # Set learning rate history
    lr_history_epochs = loss_history_record['lr_history_epochs']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return lr_scheduler_type, lr_history_epochs
# =============================================================================
def write_training_summary_file(
    device_type, seed, model_directory, load_model_state, n_max_epochs,
    is_data_normalization, batch_size, is_sampler_shuffle, loss_type,
    loss_kwargs, opt_algorithm, lr_init, lr_scheduler_type,
    lr_scheduler_kwargs, n_epochs, dataset_file_path, dataset, best_loss,
    best_training_epoch, total_time_sec, avg_time_epoch):
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
    is_data_normalization : bool
        If True, then input and output features are normalized for training
        False otherwise. Data scalers need to be fitted with fit_data_scalers()
        and are stored as model attributes.
    batch_size : int
        Number of samples loaded per batch.
    is_sampler_shuffle : bool
        If True, shuffles data set samples at every epoch.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    opt_algorithm : {'adam',}
        Optimization algorithm.
    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}
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
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['load_model_state'] = \
        load_model_state if load_model_state else None
    summary_data['n_max_epochs'] = n_max_epochs
    summary_data['is_data_normalization'] = is_data_normalization
    summary_data['batch_size'] = batch_size
    summary_data['is_sampler_shuffle'] = is_sampler_shuffle
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
    summary_data['Training data set (effective) size'] = len(dataset)
    summary_data['Best loss: '] = \
        f'{best_loss:.8e} (training epoch {best_training_epoch})'
    summary_data['Total training time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. training time per epoch'] = \
        str(datetime.timedelta(seconds=int(avg_time_epoch)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=model_directory,
        summary_title='Summary: Graph Neural Network model training',
        **summary_data)
# =============================================================================
class EarlyStopper:
    """Early stopping procedure (implicit regularizaton).
    
    Attributes
    ----------
    _validation_size : float
        Size of the validation data set for early stopping evaluation, where
        size is a fraction of the whole data set contained between 0 and 1.
    _validation_frequency : int
        Frequency of validation procedures, i.e., frequency with respect to
        training epochs at which model is validated to evaluate early stopping
        criterion.
    _trigger_tolerance : int
        Number of consecutive model validation procedures without performance
        improvement to trigger early stopping.
    _improvement_tolerance : float
        Minimum relative improvement required to count as a performance
        improvement.
    _validation_steps_history : list
        Validation steps history.
    _validation_loss_history : list
        Validation loss history.
    _min_validation_loss : float
        Minimum validation loss.
    _n_not_improve : int
        Number of consecutive model validations without improvement.
    _best_model_state : dict
        Model state corresponding to the best performance.
    _best_optimizer_state : dict
        Optimizer state corresponding to the best performance.
    _best_training_epoch : int
        Training epoch corresponding to the best performance.
            
    Methods
    -------
    get_training_dataset(self)
        Get Graph Neural Network model available training data set.
    get_validation_loss_history(self)
        Get validation loss history.
    is_evaluate_criterion(self, epoch)
        Check whether to evaluate early stopping criterion.
    evaluate_criterion(self, epoch)
        Evaluate early stopping criterion.
    _validate_model(self, model)
        Perform model validation.
    load_best_performance_state(self, model, optimizer)
        Load minimum validation loss model and optimizer states.
    """
    def __init__(self, dataset, validation_size=0.2, validation_frequency=1,
                 trigger_tolerance=1, improvement_tolerance=1e-3):
        """Constructor.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Graph Neural Network graph data set. Each sample corresponds to a
            torch_geometric.data.Data object describing a homogeneous graph.
        validation_size : float, default=0.2
            Size of the validation data set for early stopping evaluation,
            where size is a fraction of the whole data set contained between 0
            and 1.
        validation_frequency : int, default=1
            Frequency of validation procedures, i.e., frequency with respect to
            training epochs at which model is validated to evaluate early
            stopping criterion.
        trigger_tolerance : int, default=1
            Number of consecutive model validation procedures without
            performance improvement to trigger early stopping.
        improvement_tolerance : float, default=1e-3
            Minimum relative improvement required to count as a performance
            improvement.
        """
        # Set validation data set size
        self._validation_size = validation_size
        # Set validation frequency
        self._validation_frequency = validation_frequency
        # Set early stopping trigger tolerance
        self._trigger_tolerance = trigger_tolerance
        # Set minimum relative improvement tolerance
        self._improvement_tolerance = improvement_tolerance
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set split sizes
        split_sizes = {'training': (1.0 - validation_size),
                       'validation': validation_size}
        # Split data set
        dataset_split = split_dataset(dataset, split_sizes)
        # Set training and validation datasets
        self._training_dataset = dataset_split['training']
        self._validation_dataset = dataset_split['validation']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize validation training steps history
        self._validation_steps_history = []
        # Initialize validation loss history
        self._validation_loss_history = []
        # Initialize minimum validation loss
        self._min_validation_loss = np.inf
        # Initialize number of consecutive model validations without
        # improvement
        self._n_not_improve = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize minimum validation loss state (best performance)
        self._best_model_state = None
        self._best_optimizer_state = None
        self._best_training_epoch = None
    # -------------------------------------------------------------------------
    def get_training_dataset(self):
        """Get Graph Neural Network model available training data set.
        
        Returns
        -------
        training_dataset : torch.utils.data.Dataset
            Training data set.
        """
        # Get available training data set
        training_dataset = self._training_dataset
        self._training_dataset = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return training_dataset
    # -------------------------------------------------------------------------
    def get_validation_loss_history(self):
        """Get validation loss history.
        
        Returns
        -------
        validation_loss_history : list[float]
            Validation loss history.
        """
        return copy.deepcopy(self._validation_loss_history)
    # -------------------------------------------------------------------------
    def is_evaluate_criterion(self, epoch):
        """Check whether to evaluate early stopping criterion.
        
        Parameters
        ----------
        epoch : int
            Training epoch.
            
        Returns
        -------
        is_evaluate_criterion : bool
            If True, then early stopping criterion should be evaluated, False
            otherwise.
        """
        return epoch % self._validation_frequency == 0
    # -------------------------------------------------------------------------
    def evaluate_criterion(self, model, optimizer, epoch,
                           loss_type='mse', loss_kwargs={}, device_type='cpu'):
        """Evaluate early stopping criterion.
        
        Parameters
        ----------
        model : torch.nn.Module
            Graph Neural Network model.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        epoch : int
            Training epoch.
        loss_type : {'mse',}, default='mse'
            Loss function type:
            
            'mse'  : MSE (torch.nn.MSELoss)
            
        loss_kwargs : dict, default={}
            Arguments of torch.nn._Loss initializer.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
            
        Returns
        -------
        is_stop_training : bool
            True if early stopping criterion has been triggered, False
            otherwise.
        """
        # Set early stopping flag
        is_stop_training = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform model validation
        avg_valid_loss_sample = self._validate_model(
            model, optimizer, epoch, loss_type=loss_type,
            loss_kwargs=loss_kwargs, device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update minimum validation loss and performance counter
        if avg_valid_loss_sample < self._min_validation_loss:
            # Check relative performance improvement with respect to minimum
            # validation loss
            if len(self._validation_steps_history) > 1:
                # Compute relative performance improvement
                relative_improvement = \
                    (self._min_validation_loss - avg_valid_loss_sample)/ \
                    np.abs(self._min_validation_loss)
                # Update performance counter
                if relative_improvement > self._improvement_tolerance:
                    # Reset performance counter (significant improvement)
                    self._n_not_improve = 0
                else:
                    # Reset performance counter (not significant improvement)
                    self._n_not_improve += 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update minimum validation loss
            self._min_validation_loss = avg_valid_loss_sample
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save best performance state (minimum validation loss)
            self._best_model_state = copy.deepcopy(model.state_dict())
            self._best_optimizer_state = \
                dict(state=copy.deepcopy(optimizer.state_dict()), epoch=epoch)
            self._best_training_epoch = epoch
        else:
            # Increment performance counter
            self._n_not_improve += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate early stopping criterion
        if self._n_not_improve >= self._trigger_tolerance:
            # Trigger early stopping
            is_stop_training = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_stop_training
    # -------------------------------------------------------------------------
    def _validate_model(self, model, optimizer, epoch, loss_type='mse',
                        loss_kwargs={}, device_type='cpu'):
        """Perform model validation.
        
        Parameters
        ----------
        model : torch.nn.Module
            Graph Neural Network model.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        epoch : int
            Training epoch.
        loss_type : {'mse',}, default='mse'
            Loss function type:
            
            'mse'  : MSE (torch.nn.MSELoss)
            
        loss_kwargs : dict, default={}
            Arguments of torch.nn._Loss initializer.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
            
        Returns
        -------
        avg_predict_loss : float
            Average prediction loss per sample.
        """
        # Set material patch model state file name and path
        model_state_file = model.model_name + '-' + str(int(epoch))
        # Set material patch model state file path
        model_state_path = \
            os.path.join(model.model_directory, model_state_file + '.pt')
        # Set optimizer state file name and path
        optimizer_state_file = \
            model.model_name + '_optim' + '-' + str(int(epoch))
        optimizer_state_path = \
            os.path.join(model.model_directory, optimizer_state_file + '.pt')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize temporary state files flag
        is_state_file_temp = False
        # Save model and optimizer state files (required for validation)
        if not os.path.isfile(model_state_path):
            # Update temporary state files flag
            is_state_file_temp = True
            # Save state files
            save_training_state(model=model, optimizer=optimizer, epoch=epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prediction with Graph Neural Network model
        _, avg_valid_loss_sample = predict(
            self._validation_dataset, model.model_directory,
            predict_directory=None, load_model_state=epoch,
            loss_type=loss_type, loss_kwargs=loss_kwargs,
            is_normalized_loss=model.is_data_normalization,
            device_type=device_type, seed=None, is_verbose=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update validation epochs history
        self._validation_steps_history.append(epoch)
        # Propagate last validation loss until current epoch
        history_length = len(self._validation_loss_history)
        history_gap = epoch - history_length
        if history_length > 0:
            self._validation_loss_history += \
                history_gap*[self._validation_loss_history[-1],]
        # Append validation loss
        self._validation_loss_history.append(avg_valid_loss_sample)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove model and optimizer state files (required for validation)
        if is_state_file_temp:
            os.remove(model_state_path)
            os.remove(optimizer_state_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return avg_valid_loss_sample
    # -------------------------------------------------------------------------
    def load_best_performance_state(self, model, optimizer):
        """Load minimum validation loss model and optimizer states.
        
        Both model and optimizer are updated 'in-place' with stored state data.
        
        Parameters
        ----------
        model : torch.nn.Module
            Graph Neural Network model.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
            
        Returns
        -------
        best_training_epoch : int
            Training epoch corresponding to the best performance.
        """
        # Check best performance states
        if self._best_model_state is None:
            raise RuntimeError('The best performance model state has not been '
                               'stored.')
        if self._best_optimizer_state is None:
            raise RuntimeError('The best performance optimization state has '
                               'not been stored.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load material patch model state
        model.load_state_dict(self._best_model_state)
        # Set loaded optimizer state
        optimizer.load_state_dict(self._best_optimizer_state['state'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return self._best_training_epoch