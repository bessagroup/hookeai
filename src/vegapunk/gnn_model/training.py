"""Training of Graph Neural Network based material patch model.

Functions
---------
train_model
    Training of GNN-based material patch model.
get_pytorch_loss
    Get PyTorch loss function.
get_pytorch_optimizer
    Get PyTorch optimizer.
get_learning_rate_scheduler
    Get PyTorch optimizer learning rate scheduler.
save_training_state
    Save model and optimizer states at given training step.
load_training_state
    Load model and optimizer states from available training data.
remove_posterior_optim_state_files
    Delete optimizer training step state files posterior to given step.
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
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import random
import re
# Third-party
import torch
import torch_geometric.loader
import numpy as np
# Local
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def train_model(n_train_steps, dataset, model_init_args, lr_init,
                opt_algorithm='adam', lr_scheduler_type=None,
                lr_scheduler_kwargs={}, loss_type='mse', loss_kwargs={},
                batch_size=1, is_sampler_shuffle=False, load_model_state=None,
                save_every=None, device_type='cpu', seed=None,
                is_verbose=False):
    """Training of GNN-based material patch model.
    
    Parameters
    ----------
    n_train_steps : int
        Number of training steps.
    dataset : GNNMaterialPatchDataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_init_args : dict
        GNN-based material patch model class initialization parameters (check
        class GNNMaterialPatchModel).
    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    opt_algorithm : {'adam',}, default='adam'
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)
        
    lr_scheduler_type : {'steplr',}, default=None
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)

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
    load_model_state : {'best', 'last', int, 'default'}, default=None
        Load available GNN-based material patch model state from the model
        directory. Data scalers are also loaded from model initialization file.
        Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training step
        
        int         : Model state corresponding to given training step
        
        'default'   : Model default state file

    save_every : int, default=None
        Save GNN-based material patch model every save_every training steps.
        If None, then saves only last training step and best performance
        states.
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
        GNN-based material patch model.
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
    if is_verbose:
        print('\nGNN-based material patch data model training'
              '\n--------------------------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Move model to process ID
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
    # Initialize loss and learning rate histories
    loss_history = []
    lr_history = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training flag
    is_keep_training = True
    # Initialize training step counter
    step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model state
    if load_model_state is not None:
        # Initialize GNN-based material patch model
        # (includes loading of data scalers)
        model = GNNMaterialPatchModel.init_model_from_file(
            model_init_args['model_directory'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Move model to process ID
        model.to(device=device)
        # Set model in training mode
        model.train()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load GNN-based material patch model state
        loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                          load_model_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load loss history
        loss_history = load_loss_history(model, loss_type,
                                         training_step=loaded_step)
        # Load learning rate history
        lr_history = load_lr_history(model, training_step=loaded_step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update training step counter
        step = int(loaded_step)
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
    # Loop over training iterations
    while is_keep_training:
        # Loop over graph batches. A graph batch is a data object describing a
        # batch of graphs as one large (disconnected) graph.
        for pyg_graph in data_loader:
            # Move graph sample to process ID
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node internal forces predictions (forward
            # propagation). During the foward pass, PyTorch creates a
            # computation graph for the tensors that require gradients
            # (gradient flag set to True) to keep track of the operations
            # on these tensors, i.e., the model parameters. In addition,
            # PyTorch additionally stores the corresponding 'gradient
            # functions' (mathematical operator) of the executed operations
            # to the output tensor, stored in the .grad_fn attribute of the
            # corresponding tensors. Tensor.grad_fn is set to None for
            # tensors corresponding to leaf-nodes of the computation graph
            # or for tensors with the gradient flag set to False.
            node_internal_forces = model.predict_internal_forces(
                pyg_graph, is_normalized=is_data_normalization)
            # Get node internal forces ground-truth
            node_internal_forces_target = \
                model.get_output_features_from_graph(
                    pyg_graph, is_normalized=is_data_normalization)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute and accumulate loss
            loss = loss_function(node_internal_forces,
                                 node_internal_forces_target)
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
                print('> Training step: {:{width}d}/{:d} | '
                      'Loss: {:.8e}'.format(step, n_train_steps, loss,
                                            width=len(str(n_train_steps))),
                      end='\r')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save training step loss and learning rate
            loss_history.append(loss)
            lr_history.append(lr_scheduler.get_last_lr())
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save model and optimizer current states
            if save_every is not None and step % save_every == 0:
                save_training_state(model=model, optimizer=optimizer,
                                    training_step=step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save model and optimizer current best performant state
            # (criterion: minimum loss during training process)
            if loss <= min(loss_history):
                save_training_state(model=model, optimizer=optimizer,
                                    training_step=step, is_best_state=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check training process completion
            if step >= n_train_steps:
                # Completed prescribed number of training steps
                is_keep_training = False
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Increment training step counter
            step += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model and optimizer final states
    save_training_state(model=model, optimizer=optimizer, training_step=step)
    # Save loss and learning rate histories
    save_loss_history(model, n_train_steps, loss_type, loss_history,
                      lr_scheduler_type=lr_scheduler_type,
                      lr_history=lr_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        best_loss = min(loss_history)
        training_step = loss_history.index(best_loss)
        print('\n\n> Minimum loss: {:.8e} | Training step: {:d}'.format(
            best_loss, training_step))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Model directory: {model.model_directory}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model
# =============================================================================
def get_pytorch_loss(loss_type, **kwargs):
    """Get PyTorch loss function.
   
    Parameters
    ----------
    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    **kwargs
        Arguments of torch.nn._Loss initializer.
        
    Returns
    -------
    loss_function : torch.nn._Loss
        PyTorch loss function.
    """
    if loss_type == 'mse':
        loss_function = torch.nn.MSELoss(**kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch loss function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_function
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
def get_learning_rate_scheduler(optimizer, scheduler_type='steplr', **kwargs):
    """Get PyTorch optimizer learning rate scheduler.
    
    Parameters
    ----------
    scheduler_type : {'steplr',}
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    **kwargs
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        PyTorch optimizer learning rate scheduler.
    """
    if scheduler_type == 'steplr':
        # Check step size (period of learning rate decay)
        if 'step_size' not in kwargs.keys():
            raise RuntimeError('The step_size needs to be provided to '
                               'initialize step-based decay learning rate '
                               'scheduler.')
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch optimizer '
                           'learning rate scheduler.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return scheduler
# =============================================================================
def save_training_state(model, optimizer, training_step=None,
                        is_best_state=False, is_remove_posterior=True):
    """Save model and optimizer states at given training step.
    
    Material patch model state file is stored in model_directory under the
    name < model_name >.pt or < model_name >-< training_step >.pt if
    training_step is known.
    
    Material patch model state file corresponding to the best performance
    is stored in model_directory under the name < model_name >-best.pt or
    < model_name >-< training_step >-best.pt if training_step is known.
        
    Optimizer state file is stored in model_directory under the name
    < model_name >_optim-< training_step >.pt.
    
    Optimizer state file corresponding to the best performance is stored in
    model_directory under the name < model_name >_optim-best.pt or
    < model_name >_optim-< training_step >-best.pt if training_step is known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    training_step : int, default=None
        Training step.
    is_best_state : bool, default=False
        If True, save material patch model state file corresponding to the best
        performance instead of regular state file.
    is_remove_posterior : bool, default=True
        Remove material patch model and optimizer state files corresponding to
        training steps posterior to the saved state file. Effective only if
        saved training step is known.
    """
    # Save GNN-based material patch model
    model.save_model_state(training_step=training_step,
                           is_best_state=is_best_state)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file
    optimizer_state_file = model.model_name + '_optim'
    # Append training step
    if isinstance(training_step, int):
        optimizer_state_file += '-' + str(training_step)
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
            # Check if file is optimizer training step best state file
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
                           training_step=training_step)
    torch.save(optimizer_state, optimizer_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Delete model and optimizer training step state files posterior to saved
    # training step
    if isinstance(training_step, int) and is_remove_posterior:
        remove_posterior_optim_state_files(model, training_step)
# =============================================================================
def load_training_state(model, opt_algorithm, optimizer,
                        load_model_state=None, is_remove_posterior=True):
    """Load model and optimizer states from available training data.
    
    Material patch model state file is stored in model_directory under the
    name < model_name >.pt, < model_name >-< training_step >.pt,
    < model_name >-best.pt or < model_name >-< training_step >-best.pt.

    Optimizer state file is stored in model_directory under the name
    < model_name >_optim.pt or < model_name >_optim-< training_step >.pt.
    
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
        Load available GNN-based material patch model state from the model
        directory. Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training step
        
        int         : Model state corresponding to given training step
        
        None   : Model default state file

    is_remove_posterior : bool, default=True
        Remove material patch model state files corresponding to training
        steps posterior to the loaded state file. Effective only if
        loaded training step is known.

    Returns
    -------
    loaded_step : int
        Training step corresponding to loaded state data. Defaults to 0 if
        training step is unknown.
    """
    # Load GNN-based material patch model state        
    loaded_step = \
        model.load_model_state(load_model_state=load_model_state,
                               is_remove_posterior=is_remove_posterior)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file
    optimizer_state_file = model.model_name + '_optim'
    # Append training step
    if isinstance(loaded_step, int):
        optimizer_state_file += '-' + str(loaded_step)
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
        # Delete optimizer training step state files posterior to loaded
        # training step
        if isinstance(loaded_step, int) and is_remove_posterior:
            remove_posterior_optim_state_files(model, loaded_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loaded training step to 0 if unknown from state file
    if loaded_step is None:
        loaded_step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loaded_step
# =============================================================================
def remove_posterior_optim_state_files(model, training_step):
    """Delete optimizer training step state files posterior to given step.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    training_step : int
        Training step.
    """
    # Get files in material patch model directory
    directory_list = os.listdir(model.model_directory)
    # Loop over files in material patch model directory
    for filename in directory_list:
        # Check if file is optimizer training step state file
        is_state_file = bool(re.search(r'^' + model.model_name + r'_optim'
                             + r'-[0-9]+' + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete optimizer training step state file posterior to given training
        # step
        if is_state_file:
            # Get optimizer state training step
            step = int(os.path.splitext(filename)[0].split('-')[-1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete optimizer training step state file
            if step > training_step:
                os.remove(os.path.join(model.model_directory, filename))
# =============================================================================
def save_loss_history(model, total_n_train_steps, loss_type, loss_history,
                      lr_scheduler_type=None, lr_history=None):
    """Save training process loss history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.

    Overwrites existing loss history record file.
    
    Does also store learning history if provided.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    total_n_train_steps : int
        Total number of training steps prescribed for training process.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)

    loss_history : list[float]
        Training process loss history.
    lr_scheduler_type : {'steplr',}, default=None
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
    lr_history : list[float], default=None
        Training process learning rate history.
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build loss history record
    loss_history_record = {}
    loss_history_record['total_n_train_steps'] = int(total_n_train_steps)
    loss_history_record['loss_type'] = str(loss_type)
    loss_history_record['loss_history'] = list(loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store learning rate history record
    if lr_scheduler_type is not None:
        loss_history_record['lr_scheduler_type'] = str(lr_scheduler_type)
    else:
        loss_history_record['lr_scheduler_type'] = None
    if lr_history is not None:
        loss_history_record['lr_history'] = list(lr_history)
    else:
        loss_history_record['lr_history'] = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save loss history record
    with open(loss_record_path, 'wb') as loss_record_file:
        pickle.dump(loss_history_record, loss_record_file)
# =============================================================================
def load_loss_history(model, loss_type, training_step=None):
    """Load training process loss history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    training_step : int, default=None
        Training step to which loss history is loaded (included), with the
        first training step being 0. If None, then loads the full loss history.

    Returns
    -------
    loss_history : list[float]
        Training process loss history.
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model.model_directory,
                                    'loss_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training process loss history
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
        # Check loss history
        loss_record = loss_history_record['loss_history']
        if not isinstance(loss_record, list):
            raise RuntimeError('Loaded loss history is not a list[float].')
        # Load loss history
        if training_step is None or training_step + 1 == len(loss_record):
            loss_history = loss_record
        else:
            if training_step + 1 > len(loss_record):
                raise RuntimeError('Target training step is beyond available '
                                   'loss history.')
            else:
                loss_history = loss_record[:training_step + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Build loss history with None entries if loss history record file
        # cannot be found
        if training_step is None:
            raise RuntimeError('Training process loss history file has not '
                               'been found and loaded training step is '
                               'unknown.')
        else:
            loss_history = (training_step + 1)*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_history
# =============================================================================
def load_lr_history(model, training_step=None):
    """Load training process learning rate history record.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.        
    training_step : int, default=None
        Training step to which learning history is loaded (included), with the
        first training step being 0. If None, then loads the full learning rate
        history.

    Returns
    -------
    lr_history : list[float]
        Training process learning rate history.
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
        lr_record = loss_history_record['lr_history']
        if not isinstance(lr_record, list) and lr_record is not None:
            raise RuntimeError('Loaded learning rate history is not a '
                               'list[float] or None.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        # Load learning rate history
        if lr_record is None and training_step is None:
            # Build learning rate history with None entries if learning rate
            # history is not available
            lr_history = (len(loss_history_record['loss_history']))*[None,]
        elif lr_record is None and training_step is not None:
            # Build learning rate history with None entries if learning rate
            # history is not available
            lr_history = (training_step + 1)*[None,]
        else:
            if training_step is None or training_step + 1 == len(lr_record):
                lr_history = lr_record
            elif training_step + 1 > len(lr_record):
                    raise RuntimeError('Target training step is beyond '
                                       'available learning rate history.')
            else:
                lr_history = lr_record[:training_step + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Build learning rate history with None entries if loss history record
        # file cannot be found
        if training_step is None:
            raise RuntimeError('Training process loss history file has not '
                               'been found and loaded training step is '
                               'unknown.')
        else:
            lr_history = (training_step + 1)*[None,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return lr_history
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
    """Read training loss history from loss history record file.
    
    Loss history record file is stored in model_directory under the name
    loss_history_record.pkl.
    
    Detaches loss values from computation graph.
    
    Parameters
    ----------
    loss_record_path : str
        Loss history record file path.
    
    Returns
    -------
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)

    loss_history : list[float]
        Training process loss history.
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
    elif 'loss_history' not in loss_history_record.keys():
        raise RuntimeError('Loss history is not available in loss history '
                           'record.')
    elif not isinstance(loss_history_record['loss_history'], list):
        raise RuntimeError('Loss history is not a list[float].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss type
    loss_type = str(loss_history_record['loss_type'])
    # Set loss history
    loss_history = [x.detach() for x in loss_history_record['loss_history']]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_type, loss_history
