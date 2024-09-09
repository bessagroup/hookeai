"""DARPA METALS PROJECT: Material model finder learning procedure.

Functions
---------
train_model
    Training of recurrent constitutive model.
save_parameters_history
    Save model learnable parameters history record.
read_parameters_history_from_file
    Read model learnable parameters history from parameters record file.
save_best_parameters
    Save best performance state model parameters.
read_best_parameters_from_file
    Read best performance state model parameters from file.
write_training_summary_file
    Write summary data file for model training process.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import random
import time
import datetime
import pickle
# Third-party
import torch
import numpy as np
# Local
from material_model_finder.model.material_discovery import MaterialModelFinder
from gnn_base_model.train.training import get_pytorch_optimizer, \
    get_learning_rate_scheduler, save_training_state, save_loss_history
from gnn_base_model.model.model_summary import get_model_summary
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
def train_model(n_max_epochs, specimen_data, specimen_material_state,
                model_init_args, lr_init, opt_algorithm='adam',
                lr_scheduler_type=None, lr_scheduler_kwargs={},
                is_explicit_model_parameters=False,
                save_every=None, device_type='cpu', seed=None,
                is_verbose=False):
    """Training of recurrent constitutive model.
    
    Parameters
    ----------
    n_max_epochs : int
        Maximum number of training epochs.
    specimen_data : SpecimenNumericalData
        Specimen numerical data translated from experimental results.
    specimen_material_state : StructureMaterialState
        FETorch structure material state.
    model_init_args : dict
        Material model finder class initialization parameters (check class
        MaterialModelFinder).
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
    is_explicit_model_parameters : bool, default=False
        If True, then activate the explicit handling of model parameters. This
        includes enforcing available bounds on the parameters during the
        training procedure and storing the model parameters history for
        post-processing.
    save_every : int, default=None
        Save model every save_every epochs. If None, then saves only last epoch
        and best performance states.
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
        Recurrent neural network model.
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
        print('\nMaterial model finder learning procedure'
              '\n----------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material model finder
    model = MaterialModelFinder(**model_init_args)
    # Set specimen data and material state
    model.set_specimen_data(specimen_data, specimen_material_state)
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
    # Initialize learning rate
    learning_rate = lr_init
    # Set optimizer    
    if opt_algorithm == 'adam':
        # Initialize optimizer, specifying the model (and submodels) parameters
        # that should be optimized
        optimizer = get_pytorch_optimizer(algorithm=opt_algorithm,
                                          params=model_parameters,
                                          **{'lr': learning_rate})
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
    # Initialize loss and learning rate histories (per epoch)
    loss_history_epochs = []
    lr_history_epochs = []
    # Initialize loss and learning rate histories (per training step)
    loss_history_steps = []
    lr_history_steps = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize storage of model parameters history
    # (only explicit learnable parameters)
    if is_explicit_model_parameters:
        # Initialize model parameters history (per epoch)
        model_init_parameters = model.get_detached_model_parameters()
        model_parameters_history_epochs = \
            {key: [] for key, _ in model_init_parameters.items()}
        # Initialize model parameters history (per training step)
        model_parameters_history_steps = \
            {key: [] for key, _ in model_init_parameters.items()}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training flag
    is_keep_training = True
    # Initialize number of training epochs
    epoch = 0
    # Initialize number of training steps
    step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Starting training process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training iterations
    while is_keep_training:
        # Store epoch initial training step
        epoch_init_step = step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium history loss
        loss = model(sequential_mode='sequential_element_vmap')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize gradients (set to zero)
        optimizer.zero_grad()
        # Compute gradients with respect to model parameters (backward
        # propagation)
        loss.backward()
        # Perform optimization step. Gradients are stored in the .grad
        # attribute of model parameters
        optimizer.step()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce bounds on model parameters
        # (only explicit learnable parameters)
        if is_explicit_model_parameters:
            model.enforce_parameters_bounds()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            total_time_sec = time.time() - start_time_sec
            print('> Epoch: {:{width}d}/{:d} | Training step: {:d} | '
                    'Loss: {:.8e} | Elapsed time (s): {:}'.format(
                epoch, n_max_epochs, step, loss,
                str(datetime.timedelta(seconds=int(total_time_sec))),
                width=len(str(n_max_epochs))), end='\r')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save training step loss and learning rate
        loss_history_steps.append(loss.detach().clone().cpu())
        if is_lr_scheduler:
            lr_history_steps.append(lr_scheduler.get_last_lr())
        else:
            lr_history_steps.append(lr_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model parameters (only explicit learnable parameters)
        if is_explicit_model_parameters:
            # Get current model parameters
            model_parameters_step = model.get_detached_model_parameters()
            # Loop over model parameters
            for key, val in model_parameters_step.items():
                model_parameters_history_steps[key].append(val)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment training step counter
        step += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update optimizer learning rate
        if is_lr_scheduler:
            lr_scheduler.step()
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
        # Save model parameters (epoch average parameter value)
        if is_explicit_model_parameters:
            # Loop over parameters
            for key in model_init_parameters.keys():
                model_parameters_history_epochs[key].append(np.mean(
                    model_parameters_history_steps[key][epoch_init_step:]))
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
            # Save model parameters (only explicit learnable parameters)
            if is_explicit_model_parameters:
                best_model_parameters = model.get_detached_model_parameters()
            # Save material models state
            save_material_models_state(model=model, epoch=epoch,
                                       is_best_state=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check training process flow
        if epoch >= n_max_epochs:
            # Completed maximum number of epochs
            is_keep_training = False
            break
        else:
            # Increment epoch counter
            epoch += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Finished training process!')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model and optimizer final states
    save_training_state(model=model, optimizer=optimizer, epoch=epoch)
    # Save material models final states
    save_material_models_state(model=model, epoch=epoch)
    # Save loss and learning rate histories
    save_loss_history(model, n_max_epochs, 'Force equilibrium history',
                      'None', loss_history_epochs,
                      lr_scheduler_type=lr_scheduler_type,
                      lr_history_epochs=lr_history_epochs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model parameters history (only explicit learnable parameters)
    if is_explicit_model_parameters:
        save_parameters_history(model, model_parameters_history_epochs,
                                model.get_model_parameters_bounds())
        save_best_parameters(model, best_model_parameters,
                             model.get_model_parameters_bounds())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get best loss and corresponding training epoch
    best_loss = float(min(loss_history_epochs))
    best_training_epoch = loss_history_epochs.index(best_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Minimum training loss: {:.8e} | Epoch: {:d}'.format(
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
    # Display best performance state model parameters
    if is_explicit_model_parameters and is_verbose:
        print('\nBest performance state model parameters'
              '\n---------------------------------------')
        # Loop over model parameters
        for key, val in best_model_parameters.items():
            print(f'> Parameter: {key} = {val}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get summary of PyTorch model
    model_statistics = get_model_summary(model, device_type=device_type)
    # Write summary data file for model training process
    write_training_summary_file(
        device_type, seed, model.model_directory, n_max_epochs,
        opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, epoch,
        best_loss, best_training_epoch, total_time_sec, avg_time_epoch,
        best_model_parameters=best_model_parameters,
        torchinfo_summary=str(model_statistics))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model, best_loss, best_training_epoch
# =============================================================================
def save_material_models_state(model, epoch=None, is_best_state=None):
    """Save material models states at given training epoch.
    
    Material model state file is stored in the corresponding model_directory
    under the name < model_name >.pt or < model_name >-< epoch >.pt if epoch is
    known.
    
    Material model state file corresponding to the best performance
    is stored in the corresponding model_directory under the name
    < model_name >-best.pt or < model_name >-< epoch >-best.pt if epoch is
    known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    epoch : int, default=None
        Training epoch.
    is_best_state : bool, default=False
        If True, save material model state file corresponding to the best
        performance instead of regular state file.
    """
    # Get material models
    material_models = model.get_material_models()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material models
    for _, model in material_models.items():
        # Save model state
        model.save_model_state(epoch=epoch, is_best_state=is_best_state)
# =============================================================================
def save_parameters_history(model, model_parameters_history,
                            model_parameters_bounds):
    """Save model learnable parameters history record.
    
    Model parameters record file is stored in model_directory under the name
    parameters_history_record.pkl.

    Overwrites existing model parameters record file.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    model_parameters_history : dict
        Model learnable parameters history. For each model parameter
        (key, str), store the corresponding training history (item, list).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    """
    # Set model parameters record file path
    parameters_record_path = os.path.join(model.model_directory,
                                          'parameters_history_record' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model parameters history record
    parameters_history_record = {}
    parameters_history_record['model_parameters_history'] = \
        model_parameters_history
    parameters_history_record['model_parameters_bounds'] = \
        model_parameters_bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model parameters history record
    with open(parameters_record_path, 'wb') as parameters_record_file:
        pickle.dump(parameters_history_record, parameters_record_file) 
# =============================================================================
def read_parameters_history_from_file(parameters_record_path):
    """Read model learnable parameters history from parameters record file.
    
    Model parameters record file is stored in model_directory under the name
    parameters_history_record.pkl.
    
    Parameters
    ----------
    parameters_record_path : str
        Model parameters history record file path.

    Returns
    -------
    model_parameters_history : dict
        Model learnable parameters history. For each model parameter
        (key, str), store the corresponding training history (item, list).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    """
    # Check model parameters history record file
    if not os.path.isfile(parameters_record_path):
        raise RuntimeError('Model parameters history record file has not been '
                           'found:\n\n' + parameters_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model parameters history record
    with open(parameters_record_path, 'rb') as parameters_record_file:
        parameters_history_record = pickle.load(parameters_record_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model parameters history
    model_parameters_history = \
        parameters_history_record['model_parameters_history']
    model_parameters_bounds = \
        parameters_history_record['model_parameters_bounds']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_parameters_history, model_parameters_bounds
# =============================================================================
def save_best_parameters(model, best_model_parameters,
                         model_parameters_bounds):
    """Save best performance state model parameters.
    
    Best performance state model parameters file is stored in model_directory
    under the name parameters_best.pkl.

    Overwrites existing model parameters file.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    best_model_parameters : dict
        Model best performance state learnable parameters. For each model
        parameter (key, str), store the corresponding value (item, float).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    """
    # Set model best parameters file path
    parameters_file_path = os.path.join(model.model_directory,
                                        'parameters_best' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model best parameters record
    parameters_record = {}
    parameters_record['best_model_parameters'] = best_model_parameters
    parameters_record['model_parameters_bounds'] = model_parameters_bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model best parameters
    with open(parameters_file_path, 'wb') as parameters_file:
        pickle.dump(parameters_record, parameters_file)
# =============================================================================
def read_best_parameters_from_file(parameters_file_path):
    """Read best performance state model parameters from file.
    
    Best performance state model parameters file is stored in model_directory
    under the name parameters_best.pkl.
    
    Parameters
    ----------
    parameters_file_path : str
        Model parameters file path.

    Returns
    -------
    best_model_parameters : dict
        Model best performance state learnable parameters. For each model
        parameter (key, str), store the corresponding value (item, float).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    """
    # Check model best parameters file path
    if not os.path.isfile(parameters_file_path):
        raise RuntimeError('Model best parameters file has not been '
                           'found:\n\n' + parameters_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model best parameters
    with open(parameters_file_path, 'rb') as parameters_file:
        parameters_record = pickle.load(parameters_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model parameters history
    best_model_parameters = parameters_record['best_model_parameters']
    model_parameters_bounds = parameters_record['model_parameters_bounds']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return best_model_parameters, model_parameters_bounds
# =============================================================================
def write_training_summary_file(
    device_type, seed, model_directory, n_max_epochs, opt_algorithm, lr_init,
    lr_scheduler_type, lr_scheduler_kwargs, n_epochs, best_loss,
    best_training_epoch, total_time_sec, avg_time_epoch,
    best_model_parameters=None, torchinfo_summary=None):
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
        Directory where model is stored.
    n_max_epochs : int
        Maximum number of training epochs.
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
    summary_data['n_max_epochs'] = n_max_epochs
    summary_data['opt_algorithm'] = opt_algorithm
    summary_data['lr_init'] = lr_init
    summary_data['lr_scheduler_type'] = \
        lr_scheduler_type if lr_scheduler_type else None
    summary_data['lr_scheduler_kwargs'] = \
        lr_scheduler_kwargs if lr_scheduler_kwargs else None
    summary_data['Number of completed epochs'] = n_epochs
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