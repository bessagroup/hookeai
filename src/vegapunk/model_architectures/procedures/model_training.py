"""Procedures associated to model training.

Functions
---------
save_training_state
    Save model and optimizer states at given training epoch.
save_loss_history
    Save training process loss history record.
read_loss_history_from_file
    Read training process loss history from loss history record file.
read_lr_history_from_file
    Read training learning rate history from loss history record file.
write_training_summary_file
    Write summary data file for model training process.
plot_training_loss_history
    Plot model training process loss history.
plot_training_loss_and_lr_history
    Plot model training process loss and learning rate histories.
plot_model_parameters_history
    Plot model learnable parameters history.
write_cross_validation_summary_file
    Write summary data file for model cross-validation process.
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
import matplotlib.pyplot as plt
# Local
from model_architectures.procedures.model_state_files import save_model_state
from ioput.iostandard import write_summary_file
from ioput.plots import plot_xy_data, plot_xy2_data, save_figure
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
    loss_nature : str
        Loss nature.
    loss_type : str
        Loss function type.
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
    if 'loss_nature' not in loss_history_record.keys():
        raise RuntimeError('Loss nature is not available in loss history '
                           'record.')
    elif 'loss_type' not in loss_history_record.keys():
        raise RuntimeError('Loss type is not available in loss history '
                           'record.')
    elif 'training_loss_history' not in loss_history_record.keys():
        raise RuntimeError('Loss history is not available in loss history '
                           'record.')
    elif not isinstance(loss_history_record['training_loss_history'],
                        list):
        raise RuntimeError('Loss history is not a list[float].')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss nature
    loss_nature = str(loss_history_record['loss_nature'])
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
    return (loss_nature, loss_type, training_loss_history,
            validation_loss_history)
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
        Type of learning rate scheduler.
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
# =============================================================================
def plot_training_loss_history(loss_history, loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='training_loss_history',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=False):
    """Plot model training process loss history.
    
    Parameters
    ----------
    loss_history : dict
        One or more training processes loss histories, where each loss history
        (key, str) is stored as a list of epochs loss values (item, list).
        Dictionary keys are taken as labels for the corresponding training
        processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check loss history
    if not isinstance(loss_history, dict):
        raise RuntimeError('Loss history is not a dict.')
    elif not all([isinstance(x, list) for x in loss_history.values()]):
        raise RuntimeError('Data must be provided as a dict where each loss '
                           'history (key, str) is stored as a list[float] '
                           '(item, list).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of training processes
    n_loss_history = len(loss_history.keys())
    # Get maximum number of training epochs
    max_n_train_epochs = max([len(x) for x in loss_history.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_train_epochs, 2*n_loss_history), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training processes
    for i, (key, val) in enumerate(loss_history.items()):
        # Assemble loss history
        data_xy[:len(val), 2*i] = tuple([*range(0, len(val))])
        if is_log_loss:
            data_xy[:len(val), 2*i+1] = tuple(np.log(val))
        else:
            data_xy[:len(val), 2*i+1] = tuple(val)
        # Assemble data label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, max_n_train_epochs)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
    if loss_type is None:
        if is_log_loss:
            y_label = 'log(Loss)'
        else:
            y_label = 'Loss'
    else:
        if is_log_loss:
            y_label = f'log(Loss) ({loss_type})'
        else:
            y_label = f'Loss ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xy_data(data_xy, data_labels=data_labels, x_lims=x_lims,
                             y_lims=y_lims, x_label=x_label,
                             y_label=y_label, y_scale=y_scale,
                             x_tick_format='int', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_training_loss_and_lr_history(loss_history, lr_history, loss_type=None,
                                      is_log_loss=False, loss_scale='linear',
                                      lr_type=None,
                                      filename='training_loss_and_lr_history',
                                      save_dir=None, is_save_fig=False,
                                      is_stdout_display=False, is_latex=False):
    """Plot model training process loss and learning rate histories.
    
    Parameters
    ----------
    loss_history : list[float]
        Training process loss history stored as a list of training epochs
        loss values.
    lr_history : list[float]
        Training process learning rate history stored as a list of training
        epochs learning rate values.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    lr_type : str, default=None
        Learning rate scheduler type. If provided, then learning rate scheduler
        type is added to the y-axis label.    
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check loss history
    if not isinstance(loss_history, list):
        raise RuntimeError('Loss history is not a list[float].')
    # Check learning rate history
    if not isinstance(lr_history, list):
        raise RuntimeError('Learning rate history is not a list[float].')
    elif len(lr_history) != len(loss_history):
        raise RuntimeError('Number of epochs of learning rate history is not '
                           'consistent with loss history.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data arrays
    x = tuple([*range(0, len(loss_history))])
    if is_log_loss:
        data_xy1 = np.column_stack((x, tuple(np.log(loss_history))))
    else:
        data_xy1 = np.column_stack((x, tuple(loss_history)))
    data_xy2 = np.column_stack((x, tuple(lr_history)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, len(loss_history))
    y1_lims = (None, None)
    y2_lims = (None, None)
    y1_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
    if loss_type is None:
        if is_log_loss:
            y1_label = 'log(Loss)'
        else:
            y1_label = 'Loss'
    else:
        if is_log_loss:
            y1_label = f'log(Loss) ({loss_type})'
        else:
            y1_label = f'Loss ({loss_type})'
    if lr_type is None:
        y2_label = 'Learning rate'
    else:
        y2_label = f'Learning rate ({lr_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Training loss and learning rate history'    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss and learning rate history
    figure, _ = plot_xy2_data(data_xy1, data_xy2, x_lims=x_lims,
                              y1_lims=y1_lims, y2_lims=y2_lims,
                              x_label=x_label, y1_label=y1_label,
                              y2_label=y2_label, y1_scale=y1_scale,
                              x_tick_format='int', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_model_parameters_history(model_parameters_history,
                                  model_parameters_bounds,
                                  filename='model_parameter_history',
                                  save_dir=None, is_save_fig=False,
                                  is_stdout_display=False, is_latex=False):
    """Plot model learnable parameters history.
    
    Parameters
    ----------
    model_parameters_history : dict
        Model learnable parameters history. For each model parameter
        (key, str), store the corresponding training history (item, list).
    model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    filename : str, default='model_parameter_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check model parameters history
    if not isinstance(model_parameters_history, dict):
        raise RuntimeError('Model parameters history is not a dict.')
    elif not all([isinstance(x, list)
                  for x in model_parameters_history.values()]):
        raise RuntimeError('Data must be provided as a dict where each '
                           'parameter history (key, str) is stored as a '
                           'list.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters
    for name, history in model_parameters_history.items():
        # Initialize data array
        data_xy = np.zeros((len(history), 2))
        # Build data array
        data_xy[:, 0] = np.arange(len(history))
        data_xy[:, 1] = history
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        x_lims = (0, len(history))
        y_lims = (None, None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Epochs'
        y_label = f'Parameter: {name}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot parameter history
        figure, axes = plot_xy_data(data_xy, x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    x_tick_format='int', marker='o',
                                    markersize=2, is_latex=is_latex)
        # Plot parameter bounds
        axes.hlines(model_parameters_bounds[name], 0, len(history),
                    colors='k', linestyles='dashed')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, f'{filename}_{name}',
                        format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
def write_cross_validation_summary_file(
    cross_validation_dir, device_type, n_fold, n_max_epochs,
    is_model_in_normalized, is_model_out_normalized, batch_size, loss_nature,
    loss_type, loss_kwargs, dataset_file_path, dataset, k_fold_loss_array,
    total_time_sec, avg_time_fold):
    """Write summary data file for model cross-validation process.
    
    Parameters
    ----------
    cross_validation_dir : dir
        Directory where cross-validation process data is stored.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    n_fold : int
        Number of folds into which the data set is split to perform
        cross-validation.
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
    loss_nature : str
        Loss nature.
    loss_type : str
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    dataset_file_path : str
        Data set file path if such file exists. Only used for output purposes
    dataset : torch.utils.data.Dataset
        Data set.
    k_fold_loss_array : numpy.ndarray(2d)
        k-fold cross-validation loss array. For the i-th fold,
        data_array[i, 0] stores the best training loss and data_array[i, 1]
        stores the average prediction loss per sample.
    total_time_sec : int
        Total cross-validation time in seconds.
    avg_time_fold : float
        Average cross-validation time per fold.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['n_fold'] = n_fold
    summary_data['n_max_epochs'] = n_max_epochs
    summary_data['is_model_in_normalized'] = is_model_in_normalized
    summary_data['is_model_out_normalized'] = is_model_out_normalized
    summary_data['batch_size'] = batch_size
    summary_data['loss_nature'] = loss_nature
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['k-fold cross-validation data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['k-fold cross-validation data set size'] = len(dataset)
    summary_data['k-fold cross-validation results'] = k_fold_loss_array
    summary_data['Total cross-validation time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. cross-validation time per fold'] = \
        str(datetime.timedelta(seconds=int(avg_time_fold)))
    # Set summary title
    summary_title = 'Summary: Model k-fold cross-validation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(summary_directory=cross_validation_dir,
                       summary_title=summary_title, **summary_data)