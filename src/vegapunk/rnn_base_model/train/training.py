"""Training of recurrent neural network model.

Classes
-------
EarlyStopper
    Early stopping procedure (implicit regularizaton).

Functions
---------
train_model
    Training of recurrent neural network model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import random
import time
import datetime
import copy
# Third-party
import torch
import numpy as np
# Local
from rnn_base_model.data.time_dataset import get_time_series_data_loader
from rnn_base_model.model.gru_model import GRURNNModel
from rnn_base_model.predict.prediction import predict
from gnn_base_model.train.training import get_pytorch_optimizer, \
    get_learning_rate_scheduler, save_training_state, save_loss_history, \
    seed_worker, write_training_summary_file
from utilities.loss_functions import get_pytorch_loss
from gnn_base_model.model.model_summary import get_model_summary
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
                lr_scheduler_kwargs={}, loss_nature='features_out',
                loss_type='mse', loss_kwargs={},
                batch_size=1, is_sampler_shuffle=False,
                is_early_stopping=False, early_stopping_kwargs={},
                load_model_state=None, save_every=None, dataset_file_path=None,
                device_type='cpu', seed=None, is_verbose=False):
    """Training of recurrent neural network model.
    
    Parameters
    ----------
    n_max_epochs : int
        Maximum number of training epochs.
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    model_init_args : dict
        Recurrent neural network model class initialization parameters (check
        class GRURNNModel).
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
    loss_nature : {'features_out',}, default='features_out'
        Loss nature:
        
        'features_out' : Based on output features

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
        is triggered.
    early_stopping_kwargs : dict, default={}
        Early stopping criterion parameters (key, str, item, value).
    load_model_state : {'best', 'last', 'init', int, None}, default=None
        Load available model state from the model directory. Data scalers are
        also loaded from model initialization file.
        Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training epoch
        
        int         : Model state corresponding to given training epoch
        
        'init'      : Model state corresponding to initial state
        
        None        : Model default state file

    save_every : int, default=None
        Save model every save_every epochs. If None, then saves only initial,
        last epoch and best performance states.
    dataset_file_path : str, default=None
        Time series data set file path if such file exists. Only used for
        output purposes.
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
        print('\nRecurrent Neural Network model training'
              '\n---------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize recurrent neural network model state
    if load_model_state is not None:
        if is_verbose:
            print('\n> Initializing model...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize recurrent neural network model
        # (includes loading of data scalers)
        model = GRURNNModel.init_model_from_file(
            model_directory=model_init_args['model_directory'])
        # Set model device
        model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model input and output features normalization
        is_model_in_normalized = model.is_model_in_normalized
        is_model_out_normalized = model.is_model_out_normalized
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Loading model state...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load recurrent neural network model state
        _ = model.load_model_state(load_model_state=load_model_state,
                                   is_remove_posterior=True)
    else:
        if is_verbose:
            print('\n> Initializing model...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize recurrent neural network model
        model = GRURNNModel(**model_init_args)    
        # Set model device
        model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model input and output features normalization
        is_model_in_normalized = model.is_model_in_normalized
        is_model_out_normalized = model.is_model_out_normalized
        # Fit model data scalers  
        if is_model_in_normalized or is_model_out_normalized:
            model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Save model initial state
    model.save_model_init_state()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model parameters
    model_parameters = model.parameters(recurse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Move model to device
    model.to(device=device)
    # Set model in training mode
    model.train()
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
    # Initialize validation loss history
    validation_loss_history = None
    # Initialize early stopping criterion
    if is_early_stopping:
        if is_verbose:
            print('\n> Initializing early stopping criterion...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize early stopping criterion
        early_stopper = EarlyStopper(**early_stopping_kwargs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize early stopping flag
        is_stop_training = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Training data set size: {len(dataset)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    if isinstance(seed, int):
        data_loader = get_time_series_data_loader(
            dataset=dataset, batch_size=batch_size,
            is_shuffle=is_sampler_shuffle,
            kwargs={'worker_init_fn': seed_worker, 'generator': generator})
    else:
        data_loader = get_time_series_data_loader(
            dataset=dataset, batch_size=batch_size,
            is_shuffle=is_sampler_shuffle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        input_normalization_str = 'Yes' if is_model_in_normalized else 'No'
        print(f'\n> Input data normalization: {input_normalization_str}')
        output_normalization_str = 'Yes' if is_model_out_normalized else 'No'
        print(f'\n> Output data normalization: {output_normalization_str}')
        print('\n\n> Starting training process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training iterations
    while is_keep_training:
        # Store epoch initial training step
        epoch_init_step = step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over batches
        for batch in data_loader:
            # Move batch to device
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get input features
            if is_model_in_normalized:
                # Normalize features ground-truth
                features_in = \
                    model.data_scaler_transform(tensor=batch['features_in'],
                                                features_type='features_in',
                                                mode='normalize')
            else:
                features_in = batch['features_in']
            # Get initial hidden state features
            if 'hidden_features_in' in batch.keys():
                hidden_features_in = batch['hidden_features_in']
            else:
                hidden_features_in = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get output features ground-truth
            if is_model_out_normalized:
                # Normalize features ground-truth
                targets = \
                    model.data_scaler_transform(tensor=batch['features_out'],
                                                features_type='features_out',
                                                mode='normalize')
            else:
                targets = batch['features_out']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute output features predictions (forward propagation).
            # During the foward pass, PyTorch creates a computation graph for
            # the tensors that require gradients (gradient flag set to True) to
            # keep track of the operations on these tensors, i.e., the model
            # parameters. In addition, PyTorch additionally stores the
            # corresponding 'gradient functions' (mathematical operator) of the
            # executed operations to the output tensor, stored in the .grad_fn
            # attribute of the corresponding tensors. Tensor.grad_fn is set to
            # None for tensors corresponding to leaf-nodes of the computation
            # graph or for tensors with the gradient flag set to False.
            if loss_nature == 'features_out':
                # Get output features
                features_out, _ = model(features_in, hidden_features_in)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                
                # Compute loss
                loss = loss_function(features_out, targets)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown loss nature.')
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
                    model, optimizer, epoch, loss_nature=loss_nature,
                    loss_type=loss_type, loss_kwargs=loss_kwargs,
                    batch_size=batch_size, device_type=device_type)
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
    save_loss_history(model, n_max_epochs, loss_nature, loss_type,
                      loss_history_epochs, lr_scheduler_type=lr_scheduler_type,
                      lr_history_epochs=lr_history_epochs,
                      validation_loss_history=validation_loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get best loss and corresponding training epoch
    best_loss = float(min(loss_history_epochs))
    best_training_epoch = loss_history_epochs.index(best_loss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        if is_model_out_normalized:
            min_loss_str = 'Minimum training loss (normalized)'
        else:
            min_loss_str = 'Minimum training loss'
        print(f'\n\n> {min_loss_str}: {best_loss:.8e} | '
              f'Epoch: {best_training_epoch:d}')
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
    # Get summary of PyTorch model
    model_statistics = get_model_summary(model, device_type=device_type)
    # Write summary data file for model training process
    write_training_summary_file(
        device_type, seed, model.model_directory, load_model_state,
        n_max_epochs, is_model_in_normalized, is_model_out_normalized,
        batch_size, is_sampler_shuffle, loss_nature, loss_type, loss_kwargs,
        opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, epoch,
        dataset_file_path, dataset, best_loss, best_training_epoch,
        total_time_sec, avg_time_epoch,
        torchinfo_summary=str(model_statistics))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model, best_loss, best_training_epoch
# =============================================================================
class EarlyStopper:
    """Early stopping procedure (implicit regularizaton).
    
    Attributes
    ----------
    _validation_dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
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
    get_validation_loss_history(self)
        Get validation loss history.
    is_evaluate_criterion(self, epoch)
        Check whether to evaluate early stopping criterion.
    evaluate_criterion(self, model, optimizer, epoch, \
                       loss_nature='node_features_out', loss_type='mse', \
                       loss_kwargs={}, batch_size=1, device_type='cpu')
        Evaluate early stopping criterion.
    _validate_model(self, model, optimizer, epoch,
                    loss_nature='node_features_out', loss_type='mse',
                    loss_kwargs={}, batch_size=1, device_type='cpu')
        Perform model validation.
    load_best_performance_state(self, model, optimizer)
        Load minimum validation loss model and optimizer states.
    """
    def __init__(self, validation_dataset, validation_frequency=1,
                 trigger_tolerance=1, improvement_tolerance=1e-2):
        """Constructor.
        
        Parameters
        ----------
        validation_dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        validation_frequency : int, default=1
            Frequency of validation procedures, i.e., frequency with respect to
            training epochs at which model is validated to evaluate early
            stopping criterion.
        trigger_tolerance : int, default=1
            Number of consecutive model validation procedures without
            performance improvement to trigger early stopping.
        improvement_tolerance : float, default=1e-2
            Minimum relative improvement required to count as a performance
            improvement.
        """
        # Set validation data set
        self._validation_dataset = validation_dataset
        # Set validation frequency
        self._validation_frequency = validation_frequency
        # Set early stopping trigger tolerance
        self._trigger_tolerance = trigger_tolerance
        # Set minimum relative improvement tolerance
        self._improvement_tolerance = improvement_tolerance
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
                           loss_nature='features_out', loss_type='mse',
                           loss_kwargs={}, batch_size=1, device_type='cpu'):
        """Evaluate early stopping criterion.
        
        Parameters
        ----------
        model : torch.nn.Module
            Recurrent neural network model.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        epoch : int
            Training epoch.
        loss_nature : {'features_out',}, default='features_out'
            Loss nature:
            
            'features_out' : Based on output features

        loss_type : {'mse',}, default='mse'
            Loss function type:
            
            'mse'  : MSE (torch.nn.MSELoss)
            
        loss_kwargs : dict, default={}
            Arguments of torch.nn._Loss initializer.
        batch_size : int, default=1
            Number of samples loaded per batch.
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
            model, optimizer, epoch, loss_nature=loss_nature,
            loss_type=loss_type, loss_kwargs=loss_kwargs,
            batch_size=batch_size, device_type=device_type)
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
    def _validate_model(self, model, optimizer, epoch,
                        loss_nature='features_out', loss_type='mse',
                        loss_kwargs={}, batch_size=1, device_type='cpu'):
        """Perform model validation.
        
        Parameters
        ----------
        model : torch.nn.Module
            Recurrent neural network model.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        epoch : int
            Training epoch.
        loss_nature : {'features_out',}, default='features_out'
            Loss nature:
            
            'features_out' : Based on output features

        loss_type : {'mse',}, default='mse'
            Loss function type:
            
            'mse'  : MSE (torch.nn.MSELoss)
            
        loss_kwargs : dict, default={}
            Arguments of torch.nn._Loss initializer.
        batch_size : int, default=1
            Number of samples loaded per batch.
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
        # Prediction with model
        _, avg_valid_loss_sample = predict(
            self._validation_dataset, model.model_directory,
            model=model, predict_directory=None, load_model_state=epoch,
            loss_nature=loss_nature, loss_type=loss_type,
            loss_kwargs=loss_kwargs,
            is_normalized_loss=model.is_model_out_normalized,
            batch_size=batch_size,
            device_type=device_type, seed=None, is_verbose=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model in training mode
        model.train()
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
            Recurrent neural network model.
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