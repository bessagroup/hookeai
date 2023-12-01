"""Cross-validation of GNN-based material patch model.

Functions
---------
kfold_cross_validation
    k-fold cross validation of GNN-based material patch model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import time
import datetime
# Third-party
import torch
import sklearn.model_selection
import numpy as np
# Local
from gnn_model.training import train_model
from gnn_model.prediction import predict
from ioput.iostandard import make_directory, write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def kfold_cross_validation(cross_validation_dir, n_fold, n_max_epochs,
                           dataset, model_init_args, lr_init,
                           opt_algorithm='adam', lr_scheduler_type='steplr',
                           lr_scheduler_kwargs={}, loss_type='mse',
                           loss_kwargs={}, batch_size=1,
                           is_sampler_shuffle=False,
                           is_early_stopping=False, early_stopping_kwargs={},
                           dataset_file_path=None, device_type='cpu',
                           is_verbose=False):
    """k-fold cross validation of GNN-based material patch model.
    
    Data set is split into k consecutive folds. The first n_samples % n_splits
    folds have size n_samples // n_splits + 1, other folds have size
    n_samples // n_splits, where n_samples is the number of samples. Each fold
    is then used once as a validation set while the k - 1 remaining folds form
    the training set.
    
    Parameters
    ----------
    cross_validation_dir : dir
        Directory where cross-validation process data is stored.
    n_fold : int
        Number of folds into which the data set is split to perform
        cross-validation.
    n_max_epochs : int
        Maximum number of training epochs.
    dataset : torch.utils.data.Dataset
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
    dataset_file_path : str, default=None
        GNN-based material patch data set file path if such file exists. Only
        used for output purposes.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    k_fold_loss_array : numpy.ndarray(2d)
        k-fold cross-validation loss array. For the i-th fold,
        data_array[i, 0] stores the best training loss and data_array[i, 1]
        stores the average prediction loss per sample.
    """
    if is_verbose:
        print('\nGNN-based material patch data model: k-fold cross-validation'
              '\n------------------------------------------------------------')
        start_time_sec = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check cross-validation directory
    if not os.path.isdir(cross_validation_dir):
        raise RuntimeError('The cross-validation directory has not been '
                           'found:\n\n' + cross_validation_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Number of folds: {n_fold}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize folder
    k_folder = sklearn.model_selection.KFold(n_splits=n_fold, shuffle=True,
                                             random_state=None)
    # Set number of samples dummy array
    n_sample_array = np.zeros((len(dataset), 1))
    # Get cross-validation folds data set indexes
    folds_indexes = []
    for (train_ids, valid_ids) in k_folder.split(n_sample_array):
        folds_indexes.append((train_ids, valid_ids))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize k-fold cross-validation loss array
    k_fold_loss_array = np.empty((0, 2))
    # Initialize cross-validation training and validation losses
    folds_training_losses = []
    folds_validation_losses = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model data normalization
    is_data_normalization = False
    if 'is_data_normalization' in model_init_args.keys():
        is_data_normalization = model_init_args['is_data_normalization']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Starting k-fold cross-validation process...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over cross-validation folds
    for fold in range(n_fold):
        # Create cross-validation fold directory
        fold_dir = os.path.join(os.path.normpath(cross_validation_dir),
                                f'fold_{fold + 1}')
        make_directory(fold_dir, is_overwrite=True)
        # Create cross-validation fold model subdirectory
        fold_model_dir = os.path.join(os.path.normpath(fold_dir), 'model')
        make_directory(fold_model_dir, is_overwrite=True)
        # Create cross-validation fold validation subdirectory
        fold_validation_dir = \
            os.path.join(os.path.normpath(fold_dir), 'validation')
        make_directory(fold_validation_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fold training data set
        training_ids = folds_indexes[fold][0]
        training_dataset = torch.utils.data.Subset(dataset, training_ids)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fold validation data set
        validation_ids = folds_indexes[fold][1]
        validation_dataset = torch.utils.data.Subset(dataset, validation_ids)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_init_args['model_directory'] = fold_model_dir
        # Training of GNN-based material patch model
        model, best_training_loss, _ = train_model(
            n_max_epochs, training_dataset, model_init_args, lr_init,
            opt_algorithm=opt_algorithm, lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs, loss_type=loss_type,
            loss_kwargs=loss_kwargs, batch_size=batch_size,
            is_sampler_shuffle=is_sampler_shuffle,
            is_early_stopping=is_early_stopping,
            early_stopping_kwargs=early_stopping_kwargs,
            device_type=device_type, seed=None, is_verbose=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prediction with GNN-based material patch model
        _, avg_valid_loss_sample = predict(
            validation_dataset, model.model_directory,
            predict_directory=fold_validation_dir, load_model_state='best',
            loss_type=loss_type, loss_kwargs=loss_kwargs,
            is_normalized_loss=is_data_normalization, device_type=device_type,
            seed=None, is_verbose=False)
        # Check average validation loss
        if avg_valid_loss_sample is None:
            raise RuntimeError(f'The average validation loss for fold '
                               f'{fold + 1} could not be computed because '
                               f'the ground-truth is not available for all '
                               f'validation data set samples.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble fold losses
        k_fold_loss_array = np.vstack(
            (k_fold_loss_array, (best_training_loss, avg_valid_loss_sample)))
        # Store fold losses
        folds_training_losses.append(best_training_loss)
        folds_validation_losses.append(avg_valid_loss_sample)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            # Set losses output format
            train_loss_str = f'{best_training_loss:.8e}'
            valid_loss_str = f'{avg_valid_loss_sample:.8e}'
            if is_data_normalization:
                train_loss_str += ' (normalized)'
                valid_loss_str += ' (normalized)'
            # Display fold losses
            print(f'\n> Fold {fold + 1}/{n_fold}:\n'
                  f'  > Best training loss:   {train_loss_str}\n'
                  f'  > Avg. validation loss: {valid_loss_str}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished k-fold cross-validation process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute cross-validation average training loss
    avg_best_training_loss = np.mean(folds_training_losses)
    # Compute cross-validation average validation loss
    avg_validation_loss = np.mean(folds_validation_losses)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        # Set losses output format
        train_loss_str = f'{avg_best_training_loss:.8e}'
        valid_loss_str = f'{avg_validation_loss:.8e}'
        if is_data_normalization:
            train_loss_str += ' (normalized)'
            valid_loss_str += ' (normalized)'
        # Display cross-validation losses
        print(f'\n> k-fold cross-validation results:')
        print(f'\n  > Average training loss:   {train_loss_str}')
        print(f'\n  > Average validation loss: {valid_loss_str}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total cross-validation time and average cross-validation time per
    # fold
    total_time_sec = time.time() - start_time_sec
    avg_time_fold = total_time_sec/n_fold
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Cross-validation directory: {cross_validation_dir}')
        print(f'\n> Total cross-validation time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. time per fold: '
              f'{str(datetime.timedelta(seconds=int(avg_time_fold)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for model cross-validation
    write_cross_validation_summary_file(
        cross_validation_dir, device_type, n_fold, n_max_epochs,
        is_data_normalization, batch_size, loss_type, loss_kwargs, dataset,
        dataset_file_path, k_fold_loss_array, total_time_sec, avg_time_fold)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return k_fold_loss_array
# =============================================================================
def write_cross_validation_summary_file(
    cross_validation_dir, device_type, n_fold, n_max_epochs,
    is_data_normalization, batch_size, loss_type, loss_kwargs, dataset,
    dataset_file_path, k_fold_loss_array, total_time_sec, avg_time_fold):
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
    is_data_normalization : bool
        If True, then input and output features are normalized for training
        False otherwise. Data scalers need to be fitted with fit_data_scalers()
        and are stored as model attributes.
    batch_size : int
        Number of samples loaded per batch.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    dataset : torch.utils.data.Dataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    dataset_file_path : str
        GNN-based material patch data set file path if such file exists. Only
        used for output purposes.
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
    summary_data['is_data_normalization'] = is_data_normalization
    summary_data['batch_size'] = batch_size
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['Data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Data set size'] = len(dataset)
    summary_data['k-fold cross-validation results'] = k_fold_loss_array
    summary_data['Total cross-validation time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. cross-validation time per fold'] = \
        str(datetime.timedelta(seconds=int(avg_time_fold)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=cross_validation_dir,
        summary_title=
            'Summary: GNN-based material patch model k-fold cross-validation',
        **summary_data)