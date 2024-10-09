"""Prediction of hybrid material constitutive model.

Functions
---------
predict
    Make predictions with recurrent constitutive model for given dataset.
compute_sample_prediction_loss
    Compute loss of sample output features prediction.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import random
import time
import datetime
# Third-party
import torch
import tqdm
import numpy as np
# Local
from rnn_base_model.data.time_dataset import get_time_series_data_loader
from hybrid_base_model.model.hybrid_model import HybridMaterialModel
from gnn_base_model.train.torch_loss import get_pytorch_loss
from gnn_base_model.predict.prediction import make_predictions_subdir, \
    save_sample_predictions, seed_worker, write_prediction_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def predict(dataset, model_directory, predict_directory=None,
            load_model_state=None, loss_nature='features_out',
            loss_type='mse', loss_kwargs={}, is_normalized_loss=False,
            dataset_file_path=None, device_type='cpu', seed=None,
            is_verbose=False):
    """Make predictions with hybrid material model for given dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    model_directory : str
        Directory where model is stored.
    predict_directory : str, default=None
        Directory where model predictions results are stored. If None, then
        all output files are supressed.
    load_model_state : {'best', 'last', int, None}, default=None
        Load available model state from the model directory. Options:
        
        'best' : Model state corresponding to best performance available
        
        'last' : Model state corresponding to highest training epoch
        
        int    : Model state corresponding to given training epoch
        
        None   : Model default state file

    loss_nature : {'features_out',}, default='features_out'
        Loss nature:
        
        'features_out' : Based on output features

    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict, default={}
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
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
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    avg_predict_loss : float
        Average prediction loss per sample. Defaults to None if ground-truth is
        not available for all data set samples.
    """
    # Set random number generators initialization for reproducibility
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time_sec = time.time()
    if is_verbose:
        print('\nHybrid material constitutive model prediction'
              '\n---------------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model directory
    if not os.path.exists(model_directory):
        raise RuntimeError('The model directory has not been found:\n\n'
                           + model_directory)
    # Check prediction directory
    if predict_directory is not None and not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Loading Hybrid material constitutive model...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybrid material constitutive model
    model = HybridMaterialModel.init_model_from_file(model_directory)
    # Set model device
    model.set_device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load hybrid material constitutive model state
    _ = model.load_model_state(load_model_state=load_model_state,
                               is_remove_posterior=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Move model to device
    model.to(device=device)
    # Set model in evaluation mode
    model.eval()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create model predictions subdirectory for current prediction process
    predict_subdir = None
    if predict_directory is not None:
        predict_subdir = make_predictions_subdir(predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    if isinstance(seed, int):
        data_loader = get_time_series_data_loader(
            dataset=dataset,
            kwargs={'worker_init_fn': seed_worker, 'generator': generator})
    else:
        data_loader = get_time_series_data_loader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize loss function
    loss_function = get_pytorch_loss(loss_type, **loss_kwargs)
    # Initialize samples prediction loss
    loss_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Starting prediction process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set context manager to avoid creation of computation graphs during the
    # model evaluation (forward propagation)
    with torch.no_grad():
        # Loop over graph samples
        for i, batch in enumerate(tqdm.tqdm(data_loader,
                                            desc='> Predictions: ',
                                            disable=not is_verbose)):
            # Move batch to device
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize sample results
            results = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute output features predictions (forward propagation)
            if loss_nature == 'features_out':
                # Compute output features
                features_out = model(batch['features_in'],
                                     is_normalized=False)
                # Get output features ground-truth (None if not available)
                targets = batch['features_out']
                # Store sample results (removing batch dimension)
                results['features_out'] = features_out.detach().cpu()[:, 0, :]
                results['targets'] = None
                if targets is not None:
                    results['targets'] = targets.detach().cpu()[:, 0, :]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown loss nature.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample output features prediction loss
            loss = compute_sample_prediction_loss(
                model, loss_nature, loss_function, features_out, targets,
                is_normalized=is_normalized_loss)
            # Store prediction loss data
            results['prediction_loss_data'] = \
                (loss_nature, loss_type, loss, is_normalized_loss)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample prediction loss if ground-truth is available
            if loss is not None:
                loss_samples.append(loss.detach().cpu())
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save sample predictions results
            if predict_directory is not None:
                save_sample_predictions(predictions_dir=predict_subdir,
                                        sample_id=i, sample_results=results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished prediction process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute average prediction loss per sample
    avg_predict_loss = None
    if isinstance(loss_samples, list) and len(loss_samples) == len(dataset):
        avg_predict_loss = np.mean(loss_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        # Set average prediction loss output format
        if avg_predict_loss:
            loss_str = (f'{avg_predict_loss:.8e} | {loss_type}')
            if is_normalized_loss:
                loss_str += ', normalized'
        else:
            loss_str = 'Ground-truth not available'  
        # Display average loss
        print('\n> Avg. prediction loss per sample: '
              + loss_str)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total prediction time and average prediction time per sample
    total_time_sec = time.time() - start_time_sec
    if len(dataset) > 0:
        avg_time_sample = total_time_sec/len(dataset)
    else:
        avg_time_sample = float('nan')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Prediction results directory: {predict_subdir}')
        print(f'\n> Total prediction time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. prediction time per sample: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sample)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for model prediction process
    if predict_directory is not None:
        write_prediction_summary_file(
            predict_subdir, device_type, seed, model_directory,
            load_model_state, loss_type, loss_kwargs, is_normalized_loss,
            dataset_file_path, dataset, avg_predict_loss, total_time_sec,
            avg_time_sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir, avg_predict_loss
# =============================================================================
def compute_sample_prediction_loss(model, loss_nature, loss_function,
                                   features_out, targets, is_normalized=False):
    """Compute loss of sample output features prediction.
    
    Parameters
    ----------
    model : torch.nn.Module
        Recurrent neural network model.
    loss_nature : {'features_out',}
        Loss nature.
    loss_function : torch.nn._Loss
        PyTorch loss function.
    features_out : torch.Tensor
        Predicted output features stored as a torch.Tensor(2d).
    targets : {torch.Tensor, None}
        Output features ground-truth stored as a torch.Tensor(2d).
    is_normalized : bool, default=False
        If True, get normalized loss according with model fitted features data
        scalers, False otherwise.
    
    Returns
    -------
    loss : {float, None}
        Loss of sample output features prediction. Set to None if output
        features ground-truth is not available.
    """
    # Check if output features ground-truth is available
    is_ground_truth_available = targets is not None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute sample loss
    if is_ground_truth_available:
        if is_normalized:
            # Check model data normalization
            if is_normalized:
                model.check_normalized_return()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model data scaler
            if loss_nature == 'features_out':
                scaler = model.get_fitted_data_scaler('features_out')
            else:
                raise RuntimeError('Unknown loss nature.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get normalized output features predictions
            norm_features_out = scaler.transform(features_out)
            # Get normalized output features ground-truth
            norm_targets = scaler.transform(targets)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample loss (normalized data)
            loss = loss_function(norm_features_out, norm_targets)
        else:
            # Compute sample loss
            loss = loss_function(features_out, targets)
    else:
        # Set loss to None if ground-truth is not available
        loss = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss