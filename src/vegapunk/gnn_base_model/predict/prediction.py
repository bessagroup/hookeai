"""Prediction of Graph Neural Network model.

Functions
---------
predict
    Make predictions with Graph Neural Network model for given dataset.
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_predictions
    Save model prediction results for given sample.
load_sample_predictions
    Load model prediction results for given sample.
compute_sample_prediction_loss
    Compute loss of sample output features prediction.
seed_worker
    Set workers seed in PyTorch data loaders to preserve reproducibility.
write_prediction_summary_file
    Write summary data file for model prediction process.
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
# Third-party
import torch
import torch_geometric
import tqdm
import numpy as np
# Local
from gnn_base_model.model.gnn_model import GNNEPDBaseModel
from gnn_base_model.train.torch_loss import get_pytorch_loss
from ioput.iostandard import make_directory, write_summary_file
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
            load_model_state=None, loss_nature='node_features_out',
            loss_type='mse', loss_kwargs={}, is_normalized_loss=False,
            dataset_file_path=None, device_type='cpu', seed=None,
            is_verbose=False):
    """Make predictions with Graph Neural Network model for given dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Graph Neural Network graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_directory : str
        Directory where Graph Neural Network model is stored.
    predict_directory : str, default=None
        Directory where model predictions results are stored. If None, then
        all output files are supressed.
    load_model_state : {'best', 'last', int, None}, default=None
        Load available Graph Neural Network model state from the model
        directory. Options:
        
        'best' : Model state corresponding to best performance available
        
        'last' : Model state corresponding to highest training epoch
        
        int    : Model state corresponding to given training epoch
        
        None   : Model default state file

    loss_nature : {'node_features_out', 'global_features_out'}, \
                  default='node_features_out'
        Loss nature:
        
        'node_features_out' : Based on node output features

        'global_features_out' : Based on global output features

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
        print('\nGraph Neural Network model prediction'
              '\n-------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model directory
    if not os.path.exists(model_directory):
        raise RuntimeError('The Graph Neural Network model directory has not '
                           'been found:\n\n' + model_directory)
    # Check prediction directory
    if predict_directory is not None and not os.path.exists(predict_directory):
        raise RuntimeError('The Graph Neural Network model prediction '
                           'directory has not been found:\n\n'
                           + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Loading Graph Neural Network model...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Graph Neural Network model
    model = GNNEPDBaseModel.init_model_from_file(model_directory)
    # Set model device
    model.set_device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Graph Neural Network model state
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
        data_loader = torch_geometric.loader.dataloader.DataLoader(
            dataset=dataset, worker_init_fn=seed_worker, generator=generator)
    else:
        data_loader = \
            torch_geometric.loader.dataloader.DataLoader(dataset=dataset)
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
        for i, pyg_graph in enumerate(tqdm.tqdm(data_loader,
                                      desc='> Predictions: ',
                                      disable=not is_verbose)):
            # Move sample to device
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize sample results
            results = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute output features predictions (forward propagation)
            if loss_nature == 'node_features_out':
                # Compute node output features
                features_out, _, _ = model.predict_output_features(pyg_graph)
                # Get sample node output features ground-truth
                # (None if not available)
                targets, _, _ = model.get_output_features_from_graph(pyg_graph)
                # Store sample results
                results['node_features_out'] = features_out.detach().cpu()
                results['node_targets'] = None
                if targets is not None:
                    results['node_targets'] = targets.detach().cpu()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif loss_nature == 'global_features_out':
                # Compute global output features
                _, _, features_out = model.predict_output_features(pyg_graph)
                # Get sample global output features ground-truth
                # (None if not available)
                _, _, targets = \
                    model.get_output_features_from_graph(pyg_graph)
                # Store sample results
                results['global_features_out'] = features_out.detach().cpu()
                results['global_targets'] = None
                if targets is not None:
                    results['global_targets'] = targets.detach().cpu()
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
def make_predictions_subdir(predict_directory):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.

    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_directory, 'prediction_set_0')
    while os.path.exists(predict_subdir):
        predict_subdir = os.path.join(
            predict_directory,
            'prediction_set_' + str(int(predict_subdir.split('_')[-1]) + 1))
    # Create model predictions subdirectory
    predict_subdir = make_directory(predict_subdir, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
# =============================================================================
def save_sample_predictions(predictions_dir, sample_id, sample_results):
    """Save model prediction results for given sample.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where sample prediction results are stored.
    sample_id : int
        Sample ID. Sample ID is appended to sample prediction results file
        name.
    sample_results : dict
        Sample prediction results.
    """
    # Check prediction results directory
    if not os.path.exists(predictions_dir):
        raise RuntimeError('The prediction results directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set sample prediction results file path
    sample_path = os.path.join(predictions_dir,
                               'prediction_sample_'+ str(sample_id) + '.pkl')
    # Save sample prediction results
    with open(sample_path, 'wb') as sample_file:
        pickle.dump(sample_results, sample_file)
# =============================================================================
def load_sample_predictions(sample_prediction_path):
    """Load model prediction results for given sample.
    
    Parameters
    ----------
    sample_prediction_path : str
        Sample prediction results file path.
        
    Returns
    -------
    sample_results : dict
        Sample prediction results.
    """
    # Check sample prediction results file
    if not os.path.isfile(sample_prediction_path):
        raise RuntimeError('Sample prediction results file has not been '
                           'found:\n\n' + sample_prediction_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sample prediction results
    with open(sample_prediction_path, 'rb') as sample_prediction_file:
        sample_results = pickle.load(sample_prediction_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_results
# =============================================================================
def compute_sample_prediction_loss(model, loss_nature, loss_function,
                                   features_out, targets, is_normalized=False):
    """Compute loss of sample output features prediction.
    
    Parameters
    ----------
    model : GNNEPDBaseModel
        Graph Neural Network model.
    loss_nature : {'node_features_out', 'global_features_out'}
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
            if loss_nature == 'node_features_out':
                scaler = model.get_fitted_data_scaler('node_features_out')
            elif loss_nature == 'global_features_out':
                scaler = model.get_fitted_data_scaler('global_features_out')
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
def write_prediction_summary_file(
    predict_subdir, device_type, seed, model_directory, load_model_state,
    loss_type, loss_kwargs, is_normalized_loss, dataset_file_path, dataset,
    avg_predict_loss, total_time_sec, avg_time_sample):
    """Write summary data file for model prediction process.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where model is stored.
    load_model_state : {'best', 'last', int, None}
        Load availabl model state from the model directory. Data scalers are
        also loaded from model initialization file.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    dataset_file_path : str
        Data set file path if such file exists. Only used for output purposes.
    dataset : torch.utils.data.Dataset
        Data set.
    avg_predict_loss : float
        Average prediction loss per sample.
    total_time_sec : int
        Total prediction time in seconds.
    avg_time_sample : float
        Average prediction time per sample.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['load_model_state'] = load_model_state
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['is_normalized_loss'] = is_normalized_loss
    summary_data['Prediction data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Prediction data set size'] = len(dataset)
    summary_data['Avg. prediction loss per sample: '] = \
        f'{avg_predict_loss:.8e}' if avg_predict_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per sample'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=predict_subdir,
        summary_title='Summary: Model prediction',
        **summary_data)