"""Prediction of Graph Neural Network based material patch model.

Functions
---------
predict
    Make predictions with GNN-based material patch model.
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_results
    Save model prediction results for given sample.
load_sample_results
    Load model prediction results for given sample.
compute_sample_prediction_loss
    Compute loss of sample node internal forces prediction.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import random
# Third-party
import torch
import tqdm
import numpy as np
# Local
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
from gnn_model.training import get_pytorch_loss, seed_worker
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def predict(predict_directory, dataset, model_directory,
            load_model_state='default', is_compute_loss=True, loss_type='mse',
            loss_kwargs={}, device_type='cpu', seed=None, is_verbose=False):
    """Make predictions with GNN-based material patch model for given dataset.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset : GNNMaterialPatchDataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_directory : str
        Directory where material patch model is stored.
    load_model_state : {'best', 'last', int, 'default'}, default='default'
        Load available GNN-based material patch model state from the model
        directory. Options:
        
        'best'      : Model state corresponding to best performance available
        
        'last'      : Model state corresponding to highest training step
        
        int         : Model state corresponding to given training step
        
        'default'   : Model default state file
        
    is_compute_loss : bool, default=True
        If True, computes predictions average loss. The computation of the
        predictions loss is restricted to data set samples for which the
        ground-truth is available from the corresponding
        torch_geometric.data.Data object, being set to None otherwise. Each
        sample prediction loss is stored in the corresponding prediction
        results file.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict, default={}
        Arguments of torch.nn._Loss initializer.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    seed : int, default=None
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    is_verbose : bool, default=False
        If True, enable verbose output.
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
    if is_verbose:
        print('\nGNN-based material patch data model prediction'
              '\n----------------------------------------------\n')
        print(f'\n> Data set size: {len(dataset)} \n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model directory
    if not os.path.exists(model_directory):
        raise RuntimeError('The material patch model directory has not '
                           'been found:\n\n' + model_directory)
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Loading GNN-based material patch model...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    model = GNNMaterialPatchModel.init_model_from_file(model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model state
    _ = model.load_model_state(load_model_state=load_model_state,
                               is_remove_posterior=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model data normalization
    is_data_normalization = model.is_data_normalization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Move model to process ID
    model.to(device=device)
    # Set model in evaluation mode
    model.eval()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create model predictions subdirectory for current prediction process
    predict_subdir = make_predictions_subdir(predict_directory,
                                             model.model_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    if isinstance(seed, int):
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, worker_init_fn=seed_worker, generator=generator)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize predictions loss computation
    if is_compute_loss:
        # Initialize loss function
        loss_function = get_pytorch_loss(loss_type, **loss_kwargs)
        # Initialize samples losses
        loss_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting predictions computation...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set context manager to avoid creation of computation graphs during the
    # model evaluation (forward propagation)
    with torch.no_grad():
        # Loop over graph samples
        for i, pyg_graph in enumerate(tqdm.tqdm(data_loader,
                                      desc='> Predictions: ',
                                      disable=not is_verbose)):
            # Move sample to process ID
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node internal forces predictions (forward propagation)
            node_internal_forces = model.predict_internal_forces(pyg_graph)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if node internal forces ground-truth is available
            is_ground_truth_available = \
                model.get_output_features_from_graph(pyg_graph) is not None
            # Compute loss if node internal forces ground-truth is available
            if is_ground_truth_available:
                # Compute loss according with model data normalization
                loss = compute_sample_prediction_loss(
                    pyg_graph, model, loss_function, node_internal_forces)
                # Assemble sample loss
                loss_samples.append(loss)
            else:
                # Set sample loss to None if ground-truth is unavailable
                loss = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build sample results
            results = {}
            results['node_internal_forces'] = node_internal_forces
            results['node_internal_forces_loss'] = (loss_type, loss)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save sample predictions results
            save_sample_results(results_dir=predict_subdir,
                                sample_id=i, sample_results=results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        # Compute samples average loss
        if is_compute_loss:
            # Compute samples average loss
            average_loss = torch.mean(loss_samples)
            # Display average loss
            print('\n> Node internal forces predictions average loss: '
                  f'{average_loss:.8e} ({loss_type}, {len(loss_samples)} '
                  f'samples.')
        # Display prediction results directory 
        print('\n> Prediction results directory: ', predict_subdir, '\n')
# =============================================================================
def make_predictions_subdir(predict_directory, model_name):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    model_name : str
        Name of model.

    Returns
    -------
    predict_subdir : str
        Subdirectory where model predictions results are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_directory,
                                  str(model_name) + '_prediction')
    # Create model predictions subdirectory
    make_directory(predict_directory, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
# =============================================================================
def save_sample_results(results_dir, sample_id, sample_results):
    """Save model prediction results for given sample.
    
    Parameters
    ----------
    results_dir : str
        Directory where sample prediction results are stored.
    sample_id : int
        Sample ID. Sample ID is appended to sample prediction results file
        name.
    sample_results : dict
        Sample prediction results.
    """
    # Check prediction results directory
    if not os.path.exists(results_dir):
        raise RuntimeError('The prediction results directory has not been '
                           'found:\n\n' + results_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set sample prediction results file path
    sample_path = os.path.join(results_dir,
                               'prediction_sample_'+ str(sample_id) + '.pkl')
    # Save sample prediction results
    with open(sample_path, 'wb') as sample_file:
        pickle.dump(sample_results, sample_file)
# =============================================================================
def load_sample_results(sample_prediction_path):
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
def compute_sample_prediction_loss(pyg_graph, model, loss_function,
                                   node_internal_forces):
    """Compute loss of sample node internal forces prediction.
    
    Loss is computed according with model training data normalization.
    
    Parameters
    ----------
    pyg_graph : torch_geometric.data.Data
        Material patch homogeneous graph.
    model : GNNMaterialPatchModel
        GNN-based material patch model.
    loss_function : torch.nn._Loss
        PyTorch loss function.
    node_internal_forces : torch.Tensor
        Predicted nodes internal forces matrix stored as a torch.Tensor(2d) of
        shape (n_nodes, n_dim).
    
    Returns
    -------
    loss : float
        Loss of sample node internal forces prediction.
    """
    # Get model data normalization
    is_data_normalization = model.is_data_normalization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute loss according with model data normalization
    if is_data_normalization:
        # Get model data scaler
        scaler_node_out = model.get_fitted_data_scaler('node_features_out')
        # Get normalized node internal forces predictions
        norm_node_internal_forces = \
            scaler_node_out.transform(node_internal_forces)
        # Get normalized node internal forces ground-truth
        norm_node_internal_forces_target = \
            model.get_output_features_from_graph(
                pyg_graph, is_normalized=is_data_normalization)
        # Compute sample loss (normalized data)
        loss = loss_function(norm_node_internal_forces,
                             norm_node_internal_forces_target)
    else:
        # Get node internal forces ground-truth
        node_internal_forces_target = \
            model.get_output_features_from_graph(pyg_graph)
        # Compute sample loss
        loss = loss_function(node_internal_forces,
                             node_internal_forces_target)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss