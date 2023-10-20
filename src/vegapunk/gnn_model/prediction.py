"""Prediction of Graph Neural Network based material patch model.

Functions
---------
predict
    Make predictions with GNN-based material patch model.
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_predictions
    Save model prediction results for given sample.
load_sample_predictions
    Load model prediction results for given sample.
compute_sample_prediction_loss
    Compute loss of sample node internal forces prediction.
build_prediction_data_arrays
    Build samples predictions data arrays with predictions and ground-truth.
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
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
from gnn_model.training import get_pytorch_loss, seed_worker
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
def predict(predict_directory, dataset, model_directory,
            load_model_state=None, loss_type='mse', loss_kwargs={},
            is_normalized_loss=False, dataset_file_path=None,
            device_type='cpu', seed=None, is_verbose=False):
    """Make predictions with GNN-based material patch model for given dataset.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset : torch.utils.data.Dataset
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
        
        None        : Model default state file
        
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
        GNN-based material patch data set file path if such file exists. Only
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
              '\n----------------------------------------------')
        start_time_sec = time.time()
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
        print('\n> Loading GNN-based material patch model...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    model = GNNMaterialPatchModel.init_model_from_file(model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model state
    _ = model.load_model_state(load_model_state=load_model_state,
                               is_remove_posterior=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Move model to process ID
    model.to(device=device)
    # Set model in evaluation mode
    model.eval()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create model predictions subdirectory for current prediction process
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
    # Initialize samples prediction losses
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
            # Move sample to process ID
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node internal forces predictions (forward propagation)
            node_internal_forces = model.predict_internal_forces(pyg_graph)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            # Get sample node internal forces ground-truth
            node_internal_forces_target = \
                model.get_output_features_from_graph(pyg_graph)
            # Compute sample node internal forces prediction loss
            loss = compute_sample_prediction_loss(
                pyg_graph, model, loss_function, node_internal_forces,
                is_normalized=is_normalized_loss)
            # Assemble sample prediction loss if ground-truth is available
            if loss is not None:
                loss_samples.append(loss)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build sample results
            results = {}
            results['node_internal_forces'] = node_internal_forces
            results['node_internal_forces_target'] = \
                node_internal_forces_target
            results['node_internal_forces_loss'] = \
                (loss_type, loss, is_normalized_loss)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save sample predictions results
            save_sample_predictions(predictions_dir=predict_subdir,
                                    sample_id=i, sample_results=results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished prediction process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute average loss per sample
    average_loss = None
    if loss_samples:
        average_loss = np.mean(loss_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        # Set average loss output format
        if average_loss:
            loss_str = (f'{average_loss:.8e} | {loss_type}, '
                        f'{len(loss_samples)} samples')
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
    avg_time_sample = total_time_sec/len(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Prediction results directory: {predict_subdir}')
        print(f'\n> Total prediction time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. prediction time per sample: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sample)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        f'{average_loss:.8e}' if average_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per sample'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    # Write summary file
    write_summary_file(
        predict_subdir,
        summary_title='Summary: GNN-based material patch model prediction',
        **summary_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
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
        predict_subdir = os.path.join(predict_directory,
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
def compute_sample_prediction_loss(pyg_graph, model, loss_function,
                                   node_internal_forces,
                                   is_normalized=False):
    """Compute loss of sample node internal forces prediction.
    
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
    is_normalized : bool, default=False
        If True, get normalized loss according with model fitted features data
        scalers, False otherwise.
    
    Returns
    -------
    loss : {float, None}
        Loss of sample node internal forces prediction. Set to None if node
        internal forces ground-truth is not available.
    """
    # Check if node internal forces ground-truth is available
    is_ground_truth_available = \
        model.get_output_features_from_graph(pyg_graph) is not None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute sample loss
    if is_ground_truth_available:
        if is_normalized:
            # Check model data normalization
            if is_normalized:
                model.check_normalized_return()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model data scaler
            scaler_node_out = model.get_fitted_data_scaler('node_features_out')
            # Get normalized node internal forces predictions
            norm_node_internal_forces = \
                scaler_node_out.transform(node_internal_forces)
            # Get normalized node internal forces ground-truth
            norm_node_internal_forces_target = \
                model.get_output_features_from_graph(
                    pyg_graph, is_normalized=is_normalized)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample loss (normalized data)
            loss = loss_function(norm_node_internal_forces,
                                norm_node_internal_forces_target)
        else:
            # Get node internal forces ground-truth
            node_internal_forces_target = \
                model.get_output_features_from_graph(pyg_graph)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample loss
            loss = loss_function(node_internal_forces,
                                node_internal_forces_target)
    else:
        # Set loss to None if ground-truth is not available
        loss = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss
# =============================================================================
def build_prediction_data_arrays(predictions_dir, prediction_type,
                                 samples_ids='all'):
    """Build samples predictions data arrays with predictions and ground-truth.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    prediction_type : {'int_force_comps', 'int_force_norm'}
        Type of prediction data arrays:
        
        'int_force_comp' : Node internal forces (one array per dimension)
        
        'int_force_norm' : Norm of node internal forces
    samples_ids : {'all', list[int]}, default='all'
        Samples IDs whose prediction results are collated in each prediction
        data array.
    
    Returns
    -------
    prediction_data_arrays : list[numpy.ndarray(2d)]
        Prediction data arrays. Each data array collates data from all
        specified samples and is stored as a numpy.ndarray(2d) of shape
        (n_nodes, 2), where data_array[i, 0] stores the i-th node ground-truth
        and data_array[i, 1] stores the i-th node prediction.
    """
    # Check sample predictions directory
    if not os.path.isdir(predictions_dir):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # Check samples IDs
    if samples_ids != 'all' and not isinstance(samples_ids, list):
        raise RuntimeError('Samples IDs must be specified as "all" or as '
                           'list[int].')
    elif (isinstance(samples_ids, list)
          and not all([isinstance(x, int) for x in samples_ids])):
        raise RuntimeError('Samples IDs must be specified as a list of '
                           'integers.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get prediction files samples IDs
    prediction_files_ids = []
    for filename in directory_list:
        # Check if file is sample results file
        id = re.search(r'^prediction_sample_([0-9])+.pkl$', filename)
        # Assemble sample ID
        if id is not None:
            prediction_files_ids.append(int(id.groups()[0]))
    # Check prediction files
    if not prediction_files_ids:
        raise RuntimeError('No sample results files have been found in '
                           'directory where samples predictions results files '
                           'are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set all available samples
    if samples_ids == 'all':
        samples_ids = prediction_files_ids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Set number of prediction data arrays
    if prediction_type == 'int_force_comps':
        # Get sample prediction results
        sample_prediction_path = \
            os.path.join(os.path.normpath(predictions_dir),
                         f'prediction_sample_{prediction_files_ids[0]}.pkl')
        sample_results = load_sample_predictions(sample_prediction_path)
        # Get number of spatial dimensions
        n_dim = sample_results['node_internal_forces'].shape[1]  
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        # Set number of prediction data arrays
        n_data_arrays = n_dim
    elif prediction_type == 'int_force_norm':
        # Set number of prediction data arrays
        n_data_arrays = 1
    else:
        raise RuntimeError('Unknown prediction data array type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction data arrays
    prediction_data_arrays = n_data_arrays*[np.empty((0, 2)),]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample_id in samples_ids:
        # Check if sample ID prediction results file is available
        if sample_id not in prediction_files_ids:
            raise RuntimeError(f'The prediction results file for sample '
                               f'{sample_id} has not been found in directory: '
                               f'\n\n{predictions_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample predictions file path
        sample_prediction_path = \
            os.path.join(os.path.normpath(predictions_dir),
                         f'prediction_sample_{sample_id}.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = load_sample_predictions(sample_prediction_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction data arrays
        for i in range(n_data_arrays):
            # Build sample data array
            if prediction_type in ('int_force_comps', 'int_force_norm'):
                # Get node internal forces predictions
                int_forces = sample_results['node_internal_forces']
                # Get node internal forces ground-truth
                int_forces_target = \
                    sample_results['node_internal_forces_target']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check availability of ground-truth
                if int_forces_target is None:
                    raise RuntimeError('Node internal forces ground-truth is '
                                       f'not available for sample {sample_id}'
                                       '.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build sample data array
                if prediction_type == 'int_force_comps':
                    data_array = np.concatenate(
                        (int_forces_target[:, i].reshape((-1, 1)),
                         int_forces[:, i].reshape((-1, 1))), axis=1)
                elif prediction_type == 'int_force_norm':
                    data_array = np.concatenate(
                        (np.linalg.norm(int_forces_target,
                                        axis=1).reshape((-1, 1)),
                         np.linalg.norm(int_forces,
                                        axis=1).reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample prediction data
            prediction_data_arrays[i] = \
                np.append(prediction_data_arrays[i], data_array, axis=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_data_arrays