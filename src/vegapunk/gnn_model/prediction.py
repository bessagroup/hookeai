"""Prediction of Graph Neural Network based material patch model.

Functions
---------
predict
    Make predictions with GNN-based material patch model.
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_results
    Save model prediction results for given sample. 
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
# Third-party
import torch
import tqdm
# Local
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
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
def predict(predict_dir, dataset, model_directory, device='cpu',
            is_verbose=False):
    """Make predictions with GNN-based material patch model for given dataset.
    
    Parameters
    ----------
    predict_dir : str
        Directory where model predictions results are stored.
    dataset : torch_geometric.data.Dataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_directory : str
        Directory where GNN-based material patch model is stored.
    device : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    if is_verbose:
        print('\nGNN-based material patch data model prediction'
              '\n----------------------------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model directory
    if not os.path.exists(model_directory):
        raise RuntimeError('The material patch model directory has not '
                           'been found:\n\n' + model_directory)
    # Check prediction directory
    if not os.path.exists(predict_dir):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Loading GNN-based material patch model...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model class initialization parameters
    model_init_file_path = os.path.join(model_directory,
                                        'model_init_args' + '.pkl')
    if not os.path.isfile(model_init_file_path):
        raise RuntimeError('The material patch model class initialization '
                           'parameters file has not been found:\n\n'
                           + model_init_file_path)
    else:
        model_init_args = pickle.load(model_init_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model state (state file corresponding to
    # the highest training step available in model directory)
    _ = model.load_model_state(is_latest=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Move model to process ID
    model.to(device=device)
    # Set model in evaluation mode
    model.eval()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create model predictions subdirectory for current prediction process
    predict_subdir = make_predictions_subdir(predict_dir, model.model_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting predictions computation...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set context manager to avoid creation of computation graphs during the
    # model evaluation (forward propagation)
    with torch.no_grad():
        # Loop over graph samples
        for i, pyg_graph in enumerate(tqdm.tqdm(data_loader),
                                      desc='> Predictions: ',
                                      disable=not is_verbose):
            # Move sample to process ID
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node internal forces predictions (forward propagation)
            node_internal_forces = model.predict_internal_forces(pyg_graph)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build sample results
            results = {}
            results['node_internal_forces'] = node_internal_forces
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save sample predictions results
            save_sample_results(results_dir=predict_subdir,
                                sample_id=i, sample_results=results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Prediction results directory: ', predict_subdir, '\n')
# =============================================================================
def make_predictions_subdir(predict_dir, model_name):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_dir : str
        Directory where model predictions results are stored.
    model_name : str
        Name of model.

    Returns
    -------
    predict_subdir : str
        Subdirectory where model predictions results are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_dir):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_dir, str(model_name) + '_prediction')
    # Create model predictions subdirectory
    make_directory(predict_dir, is_overwrite=False)
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