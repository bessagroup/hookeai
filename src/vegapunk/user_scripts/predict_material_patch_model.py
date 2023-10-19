"""User script: Predict with GNN-based material patch model."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
# Local
from gnn_model.gnn_patch_dataset import GNNMaterialPatchDataset
from gnn_model.prediction import predict, build_prediction_data_arrays
from gnn_model.evaluation_metrics import plot_truth_vs_prediction
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def perform_model_prediction(predict_directory, dataset_file_path,
                             model_directory, device_type='cpu',
                             is_verbose=False):
    """Perform prediction with GNN-based material patch model.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset_file_path : str
        GNN-based material patch testing data set file path.        
    model_directory : str
        Directory where material patch model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based material patch model prediction options
    loss_type, loss_kwargs = set_default_prediction_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch data set
    dataset = GNNMaterialPatchDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prediction with GNN-based material patch model
    predict_subdir = predict(predict_directory, dataset, model_directory,
                             load_model_state='best', loss_type=loss_type,
                             loss_kwargs=loss_kwargs, is_normalized_loss=True,
                             device_type=device_type, seed=None,
                             is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Set prediction data arrays types and filenames
    prediction_types = {}
    prediction_types['int_force_comps'] = ('prediction_int_force_dim_1',
                                           'prediction_int_force_dim_2',
                                           'prediction_int_force_dim_3')
    prediction_types['int_force_norm'] = ('prediction_int_force_norm',)
    # Plot model predictions against ground-truth
    for key, val in prediction_types.items():
        # Build samples predictions data arrays with predictions and
        # ground-truth
        prediction_data_arrays = build_prediction_data_arrays(
            predict_subdir, prediction_type=key, samples_ids='all')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples predictions data arrays
        for i, data_array in enumerate(prediction_data_arrays):
            # Get prediction plot file name
            filename = val[i]
            # Set prediction process
            if key == 'int_force_comps':
                prediction_sets = {'$f^{\\mathrm{\; int}} (\\mathrm{dim}: '
                                   + str(i + 1) + ')$': data_array}
            elif key == 'int_force_norm':
                prediction_sets = {'$||\\mathbf{f}^{\\mathrm{\; int}}||$':
                                   data_array}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot model predictions against ground-truth
            plot_truth_vs_prediction(prediction_sets, error_bound=0.1,
                                     is_normalize_data=True,
                                     filename=filename,
                                     save_dir=predict_subdir,
                                     is_save_fig=True, is_stdout_display=True)
# =============================================================================
def set_default_prediction_options():
    """Set default GNN-based material patch model prediction options.
    
    Returns
    -------
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.   
    """
    loss_type = 'mse'
    loss_kwargs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_type, loss_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set case study name
    case_study_name = '2d_elastic'
    # Set case study directory
    case_study_base_dirs = {
        '2d_elastic': f'/home/bernardoferreira/Documents/temp',}
    case_study_dir = \
        os.path.join(os.path.normpath(case_study_base_dirs[case_study_name]),
                     f'cs_{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set directory
    dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                     '1_dataset')
    # Get files in GNN-based material patch data set directory
    directory_list = os.listdir(dataset_directory)
    # Loop over files
    is_testing_dataset = False
    for filename in directory_list:
        # Check if file is testing data set file
        is_testing_dataset = \
            bool(re.search(r'^material_patch_graph_dataset_testing_n'
                           r'[0-9]+.pkl$', filename))
        # Leave searching loop when testing data set file is found
        if is_testing_dataset:
            break
    # Set GNN-based material patch testing data set file path
    if is_testing_dataset:
        dataset_file_path = os.path.join(os.path.normpath(dataset_directory),
                                         filename)
    else:
        raise RuntimeError(f'Testing data set file has not been found in '
                           'dataset directory:\n\n{dataset_directory}')      
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                   '2_model')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model prediction directory
    prediction_directory = os.path.join(os.path.normpath(case_study_dir),
                                        '3_prediction')
    # Create prediction directory
    if not os.path.isdir(prediction_directory):
        make_directory(prediction_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    device_type = 'cpu'
    # Perform prediction with GNN-based material patch model
    perform_model_prediction(prediction_directory, dataset_file_path,
                             model_directory, device_type, is_verbose=True)