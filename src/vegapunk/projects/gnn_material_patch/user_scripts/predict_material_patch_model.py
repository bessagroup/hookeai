"""User script: Predict with GNN-based material patch model."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import torch
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset
from gnn_base_model.predict.prediction import predict
from projects.gnn_material_patch.gnn_model_tools.process_predictions import \
    build_prediction_data_arrays
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
from projects.gnn_material_patch.gnn_model_tools.quality_tests import \
    perform_quality_tests
from ioput.iostandard import make_directory, find_unique_file_with_regex, \
    write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
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
    loss_nature, loss_type, loss_kwargs = set_default_prediction_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch data set
    dataset = GNNGraphDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prediction with GNN-based material patch model
    predict_subdir, _ = \
        predict(dataset, model_directory, predict_directory=predict_directory,
                model_load_state='best', loss_nature=loss_nature,
                loss_type=loss_type, loss_kwargs=loss_kwargs,
                is_normalized_loss=True, dataset_file_path=dataset_file_path,
                device_type=device_type, seed=None, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(predict_subdir), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
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
                                     save_dir=plot_dir,
                                     is_save_fig=True, is_stdout_display=False,
                                     is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform set of GNN-based material patch model quality tests
    quality_tests_results = perform_quality_tests(
        predictions_dir=predict_subdir, quality_tests='all', is_save_fig=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set summary data
    summary_data = {}
    summary_data['Equilibrium of internal forces (mean, std)'] = \
        quality_tests_results['sum_internal_forces']
    # Write quality tests summary file
    summary_title = ('Summary: Quality tests scores of GNN-based material ' \
                     'patch model prediction')
    write_summary_file(summary_directory=predict_subdir,
                       filename='quality_tests_summary',
                       summary_title=summary_title, **summary_data)
# =============================================================================
def set_default_prediction_options():
    """Set default GNN-based material patch model prediction options.
    
    Returns
    -------
    loss_nature : {'node_features_out', 'global_features_out'}, \
                  default='node_features_out'
        Loss nature:
        
        'node_features_out' : Based on node output features

        'global_features_out' : Based on global output features

    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.   
    """
    loss_nature = 'node_features_out'
    loss_type = 'mse'
    loss_kwargs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_nature, loss_type, loss_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set in-distribution/out-of-distribution testing flag
    is_in_dist_testing = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'gnn_material_patch/case_studies/2d_elastic_infinitesimal/'
                'single_element_quad4/')
    # Set case study directory
    case_study_name = 'out_of_dist_patch_size'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch testing data set directory
    if is_in_dist_testing:
        # Set testing data set directory (in-distribution)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '1_training_dataset')
    else:
        # Set testing data set directory (out-of-distribution)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '4_testing_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get GNN-based material patch testing data set file path
    regex = (r'^material_patch_graph_dataset_testing_n[0-9]+.pkl$',
             r'^material_patch_graph_dataset_n[0-9]+.pkl$')
    is_file_found, dataset_file_path = \
        find_unique_file_with_regex(dataset_directory, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Testing data set file has not been found  '
                           f'in data set directory:\n\n'
                           f'{dataset_directory}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update internal directory of stored data set (if required)
    GNNGraphDataset.update_dataset_file_internal_directory(
        dataset_file_path, os.path.dirname(dataset_file_path))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                   '2_model')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model prediction directory
    prediction_directory = os.path.join(os.path.normpath(case_study_dir),
                                        '5_prediction')
    # Create prediction directory
    if not os.path.isdir(prediction_directory):
        make_directory(prediction_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create prediction in-distribution/out-of-distribution subdirectory
    if is_in_dist_testing:
        prediction_subdir = os.path.join(
            os.path.normpath(prediction_directory), 'in_distribution')
    else:
        prediction_subdir = os.path.join(
            os.path.normpath(prediction_directory), 'out_of_distribution')
    # Create prediction subdirectory
    if not os.path.isdir(prediction_subdir):
        make_directory(prediction_subdir)  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform prediction with GNN-based material patch model
    perform_model_prediction(prediction_subdir, dataset_file_path,
                             model_directory, device_type=device_type,
                             is_verbose=True)