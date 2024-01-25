"""User script: Predict with GNN-based model."""
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
import re
import shutil
import copy
# Third-party
import torch
import numpy as np
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset, \
    GNNGraphDatasetInMemory
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.predict.prediction import predict, load_sample_predictions
from gnn_base_model.predict.prediction_plots import predict, \
    plot_prediction_loss_history
from projects.abaqus_datasets.gnn_model_tools.process_predictions import \
    build_prediction_data_arrays
from projects.abaqus_datasets.gnn_model_tools.features import \
    FEMMeshFeaturesGenerator
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
from ioput.iostandard import make_directory, find_unique_file_with_regex
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
    """Perform prediction with GNN-based model.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset_file_path : str
        Testing data set file path.        
    model_directory : str
        Directory where GNN-based model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based model prediction options
    loss_nature, loss_type, loss_kwargs = set_default_prediction_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = GNNGraphDataset.load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prediction with GNN-based model
    predict_subdir, _ = \
        predict(dataset, model_directory, predict_directory=predict_directory,
                load_model_state='best', loss_nature=loss_nature,
                loss_type=loss_type, loss_kwargs=loss_kwargs,
                is_normalized_loss=True, dataset_file_path=dataset_file_path,
                device_type=device_type, seed=None, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of model predictions
    generate_prediction_plots(predict_subdir)
# =============================================================================
def generate_prediction_plots(predict_subdir):
    """Generate plots of model predictions.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(predict_subdir), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set prediction data arrays types and filenames
    prediction_types = {}
    prediction_types['coord_comps'] = ('prediction_coord_dim_1',
                                       'prediction_coord_dim_2',
                                       'prediction_coord_dim_3')
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
            if key == 'coord_comps':
                prediction_sets = {'$x_{n+1} (\\mathrm{dim}: '
                                   + str(i + 1) + ')$': data_array}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot model predictions against ground-truth
            plot_truth_vs_prediction(prediction_sets, error_bound=0.1,
                                     is_normalize_data=False,
                                     filename=filename,
                                     save_dir=plot_dir,
                                     is_save_fig=True, is_stdout_display=False,
                                     is_latex=True)
# =============================================================================
def perform_model_rollout(predict_directory, dataset_file_path,
                          model_directory, device_type='cpu',
                          is_verbose=False):
    """Perform prediction with GNN-based model.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset_file_path : str
        Testing data set file path.        
    model_directory : str
        Directory where GNN-based model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Get ABAQUS data file ID from data set file path
    id = re.search(r'^graph_dataset_bottle_([0-9]+).*.pkl$',
                   os.path.basename(dataset_file_path))
    if id is not None:
        bottle_id = int(id)
    else:
        raise RuntimeError('Could not extract ABAQUS data file ID from the '
                           'corresponding data set file path.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set rollout prediction directory
    rollout_directory = os.path.join(os.path.normpath(prediction_directory),
                                     f'rollout_bottle_{str(bottle_id)}')
    # Create rollout prediction directory
    make_directory(rollout_directory, is_overwrite=True)
    # Set rollout prediction plots directory
    rollout_plots_dir = \
        os.path.join(os.path.normpath(rollout_directory), 'plots')
    # Create rollout prediction plots directory
    make_directory(rollout_plots_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default GNN-based model prediction options
    loss_nature, loss_type, loss_kwargs = set_default_prediction_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = GNNGraphDataset.load_dataset(dataset_file_path)
    # Get initial time step sample graph
    time_step_graph = dataset[0]
    # Get number of dimensions and edges indexes matrix
    n_dim, edges_indexes = GraphData.extract_data_torch_data_object(
        time_step_graph, attributes=('n_dim', 'edge_indexes'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of time steps
    n_time_steps = len(dataset)
    # Initialize prediction loss history
    predict_loss_hist = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for i in range(n_time_steps):
        # Set time step data set
        time_step_dataset = GNNGraphDatasetInMemory([time_step_graph,])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Time step prediction with GNN-based model
        predict_subdir, time_step_predict_loss = \
            predict(time_step_dataset, model_directory,
                    predict_directory=predict_directory,
                    load_model_state='best', loss_nature=loss_nature,
                    loss_type=loss_type, loss_kwargs=loss_kwargs,
                    is_normalized_loss=True, 
                    dataset_file_path=dataset_file_path,
                    device_type=device_type, seed=None, is_verbose=is_verbose)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate plots of model prediction
        generate_prediction_plots(predict_subdir)
        # Store time step prediction loss
        predict_loss_hist.append(time_step_predict_loss)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set time step prediction results file path
        prediction_file_path = os.path.join(os.path.normpath(predict_subdir),
                                            f'prediction_sample_0.pkl')
        # Check prediction results file
        if not os.path.isfile(prediction_file_path):
            raise RuntimeError('The time step prediction results file has not '
                               'been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set rollout prediction results file path
        time_step_prediction_file_path = \
            f'prediction_sample_bottle_{str(bottle_id)}_tstep_{str(i)}.pkl'
        # Copy prediction results file to rollout prediction directory
        shutil.copy(prediction_file_path, time_step_prediction_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set prediction results plots directory
        plots_directory = \
            os.path.join(os.path.normpath(predict_subdir), 'plots')
        # Get prediction results plot files
        directory_list = os.listdir(plots_directory)
        # Loop over plots files
        for filename in directory_list:
            # Get file extension
            file_ext = os.path.splitext(filename)[1]
            # Set rollout prediction results plot file path
            time_step_plot_file_path = os.path.join(
                os.path.normpath(rollout_plots_dir),
                f'{filename}_{str(bottle_id)}_tstep_{str(i)}{file_ext}')
            # Copy prediction results plot file to rollout prediction plots
            # directory
            shutil.copy(
                os.path.join(os.path.normpath(plots_directory), filename),
                time_step_plot_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove time step prediction directory
        shutil.rmtree(predict_subdir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = \
            load_sample_predictions(time_step_prediction_file_path)
        # Get node coordinates predictions
        coords = sample_results['node_features_out']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get next time step sample graph
        time_step_graph = dataset[i + 1]
        # Get next time step nodes coordinates ground-truth
        node_targets_matrix = GraphData.extract_data_torch_data_object(
            time_step_graph, attributes=('node_targets_matrix'))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate graph data
        graph_data = GraphData(n_dim=n_dim, nodes_coords=coords)
        # Set graph edges
        graph_data.set_graph_edges_indexes(edges_indexes_mesh=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build nodes coordinates history with current time step predictions
        # and next time step ground-truth
        nodes_coords_hist = \
            np.concatenate((coords, node_targets_matrix), axis=1)
        # Instantiate finite element mesh features generator
        features_generator = FEMMeshFeaturesGenerator(
            n_dim=n_dim, nodes_coords_hist=nodes_coords_hist,
            edges_indexes=edges_indexes)
        # Build node features matrix
        node_features_matrix = features_generator.build_nodes_features_matrix(
            features=('coord_old',))
        # Set graph node features
        graph_data.set_node_features_matrix(node_features_matrix)
        # Build edge features matrix
        edge_features_matrix = features_generator.build_edges_features_matrix(
            features=('edge_vector_old', 'edge_vector_old_norm'))
        # Set graph edge features
        graph_data.set_edge_features_matrix(edge_features_matrix)
        # Set graph node targets
        graph_data.set_node_targets_matrix(node_targets_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set next time step sample graph
        time_step_graph = graph_data.get_torch_data_object()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build prediction process loss history
    loss_histories = {}
    loss_histories['Prediction (rollout)'] = predict_loss_hist
    # Plot model prediction process loss history
    plot_prediction_loss_history(
        loss_histories, loss_type=loss_type.upper(), loss_scale='log',
        filename=f'prediction_loss_history_bottle_{str(bottle_id)}',
        save_dir='rollout_plots_dir', is_save_fig=True,
        is_stdout_display=False, is_latex=True)
# =============================================================================
def set_default_prediction_options():
    """Set default GNN-based model prediction options.
    
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
    is_in_dist_testing = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'abaqus_datasets/case_studies')
    # Set case study directory
    case_study_name = 'incremental_model'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set testing data set directory
    if is_in_dist_testing:
        # Set testing data set directory (in-distribution)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '1_training_dataset')
    else:
        # Set testing data set directory (out-of-distribution)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '4_testing_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get testing data set file path
    regex = (r'^graph_dataset_testing_n[0-9]+.pkl$',
             r'^graph_dataset_n[0-9]+.pkl$')
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
    # Set GNN-based model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                   '2_model')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based model prediction directory
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
    # Perform prediction with GNN-based model
    perform_model_prediction(prediction_subdir, dataset_file_path,
                             model_directory, device_type, is_verbose=True)