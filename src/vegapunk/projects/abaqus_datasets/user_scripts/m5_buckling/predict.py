"""User script: Predict with GNN-based model."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import shutil
import time
import datetime
# Third-party
import torch
import numpy as np
import tqdm
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset, \
    GNNGraphDatasetInMemory
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.predict.prediction import predict, load_sample_predictions
from gnn_base_model.predict.prediction_plots import \
    plot_prediction_loss_history
from projects.abaqus_datasets.gnn_model_tools.process_predictions import \
    build_prediction_data_arrays
from projects.abaqus_datasets.gnn_model_tools.features import \
    FEMMeshFeaturesGenerator
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
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
        bottle_id = int(id.groups()[0])
    else:
        raise RuntimeError('Could not extract ABAQUS data file ID from the '
                           'corresponding data set file path.')
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
    # Get number of time steps
    n_time_steps = len(dataset)
    # Loop over time steps
    for i in range(n_time_steps):
        # Set time step plots suffix
        plot_filename_suffix = f'_bottle_{str(bottle_id)}_tstep_{str(i)}'
        # Generate plots of model predictions
        generate_prediction_plots(predict_subdir, samples_ids=[i,], 
                                  plot_filename_suffix=plot_filename_suffix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set new name for prediction results subdirectory
    predict_subdir_new = os.path.join(os.path.dirname(predict_subdir),
                                      f'prediction_bottle_{str(bottle_id)}')
    # Remove existing prediction directory
    if os.path.isdir(predict_subdir_new):
        shutil.rmtree(predict_subdir_new)
    # Rename prediction results subdirectory
    shutil.move(predict_subdir, predict_subdir_new)
# =============================================================================
def generate_prediction_plots(predict_subdir, samples_ids='all',
                              plot_filename_suffix=None):
    """Generate plots of model predictions.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    samples_ids : {'all', list[int]}, default='all'
        Samples IDs whose prediction results are collated in each prediction
        data array.
    plot_filename_suffix : str, default=None
         Suffix to each plot filename.
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
            predict_subdir, prediction_type=key, samples_ids=samples_ids)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples predictions data arrays
        for i, data_array in enumerate(prediction_data_arrays):
            # Get prediction plot file name
            filename = val[i]
            if plot_filename_suffix:
                filename += str(plot_filename_suffix)
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
                          model_directory, load_model_state=None,
                          device_type='cpu', is_normalized_loss=False,
                          seed=None, is_verbose=False):
    """Perform prediction with GNN-based model.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset_file_path : str
        Testing data set file path.        
    model_directory : str
        Directory where GNN-based model is stored.
    load_model_state : {'best', 'last', int, None}, default=None
        Load available Graph Neural Network model state from the model
        directory. Options:
        
        'best' : Model state corresponding to best performance available
        
        'last' : Model state corresponding to highest training epoch
        
        int    : Model state corresponding to given training epoch
        
        None   : Model default state file

    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    seed : int, default=None
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGraph Neural Network model prediction rollout'
              '\n---------------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get ABAQUS data file ID from data set file path
    id = re.search(r'^graph_dataset_bottle_([0-9]+).*.pkl$',
                   os.path.basename(dataset_file_path))
    if id is not None:
        bottle_id = int(id.groups()[0])
    else:
        raise RuntimeError('Could not extract ABAQUS data file ID from the '
                           'corresponding data set file path.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set prediction rollout directory
    rollout_directory = \
        os.path.join(os.path.normpath(predict_directory),
                     f'prediction_rollout_bottle_{str(bottle_id)}')
    # Create prediction rollout directory
    make_directory(rollout_directory, is_overwrite=True)
    # Set prediction rollout plots directory
    rollout_plots_dir = \
        os.path.join(os.path.normpath(rollout_directory), 'plots')
    # Create prediction rollout plots directory
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
    if is_verbose:
        print('\n> Starting prediction rollout process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for i in tqdm.tqdm(range(n_time_steps), desc='> Prediction steps: ',
                       disable=not is_verbose):
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
                    device_type=device_type, seed=seed, is_verbose=False)
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
        # Set prediction rollout results file path
        time_step_prediction_file_path = os.path.join(
            os.path.normpath(rollout_directory),
            f'prediction_sample_bottle_{str(bottle_id)}_tstep_{str(i)}.pkl')
        # Copy prediction results file to prediction rollout directory
        shutil.copy(prediction_file_path, time_step_prediction_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set prediction results plots directory
        plots_directory = \
            os.path.join(os.path.normpath(predict_subdir), 'plots')
        # Get prediction results plot files
        directory_list = os.listdir(plots_directory)
        # Loop over plots files
        for filename in directory_list:
            # Get file basename
            file_basename = os.path.splitext(filename)[0]
            # Get file extension
            file_ext = os.path.splitext(filename)[1]
            # Set prediction rollout results plot file path
            time_step_plot_file_path = os.path.join(
                os.path.normpath(rollout_plots_dir),
                f'{file_basename}_bottle_{str(bottle_id)}_tstep_{str(i)}'
                f'{file_ext}')
            # Copy prediction results plot file to prediction rollout plots
            # directory
            shutil.copy(
                os.path.join(os.path.normpath(plots_directory), filename),
                time_step_plot_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove time step prediction directory
        shutil.rmtree(predict_subdir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Finish rollout after getting prediction for the last time step
        if i == n_time_steps - 1:
            break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = \
            load_sample_predictions(time_step_prediction_file_path)
        # Get node coordinates predictions
        coords = sample_results['node_features_out'].numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get next time step sample graph
        time_step_graph = dataset[i + 1]
        # Get next time step nodes coordinates ground-truth
        node_targets_matrix = GraphData.extract_data_torch_data_object(
            time_step_graph, attributes=('node_targets_matrix',))[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate graph data
        graph_data = GraphData(n_dim=n_dim, nodes_coords=coords)
        # Set graph edges
        graph_data.set_graph_edges_indexes(edges_indexes_mesh=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build nodes coordinates history with current time step predictions
        # and next time step ground-truth
        nodes_coords_hist = np.stack((coords, node_targets_matrix), axis=2)
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
    if is_verbose:
        print('\n> Finished prediction rollout process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute average prediction loss per step
    avg_predict_loss = None
    if (isinstance(predict_loss_hist, list)
            and len(predict_loss_hist) == len(dataset)):
        avg_predict_loss = np.mean(predict_loss_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total prediction time and average prediction time per step
    total_time_sec = time.time() - start_time_sec
    if len(dataset) > 0:
        avg_time_step = total_time_sec/n_time_steps
    else:
        avg_time_step = float('nan')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Prediction rollout results directory: {rollout_directory}')
        print(f'\n> Total prediction time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. prediction time per step: '
              f'{str(datetime.timedelta(seconds=int(avg_time_step)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for model prediction rollout process
    if predict_directory is not None:
        write_prediction_rollout_summary_file(
            rollout_directory, device_type, seed, model_directory,
            load_model_state, loss_type, loss_kwargs, is_normalized_loss,
            dataset_file_path, dataset, avg_predict_loss, total_time_sec,
            avg_time_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build prediction process loss history
    loss_histories = {}
    loss_histories['Prediction (rollout)'] = predict_loss_hist
    # Plot model prediction process loss history
    plot_prediction_loss_history(
        loss_histories, loss_type=loss_type.upper(), loss_scale='log',
        filename=f'prediction_loss_history_bottle_{str(bottle_id)}',
        save_dir=rollout_plots_dir, is_save_fig=True,
        is_stdout_display=False, is_latex=True)
# =============================================================================
def write_prediction_rollout_summary_file(
    rollout_directory, device_type, seed, model_directory, load_model_state,
    loss_type, loss_kwargs, is_normalized_loss, dataset_file_path, dataset,
    avg_predict_loss, total_time_sec, avg_time_step):
    """Write summary data file for model prediction process.

    Parameters
    ----------
    rollout_directory : str
        Prediction rollout directory.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where Graph Neural Network model is stored.
    load_model_state : {'best', 'last', int, None}
        Load available Graph Neural Network model state from the model
        directory. Data scalers are also loaded from model initialization file.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    dataset_file_path : str
        Graph Neural Network model graph data set file path if such file
        exists. Only used for output purposes.
    dataset : torch.utils.data.Dataset
        Graph Neural Network model graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    avg_predict_loss : float
        Average prediction loss per sample.
    total_time_sec : int
        Total prediction time in seconds.
    avg_time_step : float
        Average prediction time per step.
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
    summary_data['Avg. prediction loss per step: '] = \
        f'{avg_predict_loss:.8e}' if avg_predict_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per step'] = \
        str(datetime.timedelta(seconds=int(avg_time_step)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=rollout_directory,
        summary_title='Summary: Graph Neural Network model prediction rollout',
        **summary_data)
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
    is_in_dist_testing = False
    # Set prediction rollout flag
    is_rollout = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/projects/'
                'abaqus_datasets/case_studies/M5_buckling/')
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
    regex = (r'^graph_dataset_.*_n[0-9]+.pkl$')
    is_file_found, dataset_file_path = \
        find_unique_file_with_regex(dataset_directory, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Testing data set file has not been found '
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform prediction rollout with GNN-based model
    if is_rollout:
        perform_model_rollout(prediction_subdir, dataset_file_path,
                              model_directory, load_model_state='best',
                              device_type=device_type, is_normalized_loss=True,
                              is_verbose=True)