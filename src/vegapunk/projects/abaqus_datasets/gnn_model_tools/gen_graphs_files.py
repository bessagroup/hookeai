"""Data set graph samples files.

Functions
---------
generate_dataset_samples_files
    Generate data set graph samples files.
extract_nodes_coords_from_input
    Extract finite element mesh nodes coordinates.
extract_connectivities_from_input
    Extract finite element mesh nodes connectivities.
extract_nodes_displacement_history
    Extract nodes displacement history.
convert_parquet_to_csv
    Convert '.parquet' file to '.csv' format and store in same directory.
"""
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
import time
import datetime
import re
# Third-party
import numpy as np
import torch
import tqdm
import pandas
import matplotlib.pyplot as plt
# Local
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.data.graph_dataset import \
    write_graph_dataset_summary_file
from ioput.plots import plot_histogram, save_figure
from projects.abaqus_datasets.gnn_model_tools.features import \
    FEMMeshFeaturesGenerator
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def generate_dataset_samples_files(dataset_directory, input_files_paths,
                                   data_files_paths,
                                   sample_file_basename='sample_graph',
                                   is_save_sample_plot=False,
                                   is_verbose=False):
    """Generate data set graph samples files.

    Parameters
    ----------
    dataset_directory : str
        Directory where the data set is stored (all data set samples files).
        All existent files are overridden when saving sample data files.
    input_files_paths : list[str]
        Data set ABAQUS input data files paths ('.inp' files).
    data_files_paths : list[str]
        Data set ABAQUS data files paths ('.parquet' > '.csv' files).
    sample_file_basename : str, default='sample_graph'
        Basename of data set sample file. The basename is appended with sample
        index.
    is_save_sample_plot : bool, default=False
        Save plot of each sample graph in the same directory where the data set
        is stored.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset_directory : str
        Directory where the data set is stored (all data set samples files).
    dataset_sample_files : list[str]
        Data set samples files paths. Each sample file contains a
        torch_geometric.data.Data object describing a homogeneous graph.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGenerate ABAQUS data set'
              '\n------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The data set directory has not been found:'
                           '\n\n' + dataset_directory)
    else:
        dataset_directory = os.path.normpath(dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(input_files_paths)
    # Check number of data files
    if len(data_files_paths) != n_sample:
        raise RuntimeError(f'Number of ABAQUS data files '
                           f'({len(data_files_paths)}) does not match number '
                           f'of ABAQUS input data files ({n_sample}).')
    # Set number of dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data set samples files
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting graphs generation process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize sample graph index
    sample_graph_id = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over ABAQUS input data files
    for i in tqdm.tqdm(range(n_sample), desc='> Generating graphs: ',
                       disable=not is_verbose):
        # Get input data file path
        input_file_path = input_files_paths[i]
        # Get ABAQUS input data file ID
        abaqus_file_id = int(os.path.basename(input_file_path).split('.')[0])
        # Get data file path
        data_file_path = data_files_paths[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input data file
        if not os.path.isfile(input_file_path):
            raise RuntimeError('ABAQUS input data file has not been found:'
                               '\n\n', input_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract finite element mesh nodes coordinates
        nodes_coords = extract_nodes_coords_from_input(input_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract finite element mesh nodes connectivities
        connected_nodes = extract_connectivities_from_input(input_file_path)
        # Get finite element mesh edges indexes matrix
        edges_indexes_mesh = GraphData.get_edges_indexes_mesh(connected_nodes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract nodes displacement history
        nodes_coords_hist, nodes_disps_hist, time_hist = \
            extract_nodes_displacement_history(data_file_path)
        # Get number of discrete times
        n_times = len(time_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node features
        node_features = ('coord_old', 'time')
        # Set edge features
        edge_features = ('edge_vector_init', 'edge_vector_init_norm')
        # Set node targets
        node_targets = ('coord',)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over time steps
        for j in tqdm.tqdm(range(n_times - 1),
                           desc='  > Processing time steps: ', leave=False):
            # Instantiate graph data
            graph_data = GraphData(n_dim=n_dim,
                                   nodes_coords=nodes_coords_hist[:, :, 0])
            # Set graph edges
            graph_data.set_graph_edges_indexes(
                edges_indexes_mesh=edges_indexes_mesh)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Instantiate finite element mesh features generator
            features_generator = FEMMeshFeaturesGenerator(
                n_dim=n_dim, nodes_coords_hist=nodes_coords_hist[:, :, 0:j+2],
                edges_indexes=graph_data.get_graph_edges_indexes(),
                nodes_disps_hist=nodes_disps_hist[:, :, 0:j+2],
                time_hist=time_hist[0:j+2])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build node features matrix
            node_features_matrix = \
                features_generator.build_nodes_features_matrix(
                    features=node_features)
            # Set graph node features
            graph_data.set_node_features_matrix(node_features_matrix)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build edge features matrix
            edge_features_matrix = \
                features_generator.build_edges_features_matrix(
                    features=edge_features)
            # Set graph edge features
            graph_data.set_edge_features_matrix(edge_features_matrix)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set node targets matrix
            node_targets_matrix = \
                features_generator.build_nodes_features_matrix(
                    features=node_targets)
            # Set graph node targets
            graph_data.set_node_targets_matrix(node_targets_matrix)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get PyG homogeneous graph data object
            pyg_graph = graph_data.get_torch_data_object()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set sample file name
            sample_file_name = \
                (f'{sample_file_basename}_{str(sample_graph_id)}'
                 f'_bottle_{str(abaqus_file_id)}_tstep_{str(j)}.pt')
            # Set sample file path
            sample_file_path = os.path.join(
                os.path.normpath(dataset_directory), sample_file_name)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save graph sample file
            torch.save(pyg_graph, sample_file_path)
            # Save graph sample file path
            dataset_sample_files.append(sample_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save graph sample plot
            if is_save_sample_plot and j == 0:
                # Set sample plot name
                sample_file_name = \
                    (f'{sample_file_basename}_{str(sample_graph_id)}'
                     f'_bottle_{str(abaqus_file_id)}_tstep_{str(j)}_plot')
                # Save sample plot
                graph_data.plot_graph(is_save_plot=is_save_sample_plot,
                                      save_directory=dataset_directory,
                                      plot_name=sample_file_name,
                                      is_overwrite_file=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Increment sample graph index
            sample_graph_id += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished graphs generation process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total generation time and average generation time per patch
    total_time_sec = time.time() - start_time_sec
    avg_time_sec = total_time_sec/n_sample
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Data set directory: ', dataset_directory)
        print(f'\n> Total generation time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. generation time per graph: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for graph data set generation
    write_graph_dataset_summary_file(
        dataset_directory, n_sample, total_time_sec, avg_time_sec,
        node_features=node_features, edge_features=edge_features,
        node_targets=node_targets,
        filename=f'summary_bottle_{str(abaqus_file_id)}_tstep_X')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_directory, dataset_sample_files
# =============================================================================
def extract_nodes_coords_from_input(input_file_path):
    """Extract finite element mesh nodes coordinates.
    
    Parameters
    ----------
    input_file_path : str
        ABAQUS input data file path ('.inp' file).
        
    Returns
    -------
    nodes_coords : numpy.ndarray(2d)
        Coordinates of nodes stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    """
    # Check input data file
    if not os.path.isfile(input_file_path):
        raise RuntimeError('ABAQUS input data file has not been found:'
                           '\n\n', input_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open input data file
    input_file = open(input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize nodes coordinates
    nodes_coords = np.empty((0, 3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for NODE keyword and collect nodes coordinates
    line_number = 0
    for line in input_file:
        if '*NODE' in line or '*Node' in line:
            # Start processing NODE section
            is_keyword_found = True
        elif is_keyword_found and bool(re.search(r'^' + r'[*][A-Z]+', line)):
            # Finished processing NODE section
            break
        elif is_keyword_found:
            # Get node coordinates
            node_coords = \
                np.array([float(x) for x in line.split(sep=',')[1:4]])
            # Store node coordinates
            nodes_coords = \
                np.append(nodes_coords, node_coords.reshape(1, -1), axis=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if NODE keyword was found
    if nodes_coords.shape[0] == 0:
        raise RuntimeError('The *NODE keyword has not been found.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords
# =============================================================================
def extract_connectivities_from_input(input_file_path):
    """Extract finite element mesh nodes connectivities.
    
    Parameters
    ----------
    input_file_path : str
        ABAQUS input data file path ('.inp' file).
        
    Returns
    -------
    connected_nodes : tuple[tuple(2)]
        A set containing all pairs of nodes that are connected by any
        relevant mesh representation (e.g., finite element mesh). Each
        connection is stored a single time as a tuple(node[int], node[int])
        and is independent of the corresponding nodes storage order.
    """
    # Check input data file
    if not os.path.isfile(input_file_path):
        raise RuntimeError('ABAQUS input data file has not been found:'
                           '\n\n', input_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open input data file
    input_file = open(input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize nodes connectivities
    connected_nodes = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for ELEMENT keyword and collect nodes connectivities
    line_number = 0
    for line in input_file:
        if '*ELEMENT, TYPE' in line or '*Element, type' in line:
            # Start processing ELEMENT section
            is_keyword_found = True
        elif is_keyword_found and bool(re.search(r'^' + r'[*][A-Z]+', line)):
            # Finished processing ELEMENT section
            is_keyword_found = False
        elif is_keyword_found:
            # Get element nodes
            elem_nodes = [int(x) for x in line.split(sep=',')[1:]]
            # Duplicate first node to process last connectivity
            elem_nodes.append(elem_nodes[0])
            # Store element connectivities
            for i in range(len(elem_nodes) - 1):
                # Set connectivity
                connectivity = (elem_nodes[i], elem_nodes[i + 1])
                # Store new connectivity
                if (connectivity not in connected_nodes
                    and connectivity[:-1] not in connected_nodes):
                    connected_nodes.append(connectivity)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if ELEMENT keyword was found
    if len(connected_nodes) == 0:
        raise RuntimeError('The *ELEMENT, TYPE keyword has not been found.')    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return connected_nodes
# =============================================================================
def extract_nodes_displacement_history(data_file_path):
    """Extract nodes displacement history.
    
    Parameters
    ----------
    data_file_path : str
        ABAQUS data file path ('.parquet' > '.csv' file).

    Returns
    -------
    nodes_coords_hist : numpy.ndarray(3d)
        Nodes coordinates history stored as a numpy.ndarray(3d) with shape
        (n_nodes, n_dim, n_time_steps). Coordinates of i-th node at k-th
        time step are stored in nodes_coords[i, :, k].
    nodes_disps_hist : numpy.ndarray(3d), default=None
        Nodes displacements history stored as a numpy.ndarray(3d) with
        shape (n_nodes, n_dim, n_time_steps). Displacements of i-th node at
        k-th time step are stored in nodes_disps_hist[i, :, k].
    time_hist : tuple
        Discrete time history.
    """
    # Check data file
    if not os.path.isfile(data_file_path):
        raise RuntimeError('ABAQUS data file has not been found:'
                           '\n\n', data_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data
    df = pandas.read_csv(data_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get node ids
    nodes_ids = tuple(set(df.loc[:, 'Node']))
    # Get number of nodes
    n_nodes = len(nodes_ids)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe data parameters and time history from first node
    df_node_id = df.loc[df['Node'] == nodes_ids[0]]
    # Check number of columns
    if df_node_id.shape[1] != 8:
        raise RuntimeError(f'Expecting data frame to have 8 columns, but '
                           f'{df_node_id.shape[1]} were found. \n\n'
                           f'Expected columns: Node | X1 X2 X3 | Time | '
                           f'U1 U2 U3')
    # Get discrete time history
    time_hist = tuple(df_node_id['Time'].tolist())
    # Get number of discrete time steps
    n_time_steps = len(time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize nodes coordinates history
    nodes_coords_hist = np.zeros((n_nodes, 3, n_time_steps))
    # Initialize nodes displacements history
    nodes_disps_hist = np.zeros_like(nodes_coords_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for node_id in nodes_ids:
        # Get node data
        df_node_id = df.loc[df['Node'] == node_id]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node index
        i = node_id - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node displacement history
        nodes_disps_hist[i, :, :] = \
            np.array(df_node_id[['U1', 'U2', 'U3']]).transpose()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node coordinates history
        nodes_coords_hist[i, :, :] = \
            (np.array(df_node_id[['X1', 'X2', 'X3']])
             + np.array(df_node_id[['U1', 'U2', 'U3']])).transpose()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check discrete time history consistency        
        if not np.allclose(time_hist, tuple(df_node_id['Time'].tolist())):
            raise RuntimeError(f'Node {node_id} and Node 1 discrete time '
                               f'histories are not consistent.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_hist, nodes_disps_hist, time_hist
# =============================================================================
def extract_time_step_history(data_file_path):
    """Extract discrete time history.
    
    Parameters
    ----------
    data_file_path : str
        ABAQUS data file path ('.parquet' > '.csv' file).

    Returns
    -------
    time_hist : tuple
        Discrete time history.
    time_step_hist : tuple
        Discrete time step history.
    """
    # Check data file
    if not os.path.isfile(data_file_path):
        raise RuntimeError('ABAQUS data file has not been found:'
                           '\n\n', data_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data
    df = pandas.read_csv(data_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get node ids
    nodes_ids = tuple(set(df.loc[:, 'Node']))
    # Get number of nodes
    n_nodes = len(nodes_ids)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe data parameters and time history from first node
    # Get data from first node (shared time history)
    df_node_id = df.loc[df['Node'] == nodes_ids[0]]
    # Check number of columns
    if df_node_id.shape[1] != 8:
        raise RuntimeError(f'Expecting data frame to have 8 columns, but '
                           f'{df_node_id.shape[1]} were found. \n\n'
                           f'Expected columns: Node | X1 X2 X3 | Time | '
                           f'U1 U2 U3')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get discrete time history
    time_hist = tuple(df_node_id['Time'].tolist())
    # Get number of discrete times
    n_times = len(time_hist)
    # Compute discrete time step history
    time_step_hist = [time_hist[i + 1] - time_hist[i]
                      for i in range(n_times - 2)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return time_hist, time_step_hist
# =============================================================================
def plot_time_step_distribution(data_files_paths,
                                filename='discrete_time_step_distribution',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Generate histogram of discrete time step size.
    
    Parameters
    ----------
    data_files_paths : list[str]
        Data set ABAQUS data files paths ('.parquet' > '.csv' files).
    filename : str, default='discrete_time_step_distribution'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Initialize discrete time step population
    time_step_population = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(data_files_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over ABAQUS data files
    for i in range(n_sample):
        # Get data file path
        data_file_path = data_files_paths[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input data file
        if not os.path.isfile(data_file_path):
            raise RuntimeError('ABAQUS input data file has not been found:'
                               '\n\n', data_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract discrete time step history
        _, time_step_hist = extract_time_step_history(data_file_path)
        # Append to population
        time_step_population += list(time_step_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set histogram data sets
    hist_data = (np.array(time_step_population),)
    # Set histogram data sets labels
    hist_data_labels = None
    # Set probability density flag
    density = False
    # Set title
    title = 'Discrete time step distribution'
    # Set histogram axes labels
    x_label = 'Discrete time step'
    if density:
        y_label = 'Probability density'
    else:
        y_label = 'Frequency'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot histogram of discrete time step size
    figure, _ = plot_histogram(hist_data, data_labels=hist_data_labels,
                               density=False, title=title, x_label=x_label,
                               y_label=y_label, is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def convert_parquet_to_csv(parquet_file_path):
    """Convert '.parquet' file to '.csv' format and store in same directory.
    
    Parameters
    ----------
    parquet_file_path : str
        Source '.parquet' file path.

    Returns
    -------
    csv_file_path : str
        Converted '.csv' file path.
    """
    # Check parquet data file
    if not os.path.isfile(parquet_file_path):
        raise RuntimeError('The \'.parquet\' data file has not been found:'
                           '\n\n', parquet_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load '.parquet' file into data frame
    df = pandas.read_parquet(parquet_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set '.csv' file path
    csv_file_path = parquet_file_path.replace('.parquet', '.csv')
    # Store data into '.csv' file format
    df.to_csv(csv_file_path, encoding='utf-8', index=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return csv_file_path