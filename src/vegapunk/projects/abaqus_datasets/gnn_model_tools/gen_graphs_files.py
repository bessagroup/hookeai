"""Data set graph samples files.

Functions
---------
generate_dataset_samples_files
    Generate data set graph samples files.
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
# Local
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.data.graph_dataset import \
    write_graph_dataset_summary_file
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
        Data set ABAQUS input data files paths.
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
    # Initialize data set samples files
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting graphs generation process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over ABAQUS input data files
    for i, input_file_path in enumerate(tqdm.tqdm(input_files_paths,
                                        desc='> Generating graphs: ',
                                        disable=not is_verbose)):
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
        # Instantiate graph data
        graph_data = GraphData(n_dim=3, nodes_coords=nodes_coords)
        # Set graph edges
        graph_data.set_graph_edges_indexes(
            edges_indexes_mesh=edges_indexes_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get PyG homogeneous graph data object
        pyg_graph = graph_data.get_torch_data_object()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample file name
        sample_file_name = sample_file_basename + '_' + str(i) + '.pt'
        # Set sample file path
        sample_file_path = os.path.join(os.path.normpath(dataset_directory),
                                        sample_file_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save graph sample file
        torch.save(pyg_graph, sample_file_path)
        # Save graph sample file path
        dataset_sample_files.append(sample_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save graph sample plot
        if is_save_sample_plot:
            # Set sample plot name
            sample_file_name = sample_file_basename + '_' + str(i) + '_plot'
            # Save sample plot
            graph_data.plot_material_patch_graph(
                is_save_plot=is_save_sample_plot,
                save_directory=dataset_directory,
                plot_name=sample_file_name,
                is_overwrite_file=True)
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
    write_graph_dataset_summary_file(dataset_directory, n_sample,
                                     total_time_sec, avg_time_sec)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =============================================================================
def extract_nodes_coords_from_input(input_file_path):
    """Extract finite element mesh nodes coordinates.
    
    Parameters
    ----------
    input_file_path : str
        ABAQUS input data file path.
        
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
        if '*NODE' in line:
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
        ABAQUS input data file path.
        
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
        if '*ELEMENT, TYPE' in line:
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
if __name__ == "__main__":
    
    dataset_directory = '/home/bernardoferreira/Documents/brown/projects/abaqus_datasets/datasets/ABAQUS_M5_buckling/graph_dataset'
    
    input_files_paths = ['/home/bernardoferreira/Documents/brown/projects/abaqus_datasets/datasets/ABAQUS_M5_buckling/M5_buckling/bottle_inp/1.inp',]
    
    generate_dataset_samples_files(dataset_directory, input_files_paths,
                                   is_save_sample_plot=True,
                                   is_verbose=True)