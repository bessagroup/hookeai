"""GNN-based material patch data set graph samples files.

Functions
---------
generate_dataset_samples_files
    Generate GNN-based material patch data set graph samples files.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import time
import datetime
# Third-party
import numpy as np
import torch
import tqdm
# Local
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.data.graph_dataset import \
    write_graph_dataset_summary_file
from projects.gnn_material_patch.gnn_model_tools.features import \
    GNNPatchFeaturesGenerator
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
#
# =============================================================================
def generate_dataset_samples_files(dataset_directory, node_features,
                                   edge_features, dataset_simulation_data,
                                   sample_file_basename='material_patch_graph',
                                   is_save_plot_patch=False, is_verbose=False):
    """Generate GNN-based material patch data set samples files.

    Parameters
    ----------
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files). All existent files are overridden when saving
        sample data files.
    node_features : tuple[str]
        GNN-based material patch data set nodes features. Check class
        GNNPatchFeaturesGenerator for available node features.
    edge_features : tuple[str]
        GNN-based material patch data set edges features. Check class
        GNNPatchFeaturesGenerator for available edge features.
    dataset_simulation_data : list[dict]
        Material patches finite element simulations output data. Output data of
        each material patch is stored in a dict, where:
        
        'patch' : Instance of FiniteElementPatch, the simulated finite \
                  element material patch.
        
        'node_data' : numpy.ndarray(3d) of shape \
                      (n_nodes, n_data_dim, n_time_steps), where the i-th \
                      node output data at the k-th time step is stored in \
                      indexes [i, :, k].
        
        'global_data' : numpy.ndarray(3d) of shape \
                        (1, n_data_dim, n_time_steps) where the global output \
                        data at the k-th time step is stored in [0, :, k].
    sample_file_basename : str, default='material_patch_graph'
        Basename of GNN-based material patch data set sample file. The basename
        is appended with sample index.
    is_save_plot_patch : bool, default=False
        Save plot of each material patch graph in the same directory where the
        GNN-based material patch data set is stored.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files).
    dataset_sample_files : list[str]
        GNN-based material patch data set samples files paths. Each sample file
        contains a torch_geometric.data.Data object describing a homogeneous
        graph.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGenerate GNN-based material patch data set'
              '\n------------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The GNN-based material patch data set directory '
                           'has not been found:\n\n' + dataset_directory)
    else:
        dataset_directory = os.path.normpath(dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(dataset_simulation_data)
    # Initialize material patch data set samples files
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting graphs generation process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in tqdm.tqdm(range(n_sample), desc='> Generating graphs: ',
                       disable=not is_verbose):
        # Check sample
        patch = dataset_simulation_data[i]['patch']
        node_data = dataset_simulation_data[i]['node_data']
        if patch is None or node_data is None:
            raise RuntimeError('Failed material patch sample (index '
                               + str(i) + ' of data set).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get sample data
        nodes_coords_hist = node_data[:, 3:6, :]
        nodes_disps_hist=node_data[:, 6:9, :]
        nodes_int_forces_hist=node_data[:, 9:12, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch reference node coordinates
        node_coord_ref = nodes_coords_hist[:, 0:3, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GraphData(n_dim=patch.get_n_dim(),
                                   nodes_coords=node_coord_ref)
        # Set connectivity radius (maximum distance between two nodes leading
        # to an edge) based on finite element size
        connect_radius = \
            1.5*np.sqrt(np.sum([x**2 for x in patch.get_elem_size_dims()]))
        # Get finite element mesh edges indexes matrix
        edges_indexes_mesh = GraphData.get_edges_indexes_mesh(
            patch.get_mesh_connected_nodes())
        # Set GNN-based material patch graph edges
        gnn_patch_data.set_graph_edges_indexes(
            connect_radius=connect_radius,
            edges_indexes_mesh=edges_indexes_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch features generator
        features_generator = GNNPatchFeaturesGenerator(
            n_dim=patch.get_n_dim(),
            nodes_coords_hist=nodes_coords_hist,
            edges_indexes=gnn_patch_data.get_graph_edges_indexes(),
            nodes_disps_hist=nodes_disps_hist,
            nodes_int_forces_hist=nodes_int_forces_hist)
        # Compute node features matrix
        node_features_matrix = features_generator.build_nodes_features_matrix(
            features=node_features, n_time_steps=1)
        # Compute edge features matrix
        edge_features_matrix = features_generator.build_edges_features_matrix(
            features=edge_features, n_time_steps=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch graph node and edges features
        gnn_patch_data.set_node_features_matrix(node_features_matrix)
        gnn_patch_data.set_edge_features_matrix(edge_features_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute node targets matrix
        node_targets_matrix = features_generator.build_nodes_features_matrix(
            features=('int_force',), n_time_steps=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch graph node targets
        gnn_patch_data.set_node_targets_matrix(node_targets_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get PyG homogeneous graph data object
        pyg_graph = gnn_patch_data.get_torch_data_object()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample file name
        sample_file_name = sample_file_basename + '_' + str(i) + '.pt'
        # Set sample file path
        sample_file_path = os.path.join(os.path.normpath(dataset_directory),
                                        sample_file_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save GNN-based material patch graph sample file
        torch.save(pyg_graph, sample_file_path)
        # Save GNN-based material patch graph sample file path
        dataset_sample_files.append(sample_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save GNN-based material patch graph sample plot
        if is_save_plot_patch:
            # Set sample plot name
            sample_file_name = sample_file_basename + '_' + str(i) + '_plot'
            # Save sample plot
            gnn_patch_data.plot_graph(
                is_save_plot=is_save_plot_patch,
                save_directory=dataset_directory,
                plot_name=sample_file_name,
                is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished graphs generation process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Data set size (number of graphs): {n_sample}')
        if node_features_matrix is not None:
            print(f'\n> Node features ({node_features_matrix.shape[1]}): '
                f'{" || ".join([x for x in node_features])}')
        else:
            print(f'\n> Node features (0): None')
        if edge_features_matrix is not None:
            print(f'\n> Edge features ({edge_features_matrix.shape[1]}): '
                f'{" || ".join([x for x in edge_features])}')
        else:
            print(f'\n> Edge features (0): None')
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
                                     total_time_sec, avg_time_sec,
                                     node_features=node_features,
                                     edge_features=edge_features)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    return dataset_directory, dataset_sample_files