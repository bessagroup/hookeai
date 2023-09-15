"""GNN-based material patch data set.

Functions
---------
generate_gnn_material_patch_dataset
    Generate GNN-based material patch data set.
get_gnn_material_patch_data_loader
    Get GNN-based material patch data set PyG data loader.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
import torch
import torch_geometric.loader
import tqdm
# Local
from gnn_model.gnn_patch_data import GNNPatchGraphData, \
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
def generate_gnn_material_patch_dataset(dataset_directory,
    dataset_simulation_data, sample_file_basename='gnn_patch_sample',
    is_save_plot_patch=False, is_verbose=False):
    """Generate GNN-based material patch data set.

    Parameters
    ----------
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored.
        All existent files are overridden when saving sample data files.
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
    sample_file_basename : str, default='gnn_patch_sample'
        Basename of GNN-based material patch data set sample file. The basename
        is appended with sample index.
    is_save_plot_patch : bool, default=False
        Save plot of each material patch design sample in the same directory
        where the GNN-based material patch data set is stored.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset_sample_files : list[str]
        GNN-based material patch data set samples file paths. Each sample file
        contains a torch_geometric.data.Data object describing a homogeneous
        graph.
    """
    if is_verbose:
        print('\nGenerate GNN-based material patch data set'
              '\n------------------------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(dataset_simulation_data)
    # Initialize material patch data set samples files
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in tqdm.tqdm(range(n_sample), desc='> Generating data set: ',
                       disable=not is_verbose):
        # Check sample
        patch = dataset_simulation_data[i]['patch']
        node_data = dataset_simulation_data[i]['node_data']
        if patch is None or node_data is None:
            raise RuntimeError('Failed material patch sample (index '
                               + str(i) + ' of data set).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get sample data
        nodes_coords_hist = node_data[:, 1:4, :]
        nodes_disps_hist=node_data[:, 4:7, :]
        nodes_int_forces_hist=node_data[:, 7:10, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch reference node coordinates
        node_coord_ref = nodes_coords_hist[:, 1:4, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GNNPatchGraphData(n_dim=patch.get_n_dim(),
                                           nodes_coords=node_coord_ref)
        # Set connectivity radius (maximum distance between two nodes leading
        # to an edge) based on finite element size
        connect_radius = \
            1.5*np.sqrt(np.sum([x**2 for x in patch.get_elem_size_dims()]))
        # Get finite element mesh edges indexes matrix
        edges_indexes_mesh = GNNPatchGraphData.get_edges_indexes_mesh(
            patch.get_mesh_connected_nodes())
        # Set GNN-based material patch graph edges
        gnn_patch_data.set_graph_edges_indexes(
            connect_radius=connect_radius,
            edges_indexes_mesh=edges_indexes_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch features generator
        features_generator = GNNPatchFeaturesGenerator(
            nodes_coords_hist=nodes_coords_hist,
            edges_indexes=gnn_patch_data.get_graph_edges_indexes(),
            nodes_disps_hist=nodes_disps_hist,
            nodes_int_forces_hist=nodes_int_forces_hist)
        # Compute node features matrix
        node_features_matrix = features_generator.build_nodes_features_matrix(
            features=('coord_hist', 'disp_hist'), n_time_steps=1)
        # Get available edges features
        edge_features = \
            GNNPatchFeaturesGenerator.get_available_edges_features()
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
        gnn_patch_data.set_node_features_matrix(node_targets_matrix)
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
            sample_file_name = sample_file_basename + '_plot_' + str(i)
            # Save sample plot
            patch.plot_deformed_patch(is_save_plot=is_save_plot_patch,
                save_directory=dataset_directory,
                plot_name=sample_file_name,
                is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Data set directory: ', dataset_directory, '\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    return dataset_sample_files
# =============================================================================
def get_gnn_material_patch_data_loader(dataset_sample_files, batch_size=1,
                                       is_shuffle=False, is_verbose=False,
                                       **kwargs):
    """Get GNN-based material patch data set PyG data loader.
    
    Parameters
    ----------
    dataset_sample_files : list[str]
        GNN-based material patch data set samples file paths. Each sample file
        contains a torch_geometric.data.Data object describing a homogeneous
        graph.
    batch_size : int, default=1
        Number of samples loaded per batch.
    is_shuffle : bool, default=False
        Reshuffle data set at every epoch.
    is_verbose : bool, default=False
        If True, enable verbose output.
    **kwargs
        Arguments of torch.utils.data.DataLoader.
    
    Returns
    -------
    data_loader : torch_geometric.loader.DataLoader
        GNN-based material patch data set PyG data loader.
    """
    if is_verbose:
        print('\nBuild GNN-based material patch data loader'
              '\n------------------------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch graph sample data set
    dataset = []
    # Read GNN-based material patch graph sample data set
    for sample_file_path in tqdm.tqdm(dataset_sample_files,
                                      desc='> Loading samples: ',
                                      disable=not is_verbose):
        # Check sample file
        if not os.path.isfile(sample_file_path):
            raise RuntimeError('GNN-based material patch graph sample file '
                               'has not been found:\n\n', sample_file_path)
        # Load sample
        dataset.append(torch.load(sample_file_path))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Building data loader...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GNN-based material patch data set PyG data loader
    data_loader = torch_geometric.loader.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_loader