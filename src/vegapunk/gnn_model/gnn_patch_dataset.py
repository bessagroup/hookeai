"""GNN-based material patch data set.

Classes
-------
GNNMaterialPatchDataset(torch.utils.data.Dataset)
    GNN-based material patch data set.

Functions
---------
generate_dataset_samples_files
    Generate GNN-based material patch data set samples files.
get_dataset_sample_files_from_dir
    Get GNN-based material patch data set samples files from directory.
split_dataset
    Randomly split data set into non-overlapping parts.
get_pyg_data_loader
    Get GNN-based material patch data set PyG data loader.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import time
import datetime
import re
import pickle
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
def generate_dataset_samples_files(dataset_directory, dataset_simulation_data,
                                   sample_file_basename='material_patch_graph',
                                   is_save_plot_patch=False, is_verbose=False):
    """Generate GNN-based material patch data set samples files.

    Parameters
    ----------
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files). All existent files are overridden when saving
        sample data files.
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
    if is_verbose:
        print('\nGenerate GNN-based material patch data set'
              '\n------------------------------------------\n')
        start_time_sec = time.time()
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
        node_coord_ref = nodes_coords_hist[:, 0:3, 0]
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
            n_dim=patch.get_n_dim(),
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
            gnn_patch_data.plot_material_patch_graph(
                is_save_plot=is_save_plot_patch,
                save_directory=dataset_directory,
                plot_name=sample_file_name,
                is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Data set directory: ', dataset_directory)
        total_time_sec = time.time() - start_time_sec
        print(f'\n> Total generation time (s): '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    return dataset_directory, dataset_sample_files
# =============================================================================
def get_dataset_sample_files_from_dir(dataset_directory, sample_file_basename):
    """Get GNN-based material patch data set samples files from directory.
    
    Parameters
    ----------
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files).
    sample_file_basename : str
        Basename of GNN-based material patch data set sample file. The basename
        is appended with sample index.
        
    Returns
    -------
    dataset_sample_files : list[str]
        GNN-based material patch data set samples files paths. Each sample file
        contains a torch_geometric.data.Data object describing a homogeneous
        graph.
    """
    # Check GNN-based material patch data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The GNN-based material patch data set directory '
                           'has not been found:\n\n' + dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch data set samples files paths
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in GNN-based material patch data set directory
    directory_list = os.listdir(dataset_directory)
    # Loop over files
    for filename in directory_list:
        # Check if file is GNN-based material patch data set samples file
        is_sample_file = bool(re.search(r'^' + sample_file_basename
                                        + r'_[0-9]+' + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append sample file path
        if is_sample_file:
            dataset_sample_files.append(filename)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort GNN-based material patch data set samples
    dataset_sample_files = sorted(dataset_sample_files)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_sample_files
# =============================================================================
class GNNMaterialPatchDataset(torch.utils.data.Dataset):
    """GNN-based material patch data set.
    
    Attributes
    ----------
    _dataset_directory : str
        Directory where the GNN-based material patch data set is stored
        (all data set samples files).
    _dataset_sample_files : list[str]
        GNN-based material patch data set samples files paths. Each sample file
        contains a torch_geometric.data.Data object describing a homogeneous
        graph.
    _is_store_dataset : bool, default=False
        If True, then the GNN-based material patch data set samples are loaded
        and stored in attribute dataset_samples_data. If False, the dataset
        object holds solely the samples data files paths and load the
        corresponding files when accessing a given sample data.
    
    Methods
    -------
    __len__(self):
        Return size of data set (number of samples).
    __getitem__(self, index)
        Return data set sample from corresponding index.
    get_dataset_directory(self)
        Get directory where the GNN-based material patch data set is stored.
    get_dataset_sample_files(self)
        Get GNN-based material patch data set samples files paths.
    save_dataset(self)
        Save GNN-based material patch data set to file.
    update_dataset_directory(self, dataset_directory, is_reload_data=False)
        Update directory where GNN-based material patch data set is stored.
    """
    def __init__(self, dataset_directory, dataset_sample_files,
                 is_store_dataset=False):
        """Constructor.
        
        Parameters 
        ----------
        dataset_directory : str
            Directory where the GNN-based material patch data set is stored
            (all data set samples files).
        dataset_sample_files : list[str]
            GNN-based material patch data set samples file paths. Each sample
            file contains a torch_geometric.data.Data object describing a
            homogeneous graph.
        is_store_dataset : bool, default=False
            If True, then the GNN-based material patch data set samples are
            loaded and stored in attribute dataset_samples_data. If False,
            the dataset object holds solely the samples data files paths and
            load the corresponding files when accessing a given sample data.
        """
        # Initialize data set from base class
        super(GNNMaterialPatchDataset, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The GNN-based material patch data set '
                               'directory has not been found:\n\n'
                               + dataset_directory)
        else:
            self._dataset_directory = dataset_directory
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set sample files
        for file_path in dataset_sample_files:
            # Check if sample file exists
            if not os.path.isfile(file_path):
                raise RuntimeError('GNN-based material patch data set sample '
                                   'file has not been found:\n\n' + file_path)
            elif os.path.dirname(file_path) \
                    != os.path.normpath(dataset_directory):
                raise RuntimeError('GNN-based material patch data set sample '
                                   'file is not in dataset directory:\n\n'
                                   + dataset_directory)
        # Store data set samples file paths
        self._dataset_sample_files = dataset_sample_files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set sample data storange flag
        self._is_store_dataset = is_store_dataset
        # Load data set sample files data
        self._dataset_samples = []
        if is_store_dataset:
            # Loop over sample files
            for file_path in dataset_sample_files:
                # Load sample
                self._dataset_samples.append(torch.load(file_path))       
    # -------------------------------------------------------------------------
    def __len__(self):
        """Return size of data set (number of samples).
        
        Returns
        -------
        n_sample : int
            Data set size (number of samples).
        """
        return len(self._dataset_sample_files)
    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        """Return data set sample from corresponding index.
        
        Parameters
        ----------
        index : int
            Index of returned data set sample (index must be in [0, n_sample]).
            
        Returns
        -------
        pyg_graph : torch_geometric.data.Data
            Data set sample defined as a PyG data object describing a
            homogeneous graph.
        """
        # Get data set sample
        if self._is_store_dataset:
            pyg_graph = self._dataset_samples[index].clone()
        else:
            pyg_graph = torch.load(self._dataset_sample_files[index])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return pyg_graph
    # -------------------------------------------------------------------------
    def get_dataset_directory(self):
        """Get directory where the GNN-based material patch data set is stored.
        
        Returns
        -------
        dataset_directory : str
            Directory where the GNN-based material patch data set is stored
            (all data set samples files).
        """
        return self._dataset_directory
    # -------------------------------------------------------------------------
    def get_dataset_sample_files(self):
        """Get GNN-based material patch data set samples files paths.
        
        Returns
        -------
        dataset_sample_files : list[str]
            GNN-based material patch data set samples files paths. Each sample
            file contains a torch_geometric.data.Data object describing a
            homogeneous graph.
        """
        return self._dataset_sample_files
    # -------------------------------------------------------------------------
    def save_dataset(self):
        """Save GNN-based material patch data set to file.
        
        GNN-based material patch data set is stored in dataset_directory as a
        pickle file named 'material_patch_graph_dataset.pkl'.
        """
        # Check data set directory
        if not os.path.isdir(self._dataset_directory):
            raise RuntimeError('The GNN-based material patch data set '
                               'directory has not been found:\n\n'
                               + self._dataset_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch data set file path
        dataset_path = os.path.join(self._dataset_directory,
                                    'material_patch_graph_dataset' + '.pkl')
        # Save GNN-based material patch data set
        with open(dataset_path, 'wb') as dataset_file:
            pickle.dump(self, dataset_file)
    # -------------------------------------------------------------------------
    def update_dataset_directory(self, dataset_directory,
                                 is_reload_data=False):
        """Update directory where GNN-based material patch data set is stored.
        
        Stored data set samples files paths directory is updated according
        with new directory.
        
        Parameters
        ----------
        dataset_directory : str
            Directory where the GNN-based material patch data set is stored
            (all data set samples files).
        is_reload_data : bool, default=False
            Reload and store data set samples in attribute
            dataset_samples_data. Only effective if is_store_dataset=True.
        """
        # Set new data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The new GNN-based material patch data set '
                               'directory has not been found:\n\n'
                               + dataset_directory)
        else:
            self._dataset_directory = dataset_directory
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples files
        for i, file_path in enumerate(self._dataset_sample_files):
            # Set new sample file path (update directory)
            new_file_path = os.path.join(os.path.dirname(dataset_directory),
                                         os.path.basename(file_path))
            # Update sample file path
            if not os.path.isfile(new_file_path):
                raise RuntimeError('GNN-based material patch data set sample '
                                   'file is not in new dataset directory:\n\n'
                                   + dataset_directory)
            else:
                self._dataset_sample_files[i] = new_file_path
                # Reload sample data
                if self._is_store_dataset and is_reload_data:
                    self._dataset_samples[i] = torch.load(file_path)
# =============================================================================
def split_dataset(dataset, split_sizes, seed=None):
    """Randomly split data set into non-overlapping parts.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Data set.
    split_sizes : dict
        Size (item, float) for each data set split part name (key, str), where
        size is a fraction contained between 0 and 1. The sum of all sizes must
        equal 1.
    seed : int, default=None
        Seed for random data set split generator.
    
    Returns
    -------
    dataset_split : dict
        Split data set part (item, torch.utils.data.Dataset) for each data set
        split part name (key, str).
    """
    # Initialize data set split parts names and sizes
    parts_names = []
    parts_sizes = []
    # Assemble data set split parts names and sizes
    for key, val in split_sizes.items():
        # Check if part size is valid
        if val < 0.0 or val > 1.0:
            raise RuntimeError(f'Part size must be contained between 0 and 1. '
                               f'Check part (size): {key} ({val})')
        # Assemble part name and size
        parts_names.append(str(key))
        parts_sizes.append(val)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if split sizes are valid
    if not np.isclose(np.sum(parts_sizes), 1.0):
        raise RuntimeError('Sum of part split sizes must equal 1. '
                           f'Current sum: {np.sum(parts_sizes):.2f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random split generator
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Randomly split data set into non-overlapping parts
    subsets_list = torch.utils.data.random_split(dataset, parts_sizes,
                                                 generator=generator)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data set split parts
    dataset_split = {}
    for i, part in enumerate(parts_names):
        dataset_split[str(part)] = subsets_list[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    return dataset_split
# =============================================================================
def get_pyg_data_loader(dataset, batch_size=1, is_shuffle=False,
                        is_verbose=False, **kwargs):
    """Get GNN-based material patch data set PyG data loader.
    
    Parameters
    ----------
    dataset : {GNNMaterialPatchDataset, list[str]}
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
        Accepts GNNMaterialPatchDataset or a list of data set samples files
        paths, where each sample file contains a torch_geometric.data.Data
        object describing a homogeneous graph.
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
    # Build data set from list of data set samples files paths
    if isinstance(dataset, list):
        # Get data set samples files paths
        dataset_sample_files = dataset[:]
        # Get data set directory
        dataset_directory = \
            os.path.normpath(os.path.dirname(dataset_sample_files[0]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build data set
        dataset = GNNMaterialPatchDataset(
            dataset_directory, dataset_sample_files, is_store_dataset=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Building data loader...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GNN-based material patch data set PyG data loader
    data_loader = torch_geometric.loader.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_loader