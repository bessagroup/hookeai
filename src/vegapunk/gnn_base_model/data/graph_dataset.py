"""Graph Neural Network graph data set.

Classes
-------
GNNGraphDataset(torch.utils.data.Dataset)
    Graph Neural Network graph data set.
GNNGraphDatasetInMemory(torch.utils.data.Dataset)
    Graph Neural Network graph data set (in-memory storage only).

Functions
---------
get_dataset_sample_files_from_dir
    Get Graph Neural Network graph data set samples files from directory.
split_dataset
    Randomly split data set into non-overlapping parts.
get_subset_indices_mapping
    Get mapping from subset indexes to whole parent data set indices.
get_pyg_data_loader
    Get Graph Neural Network graph data set PyG data loader.
write_graph_dataset_summary_file
    Write summary data file for Graph Neural Network graph data set generation.
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
# Local
from ioput.iostandard import write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def get_dataset_sample_files_from_dir(dataset_directory, sample_file_basename):
    """Get Graph Neural Network graph data set samples files from directory.
    
    Parameters
    ----------
    dataset_directory : str
        Directory where the Graph Neural Network graph data set is stored (all
        data set samples files).
    sample_file_basename : str
        Basename of Graph Neural Network graph data set sample file. The
        basename is appended with sample index.
        
    Returns
    -------
    dataset_sample_files : list[str]
        Graph Neural Network graph data set samples files paths. Each sample
        file contains a torch_geometric.data.Data object describing a
        homogeneous graph.
    """
    # Check Graph Neural Network graph data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The Graph Neural Network graph data set directory '
                           'has not been found:\n\n' + dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Graph Neural Network graph data set samples files paths
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in Graph Neural Network graph data set directory
    directory_list = os.listdir(dataset_directory)
    # Loop over files
    for filename in directory_list:
        # Check if file is Graph Neural Network graph data set samples file
        is_sample_file = bool(re.search(r'^' + sample_file_basename
                                        + r'_[0-9]+' + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append sample file path
        if is_sample_file:
            dataset_sample_files.append(filename)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort Graph Neural Network graph data set samples
    dataset_sample_files = sorted(dataset_sample_files)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_sample_files
# =============================================================================
class GNNGraphDataset(torch.utils.data.Dataset):
    """Graph Neural Network graph data set.
    
    Attributes
    ----------
    _dataset_directory : str
        Directory where the Graph Neural Network graph data set is stored
        (all data set samples files).
    _dataset_sample_files : list[str]
        Graph Neural Network graph data set samples files paths. Each sample
        file contains a torch_geometric.data.Data object describing a
        homogeneous graph.
    _dataset_samples : list
        Graph Neural Network graph data set samples data. Each sample is stored
        as a torch_geometric.data.Data object describing a homogeneous graph.
        Only populated if _is_store_dataset is True, otherwise is set as an
        empty list.
    _is_store_dataset : bool, default=False
        If True, then the Graph Neural Network graph data set samples are
        loaded and stored in attribute dataset_samples_data. If False, the
        dataset object holds solely the samples data files paths and load the
        corresponding files when accessing a given sample data.
    _dataset_basename : str
        Data set file base name.
    
    Methods
    -------
    __len__(self):
        Return size of data set (number of samples).
    __getitem__(self, index)
        Return data set sample from corresponding index.
    get_dataset_directory(self)
        Get directory where the Graph Neural Network graph data set is stored.
    get_dataset_sample_files(self)
        Get Graph Neural Network graph data set data set samples files paths.
    set_dataset_basename(self, dataset_basename)
        Set data set file base name.
    get_dataset_basename(self)
        Get data set file base name.
    save_dataset(self)
        Save Graph Neural Network graph data set to file.
    load_dataset(dataset_file_path)
        Load PyTorch data set.
    _update_dataset_directory(self, dataset_directory, is_reload_data=False)
        Update directory where Graph Neural Network graph data set is stored.
    update_dataset_file_internal_directory(dataset_file_path, new_directory, \
                                           is_reload_data=False)
        Update internal directory of stored data set in provided file.
    """
    def __init__(self, dataset_directory, dataset_sample_files,
                 dataset_basename='graph_dataset', is_store_dataset=False):
        """Constructor.
        
        Parameters 
        ----------
        dataset_directory : str
            Directory where the Graph Neural Network graph data set is stored
            (all data set samples files).
        dataset_sample_files : list[str]
            Graph Neural Network graph data set samples file paths. Each sample
            file contains a torch_geometric.data.Data object describing a
            homogeneous graph.
        dataset_basename : str, default='graph_dataset'
            Data set file base name.
        is_store_dataset : bool, default=False
            If True, then the Graph Neural Network graph data set samples are
            loaded and stored in attribute dataset_samples_data. If False,
            the dataset object holds solely the samples data files paths and
            load the corresponding files when accessing a given sample data.
        """
        # Initialize data set from base class
        super(GNNGraphDataset, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The Graph Neural Network graph data set '
                               'directory has not been found:\n\n'
                               + dataset_directory)
        else:
            self._dataset_directory = os.path.normpath(dataset_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set sample files
        for file_path in dataset_sample_files:
            # Check if sample file exists
            if not os.path.isfile(file_path):
                raise RuntimeError('Graph Neural Network graph data set '
                                   'sample file has not been found:\n\n'
                                   + file_path)
            elif os.path.dirname(file_path) \
                    != os.path.normpath(dataset_directory):
                raise RuntimeError('Graph Neural Network graph data set '
                                   'sample file is not in dataset directory:'
                                   '\n\n' + dataset_directory)
        # Store data set samples file paths
        self._dataset_sample_files = dataset_sample_files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set sample data storage flag
        self._is_store_dataset = is_store_dataset
        # Load data set sample files data
        self._dataset_samples = []
        if is_store_dataset:
            # Loop over sample files
            for file_path in dataset_sample_files:
                # Load sample
                self._dataset_samples.append(torch.load(file_path))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set file base name
        self._dataset_basename = str(dataset_basename)
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
        """Get directory where Graph Neural Network graph data set is stored.
        
        Returns
        -------
        dataset_directory : str
            Directory where the Graph Neural Network graph data set is stored
            (all data set samples files).
        """
        return self._dataset_directory
    # -------------------------------------------------------------------------
    def get_dataset_sample_files(self):
        """Get Graph Neural Network graph data set samples files paths.
        
        Returns
        -------
        dataset_sample_files : list[str]
            Graph Neural Network graph data set samples files paths. Each
            sample file contains a torch_geometric.data.Data object describing
            a homogeneous graph.
        """
        return self._dataset_sample_files
    # -------------------------------------------------------------------------
    def set_dataset_basename(self, dataset_basename):
        """Set data set file base name.
        
        Parameters
        ----------
        dataset_basename : str
            Data set file base name.
        """
        self._dataset_basename = str(dataset_basename)
    # -------------------------------------------------------------------------
    def get_dataset_basename(self):
        """Get data set file base name.
        
        Returns
        -------
        dataset_basename : str
            Data set file base name.
        """
        return self._dataset_basename
    # -------------------------------------------------------------------------
    def save_dataset(self, is_append_n_sample=True):
        """Save Graph Neural Network graph data set to file.
        
        Graph Neural Network graph data set is stored in dataset_directory as a
        pickle file named material_patch_graph_dataset.pkl or
        material_patch_graph_dataset_n< n_sample >.pkl.
        
        Parameters
        ----------
        is_append_n_sample : bool, default=True
            If True, then data set size (number of samples) is appended to
            Graph Neural Network graph data set filename.
            
        Returns
        -------
        dataset_file_path : str
            PyTorch data set file path.
        """
        # Check data set directory
        if not os.path.isdir(self._dataset_directory):
            raise RuntimeError('The Graph Neural Network graph data set '
                               'directory has not been found:\n\n'
                               + self._dataset_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Graph Neural Network graph data set file
        dataset_file = self._dataset_basename
        # Append data set size
        if is_append_n_sample:
            dataset_file += f'_n{len(self._dataset_sample_files)}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Graph Neural Network graph data set file path
        dataset_file_path = os.path.join(self._dataset_directory,
                                         dataset_file + '.pkl')   
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save Graph Neural Network graph data set
        with open(dataset_file_path, 'wb') as dataset_file:
            pickle.dump(self, dataset_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dataset_file_path
    # -------------------------------------------------------------------------
    @staticmethod
    def load_dataset(dataset_file_path):
        """Load PyTorch data set.
        
        Parameters
        ----------
        dataset_file_path : str
            PyTorch data set file path.
        
        Returns
        -------
        dataset : torch.utils.data.Dataset
            PyTorch data set.
        """
        # Check PyTorch data set file
        if not os.path.isfile(dataset_file_path):
            raise RuntimeError('PyTorch data set file has not been found:\n\n'
                               + dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load PyTorch data set
        with open(dataset_file_path, 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check PyTorch data set
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise RuntimeError('Loaded data set is not a '
                               'torch.utils.data.Dataset.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dataset
    # -------------------------------------------------------------------------
    def _update_dataset_directory(self, dataset_directory,
                                  is_reload_data=False):
        """Update directory where GNN graph data set is stored.
        
        Stored data set samples files paths directory is updated according
        with the new directory.
        
        Parameters
        ----------
        dataset_directory : str
            Directory where the Graph Neural Network graph data set is stored
            (all data set samples files).
        is_reload_data : bool, default=False
            Reload and store data set samples in attribute
            dataset_samples_data. Only effective if is_store_dataset=True.
        """
        # Set new data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The new Graph Neural Network graph data set '
                               'directory has not been found:\n\n'
                               + dataset_directory)
        else:
            self._dataset_directory = dataset_directory
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples files
        for i, file_path in enumerate(self._dataset_sample_files):
            # Set new sample file path (update directory)
            new_file_path = os.path.join(os.path.normpath(dataset_directory),
                                         os.path.basename(file_path))
            # Update sample file path
            if not os.path.isfile(new_file_path):
                raise RuntimeError('Graph Neural Network graph data set '
                                   'sample file is not in new dataset '
                                   'directory:\n\n' + dataset_directory)
            else:
                self._dataset_sample_files[i] = new_file_path
                # Reload sample data
                if self._is_store_dataset and is_reload_data:
                    self._dataset_samples[i] = torch.load(file_path)
    # -------------------------------------------------------------------------
    @staticmethod
    def update_dataset_file_internal_directory(
        dataset_file_path, new_directory, is_reload_data=False):
        """Update internal directory of stored data set in provided file.
        
        Update is only performed if the new directory does not match the
        internal directory of the stored data set.
        
        Parameters
        ----------
        dataset_file_path : str
            PyTorch data set file path.
        new_directory : str
            PyTorch data set new directory.
        is_reload_data : bool, default=False
            Reload and store data set samples in attribute
            dataset_samples_data. Only effective if is_store_dataset=True.
        """
        # Check new data set directory
        if not os.path.isdir(new_directory):
            raise RuntimeError('The new data set directory has not been '
                               'found:\n\n' + new_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load PyTorch data set
        loaded_dataset = GNNGraphDataset.load_dataset(dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set type
        if isinstance(loaded_dataset, torch.utils.data.Subset):
            # Get parent data set
            dataset = loaded_dataset.dataset
        elif isinstance(loaded_dataset, GNNGraphDataset):
            # Get data set
            dataset = loaded_dataset
        else:
            raise RuntimeError('The data set must be either '
                               'torch.utils.data.Subset (extracted from '
                               'GNNGraphDataset data set) or '
                               'GNNGraphDataset data set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get loaded PyTorch data set internal directory
        stored_directory = dataset.get_dataset_directory()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update PyTorch data set file if the new directory does not match the
        # internal directory of the loaded data set
        if stored_directory != new_directory:
            # Update directory of PyTorch data set
            dataset._update_dataset_directory(
                new_directory, is_reload_data=is_reload_data)
            # Save updated PyTorch data set (overwrite existing data set file)
            with open(dataset_file_path, 'wb') as dataset_file:
                pickle.dump(loaded_dataset, dataset_file)
# =============================================================================
def split_dataset(dataset, split_sizes, is_save_subsets=False,
                  subsets_directory=None, subsets_basename=None, seed=None):
    """Randomly split data set into non-overlapping subsets.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Data set.
    split_sizes : dict
        Size (item, float) of each data subset name (key, str), where size is a
        fraction contained between 0 and 1. The sum of all sizes must equal 1.
    is_save_subsets : bool, False
        If True, then save data subsets to files.
    subsets_directory : str, default=None
        Directory where the data subsets files are stored.
    subset_basename : str, default=None
        Subset file base name.
    seed : int, default=None
        Seed for random data set split generator.
    
    Returns
    -------
    dataset_split : dict
        Data subsets (key, str, item, torch.utils.data.Subset).
    """
    # Initialize data subsets names and sizes
    subsets_names = []
    subsets_sizes = []
    # Assemble data subsets names and sizes
    for key, val in split_sizes.items():
        # Check if subset size is valid
        if val < 0.0 or val > 1.0:
            raise RuntimeError(f'Subset size must be contained between 0 and '
                               f'1. Check subset (size): {key} ({val})')
        # Assemble subset name and size
        subsets_names.append(str(key))
        subsets_sizes.append(val)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if split sizes are valid
    if not np.isclose(np.sum(subsets_sizes), 1.0):
        raise RuntimeError('Sum of subset split sizes must equal 1. '
                           f'Current sum: {np.sum(subsets_sizes):.2f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random split generator
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Randomly split data set into non-overlapping subsets
    subsets_list = torch.utils.data.random_split(dataset, subsets_sizes,
                                                 generator=generator)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data subsets
    dataset_split = {}
    for i, subset in enumerate(subsets_names):
        # Check subset
        if len(subsets_list[i]) < 1:
            raise RuntimeError(f'Subset "{subset}" resulting from data set '
                               f'split is empty.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble data subet
        dataset_split[str(subset)] = subsets_list[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save data subsets files
    if is_save_subsets:
        # Check subsets directory
        if not os.path.isdir(subsets_directory):
            raise RuntimeError('The data subsets directory has not been '
                               'specified or found:\n\n'
                               + subsets_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over data subsets
        for key, val in dataset_split.items():
            # Set data subset file name
            subset_file = ''
            if subsets_basename is not None:
                subset_file += f'{subsets_basename}_'
            subset_file += f'{str(key)}_n{len(val)}'
            # Set data subset file path
            subset_path = os.path.join(subsets_directory,
                                       subset_file + '.pkl')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save data subset
            with open(subset_path, 'wb') as subset_file:
                pickle.dump(val, subset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    return dataset_split
# =============================================================================
def get_subset_indices_mapping(subset):
    """Get mapping from subset indexes to whole parent data set indices.
    
    Parameters
    ----------
    subset : torch.utils.data.Subset
        Subset of data set.
        
    Returns
    -------
    indices_map : dict
        Mapping between each subset index (key, str[int]) and the corresponding
        whole data set index (item, str[int]).
    """
    # Check subset
    if not isinstance(subset, torch.utils.data.Subset):
        raise RuntimeError(f'The provided subset ({type(subset)}) must be of '
                           f'type torch.utils.data.Subset.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Initialize mapping indices
    indices_map = {}
    # Loop over subset indices
    for i, index in enumerate(subset.indices):
        # Assign parent data set index
        indices_map[str(i)] = index
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    return indices_map
# =============================================================================
def get_pyg_data_loader(dataset, batch_size=1, is_shuffle=False,
                        is_verbose=False, **kwargs):
    """Get Graph Neural Network graph data set PyG data loader.
    
    Parameters
    ----------
    dataset : {GNNGraphDataset, list[str]}
        Graph Neural Network graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
        Accepts GNNGraphDataset or a list of data set samples files paths,
        where each sample file contains a torch_geometric.data.Data object
        describing a homogeneous graph.
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
        Graph Neural Network graph data set PyG data loader.
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
        dataset = GNNGraphDataset(
            dataset_directory, dataset_sample_files, is_store_dataset=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Building data loader...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graph Neural Network graph data set PyG data loader
    data_loader = torch_geometric.loader.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_loader
# =============================================================================
def write_graph_dataset_summary_file(
    dataset_directory, n_sample, total_time_sec, avg_time_sample,
    node_features=None, edge_features=None, global_features=None,
    node_targets=None, edge_targets=None, global_targets=None,
    filename='summary'):
    """Write summary data file for GNN-based data set generation.
    
    Parameters
    ----------
    dataset_directory : str
        Directory where the Graph Neural Network graph data set is stored (all
        data set samples files). All existent files are overridden when saving
        sample data files.
    n_sample : int
        Data set size (number of samples).
    total_time_sec : int
        Total generation time in seconds.
    avg_time_sample : float
        Average generation time per sample.
    node_features : tuple[str], default=None
        Graph Neural Network graph data set nodes features.
    edge_features : tuple[str], default=None
        Graph Neural Network graph data set edges features.
    global_features : tuple[str], default=None
        Graph Neural Network graph data set global features.
    node_targets : tuple[str], default=None
        Graph Neural Network graph data set nodes targets.
    edge_targets : tuple[str], default=None
        Graph Neural Network graph data set edges targets.
    global_targets : tuple[str], default=None
        Graph Neural Network graph data set global targets.
    filename : str, default='summary'
        Summary file name.
    """
    # Set summary data
    summary_data = {}
    summary_data['n_sample'] = n_sample
    summary_data['node_features'] = node_features
    summary_data['edge_features'] = edge_features
    summary_data['global_features'] = global_features
    summary_data['node_targets'] = node_targets
    summary_data['edge_targets'] = edge_targets
    summary_data['global_targets'] = global_targets
    summary_data['Total generation time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. generation time per graph'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # Set summary title
    summary_title = 'Summary: Graph Neural Network graph data set generation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=dataset_directory, filename=filename,
        summary_title=summary_title, **summary_data)
# =============================================================================
class GNNGraphDatasetInMemory(torch.utils.data.Dataset):
    """Graph Neural Network graph data set (in-memory storage only).
    
    Attributes
    ----------
    _dataset_samples : list
        Graph Neural Network graph data set samples data. Each sample is stored
        as a torch_geometric.data.Data object describing a homogeneous graph.
    
    Methods
    -------
    __len__(self):
        Return size of data set (number of samples).
    __getitem__(self, index)
        Return data set sample from corresponding index.
    """
    def __init__(self, dataset_samples):
        """Constructor.
        
        Parameters 
        ----------
        dataset_samples : list
            Graph Neural Network graph data set samples data. Each sample is
            stored as a torch_geometric.data.Data object describing a
            homogeneous graph.
        """
        # Initialize data set from base class
        super(GNNGraphDatasetInMemory, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set samples
        self._dataset_samples = list(dataset_samples)
    # -------------------------------------------------------------------------
    def __len__(self):
        """Return size of data set (number of samples).
        
        Returns
        -------
        n_sample : int
            Data set size (number of samples).
        """
        return len(self._dataset_samples)
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
        pyg_graph = self._dataset_samples[index].clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return pyg_graph