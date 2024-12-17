"""Time series data set.

Classes
-------
TimeSeriesDatasetInMemory(torch.utils.data.Dataset)
    Time series data set (in-memory storage only).
TimeSeriesDataset(torch.utils.data.Dataset)
    Time series data set.

Functions
---------
time_collator
    Collate time series batch data.
get_time_series_data_loader
    Get time series data set data loader.
save_dataset
    Save PyTorch time series data set to file.
load_dataset
    Load PyTorch time series data set.
change_dataset_features_labels
    Change time series data set features labels.
add_dataset_feature_init
    Add feature initialization to all samples of time series data set.
concatenate_dataset_features
    Concatenate existing features of time series data set into new feature.
sum_dataset_features(dataset, new_feature_label, sum_features_labels,
                     features_weights=None, is_remove_features=False)
    Sum existing features of time series data set into new feature.
write_time_series_dataset_summary_file
    Write summary data file for time series data set generation.
split_dataset(dataset, split_sizes, is_save_subsets=False, \
              subsets_directory=None, subsets_basename=None, seed=None)
    Randomly split data set into non-overlapping subsets.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import datetime
import copy
import pickle
# Third-party
import torch
import numpy as np
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
def time_collator(batch_data):
    """Collate time series batch data.
    
    Ignores all features that are not stored as a torch.Tensor(2d) of shape
    (sequence_length, n_features).
    
    Parameters
    ----------
    batch_data : list[dict]
        Each batch sample data is stored as a dictionary where each feature
        (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
        
    Returns
    -------
    batch : dict
        Collated batch sample data for each feature (key, str), stored as
        torch.Tensor(3d) of shape (sequence_length, batch_size, n_features).
    """
    # Probe features from first batch sample
    features = tuple(batch_data[0].keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize collated batch sample data
    batch = {}
    # Loop over features
    for feature in features:
        # Probe feature type and shape from first batch sample
        is_batchable = (isinstance(batch_data[0][feature], torch.Tensor)
                        and len(batch_data[0][feature].shape) == 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collate batch sample feature data
        if is_batchable:
            batch[feature] = torch.stack([x[feature] for x in batch_data], 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return batch
# =============================================================================
def get_time_series_data_loader(dataset, batch_size=1, is_shuffle=False,
                                **kwargs):
    """Get time series data set data loader.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    batch_size : int, default=1
        Number of samples loaded per batch.
    is_shuffle : bool, default=False
        Reshuffle data set at every epoch.
    **kwargs
        Arguments of torch.utils.data.DataLoader.

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        Time series data set data loader.
    """
    # Time series data set data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle,
                                              collate_fn=time_collator,
                                              **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_loader
# =============================================================================
def save_dataset(dataset, dataset_basename, dataset_directory,
                 is_append_n_sample=True):
    """Save PyTorch time series data set to file.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    dataset_basename : str
        Data set file basename.
    dataset_directory : str
        Directory where the times series data set is stored.
    is_append_n_sample : bool, default=True
        If True, then data set size (number of samples) is appended to
        data set filename.
        
    Returns
    -------
    dataset_file_path : str
        PyTorch data set file path.
    """
    # Check data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The data set directory has not been found:\n\n'
                           + dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file
    dataset_file = dataset_basename
    # Append data set size
    if is_append_n_sample:
        dataset_file += f'_n{len(dataset)}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file path
    dataset_file_path = os.path.join(dataset_directory,
                                     dataset_file + '.pkl')   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save data set
    with open(dataset_file_path, 'wb') as dataset_file:
        pickle.dump(dataset, dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_file_path
# =============================================================================
def load_dataset(dataset_file_path):
    """Load PyTorch time series data set.
    
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load PyTorch data set
    with open(dataset_file_path, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check PyTorch data set
    if not isinstance(dataset, torch.utils.data.Dataset):
        raise RuntimeError('Loaded data set is not a '
                           'torch.utils.data.Dataset.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def change_dataset_features_labels(dataset, features_label_map):
    """Change time series data set features labels.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_label_map : dict
        Features labels mapping. For each original feature label
        (key, str), specifies the new label (item, str). Nonexistent
        features are silently ignored.
        
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Loop over features
    for old_label, new_label in features_label_map.items():        
        # Probe feature existence
        if old_label not in tuple(dataset[0].keys()):
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples
        for i in range(len(dataset)):
            # Get sample
            sample = dataset[i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Change feature label
            sample[new_label] = sample.pop(old_label)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update data set sample
            if isinstance(dataset, TimeSeriesDataset):
                dataset.update_dataset_sample(i, sample)
            else:
                dataset[i] = sample      
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def add_dataset_feature_init(dataset, feature_label, feature_init):
    """Add feature initialization to all samples of time series data set.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    feature_label : str
        Feature label.
    feature_init : torch.Tensor(2d)
        Feature initialization data stored as torch.Tensor(2d) of shape
        (n, n_features), where n is a general size.
    
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Check feature initialization
    if not isinstance(feature_init, torch.Tensor):
        raise RuntimeError('Feature initialization was not provided as '
                           'torch.Tensor(2d).')
    elif len(feature_init.shape) != 2:
        raise RuntimeError('Feature initialization was not provided as '
                           'torch.Tensor(2d).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add new feature
        sample[str(feature_label)] = feature_init
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data set sample
        if isinstance(dataset, TimeSeriesDataset):
            dataset.update_dataset_sample(i, sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def concatenate_dataset_features(dataset, new_feature_label,
                                 cat_features_labels,
                                 is_remove_features=False):
    """Concatenate existing features of time series data set into new feature.
    
    The new feature is stored as a torch.Tensor(2d) of shape
    (sequence_length, n_concat_features) resulting from the concatenation of
    the existing features tensors along the second dimension.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    new_feature_label : str
        New feature label.
    cat_features_labels : tuple[str]
        Labels of existing features to be concatenated into new feature.
        Concatenation is sorted accordingly.
    is_remove_features : bool, default=False
        If True, then remove concatenated features from data set after
        concatenating the new feature.
    
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Check concatenation features
    for label in cat_features_labels:
        # Probe sample existence from first sample
        if label not in dataset[0].keys():
            raise RuntimeError(f'The feature "{label}" cannot be concatenated '
                               f'because it does not exist in the data set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set storage type
    if isinstance(dataset, TimeSeriesDataset):
        is_in_memory_dataset = False
    else:
        is_in_memory_dataset = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate features
        if len(cat_features_labels) > 1:
            sample[str(new_feature_label)] = torch.cat(
                [sample[label] for label in cat_features_labels], dim=1)
        else:
            sample[str(new_feature_label)] = \
                sample[cat_features_labels[0]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove concatenated features (in-memory data set only)
        if is_remove_features and is_in_memory_dataset:
            # Loop over concatenated features
            for label in cat_features_labels:
                sample.pop(label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data set sample
        if isinstance(dataset, TimeSeriesDataset):
            dataset.update_dataset_sample(i, sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def sum_dataset_features(dataset, new_feature_label, sum_features_labels,
                         features_weights=None, is_remove_features=False):
    """Sum existing features of time series data set into new feature.
    
    The new feature is stored as a torch.Tensor(2d) of shape
    (sequence_length, n_features) resulting from the sum of the existing
    features tensors along the second dimension.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    new_feature_label : str
        New feature label.
    sum_features_labels : tuple[str]
        Labels of existing features to be summed into new feature.
    features_weights : dict, default=None
        Scalar weights (item, float) multiplied by each existing feature
        (key, str) in the summing process. If None, then defaults to 1.0 for
        all features.
    is_remove_features : bool, default=False
        If True, then remove summed features from data set after computing the
        new feature.
    
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Check summing features
    for label in sum_features_labels:
        # Probe sample existence from first sample
        if label not in dataset[0].keys():
            raise RuntimeError(f'The feature "{label}" cannot be summed '
                               f'because it does not exist in the data set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize sum weights
    sum_weights = {}
    # Set sum weights
    for feature_label in sum_features_labels:
        if feature_label in features_weights.keys():
            sum_weights[feature_label] = float(features_weights[feature_label])
        else:
            sum_weights[feature_label] = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set storage type
    if isinstance(dataset, TimeSeriesDataset):
        is_in_memory_dataset = False
    else:
        is_in_memory_dataset = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add features
        if len(sum_features_labels) > 1:
            # Compute new feature
            sample[str(new_feature_label)] = \
                torch.sum(torch.stack([sum_weights[label]*sample[label]
                                       for label in sum_features_labels],
                                      dim=0), dim=0)
        else:
            # Get feature label
            label = sum_features_labels[0]
            # Set new feature
            sample[str(new_feature_label)] = sum_weights[label]*sample[label]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove summed features (in-memory data set only)
        if is_remove_features and is_in_memory_dataset:
            # Loop over summed features
            for label in sum_features_labels:
                sample.pop(label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data set sample
        if isinstance(dataset, TimeSeriesDataset):
            dataset.update_dataset_sample(i, sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def write_time_series_dataset_summary_file(
    dataset_directory, n_sample, total_time_sec, avg_time_sample,
    features=None, targets=None, filename='summary'):
    """Write summary data file for time series data set generation.
    
    Parameters
    ----------
    dataset_directory : str
        Directory where the times series data set is stored.
    n_sample : int
        Data set size (number of samples).
    total_time_sec : int
        Total generation time in seconds.
    avg_time_sample : float
        Average generation time per sample.
    features : tuple[str], default=None
        Time series data set features.
    targets : tuple[str], default=None
        Time series data set targets.
    filename : str, default='summary'
        Summary file name.
    """
    # Set summary data
    summary_data = {}
    summary_data['n_sample'] = n_sample
    summary_data['features'] = features
    summary_data['targets'] = targets
    summary_data['Total generation time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. generation time per path'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # Set summary title
    summary_title = 'Summary: Time series data set generation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=dataset_directory, filename=filename,
        summary_title=summary_title, **summary_data)
# =============================================================================
class TimeSeriesDatasetInMemory(torch.utils.data.Dataset):
    """Time series data set (in-memory storage only).
    
    Attributes
    ----------
    _dataset_samples : list
        Time series data set samples data. Each sample is stored as a
        dictionary where each feature (key, str) data is a torch.Tensor(2d) of
        shape (sequence_length, n_features).
    
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
        dataset_samples : list[dict]
            Time series data set samples data. Each sample is stored as a
            dictionary where each feature (key, str) data is a torch.Tensor(2d)
            of shape (sequence_length, n_features).
        """
        # Initialize data set from base class
        super(TimeSeriesDatasetInMemory, self).__init__()
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
        sample_data : dict
            Data set sample stored as a dictionary where each feature
            (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        # Get data set sample
        sample_data = self._dataset_samples[index]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return sample_data
# =============================================================================
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Time series data set.
    
    Attributes
    ----------
    _dataset_directory : str
        Directory where the time series data set is stored (all data set
        samples files).
    _dataset_sample_files : list[str]
        Time series data set samples files paths. Each sample file contains a
        dictionary where each feature (key, str) data is a torch.Tensor(2d) of
        shape (sequence_length, n_features).
    _dataset_basename : str
        Data set file base name.
    
    Methods
    -------
    __len__(self)
        Return size of data set (number of samples).
    __getitem__(self, index)
        Return data set sample from corresponding index.
    update_dataset_sample(self, index, time_series)
        Update data set sample time series data.
    get_dataset_directory(self)
        Get directory where time series data set is stored.
    get_dataset_sample_files(self)
        Get time series data set samples files paths.
    set_dataset_basename(self, dataset_basename)
        Set data set file base name.
    get_dataset_basename(self)
        Get data set file base name.
    update_dataset_file_internal_directory(dataset_file_path, \
                                           new_directory, \
                                           is_reload_data=False)
        Update internal directory of stored data set in provided file.
    _update_dataset_directory(self, dataset_directory, is_reload_data=False)
        Update directory where time series data set is stored.
    """
    def __init__(self, dataset_directory, dataset_sample_files,
                 dataset_basename='time_series_dataset'):
        """Constructor.
        
        Parameters
        ----------
        dataset_directory : str
            Directory where the time series data set is stored (all data set
            samples files).
        dataset_sample_files : list[str]
            Time series data set samples files paths. Each sample file contains
            a dictionary where each feature (key, str) data is a
            torch.Tensor(2d) of shape (sequence_length, n_features).
        dataset_basename : str, default='time_series_dataset'
            Data set file base name.
        dataset_samples : list[dict]
            Time series data set samples data. Each sample is stored as a
            dictionary where each feature (key, str) data is a torch.Tensor(2d)
            of shape (sequence_length, n_features).
        """
        # Initialize data set from base class
        super(TimeSeriesDataset, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The time series data set directory has not '
                               'been found:\n\n' + dataset_directory)
        else:
            self._dataset_directory = os.path.normpath(dataset_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set sample files
        for file_path in dataset_sample_files:
            # Check if sample file exists
            if not os.path.isfile(file_path):
                raise RuntimeError('Time series data set sample file has not '
                                   'been found:\n\n' + file_path)
            elif os.path.dirname(file_path) \
                    != os.path.normpath(dataset_directory):
                raise RuntimeError('Time series data set sample file is not '
                                   'in dataset directory:\n\n'
                                   + dataset_directory)
        # Store data set samples file paths
        self._dataset_sample_files = dataset_sample_files
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
        time_series : dict
            Data set sample defined as a dictionary where each feature
            (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        # Get data set sample
        time_series = torch.load(self._dataset_sample_files[index])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return time_series
    # -------------------------------------------------------------------------
    def update_dataset_sample(self, index, time_series):
        """Update data set sample time series data.
        
        Parameters
        ----------
        index : int
            Index of returned data set sample (index must be in [0, n_sample]).
        time_series : dict
            Data set sample defined as a dictionary where each feature
            (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        # Update data set sample time series data
        torch.save(time_series, self._dataset_sample_files[index])
    # -------------------------------------------------------------------------
    def get_dataset_directory(self):
        """Get directory where time series data set is stored.
        
        Returns
        -------
        dataset_directory : str
            Directory where the time series data set is stored (all data set
            samples files).
        """
        return self._dataset_directory
    # -------------------------------------------------------------------------
    def get_dataset_sample_files(self):
        """Get time series data set samples files paths.
        
        Returns
        -------
        dataset_sample_files : list[str]
            Time series data set samples files paths. Each sample file contains
            a dictionary where each feature (key, str) data is a
            torch.Tensor(2d) of shape (sequence_length, n_features).
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
    @staticmethod
    def update_dataset_file_internal_directory(
        dataset_file_path, new_directory, is_reload_data=False):
        """Update internal directory of stored data set in provided file.
        
        Update is only performed if the new directory does not match the
        internal directory of the stored data set.
        
        Parameters
        ----------
        dataset_file_path : str
            Data set file path.
        new_directory : str
            Data set new directory.
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
        loaded_dataset = load_dataset(dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set type
        if isinstance(loaded_dataset, torch.utils.data.Subset):
            # Get parent data set
            dataset = loaded_dataset.dataset
        elif isinstance(loaded_dataset, TimeSeriesDataset):
            # Get data set
            dataset = loaded_dataset
        else:
            raise RuntimeError('The data set must be either '
                               'torch.utils.data.Subset (extracted from '
                               'TimeSeriesDataset data set) or '
                               'TimeSeriesDataset data set.')
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
    # -------------------------------------------------------------------------
    def _update_dataset_directory(self, dataset_directory):
        """Update directory where time series data set is stored.
        
        Stored data set samples files paths directory is updated according
        with the new directory.
        
        Parameters
        ----------
        dataset_directory : str
            Directory where the time series data set is stored (all data set
            samples files).
        """
        # Set new data set directory
        if not os.path.isdir(dataset_directory):
            raise RuntimeError('The new time series data set directory has '
                               'not been found:\n\n' + dataset_directory)
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
                raise RuntimeError('Time series data set sample file is not '
                                   'in new dataset directory:\n\n'
                                   + dataset_directory)
            else:
                self._dataset_sample_files[i] = new_file_path
# =============================================================================
def split_dataset(dataset, split_sizes, is_save_subsets=False,
                  subsets_directory=None, subsets_basename=None, seed=None):
    """Randomly split data set into non-overlapping subsets.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
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