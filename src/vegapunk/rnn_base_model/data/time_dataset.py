"""Time series data set.

Classes
-------
TimeSeriesDatasetInMemory(torch.utils.data.Dataset)
    Time series data set (in-memory storage only).

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
write_time_series_dataset_summary_file
    Write summary data file for time series data set generation.
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
    summary_data['Avg. generation time per graph'] = \
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
    dataset_samples : list
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
        sample_data = copy.deepcopy(self._dataset_samples[index])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return sample_data