# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pickle
# Third-party
import torch
# Local
from time_series_data.time_dataset import save_dataset, load_dataset
# =============================================================================
# Summary: Convert torch.utils.data.Dataset file to given sharable format
# =============================================================================
class TorchDatasetInMemory(torch.utils.data.Dataset):
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
        super(TorchDatasetInMemory, self).__init__()
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
# Set share format
shared_format = ('list', 'TorchDatasetInMemory')[0]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set torch.utils.data.Dataset data set file path
src_dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                         'darpa_project/2_local_rnn_training/composite_rve/'
                         'dataset_07_2024/deliverable_shelly_09_2024/'
                         '5_testing_id_dataset/ss_paths_dataset_n549.pkl')
# Set torch.utils.data.Dataset directory
dataset_directory = os.path.dirname(src_dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize time series data set samples
dataset_sample_files = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
# Load data set
src_dataset = load_dataset(src_dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check data set
if not isinstance(src_dataset, torch.utils.data.Dataset):
    raise RuntimeError('Data set file path does not contain a '
                       'torch.utils.data.Dataset data set.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize data set samples
dataset_samples = []
# Set number of samples
n_sample = len(src_dataset)
# Loop over samples
for i in range(n_sample):
    # Collect sample
    dataset_samples.append(src_dataset[i])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store data set in share format
if shared_format == 'list':
    # Set data set file basename
    dataset_basename = 'ss_paths_list' + f'_n{n_sample}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file path
    dataset_file_path = os.path.join(
        os.path.normpath(dataset_directory), dataset_basename + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save data set
    with open(dataset_file_path, 'wb') as dataset_file:
        pickle.dump(dataset_samples, dataset_file)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
elif shared_format == 'TorchDatasetInMemory':
    # Set data set file basename
    dataset_basename = 'ss_paths_im_torch_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create TorchDatasetInMemory data set
    dataset = TorchDatasetInMemory(dataset_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save TorchDatasetInMemory data set
    save_dataset(dataset, dataset_basename, dataset_directory,
                is_append_n_sample=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
else:
    raise RuntimeError('Unknown data set share format.')