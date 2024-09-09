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
# Third-party
import torch
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory, \
    TimeSeriesDataset, save_dataset, load_dataset
# =============================================================================
# Summary: Convert TimeSeriesDatasetInMemory to sharable TorchDatasetInMemory
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
# Set TimeSeriesDatasetInMemory data set file path
im_dataset_file_path = ('/home/bernardoferreira/Desktop/shunyu_dataset/'
                        'ss_paths_dataset_n2560.pkl')
# Set sample file base name
sample_basename = 'ss_response_path'
# Set TimeSeriesDataset directory
dataset_directory = os.path.dirname(im_dataset_file_path)
# Set TimeSeriesDataset data set file basename
dataset_basename = 'ss_paths_torch_dataset'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize time series data set samples
dataset_sample_files = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
# Load TimeSeriesDatasetInMemory data set
im_dataset = load_dataset(im_dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check TimeSeriesDatasetInMemory data set
if not isinstance(im_dataset, TimeSeriesDatasetInMemory):
    raise RuntimeError('Data set file path does not contain a '
                       'TimeSeriesDatasetInMemory data set.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize data set samples
dataset_samples = []
# Loop over samples
for i in range(len(im_dataset)):
    # Collect sample
    dataset_samples.append(im_dataset[i])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create TorchDatasetInMemory data set
dataset = TorchDatasetInMemory(dataset_samples)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save TorchDatasetInMemory data set
save_dataset(dataset, dataset_basename, dataset_directory,
             is_append_n_sample=True)