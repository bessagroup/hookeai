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
# Summary: Convert TimeSeriesDatasetInMemory to TimeSeriesDataset
# =============================================================================
# Set TimeSeriesDatasetInMemory data set file path
im_dataset_file_path = ('/home/bernardoferreira/Desktop/test_dataset/n10/'
                        '1_training_dataset/ss_paths_dataset_n10.pkl')
# Set sample file base name
sample_basename = 'ss_response_path'
# Set TimeSeriesDataset directory
dataset_directory = os.path.dirname(im_dataset_file_path)
# Set TimeSeriesDataset data set file basename
dataset_basename = 'ss_paths_dataset'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check TimeSeriesDatasetInMemory data set
if not isinstance(im_dataset_file_path, TimeSeriesDatasetInMemory):
    raise RuntimeError('Data set file path does not contain a '
                       'TimeSeriesDatasetInMemory data set.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize time series data set samples
dataset_sample_files = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
# Load TimeSeriesDatasetInMemory data set
im_dataset = load_dataset(im_dataset_file_path)
# Loop over samples
for i in range(len(im_dataset)):
    # Get sample
    sample = im_dataset[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set sample file path
    sample_file_path = os.path.join(dataset_directory,
                                    f'{sample_basename}_{i}.pt')
    # Store sample (local directory)
    torch.save(sample, sample_file_path)
    # Append material response path file path
    dataset_sample_files.append(sample_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create TimeSeriesDataset data set
dataset = TimeSeriesDataset(dataset_directory,
                            dataset_sample_files,
                            dataset_basename=dataset_basename)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save TimeSeriesDataset data set
save_dataset(dataset, dataset_basename, dataset_directory,
             is_append_n_sample=True)