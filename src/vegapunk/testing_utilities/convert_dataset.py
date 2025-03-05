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
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
# =============================================================================
# Summary: Convert dataset file to given format
# =============================================================================
# Set conversion format
conversion_format = ('list', 'TimeSeriesDatasetInMemory')[1]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set source data set file paths
src_dataset_file_paths = (
    '/home/bernardoferreira/Documents/brown/projects/colaboration_shunyu/'
    'deliverable_04_02_2025/deliverable_bernardo/'
    '8_gru_train_shunyu_test_shunyu/5_testing_id_dataset/'
    'ss_paths_dataset_n260.pkl',)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over source data set file paths
for src_dataset_file_path in src_dataset_file_paths:
    # Set data set directory
    dataset_directory = os.path.dirname(src_dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize time series data set samples
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
    # Load data set
    with open(src_dataset_file_path, 'rb') as dataset_file:
        src_dataset = pickle.load(dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data set samples
    dataset_samples = []
    # Set number of samples
    n_sample = len(src_dataset)
    # Loop over samples
    for i in range(n_sample):
        # Collect sample
        dataset_samples.append(src_dataset[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store data set in share format
    if conversion_format == 'list':
        # Set data set file basename
        dataset_basename = 'ss_paths_list' + f'_n{n_sample}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set file path
        dataset_file_path = os.path.join(
            os.path.normpath(dataset_directory), dataset_basename + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save data set
        with open(dataset_file_path, 'wb') as dataset_file:
            pickle.dump(dataset_samples, dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif conversion_format == 'TimeSeriesDatasetInMemory':
        # Set data set file basename
        dataset_basename = 'ss_paths_dataset'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create TimeSeriesDatasetInMemory data set
        dataset = TimeSeriesDatasetInMemory(dataset_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save TimeSeriesDatasetInMemory data set
        save_dataset(dataset, dataset_basename, dataset_directory,
                     is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown data set share format.')