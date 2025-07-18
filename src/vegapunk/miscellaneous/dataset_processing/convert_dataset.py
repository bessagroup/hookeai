"""Convert time series data set file to given format."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pickle
import re
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
# Set conversion format
conversion_format = ('list', 'TimeSeriesDatasetInMemory')[1]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize source data set file paths
src_dataset_file_paths = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set source data sets base directories
src_base_dirs = ('/home/bernardoferreira/Documents/brown/projects/colaboration_shunyu/deliverable_07_17_2025/Main_design/n10240',)
# Loop over base directories
for src_base_dir in src_base_dirs:
    # Set data set file regex
    sample_regex = re.compile(r'^ss_paths_dataset_n\d+\.pkl$')
    # Walk through directory recursively
    for root, dirs, files in os.walk(src_base_dir):
        # Loop over directory files
        for file in files:
            # Store data set file path
            if sample_regex.match(file):
                src_dataset_file_paths.append(os.path.join(root, file))
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