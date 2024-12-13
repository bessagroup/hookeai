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
import re
# Local
from rnn_base_model.data.time_dataset import load_dataset, split_dataset
# =============================================================================
# Summary: Split time series data set into non-overlapping subsets
# =============================================================================
# Set data set file path
dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                     'darpa_project/8_global_random_specimen/von_mises/'
                     '2_random_specimen_hexa8/solid/'
                     '1_local_vanilla_GRU_specimen_dataset/strain_to_stress/'
                     '1_training_dataset/ss_paths_dataset_n2664.pkl')
# Set data set split sizes
split_sizes = {'training': 0.7, 'validation': 0.2, 'testing_id': 0.1}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get data set file basename
dataset_basename = \
    re.sub(r'_n[0-9]+$', '', os.path.splitext(dataset_file_path)[0])
# Set data set file directory
dataset_directory = os.path.dirname(dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data set
dataset = load_dataset(dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Split data set into non-overlapping subsets
dataset_split = split_dataset(dataset, split_sizes, is_save_subsets=True,
                              subsets_directory=dataset_directory,
                              subsets_basename=dataset_basename)