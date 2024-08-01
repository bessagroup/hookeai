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
import re
# Local
from rnn_base_model.data.time_dataset import save_dataset
from gnn_base_model.data.graph_dataset import split_dataset
from projects.darpa_metals.rnn_material_model.user_scripts. \
    gen_response_dataset import MaterialResponseDatasetGenerator
# =============================================================================
# Summary: Generate strain-stress response path data set from .csv files.
# =============================================================================
# Set response paths data directory
response_path_files_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                           'darpa_project/2_local_rnn_training/composite_rve/'
                           'dataset_07_2024/1_rve_simulations')
# Set response file basename
response_file_basename = 'composite_rve_path'
# Set data set file basename
dataset_basename = 'ss_paths_dataset'
# Set data set directory
dataset_directory = response_path_files_dir
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set strain formulation
strain_formulation = 'infinitesimal'
# Set problem type
problem_type = 4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check response paths data directory
if not os.path.isdir(response_path_files_dir):
    raise RuntimeError('The response paths data directory has not been '
                       'found:\n\n' + response_path_files_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize response paths data file paths (.csv files)
response_file_paths = []
# Get files in response paths data directory
directory_list = os.listdir(response_path_files_dir)
# Loop over files
for filename in directory_list:
    # Check if file is response path file
    is_response_file = bool(
        re.search(r'^' + response_file_basename + r'_[0-9]+' + r'\.csv',
                  filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append response path file
    if is_response_file:
        response_file_paths.append(
            os.path.join(os.path.normpath(response_path_files_dir), filename))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sort response path files
response_file_paths = tuple(
    sorted(response_file_paths,
           key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize strain-stress material response path data set generator
dataset_generator = \
    MaterialResponseDatasetGenerator(strain_formulation, problem_type)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate
dataset = dataset_generator.gen_response_dataset_from_csv(
    response_file_paths, save_dir=response_path_files_dir, is_save_fig=False,
    is_verbose=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save data set
save_dataset(dataset, dataset_basename, dataset_directory,
             is_append_n_sample=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set split data set flag
is_split_dataset = True
# Split data set
if is_split_dataset:
    # Set data set split sizes
    split_sizes = {'training': 0.8, 'validation': 0.1, 'testing': 0.1}
    # Split data set
    dataset_split = \
        split_dataset(dataset, split_sizes, is_save_subsets=True,
                      subsets_basename=dataset_basename,
                      subsets_directory=dataset_directory)