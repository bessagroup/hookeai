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
import shutil
# Local
from ioput.iostandard import make_directory
# =============================================================================
# Summary: Build convergence analyses directory for noisy data sets
# =============================================================================
# Set operations
is_build_noise_dirs = False
is_set_shared_testing_dataset = True
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set source directory
src_dir = ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
           '6_local_rnn_training_noisy/von_mises/datasets_base')
# Set destination directory
dest_dir = ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
            '6_local_rnn_training_noisy/von_mises/'
            'convergence_analyses_homoscedastic_gaussian')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize source directory data sets directories
src_dataset_dirs = []
dataset_dir_names = []
# Get files and directories in source directory
src_dir_list = os.listdir(src_dir)
# Loop over files and directories
for dir_name in src_dir_list:
    # Check if data set directory
    is_dataset_dir= bool(re.search(r'^n[0-9]+$', dir_name))
    # Store data set directory
    if is_dataset_dir:
        src_dataset_dirs.append(
            os.path.join(os.path.normpath(src_dir), dir_name))
        dataset_dir_names.append(dir_name)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize noise case directories names
noise_dir_names = []
# Get files and directories from first data set directory
dataset_dir_list = os.listdir(src_dataset_dirs[0])
# Loop over files and directories
for dir_name in dataset_dir_list:
    # Check if noise case directory
    is_dataset_dir= bool(re.search(r'noise', dir_name))
    # Store noise cases
    if is_dataset_dir:
        noise_dir_names.append(dir_name)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build noise cases directories
if is_build_noise_dirs:
    # Loop over noise cases
    for noise_case in noise_dir_names:
        # Set destination noise case directory
        dest_noise_case_dir = \
            os.path.join(os.path.normpath(dest_dir), noise_case)
        # Create destination noise case directory
        if not os.path.isdir(dest_noise_case_dir):
            make_directory(dest_noise_case_dir)
        # Loop over data sets
        for i, dataset_dir in enumerate(src_dataset_dirs):
            # Get source data set directory
            src_dataset_dir = \
                os.path.join(os.path.normpath(dataset_dir), noise_case)
            # Set destination data set directory
            dest_dataset_dir = \
                os.path.join(os.path.normpath(dest_noise_case_dir),
                            os.path.basename(dataset_dir_names[i]))
            # Remove existing destination data set directory
            if os.path.isdir(dest_dataset_dir):
                shutil.rmtree(dest_dataset_dir)
            # Copy source noise case directory to destination directory
            shutil.copytree(src_dataset_dir, dest_dataset_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set shared testing data set
if is_set_shared_testing_dataset:
    # Set source testing data set directory
    src_testing_dataset_dir = \
        ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
         '6_local_rnn_training_noisy/von_mises/datasets_base/'
         'noiseless_testing_dataset/5_testing_id_dataset')
    # Set testing data set type
    dataset_type = ('testing_id', 'testing_od')[0]
    # Set testing data set type base name
    if dataset_type == 'testing_id':
        dataset_type_basename = '5_testing_id_dataset'
    elif dataset_type == 'testing_od':
        dataset_type_basename = '6_testing_od_dataset'
    # Loop over noise cases
    for noise_case in noise_dir_names:
        # Loop over data sets
        for dataset_name in dataset_dir_names:
            # Set destination testing data set directory
            dest_testing_dataset_dir = os.path.join(os.path.normpath(dest_dir),
                                                    noise_case, dataset_name,
                                                    dataset_type_basename)
            # Remove existing destination data set directory
            if os.path.isdir(dest_testing_dataset_dir):
                shutil.rmtree(dest_testing_dataset_dir)
            # Copy source testing data set directory to destination directory
            shutil.copytree(src_testing_dataset_dir, dest_testing_dataset_dir)