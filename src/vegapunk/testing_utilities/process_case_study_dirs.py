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
# Summary: Process case study directories and perform some task
# =============================================================================
# Set available tasks
available_tasks = {'1': 'remove_plots_folders'}
# Set task
task = available_tasks['1']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set source directory
src_dir = ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
           '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
           '0_datasets/datasets_drucker_prager_10deg/proportional/no_plots')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize case study directories
case_study_dirs = []
# Set case study directories structure option
dirs_structure_type = 1
# Set case study directories
if dirs_structure_type == 1:
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # Set case study directories
    case_study_dirs += [os.path.join(os.path.normpath(src_dir), f'n{n}/')
                        for n in training_sizes]
else:
    case_study_dirs += [src_dir,]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\nStarting task automatic procedures...')
print(f'\n  > Task: \'{task}\'')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over case study directories
for case_study_dir in case_study_dirs:
    # Get files and directories in case study directory
    l1_names = os.listdir(case_study_dir)
    l1_paths = [os.path.join(os.path.normpath(case_study_dir), x)
                for x in l1_names]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform task
    if task == 'remove_plots_folders':
        # Set data sets types
        dataset_types = ('1_training_dataset', '2_validation_dataset',
                         '5_testing_id_dataset', '6_testing_od_dataset')
        # Loop over case study directories
        for i, l1_name in enumerate(l1_names):
            # Check if data set folder
            if l1_name not in dataset_types:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set type plots directory
            plots_dir = os.path.join(os.path.normpath(l1_paths[i]), 'plots')
            # Remove existing plots directory
            if os.path.isdir(plots_dir):
                shutil.rmtree(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown task.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\nFinished task successfuly!\n')