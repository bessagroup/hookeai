"""Process case study directories."""
#
#                                                                       Modules
# =============================================================================
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
import pickle
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    # Set available tasks
    available_tasks = {'1': 'remove_plots_dirs',
                       '2': 'remove_sample_prediction_files',
                       '3': 'remove_model_dirs',
                       '4': 'redump_sample_prediction_files'}
    # Set task
    task = available_tasks['4']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set source directory
    src_dir = ('/home/username/Documents/brown/projects/'
               'darpa_paper_examples/local/ml_models/polynomial/'
               'convergence_analysis_noise/'
               'convergence_analyses_homoscedastic_gaussian/noiseless')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # Set uncertainty quantification flag
    is_uncertainty_quantification = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nStarting task automatic procedures...')
    print(f'\n  > Task: \'{task}\'')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over case study directories
    for case_study_dir in case_study_dirs:
        # Get files and directories in case study directory
        l1_names = os.listdir(case_study_dir)
        l1_paths = [os.path.join(os.path.normpath(case_study_dir), x)
                    for x in l1_names]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Task: Remove plot directory from data set directories
        if task == 'remove_plots_dirs':
            # Set data sets types
            dataset_types = ('1_training_dataset', '2_validation_dataset',
                             '5_testing_id_dataset', '6_testing_od_dataset')
            # Loop over case study directories
            for i, l1_name in enumerate(l1_names):
                # Check if data set directory
                if l1_name not in dataset_types:
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set data set type plots directory
                plots_dir = \
                    os.path.join(os.path.normpath(l1_paths[i]), 'plots')
                # Remove existing plots directory
                if os.path.isdir(plots_dir):
                    shutil.rmtree(plots_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Task: Remove sample prediction files
        elif task == 'remove_sample_prediction_files':
            # Set sample prediction file regex
            sample_regex = re.compile(r'^prediction_sample_\d+\.pkl$')
            # Set prediction folder name
            if is_uncertainty_quantification:
                prediction_dirname = 'uncertainty_quantification'
            else:
                prediction_dirname = '7_prediction'
            # Set prediction directory
            prediction_dir = os.path.join(os.path.normpath(case_study_dir),
                                          prediction_dirname)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check prediction directory
            if prediction_dir in l1_paths:
                # Walk through prediction directory recursively
                for root, dirs, files in os.walk(prediction_dir):
                    # Loop over directory files
                    for file in files:
                        # Remove sample prediction file
                        if sample_regex.match(file):
                            # Set sample prediction file path
                            sample_file_path = os.path.join(root, file)
                            # Remove sample prediction file
                            os.remove(sample_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Task: Remove specific directories from model directory
        elif task == 'remove_model_dirs':
            # Set directories to be removed
            remove_dirnames = ('3_model', '7_prediction')
            # Loop over case study directories
            for i, l1_name in enumerate(l1_names):
                # Remove model or prediction directory
                if l1_name in remove_dirnames:
                    shutil.rmtree(l1_paths[i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Task: Redump bloated sample prediction files
        elif task == 'redump_sample_prediction_files':
            # Set sample prediction file regex
            sample_regex = re.compile(r'^prediction_sample_\d+\.pkl$')
            # Set prediction folder name
            if is_uncertainty_quantification:
                prediction_dirname = 'uncertainty_quantification'
            else:
                prediction_dirname = '7_prediction'
            # Set prediction directory
            prediction_dir = os.path.join(os.path.normpath(case_study_dir),
                                          prediction_dirname)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check prediction directory
            if prediction_dir in l1_paths:
                # Walk through prediction directory recursively
                for root, dirs, files in os.walk(prediction_dir):
                    # Loop over directory files
                    for file in files:
                        # Remove sample prediction file
                        if sample_regex.match(file):
                            # Set sample prediction file path
                            sample_file_path = os.path.join(root, file)
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Open sample prediction file
                            with open(sample_file_path, 'rb') as sample_file:
                                # Load sample prediction data
                                sample_results = pickle.load(sample_file)
                            # Clean metadata from torch Tensors
                            sample_results = {k: v.detach().clone().cpu()
                                              if isinstance(v, torch.Tensor)
                                              else v for k, v in
                                              sample_results.items()}
                            # Redump sample prediction data
                            with open(sample_file_path, 'wb') as sample_file:
                                pickle.dump(sample_results, sample_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown task.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nFinished task successfuly!\n')