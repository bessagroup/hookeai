"""DARPA METALS PROJECT: Uncertainty quantification of RNN material model.

Functions
---------
perform_model_uq
    Perform model uncertainty quantification.
gen_model_uq_plots
    Generate plots of model uncertainty quantification.
plot_prediction_loss_uq
    Plot model average prediction loss uncertainty quantification.
plot_time_series_uq
    Plot time series model uncertainty quantification.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[5])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import shutil
import re
# Third-party
import torch
# Local
from projects.darpa_metals.rnn_material_model.user_scripts \
    .uncertainty_quantification import perform_model_uq, gen_model_uq_plots
from ioput.iostandard import make_directory, find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    # Set number of model samples for uncertainty quantification
    n_model_sample = 5
    # Set computation processes
    is_model_training = False
    # Set testing type
    testing_type = ('training', 'validation', 'in_distribution',
                    'out_distribution')[3]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set noise cases
    noise_cases = ('noiseless', 'homgau_noise_1e-2', 'homgau_noise_2d5e-2',
                   'homgau_noise_5e-2', 'homgau_noise_1e-1')
    # Set training data set sizes
    training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over noise cases
    for noise_case in noise_cases:
        # Loop over training data set sizes
        for n_path in training_sizes:
            # Set case studies base directory
            base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                        'darpa_project/6_local_rnn_training_noisy/von_mises/'
                        'convergence_analyses_homoscedastic_gaussian/'
                        f'{noise_case}')
            # Set case study directory
            case_study_name = f'n{n_path}'
            case_study_dir = os.path.join(os.path.normpath(base_dir),
                                        f'{case_study_name}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check case study directory
            if not os.path.isdir(case_study_dir):
                raise RuntimeError('The case study directory has not been '
                                   'found:\n\n' + case_study_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
            # Set training data set directory
            training_dataset_dir = os.path.join(
                os.path.normpath(case_study_dir), '1_training_dataset')
            # Get training data set file path
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, train_dataset_file_path = \
                find_unique_file_with_regex(training_dataset_dir, regex)
            # Check data set file
            if not is_file_found:
                raise RuntimeError(f'Training data set file has not been '
                                   f'found in data set directory:\n\n'
                                   f'{training_dataset_dir}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model directory
            model_directory = \
                os.path.join(os.path.normpath(case_study_dir), '3_model')
            # Create model directory (overwrite)
            make_directory(model_directory, is_overwrite=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set validation data set directory
            val_dataset_directory = os.path.join(
                os.path.normpath(case_study_dir), '2_validation_dataset')
            # Get validation data set file path
            val_dataset_file_path = None
            if os.path.isdir(val_dataset_directory):
                regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
                is_file_found, val_dataset_file_path = \
                    find_unique_file_with_regex(val_dataset_directory, regex)
                # Check data set file
                if not is_file_found:
                    raise RuntimeError(f'Validation data set file has not '
                                       f'been found  in data set '
                                       f'directory:\n\n '
                                       f'{val_dataset_directory}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set testing data set directory
            if testing_type == 'training':
                # Set testing data set directory (training data set)
                testing_dataset_dir = os.path.join(
                    os.path.normpath(case_study_dir), '1_training_dataset')
            elif testing_type == 'validation':
                # Set testing data set directory (validation data set)
                testing_dataset_dir = os.path.join(
                    os.path.normpath(case_study_dir), '2_validation_dataset')
            elif testing_type == 'in_distribution':
                # Set testing data set directory (in-distribution testing data
                # set)
                testing_dataset_dir = os.path.join(
                    os.path.normpath(case_study_dir), '5_testing_id_dataset')
            elif testing_type == 'out_distribution':
                # Set testing data set directory (out-of-distribution testing
                # data set)
                testing_dataset_dir = os.path.join(
                    os.path.normpath(case_study_dir), '6_testing_od_dataset')
            else:
                raise RuntimeError('Unknown testing type.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get testing data set file path
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, test_dataset_file_path = \
                find_unique_file_with_regex(testing_dataset_dir, regex)
            # Check data set file
            if not is_file_found:
                raise RuntimeError(f'Testing data set file has not been found '
                                   f'in data set directory:\n\n'
                                   f'{testing_dataset_dir}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model predictions directory
            prediction_directory = os.path.join(
                os.path.normpath(case_study_dir), '7_prediction')
            # Create model predictions directory
            if not os.path.isdir(prediction_directory):
                make_directory(prediction_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create model predictions subdirectory
            prediction_subdir = os.path.join(
                os.path.normpath(prediction_directory), testing_type)
            # Create prediction subdirectory
            if not os.path.isdir(prediction_subdir):
                make_directory(prediction_subdir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set uncertainty quantification directory
            uq_directory = os.path.join(os.path.normpath(case_study_dir),
                                        'uncertainty_quantification')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set device type
            if torch.cuda.is_available():
                device_type = 'cuda'
            else:
                device_type = 'cpu'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform model uncertainty quantification
            perform_model_uq(uq_directory, n_model_sample,
                             train_dataset_file_path, model_directory,
                             prediction_subdir, test_dataset_file_path,
                             is_model_training=is_model_training,
                             val_dataset_file_path=val_dataset_file_path,
                             device_type=device_type, is_verbose=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate plots of model uncertainty quantification
            gen_model_uq_plots(uq_directory, n_model_sample,
                               testing_dataset_dir, testing_type,
                               is_save_fig=True, is_stdout_display=False,
                               is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove model directory
            if os.path.isdir(model_directory):
                shutil.rmtree(model_directory)
            # Remove model predictions directory
            if os.path.isdir(prediction_directory):
                shutil.rmtree(prediction_directory)