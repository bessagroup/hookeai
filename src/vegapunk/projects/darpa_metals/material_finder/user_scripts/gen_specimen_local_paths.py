"""DARPA METALS PROJECT: Generate specimen local strain-stress paths data set.

Functions
---------
gen_specimen_local_dataset(specimen_data_path, specimen_material_state_path, \
                           model_directory, device_type='cpu', \
                           is_verbose=False)
    Generate specimen local strain-stress paths data set.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import torch
# Local
from material_model_finder.model.material_discovery import MaterialModelFinder
from rnn_base_model.data.time_dataset import load_dataset
from projects.darpa_metals.rnn_material_model.user_scripts. \
    gen_response_dataset import generate_dataset_plots
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
def gen_specimen_local_dataset(specimen_data_path,
                               specimen_material_state_path,
                               model_directory, device_type='cpu',
                               is_verbose=False):
    """Generate specimen local strain-stress paths data set.
    
    Parameters
    ----------
    specimen_data_path : str
        Path of file containing the specimen numerical data translated from
        experimental results.
    specimen_material_state_path : str
        Path of file containing the FETorch specimen material state.
    model_directory : str
        Directory where model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Load specimen numerical data
    specimen_data = torch.load(specimen_data_path)
    # Load specimen material state
    specimen_material_state = torch.load(specimen_material_state_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model finder name
    model_finder_name = 'material_model_finder'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material model finder
    material_finder = MaterialModelFinder(
        model_directory, model_name=model_finder_name,
        is_force_normalization=False, device_type=device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen data and material state
    material_finder.set_specimen_data(specimen_data, specimen_material_state)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute force equilibrium history loss
    # (need to manually set is_store_local_paths=True in material_finder!)
    force_equilibrium_hist_loss = \
        material_finder(sequential_mode='sequential_element')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set directory
    dataset_directory = os.path.join(os.path.normpath(model_directory),
                                     'local_response_dataset')
    # Get data set file path
    regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
    is_file_found, dataset_file_path = \
        find_unique_file_with_regex(dataset_directory, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Data set file has not been found in data set '
                           f'directory:\n\n'
                           f'{dataset_directory}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display results
    if is_verbose:
        print('')
        print('DARPA METALS PROJECT: Generate specimen local strain-stress '
              'paths data set')
        print('------------------------------------------------------------'
              '--------------')
        print(f'Specimen numerical data: {specimen_data_path}')
        print(f'\nSpecimen material state: {specimen_material_state_path}')
        print(f'\nForce equilibrium history loss: '
              f'{force_equilibrium_hist_loss:.4e}')
        print(f'Force normalization:  {str(False):>15s}')
        print(f'\nSpecimen local data set: {dataset_file_path}')
        print('-------------------------------------------------------------'
              '--------------')
        print('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set plots directory
    plots_dir = os.path.join(dataset_directory, 'plots')
    # Create plots directory
    plots_dir = make_directory(plots_dir)
    # Generate data set plots
    generate_dataset_plots(specimen_material_state.get_strain_formulation(),
                           specimen_data.get_n_dim(), dataset,
                           save_dir=plots_dir, is_save_fig=True,
                           is_stdout_display=False)
# =============================================================================
def set_default_model_parameters(model_directory, device_type='cpu'):
    """Set default model initialization parameters.
    
    Parameters
    ----------
    model_directory : str
        Directory where material patch model is stored.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    model_init_args : dict
        Material model finder class initialization parameters (check class
        MaterialModelFinder).
    """
    # Set model name
    model_name = 'material_model_finder'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model initialization parameters
    model_init_args = {'model_directory': model_directory,
                       'model_name': model_name,
                       'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_init_args
# =============================================================================
if __name__ == "__main__":
    # Set case study base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/4_global_toy_uniaxial_specimen/'
                '2d_toy_uniaxial_specimen_tri3_rc_elastic/'
                '0_elastic_properties_E/')
    # Set case study directory
    case_study_name = 'material_model_finder'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Set training data set directory
    training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                        '1_training_dataset')
    # Set specimen numerical data file path
    specimen_data_path = os.path.join(os.path.normpath(training_dataset_dir),
                                      'specimen_data.pt')
    # Check specimen numerical data file
    if not os.path.isfile(specimen_data_path):
        raise RuntimeError(f'The specimen numerical data file ('
                           f'specimen_data.pt) has not been found in data set '
                           f'directory:\n\n'
                           f'{training_dataset_dir}')
    # Set specimen material state file path
    specimen_material_state_path = \
        os.path.join(os.path.normpath(training_dataset_dir),
                                      'specimen_material_state.pt')
    # Check specimen material state file
    if not os.path.isfile(specimen_material_state_path):
        raise RuntimeError(f'The specimen material state file ('
                           f'specimen_material_state.pt) has not been found '
                           f'in data set directory:\n\n'
                           f'{training_dataset_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir), '3_model')
    # Create model directory
    if not os.path.isdir(model_directory):
        make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate specimen local strain-stress paths data set
    gen_specimen_local_dataset(
        specimen_data_path, specimen_material_state_path, model_directory,
        device_type=device_type, is_verbose=True)