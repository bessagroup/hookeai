"""Build data set from material patches local strain-stress paths.

Functions
---------
build_local_paths_dataset
    Build data set from material patches local strain-stress paths.
compute_patch_local_paths
    Compute material patch local strain-stress paths data set.
get_patch_analysis_dirs
    Get material patch analysis data directories.
gen_patch_formatted_data
    Generate material patch formatted data.
set_analysis_dir
    Set material patch analysis directory.
set_simulation_dir
    Set material patch simulation data directory.
gen_patch_data
    Generate material patch data and material state files.
gen_patch_local_dataset
    Generate material patch local strain-stress paths data set.
"""
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
import re
import shutil
# Third-party
import torch
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
from user_scripts.global_model_update.material_finder.gen_specimen_data \
    import gen_specimen_dataset, get_specimen_history_paths, \
        set_material_model_parameters
from user_scripts.global_model_update.material_finder. \
    gen_specimen_local_paths import gen_specimen_local_dataset
from user_scripts.synthetic_data.gen_response_dataset import \
    generate_dataset_plots
from simulators.links.utilities.links_dat_to_inp import links_dat_to_abaqus_inp
from simulators.links.utilities.links_nodedata_to_csv import \
    links_node_output_to_csv
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def build_local_paths_dataset(patches_data_dir, patch_name,
                              device_type='cpu', is_verbose=False):
    """Build data set from material patches local strain-stress paths.
    
    Parameters
    ----------
    patches_data_dir : str
        Directory where material patches data and simulations are stored.
    patch_name : str
        Material patch name.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Get material patch analysis data directories
    patch_data_dirs = get_patch_analysis_dirs(patches_data_dir, patch_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of material patches
    n_patch = len(patch_data_dirs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data set samples
    dataset_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material patches
    for i in range(n_patch):
        # Get material patch analysis data directory
        patch_data_dir = patch_data_dirs[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate material patch analysis formatted data
        abaqus_inp_file_path, node_data_csv_dir = \
            gen_patch_formatted_data(patch_data_dir, patch_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute material patch local strain-stress paths data set
        local_dataset = compute_patch_local_paths(
            patch_data_dir, patch_name, abaqus_inp_file_path,
            node_data_csv_dir, device_type=device_type, is_verbose=is_verbose)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect material patch data set samples
        dataset_samples += local_dataset.get_dataset_samples()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create material patches local data set
    dataset = TimeSeriesDatasetInMemory(dataset_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def compute_patch_local_paths(base_dir, patch_name, abaqus_inp_file_path,
                              node_data_csv_dir, device_type='cpu',
                              is_verbose=False):
    """Compute material patch local strain-stress paths data set.
    
    Parameters
    ----------
    base_dir : str
        Directory where the material patch analysis data is stored.
    patch_name : str
        Material patch name.
    abaqus_inp_file_path : str
        Abaqus input data file path.
    node_data_csv_dir : str
        Directory with node data '.csv' files.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    
    Returns
    -------
    local_dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Check base directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The base directory where the material patch '
                           'analysis data is stored has not been found:'
                           '\n\n' + base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch analysis directory
    _, model_finder_dir, _ = set_analysis_dir(base_dir, patch_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch simulation data directory
    simulation_dir, abaqus_inp_file_path, node_data_csv_dir = \
        set_simulation_dir(model_finder_dir, abaqus_inp_file_path,
                           node_data_csv_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch data and material state files
    patch_data_path, patch_material_state_path = \
        gen_patch_data(model_finder_dir, patch_name, simulation_dir,
                       abaqus_inp_file_path, node_data_csv_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate specimen local strain-stress paths data set
    local_dataset = gen_patch_local_dataset(
        model_finder_dir, patch_data_path, patch_material_state_path,
        device_type=device_type, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return local_dataset
# =============================================================================
def get_patch_analysis_dirs(patches_data_dir, patch_name):
    """Get material patch analysis data directories.
    
    Parameters
    ----------
    patches_data_dir : str
        Directory where material patches data and simulations are stored.
    patch_name : str
        Material patch name.

    Returns
    -------
    patch_data_dirs : list[str]
        Directory of each material patch analysis data.
    """
    # Check patches directory
    if not os.path.isdir(patches_data_dir):
        raise RuntimeError('The material patches analysis data directory has '
                           'not been found:\n\n' + patches_data_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in patches directory
    directory_list = os.listdir(patches_data_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'material patches analysis data is stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patches analysis data directories 
    patch_data_dirs = []
    # Get material patches analysis data directories
    for dirname in directory_list:
        # Check if directory is material patch analysis data directory
        id = re.search(r'^' + f'{patch_name}' + r'_([0-9]+)$', dirname)
        # Store material patch analysis data directory
        if id is not None:
            patch_data_dirs.append(
                os.path.join(os.path.normpath(patches_data_dir), dirname))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check material patches analysis data directories 
    if not patch_data_dirs:
        raise RuntimeError('No material patch analysis data directories have '
                           'been found in directory where material patches '
                           'analysis data is stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort material patches analysis data directories 
    patch_data_dirs = tuple(
        sorted(patch_data_dirs,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_data_dirs
# =============================================================================
def gen_patch_formatted_data(patch_data_dir, patch_name):
    """Generate material patch formatted data.
    
    Parameters
    ----------
    patch_data_dir : list[str]
        Material patch analysis data directory.
    patch_name : str
        Material patch name.
        
    Returns
    -------
    abaqus_inp_file_path : str
        Abaqus input data file path.
    node_data_csv_dir : str
        Directory with node data '.csv' files.
    """
    # Set material patch simulation data directory
    simulations_dir = \
        os.path.join(os.path.normpath(patch_data_dir), 'simulations')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check material patch simulation directory
    if not os.path.isdir(simulations_dir):
        raise RuntimeError('The material patch simulation data directory '
                           'has not been found:\n\n' + simulations_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links simulation input data file path
    links_input_file_path = \
        os.path.join(os.path.normpath(simulations_dir), f'{patch_name}.dat')
    # Set Links simulation output directory
    links_output_dir = \
        os.path.join(os.path.normpath(simulations_dir), f'{patch_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert Links input data file to Abaqus input data file    
    abaqus_inp_file_path = links_dat_to_abaqus_inp(links_input_file_path)
    # Convert Links node output data files to '.csv' files
    node_data_csv_dir = links_node_output_to_csv(links_output_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return abaqus_inp_file_path, node_data_csv_dir
# =============================================================================
def set_analysis_dir(base_dir, analysis_name):
    """Set material patch analysis directory.
    
    Parameters
    ----------
    base_dir : str
        Directory where the material patch analysis directory is created.
    analysis_name : str
        Name of material patch analysis directory.
    
    Returns
    -------
    analysis_dir : str
        Material patch analysis directory.
    model_finder_dir : str
        Material model finder directory.
    model_performance_dir : str
        Material model performance directory.
    """
    # Check base directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The base directory where the material patch '
                           'analysis directory is created has not been found:'
                           '\n\n' + base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch analysis directory
    analysis_dir = os.path.join(os.path.normpath(base_dir), analysis_name)
    # Create material patch analysis directory
    make_directory(analysis_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model finder directory
    model_finder_dir = os.path.join(os.path.normpath(analysis_dir),
                                    'material_model_finder')
    # Create material model finder directory
    make_directory(model_finder_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model performance directory
    model_performance_dir = os.path.join(os.path.normpath(analysis_dir),
                                         'material_model_performance')
    # Create material model performance directory
    make_directory(model_performance_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return analysis_dir, model_finder_dir, model_performance_dir
# =============================================================================
def set_simulation_dir(model_finder_dir, abaqus_inp_file_path,
                       node_data_csv_dir):
    """Set material patch simulation data directory.
    
    Parameters
    ----------
    model_finder_dir : str
        Material model finder directory.
    abaqus_inp_file_path : str
        Abaqus input data file path.
    node_data_csv_dir : str
        Directory with node data '.csv' files.

    Returns
    -------
    simulation_dir : str
        Material patch simulation data directory.
    new_abaqus_inp_file_path : str
        Abaqus input data file path.
    new_node_data_csv_dir : str
        Directory with node data '.csv' files.
    """
    # Check material model directory
    if not os.path.isdir(model_finder_dir):
        raise RuntimeError('The material model finder directory has not been '
                           'found:\n\n' + model_finder_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch simulation data directory
    simulation_dir = os.path.join(
        os.path.normpath(model_finder_dir), '0_simulation')
    # Create material patch simulation data directory
    make_directory(simulation_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Abaqus input data file path
    new_abaqus_inp_file_path = \
        os.path.join(os.path.normpath(simulation_dir),
                     os.path.basename(abaqus_inp_file_path))
    # Copy Abaqus input data file path
    shutil.copy(abaqus_inp_file_path, new_abaqus_inp_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set node data directory
    new_node_data_csv_dir = os.path.join(os.path.normpath(simulation_dir),
                                         os.path.basename(node_data_csv_dir))
    # Copy node data directory
    shutil.copytree(node_data_csv_dir, new_node_data_csv_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return simulation_dir, new_abaqus_inp_file_path, new_node_data_csv_dir
# =============================================================================
def gen_patch_data(model_finder_dir, patch_name, simulation_dir,
                   abaqus_inp_file_path, node_data_csv_dir):
    """Generate material patch data and material state files.
    
    Parameters
    ----------
    model_finder_dir : str
        Material model finder directory.
    patch_name : str
        Material patch name.
    simulation_dir : str
        Material patch simulation data directory.
    abaqus_inp_file_path : str
        Abaqus input data file path.
    node_data_csv_dir : str
        Directory with node data '.csv' files.
    
    Returns
    -------
    patch_data_path : str
        Material patch numerical data file path.
    patch_material_state_path : str
        Material patch material state data file path.
    """
    # Check material model finder directory
    if not os.path.isdir(model_finder_dir):
        raise RuntimeError('The material model finder directory has not been '
                           'found:\n\n' + model_finder_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch training data directory
    training_dir = \
        os.path.join(os.path.normpath(model_finder_dir), '1_training_dataset')
    # Create material patch training data directory
    make_directory(training_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material patch history time step file paths
    patch_history_paths = \
        get_specimen_history_paths(node_data_csv_dir, patch_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model name and parameters
    strain_formulation, problem_type, n_dim, model_name, \
        model_parameters, model_kwargs = set_material_model_parameters()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set save material patch data file flag
    is_save_patch_data = True
    # Set save material patch material state file flag
    is_save_patch_material_state = True
    # Create material models initialization subdirectory
    if is_save_patch_material_state:
        # Set material models initialization subdirectory
        material_models_dir = os.path.join(
            os.path.normpath(training_dir), 'material_models_init')
        # Create material models initialization subdirectory
        make_directory(material_models_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material model directory
        material_model_dir = os.path.join(
            os.path.normpath(material_models_dir), 'model_1')
        # Create material model directory
        make_directory(material_model_dir, is_overwrite=True)
        # Assign material model directory
        model_kwargs['model_directory'] = material_model_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch data and material state files
    patch_data_path, patch_material_state_path = gen_specimen_dataset(
        patch_name, simulation_dir, abaqus_inp_file_path, patch_history_paths,
        strain_formulation, problem_type, n_dim, model_name, model_parameters,
        model_kwargs, training_dir, is_save_patch_data,
        is_save_patch_material_state)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_data_path, patch_material_state_path
# =============================================================================
def gen_patch_local_dataset(model_finder_dir, patch_data_path,
                            patch_material_state_path, device_type='cpu',
                            is_verbose=False):
    """Generate material patch local strain-stress paths data set.
    
    Parameters
    ----------
    model_finder_dir : str
        Material model finder directory.
    patch_data_path : str
        Material patch numerical data file path.
    patch_material_state_path : str
        Material patch material state data file path.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.

    Returns
    -------
    local_dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Check material model finder directory
    if not os.path.isdir(model_finder_dir):
        raise RuntimeError('The material model finder directory has not been '
                           'found:\n\n' + model_finder_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model directory
    model_directory = os.path.join(
        os.path.normpath(model_finder_dir), '3_model')
    # Create model directory
    if not os.path.isdir(model_directory):
        make_directory(model_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch local strain-stress paths data set
    local_dataset, _ = gen_specimen_local_dataset(
        patch_data_path, patch_material_state_path, model_directory,
        device_type=device_type, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return local_dataset
# =============================================================================
if __name__ == "__main__":
    # Set directory where material patches data and simulations are stored
    patches_data_dir = ('/home/username/Documents/brown/projects/'
                        'test_patches/base_directory/'
                        'material_patches_generation')
    # Set material patch name
    patch_name = 'random_specimen'
    # Set data set file basename
    dataset_basename = 'ss_paths_dataset'
    # Set data set directory
    dataset_dir = os.path.join(os.path.normpath(patches_data_dir),
                               'local_paths_dataset')
    # Set plot data set generation flag
    is_plot_dataset = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create data set directory
    make_directory(dataset_dir, is_overwrite=True)
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data set from material patches local strain-stress paths
    dataset = build_local_paths_dataset(patches_data_dir, patch_name,
                                        device_type=device_type,
                                        is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save data set
    save_dataset(dataset, dataset_basename, dataset_dir,
                 is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patches local data set plots
    if is_plot_dataset:
        # Set data set plots directory
        plots_dir = os.path.join(dataset_dir, 'plots')
        # Create plots directory
        plots_dir = make_directory(plots_dir)
        # Get material parameters
        strain_formulation, _, n_dim, _, _, _ = set_material_model_parameters()
        # Generate data set plots
        generate_dataset_plots(strain_formulation, n_dim, dataset,
                               save_dir=plots_dir, is_save_fig=True,
                               is_stdout_display=False)