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
# Third-party
import numpy as np
import pandas
# Local
from projects.darpa_metals.rnn_material_model.user_scripts.\
    noise_analyses.gen_noisy_response_datasets import NoiseGenerator
from ioput.iostandard import make_directory
# =============================================================================
# Summary: Inject synthetic noise into mesh nodes displacements ('.csv' files)
# =============================================================================
def inject_displacements_noise(csv_data_dir, noise_parameters):
    """Inject synthetic noise into structure nodes displacements.
    
    Parameters
    ----------
    csv_data_dir : str
        Directory with '.csv' data files.
    """
    # Check if data files directory exists
    if not os.path.exists(csv_data_dir):
        raise RuntimeError('Data files directory has not been found:'
                           '\n\n{csv_data_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize noise generator
    noise_generator = NoiseGenerator()
    # Get noise distribution and variability
    noise_distribution = noise_parameters['distribution']
    # Set noise distribution and parameters
    noise_generator.set_noise_distribution(noise_distribution)
    noise_generator.set_noise_parameters(noise_parameters)
    # Get noise directory name
    noise_dirname = noise_parameters['dirname']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set noisy data files directory
    csv_noisy_data_dir = os.path.join(os.path.normpath(csv_data_dir),
                                      f'{noise_dirname}')
    # Create noisy data files directory
    make_directory(csv_noisy_data_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get directory data files
    directory_list = os.listdir(csv_data_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data files
    data_file_paths = []
    # Loop over files
    for filename in directory_list:
        # Check if data time step file
        is_step_data_file = bool(re.search(r'.*_tstep_\d+\.csv$', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append data time step file
        if is_step_data_file:
            data_file_paths.append(
                os.path.join(os.path.normpath(csv_data_dir), filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort data time step files
    data_file_paths = tuple(
        sorted(data_file_paths, key=lambda x:
            int(re.search(r'(.*)_tstep_(\d+)\.csv$', x).groups()[-1])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time history length
    n_time = len(data_file_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for t in range(1, n_time):
        # Get current and previous time step data files
        data_file_path = data_file_paths[t]
        data_file_path_old = data_file_paths[t - 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data
        df = pandas.read_csv(data_file_path)
        df_old = pandas.read_csv(data_file_path_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data
        if df.shape[1] != 10 or df_old.shape[1] != 10:
            raise RuntimeError(f'Expecting data frame to have 10 columns, but '
                               f'{df.shape[1]} were found. \n\n'
                               f'Expected columns: NODE | X1 X2 X3 | '
                               f'U1 U2 U3 | RF1 RF2 RF3')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node displacements
        nodes_disps_mesh = np.array(df.values)[:, 4:7]
        nodes_disps_mesh_old = np.array(df_old.values)[:, 4:7]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental node displacements
        nodes_inc_disps_mesh = nodes_disps_mesh - nodes_disps_mesh_old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate noise
        noise = noise_generator.generate_noise_path(nodes_inc_disps_mesh)
        # Inject noise into incremental node displacements
        nodes_inc_disps_mesh_noisy = (1.0 + noise)*nodes_inc_disps_mesh
        # Compute noisy node displacements
        nodes_disps_mesh_noisy = \
            nodes_disps_mesh_old + nodes_inc_disps_mesh_noisy
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data with noisy node displacements
        df.iloc[:, 4:7] = nodes_disps_mesh_noisy
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set noisy data '.csv' file path
        csv_noisy_file_path = \
            os.path.join(os.path.normpath(csv_noisy_data_dir),
                         os.path.basename(data_file_path))
        # Store noisy data into '.csv' file format
        df.to_csv(csv_noisy_file_path, encoding='utf-8', index=False)
# =============================================================================
if __name__ == "__main__":
    # Set directory with data files
    csv_data_dir = ('/home/bernardoferreira/Desktop/test_sample_memory/'
                    'specimen_history_data')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize noise cases
    gaussian_std_values = [0.01, 0.025, 0.05, 0.1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over noise cases
    for gaussian_std in gaussian_std_values:
        # Set noise parameters
        noise_parameters = {
            'distribution': 'gaussian',
            'std': gaussian_std,
            'dirname': f'noisy_disps_gaussian_std_'
                       f'{gaussian_std}'.replace('.', 'd')}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Inject synthetic noise into structure nodes displacements
        inject_displacements_noise(csv_data_dir, noise_parameters)
    