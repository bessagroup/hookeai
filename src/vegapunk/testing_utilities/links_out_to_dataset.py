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
import torch
import numpy as np
import pandas
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
from ioput.iostandard import make_directory
from projects.darpa_metals.rnn_material_model.user_scripts\
    .gen_response_dataset import generate_dataset_plots
# =============================================================================
# Summary: Convert Links Gauss point '.out' files to time series data set
# =============================================================================
def build_dataset(links_out_file_paths, n_dim):
    """Build time series data set from set of Links Gauss point '.out' files.
    
    Parameters
    ----------
    links_out_file_paths : tuple
        Links Gauss point '.out' data file paths.
    n_dim : int
        Number of spatial dimensions.
    
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Initialize data set samples
    dataset_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data file paths
    for links_out_file_path in links_out_file_paths:
        # Build material response path data
        response_path = extract_sample_path(links_out_file_path, n_dim)
        # Store material response path data
        dataset_samples.append(response_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create strain-stress material response path data set
    dataset = TimeSeriesDatasetInMemory(dataset_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def extract_sample_path(links_out_file_path, n_dim):
    """Build material response path data from Links Gauss point '.out' file.
    
    Parameters
    ----------
    links_out_file_path : str
        Links Gauss point '.out' data file path.
    n_dim : int
        Number of spatial dimensions.

    Returns
    -------
    response_path : dict
        Material response path.
    """
    # Check data file path
    if not os.path.isfile(links_out_file_path):
        raise RuntimeError('Links Gauss point \'.out\' data file has not '
                           'been found: \n\n', links_out_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load path data
    df_path_data = pandas.read_csv(links_out_file_path, sep='\s+', header=1)
    # Convert to array
    full_data_array = np.array(df_path_data.values)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links strain components symmetric order
    if n_dim == 2:
        strain_comps_order = None
    else:
        strain_comps_order = ('11', '22', '33', '12', '23', '13')
    n_strain_comp = len(strain_comps_order)
    # Set Links stress components symmetric order
    if n_dim == 2:
        stress_comps_order = None
    else:
        stress_comps_order = ('11', '22', '33', '12', '23', '13')
    n_stress_comp = len(stress_comps_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time path
    time_hist = full_data_array[:, 0].astype(float)
    n_time = len(time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set total strain starting index
    if n_dim == 2:
        strain_idx_init = 49
    else:
        strain_idx_init = 53
    # Initialize strain path
    strain_path = np.zeros((n_time, n_strain_comp))
    # Loop over strain components
    for i, comp in enumerate(strain_comps_order):
        # Set Voigt factor
        if comp[0] != comp[1]:
            factor = 0.5
        else:
            factor = 1.0
        # Get strain component
        strain_path[:, i] = \
            factor*full_data_array[:, strain_idx_init + i].astype(float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set stress starting index
    stress_idx_init = 11
    # Initialize stress path
    stress_path = np.zeros((n_time, n_stress_comp))
    # Loop over stress components
    for i, comp in enumerate(stress_comps_order):
        # Get stress component
        stress_path[:, i] = \
            full_data_array[:, stress_idx_init + i].astype(float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material response path data
    response_path = {}
    # Assemble strain-stress material response path
    response_path['strain_comps_order'] = strain_comps_order
    response_path['strain_path'] = \
        torch.tensor(strain_path, dtype=torch.get_default_dtype())
    response_path['stress_comps_order'] = stress_comps_order
    response_path['stress_path'] = \
        torch.tensor(stress_path, dtype=torch.get_default_dtype())
    # Assemble time path
    response_path['time_hist'] = torch.tensor(
        time_hist, dtype=torch.get_default_dtype()).reshape(-1, 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return response_path
# =============================================================================
if __name__ == '__main__':
    # Set save data set plots flags
    is_save_dataset_plots = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set directory
    dataset_dir = ('/home/bernardoferreira/Desktop/test_fortran/'
                   'test_links_lou/3D_HEXA8_LOU_Mixed')
    # Set Links Gauss points '.out' data file paths
    links_out_file_paths = \
        ('/home/bernardoferreira/Desktop/test_fortran/test_links_lou/'
         '3D_HEXA8_LOU_Mixed/3D_HEXA8_LOU_Mixed_ELEM_1_GP_5.out',
         '/home/bernardoferreira/Desktop/test_fortran/test_links_lou/'
         '3D_HEXA8_LOU_Mixed/3D_HEXA8_LOU_Mixed_ELEM_1_GP_7.out',
         '/home/bernardoferreira/Desktop/test_fortran/test_links_lou/'
         '3D_HEXA8_LOU_Mixed/3D_HEXA8_LOU_Mixed_ELEM_1_GP_8.out')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build time series data set from set of Links Gauss point '.out' files
    dataset = build_dataset(links_out_file_paths, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file basename
    dataset_basename = 'ss_paths_dataset'
    # Save data set
    save_dataset(dataset, dataset_basename, dataset_dir,
                 is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots
    if is_save_dataset_plots:
        # Set data set plots directory
        plots_dir = os.path.join(dataset_dir, 'plots')
        # Create plots directory
        plots_dir = make_directory(plots_dir, is_overwrite=True)
        # Generate data set plots
        generate_dataset_plots(strain_formulation, n_dim, dataset,
                               save_dir=plots_dir, is_save_fig=True,
                               is_stdout_display=False)