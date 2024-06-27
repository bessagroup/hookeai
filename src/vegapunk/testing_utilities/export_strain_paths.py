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
import pandas
import torch
import tqdm
# Local
from projects.darpa_metals.rnn_material_model.strain_paths.random_path import \
    RandomStrainPathGenerator
# =============================================================================
# Summary: Generate and export set of strain deformation paths (.csv files)
# =============================================================================
# Set strain paths directory
strain_paths_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                    'darpa_project/2_local_rnn_training/'
                    'composite_rve/rve_simulations')
# Set strain path basename
strain_path_basename = 'composite_rve'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set strain formulation
strain_formulation = 'infinitesimal'
# Set problem type
problem_type = 4
# Set number of spatial dimensions
n_dim = 3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set number of discrete times
n_time = 200
# Set initial and final time
time_init = 0.0
time_end = 1.0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set strain components bounds
if n_dim == 2:
    strain_bounds = {x: (-0.05, 0.05)
                     for x in ('11', '22', '12')}
else:
    strain_bounds = {x: (-0.05, 0.05)
                     for x in ('11', '22', '33', '12', '23', '13')}
# Set incremental strain norm
inc_strain_norm = None
# Set strain noise
strain_noise_std = None
# Set cyclic loading
is_cyclic_loading = False    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set strain path generators parameters
strain_path_kwargs = {'n_control': (4, 7),
                      'strain_bounds': strain_bounds,
                      'n_time': n_time,
                      'generative_type': 'polynomial',
                      'time_init': time_init,
                      'time_end': time_end,
                      'inc_strain_norm': inc_strain_norm,
                      'strain_noise_std': strain_noise_std,
                      'is_cyclic_loading': is_cyclic_loading}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize strain path generator
strain_path_generator = RandomStrainPathGenerator(strain_formulation, n_dim)
# Set number of strain paths
n_path = 1
# Initialize strain paths data
time_hists = []
strain_paths = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over strain paths
for i in tqdm.tqdm(range(n_path), desc='> Generating strain-stress paths: '):
    # Generate strain path
    strain_comps_order, time_hist, strain_path = \
        strain_path_generator.generate_strain_path(**strain_path_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store strain path data
    time_hists.append(time_hist)
    strain_paths.append(strain_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize dataframe
    df = pandas.DataFrame(torch.zeros((n_time, 13)),
                          columns=['TIME', 'E11', 'E22', 'E33', 'E12', 'E23',
                                   'E13', 'S11', 'S22', 'S33', 'S12', 'S23',
                                   'S13'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble discrete time history
    df['TIME'] = time_hist
    # Assemble strain path components
    if n_dim == 2:
        df[['E11', 'E22', 'E12']] = strain_path
    else:
        df[['E11', 'E22', 'E33', 'E12', 'E23', 'E13']] = strain_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path '.csv' file path
    csv_file_path = os.path.join(os.path.normpath(strain_paths_dir),
                                 f'{strain_path_basename}_path_{i}.csv')
    # Store data into '.csv' file format
    df.to_csv(csv_file_path, encoding='utf-8', index=False)