# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import re
# Third-party
import torch
# Local
from projects.darpa_metals.rnn_material_model.user_scripts. \
    gen_response_dataset import MaterialResponseDatasetGenerator
# =============================================================================
# Summary: Generate strain-stress material response path sample plots
# =============================================================================
# Set material response path sample file path
sample_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                    'colaboration_bazant_m7/1_bazant_m7_gru_model/'
                    '5_testing_id_dataset/ss_response_path_458.pt')
# Set plots directory
plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_bazant_m7/deliverable_yuri_08_2024')
# Set strain formulation
strain_formulation = 'infinitesimal'
# Set number of dimensions
n_dim = 3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get sample index
sample_index = int(re.search(r'_(\d+).pt$', sample_file_path).groups()[-1])
# Load material response path
response_path = torch.load(sample_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract strain-stress data
strain_comps_order = response_path['strain_comps_order']
strain_path = response_path['strain_path']
stress_comps_order = response_path['stress_comps_order']
stress_path = response_path['stress_path']
time_hist = response_path['time_hist']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot material response path
MaterialResponseDatasetGenerator.plot_material_response_path(
    strain_formulation, n_dim, strain_comps_order, strain_path,
    stress_comps_order, stress_path, time_hist,
    is_plot_strain_stress_paths=True, is_plot_eq_strain_stress=False,
    filename=f'strain_path_sample_{sample_index}',
    save_dir=plots_dir, is_save_fig=True,
    is_stdout_display=False, is_latex=True)