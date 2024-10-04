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
import matplotlib.pyplot as plt
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory, \
    load_dataset
from utilities.data_denoisers import Denoiser
from ioput.plots import plot_xy_data
from projects.darpa_metals.rnn_material_model.strain_paths.interface import \
    StrainPathGenerator
# =============================================================================
# Summary: Test denoising methods in noisy strain paths
# =============================================================================
# Set reference noiseless strain-stress data set file path
noiseless_dataset_file_path = \
    ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
     '6_local_rnn_training_noisy/von_mises/'
     'convergence_analyses_homoscedastic_gaussian/noiseless/n10/'
     '1_training_dataset/ss_paths_dataset_n10.pkl')
# Set noisy strain-stress data set file path
noisy_dataset_file_path = \
    ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
     '6_local_rnn_training_noisy/von_mises/'
     'convergence_analyses_homoscedastic_gaussian/homgau_noise_1e-1/n10/'
     '1_training_dataset/ss_paths_dataset_n10.pkl')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load reference noiseless data set
noiseless_dataset = load_dataset(noiseless_dataset_file_path)
# Check noiseless dataset data set
if not isinstance(noiseless_dataset, TimeSeriesDatasetInMemory):
    raise RuntimeError('Noiseless data set file path does not contain a '
                       'TimeSeriesDatasetInMemory data set.')
# Load noisy data set
noisy_dataset = load_dataset(noisy_dataset_file_path)
# Check noisy_dataset data set
if not isinstance(noiseless_dataset, TimeSeriesDatasetInMemory):
    raise RuntimeError('Noisy data set file path does not contain a '
                       'TimeSeriesDatasetInMemory data set.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize denoiser
denoiser = Denoiser()
# Set denoise method
denoise_method = ('moving_average', 'savitzky_golay', 'frequency_low_pass')[0]
# Set denoise method parameters
if denoise_method == 'moving_average':
    denoise_parameters = {'window_size': 5}
elif denoise_method == 'savitzky_golay':
    denoise_parameters = {'window_size': 10, 'poly_order': 2}
elif denoise_method == 'frequency_low_pass':
    denoise_parameters = {'cutoff_frequency': 0.1,
                          'is_plot_magnitude_spectrum': True}
else:
    raise RuntimeError('Unknown denoise method.')
# Set number of denoising cyles
n_denoise_cycle = 2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set MSE loss function
loss_function = torch.nn.MSELoss()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get number of samples
n_sample = 1
# Set strain components
strain_comps = ('11',)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over samples
for i in range(n_sample, n_sample + 1):
    # Get discrete time path
    time_path = noiseless_dataset[i]['time_hist']
    # Get noiseless strain path
    noiseless_strain_path = noiseless_dataset[i]['strain_path']
    # Get noisy strain path
    noisy_strain_path = noisy_dataset[i]['strain_path']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get denoised strain path
    denoised_strain_path = denoiser.denoise(noisy_strain_path, denoise_method,
                                            denoise_parameters,
                                            n_denoise_cycle=n_denoise_cycle)
    # Enforce null initial noise
    denoised_strain_path[0, :] = noiseless_strain_path[0, :]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute RMSE of denoised strain path with respect to noiseless path
    denoised_rmse = \
        torch.sqrt(loss_function(denoised_strain_path, noiseless_strain_path))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over strain components
    for j, strain_comp in enumerate(strain_comps):
        # Initialize data array
        data_xy = torch.zeros((len(time_path), 6))
        # Build data array
        data_xy[:, 0] = time_path[:, 0]
        data_xy[:, 1] = noiseless_strain_path[:, j]
        data_xy[:, 2] = time_path[:, 0]
        data_xy[:, 3] = noisy_strain_path[:, j]
        data_xy[:, 4] = time_path[:, 0]
        data_xy[:, 5] = denoised_strain_path[:, j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        data_labels = ('Noiseless', 'Noisy', 'Denoised')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes limits
        x_lims = (0, time_path[-1])
        y_lims = (None, None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Time'
        y_label = f'Strain {strain_comp}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain paths
        figure, axes = plot_xy_data(data_xy, data_labels=data_labels,
                                    is_reference_data=True,
                                    x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set RMSE box properties
        text_box_props = dict(boxstyle='round', facecolor='#ffffff',
                              edgecolor='#000000', alpha=1.0)
        # Plot RMSE of denoised strain path
        rmse_str = r'RMSE = $' + f'{denoised_rmse:.2e}' + '$'
        axes.text(0.03, 0.03, rmse_str, fontsize=10, ha='left', va='bottom',
                  transform=axes.transAxes, bbox=text_box_props, zorder=20)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        plt.show()