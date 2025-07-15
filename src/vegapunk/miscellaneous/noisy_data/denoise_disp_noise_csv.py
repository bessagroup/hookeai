"""Denoise synthetic noise from mesh nodes displacements ('.csv' files).

Functions
---------
inject_displacements_noise
    Inject synthetic noise into structure nodes displacements.
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
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
# Local
from utilities.data_denoisers import Denoiser
from ioput.plots import plot_xy_data, save_figure
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
def denoise_displacements_noise(csv_data_dir, denoise_parameters={},
                                plot_parameters=None):
    """Denoise synthetic noise from structure nodes displacements.
    
    Parameters
    ----------
    csv_data_dir : str
        Directory with '.csv' data files.
    denoise_parameters : dict, default={}
        Denoising method parameters.
    plot_parameters : dict, default=None
        Node displacement component plot parameters.
    """
    # Check if data files directory exists
    if not os.path.exists(csv_data_dir):
        raise RuntimeError('Data files directory has not been found:'
                           '\n\n{csv_data_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize denoiser
    denoiser = Denoiser()
    # Get denoise method
    denoise_method = denoise_parameters['denoise_method']
    denoiser_parameters = denoise_parameters['denoiser_parameters']
    n_denoise_cycle = denoise_parameters['n_denoise_cycle']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get denoise directory name
    denoise_dirname = denoise_parameters['denoise_dirname']
    # Set denoised data files directory
    csv_denoised_data_dir = os.path.join(os.path.normpath(csv_data_dir),
                                         f'{denoise_dirname}')
    # Create denoised data files directory
    make_directory(csv_denoised_data_dir, is_overwrite=True)
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
    # Initialize nodes displacements history data
    nodes_disps_mesh_time_steps = []
    # Loop over discrete time
    for t in range(0, n_time):
        # Get current time step data file
        data_file_path = data_file_paths[t]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data
        df = pandas.read_csv(data_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data
        if df.shape[1] != 10:
            raise RuntimeError(f'Expecting data frame to have 10 columns, but '
                               f'{df.shape[1]} were found. \n\n'
                               f'Expected columns: NODE | X1 X2 X3 | '
                               f'U1 U2 U3 | RF1 RF2 RF3')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node displacements
        nodes_disps_mesh = np.array(df.values)[:, 4:7]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store node displacements
        nodes_disps_mesh_time_steps.append(nodes_disps_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stack nodes displacements history data (n_time, n_node, n_features)
    nodes_disps_mesh_hist = np.stack(nodes_disps_mesh_time_steps, axis=0)
    # Save nodes noisy displacements history
    nodes_noisy_disps_mesh_hist = nodes_disps_mesh_hist.copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for j in range(nodes_disps_mesh_hist.shape[1]):
        # Get node displacements history (n_time, n_features)
        node_disps_hist = nodes_disps_mesh_hist[:, j, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get denoised node displacements history
        node_disps_hist = denoiser.denoise(
            torch.tensor(node_disps_hist),
            denoise_method, denoise_parameters=denoiser_parameters,
            n_denoise_cycle=n_denoise_cycle).numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce null initial displacement
        node_disps_hist[0, :] = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update node displacements history
        nodes_disps_mesh_hist[:, j, :] = node_disps_hist
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get denoised nodes displacements history data
    nodes_disps_mesh_time_steps = list(nodes_disps_mesh_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for t in range(0, n_time):
        # Get current time step data file
        data_file_path = data_file_paths[t]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data
        df = pandas.read_csv(data_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data with denoised node displacements
        df.iloc[:, 4:7] = nodes_disps_mesh_time_steps[t]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set denoised data '.csv' file path
        csv_denoised_file_path = \
            os.path.join(os.path.normpath(csv_denoised_data_dir),
                         os.path.basename(data_file_path))
        # Store denoised data into '.csv' file format
        df.to_csv(csv_denoised_file_path, encoding='utf-8', index=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot node displacement component path
    if isinstance(plot_parameters, dict):
        # Get node label
        node_id = int(plot_parameters['node_id'])
        # Get displacement component
        disp_comp = int(plot_parameters['disp_comp'])
        # Get noisy and denoised displacement component paths
        disp_paths = [nodes_noisy_disps_mesh_hist[:, node_id-1, disp_comp-1],
                      nodes_disps_mesh_hist[:, node_id-1, disp_comp-1]]
        # Set displacement component paths labels
        disp_paths_labels = ['Noisy', 'Denoised']
        # Plot displacement component paths
        plot_disp_component_paths(
            disp_paths, disp_paths_labels=disp_paths_labels,
            disp_comp_label='Displacement', is_stdout_display=True,
            is_latex=True)
# =============================================================================
def plot_disp_component_paths(disp_paths, disp_paths_labels=None,
                              disp_comp_label='Displacement',
                              filename='displacement_component_paths',
                              save_dir=None, is_save_fig=False,
                              is_stdout_display=False, is_latex=False):
    """Plot displacement component paths.
    
    Parameters
    ----------
    disp_paths : list
        One or more displacement component paths, each stored as
        numpy.ndarray(1d) of shape (n_time,).
    disp_paths_labels : list[str], default=None
        Label of each displacement component path.
    disp_comp_label : str, default='Displacement'
        Displacement component label.
    filename : str, default='displacement_component_paths'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Get number of paths
    n_path = len(disp_paths)
    # Probe time history length from first path
    n_time = disp_paths[0].shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete time history
    time_hist = np.linspace(0.0, 1.0, n_time, endpoint=True, dtype=float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_xy = np.zeros((n_time, 2*n_path))
    # Loop over displacements paths
    for i in range(n_path):
        # Assemble displacement component path data
        data_xy[:, 2*i] = time_hist
        data_xy[:, 2*i + 1] = disp_paths[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    x_lims = (0, 1.0)
    y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Time'
    y_label = str(disp_comp_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot displacement component paths
    figure, _ = plot_xy_data(data_xy, data_labels=disp_paths_labels,
                             x_lims=x_lims, y_lims=y_lims,
                             x_label=x_label, y_label=y_label,
                             is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
if __name__ == "__main__":
    # Set float precision
    is_double_precision = True
    if is_double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set directory with data files
    csv_data_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                    'darpa_paper_examples/global/tensile_dogbone_von_mises/'
                    'hexa8_n6493_e1200/1_discover_rc_von_mises/E_v_s0_a_b/'
                    'noisy_displacements/micro_resolution/'
                    'noisy_uniform_amp_0d00027/material_model_finder/'
                    '0_simulation/specimen_history_data')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set denoise method
    denoise_method = 'moving_average'
    # Set denoise method parameters
    if denoise_method == 'moving_average':
        denoiser_parameters = {'window_size': 5,}
    # Set denoise parameters
    denoise_parameters = {'denoise_dirname': f'denoised_{denoise_method}',
                          'denoise_method': denoise_method,
                          'denoiser_parameters': denoiser_parameters,
                          'n_denoise_cycle': 2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plot parameters
    plot_parameters = {'node_id': 1761, 'disp_comp': '2'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Denoise synthetic noise from structure nodes displacements
    denoise_displacements_noise(csv_data_dir,
                                denoise_parameters=denoise_parameters,
                                plot_parameters=plot_parameters)
    