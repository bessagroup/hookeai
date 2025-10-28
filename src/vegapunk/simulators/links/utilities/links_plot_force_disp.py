"""Generate force-displacement plot from Links simulation."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
# Third-party
import numpy as np
import pandas
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, save_figure
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
    # Set displacement multiplier (accounting for model symmetries)
    displacement_multiplier = 2.0
    # Set reaction force multiplier (accounting for model symmetries)
    reaction_multiplier = 4.0*1e-3
    # Set reaction forces node group
    reaction_node_group = 1
    # Set prescribed loading dimension (1-x, 2-y, 3-z)
    load_direction = 2
    # Set displacement reference node label
    ref_node_label = 607
    # Set Links simulation output directory
    links_output_directory = (
        '/home/bernardoferreira/Documents/brown/projects/darpa_project/'
        '5_global_specimens/rowan_specimen_tension_bv_hexa8_rc_von_mises/'
        'links_simulation/3D_rowan_specimen_tension_bv')
    # Set display figure flag
    is_stdout_display = True
    # Set save figure flag
    is_save_fig = True
    # Set displacement and reaction force units
    disp_units = 'mm'
    reac_units = 'kN'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if simulation output directory exists
    if not os.path.exists(links_output_directory):
        raise RuntimeError('Links simulation output directory has not been '
                           'found.')
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(links_output_directory))[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reaction force file path
    reaction_file_path = os.path.join(links_output_directory,
                                    f'{simulation_name}.reac')
    # Check if reaction force file exists
    if not os.path.isfile(reaction_file_path):
        raise RuntimeError('Links reaction data file has not been found.')
    # Set node data output directory
    node_data_dir = os.path.join(links_output_directory, 'NODE_DATA')
    # Check if node data directory exists
    if not os.path.isdir(node_data_dir):
        raise RuntimeError('Links node data directory has not been found.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load reaction force data
    df_reaction = pandas.read_csv(reaction_file_path, sep='\s+', header=None)
    # Convert to array
    reaction_array = np.array(df_reaction.values)
    # Get node group mask
    node_group_mask = reaction_array[:, 0] == reaction_node_group
    # Set prescribed loading direction index
    load_direction_index = load_direction - 1
    # Extract reaction force history data for given node group and dimension
    reaction_array_dim = \
        reaction_array[node_group_mask, 2 + load_direction_index]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in node data output directory
    directory_list = os.listdir(node_data_dir)
    # Initialize node data files
    node_data_paths = []
    # Loop over files
    for filename in directory_list:    
        # Check if is node data time step file
        is_node_data_file = bool(re.search(r'^' + simulation_name + r'(.*)'
                                           + r'\.nodedata$', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append node data time step file
        if is_node_data_file:
            node_data_paths.append(
                os.path.join(os.path.normpath(node_data_dir), filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort node data time step files
    node_data_paths = tuple(
        sorted(node_data_paths,
            key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time history length
    n_time = len(node_data_paths)
    # Check time history length
    if n_time == 0:
        raise RuntimeError('No node data time step files have been collected '
                           'when processing node data output directory.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prescribed displacement history data
    disp_array_dim = []
    # Set displacement reference node index
    ref_node_index = ref_node_label - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for t in range(n_time):
        # Load data from node data file
        df_node = pandas.read_csv(node_data_paths[t], sep='\s+', header=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check node data
        if df_node.shape[1] != 18:
            raise RuntimeError(f'Expecting data frame to have 18 columns, but '
                               f'{df_node.shape[1]} were found. \n\n'
                               f'Expected columns: NODE | LOAD FACTOR | '
                               f'TOTAL TIME | X1 X2 X3 | U1 U2 U3 | '
                               f'INTF1 INTF2 INTF3 | EXTF1 EXTF2 EXTF3 | '
                               f'RF1 RF2 RF3')
        # Check reference node
        if ref_node_index > df_node.shape[0]:
            raise RuntimeError(f'Reference node {ref_node_label} does not '
                               f'exist in {simulation_name} simulation mesh.') 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract reference node displacement for given dimension
        disp_array_dim.append(
            df_node.iloc[ref_node_index, 6 + load_direction_index])
    # Build reference node displacement history data
    disp_array_dim = np.array(disp_array_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build force-displacement data array
    data_array = np.zeros((n_time, 2))
    data_array[:, 0] = disp_array_dim
    data_array[:, 1] = reaction_array_dim
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply displacement multiplier
    data_array[:, 0] = data_array[:, 0]*displacement_multiplier
    # Apply reaction force multiplier
    data_array[:, 1] = data_array[:, 1]*reaction_multiplier
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set force-displacement history data file path
    force_disp_path = os.path.join(os.path.dirname(links_output_directory),
                                   f'{simulation_name}_force_displacement.dat')
    # Open force-displacement history data file
    force_disp_file = open(force_disp_path, 'w')
    # Build force-displacement history file content
    write_lines = \
        ['\nFORCE-DISPLACEMENT HISTORY' + '\n'] \
        + [f'\nNumber of time steps = {n_time}'] \
        + [f'\n\nForce-displacement history data:'] \
        + [f'\n\n{"Displacement":^16s}   {"Force":^16s}'] \
        + [f'\n{data_array[i, 0]:^16.8e}   {data_array[i, 1]:^16.8e}'
        for i in range(n_time)]
    # Write force-displacement history data file
    force_disp_file.writelines(write_lines)
    # Close force-displacement history data file
    force_disp_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot force-displacement data
    figure, axes = plot_xy_data(data_array,
                                x_label=f'Displacement ({disp_units})',
                                y_label=f'Reaction force ({reac_units})',
                                is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure name
    filename = f'{simulation_name}_force_displacement'
    # Set figure directory
    save_dir = os.path.join(os.path.dirname(links_output_directory))
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)