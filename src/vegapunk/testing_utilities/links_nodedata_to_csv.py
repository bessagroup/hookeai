# Standard
import os
import re
import shutil
# Third-party
import pandas
# =============================================================================
# Summary: Convert Links '.nodedata' files to '.csv' files
# =============================================================================
def links_node_output_to_csv(links_output_dir):
    """Convert Links '.nodedata' output files to '.csv' files.
    
    Parameters
    ----------
    links_output_dir : str
        Links simulation output directory.
        
    Returns
    -------
    csv_output_dir : str
        Directory with '.csv' output files.
    """
    # Check if Links simulation output directory exists
    if not os.path.exists(links_output_dir):
        raise RuntimeError('Links simulation output directory has not been '
                           'found:\n\n{links_output_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = os.path.splitext(os.path.basename(links_output_dir))[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get node data output directory
    node_data_dir = os.path.join(links_output_dir, 'NODE_DATA')
    # Get files in node data output directory
    directory_list = os.listdir(node_data_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set '.csv' files directory
    csv_output_dir = os.path.join(node_data_dir, 'specimen_history_data')
    # Check existing directory
    if os.path.isdir(csv_output_dir):
        # Remove existing directory
        shutil.rmtree(csv_output_dir)
    # Create directory
    os.makedirs(csv_output_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for t in range(n_time):
        # Load data from node data files
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract node data
        df_csv = df_node.iloc[:, [0, 3, 4, 5, 6, 7, 8, 15, 16, 17]]
        # Set node data headers
        df_csv.columns = ['NODE', 'X1', 'X2', 'X3', 'U1', 'U2', 'U3',
                          'RF1', 'RF2', 'RF3']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node data '.csv' file path
        csv_file_path = os.path.join(os.path.normpath(csv_output_dir),
                                    f'{simulation_name}_tstep_{t}.csv')
        # Store data into '.csv' file format
        df_csv.to_csv(csv_file_path, encoding='utf-8', index=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return csv_output_dir
# =============================================================================
if __name__ == "__main__":
    # Set Links simulation output directory
    links_output_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                        'darpa_project/8_global_random_specimen/von_mises/'
                        '2_random_specimen_hexa8/'
                        'voids_lattice/'
                        '0_links_simulation/random_specimen/simulations/'
                        'random_specimen')