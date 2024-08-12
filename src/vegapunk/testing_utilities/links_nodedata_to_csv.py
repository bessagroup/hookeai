# Standard
import os
import re
import shutil
# Third-party
import pandas
# =============================================================================
# Summary: Convert Links '.nodedata' files to '.csv' files
# =============================================================================
# Set Links simulation output directory
links_output_directory = ('/home/bernardoferreira/Documents/brown/projects/'
                          'darpa_project/5_global_specimens/'
                          'rowan_specimen_tension_bv/'
                          '2D_toy_uniaxial_specimen_quad4')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check if simulation output directory exists
if not os.path.exists(links_output_directory):
    raise RuntimeError('Links simulation output directory has not been found.')
# Get simulation name
simulation_name = os.path.splitext(os.path.basename(links_output_directory))[0]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get node data output directory
node_data_dir = os.path.join(links_output_directory, 'NODE_DATA')
# Get files in node data output directory
directory_list = os.listdir(node_data_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize node data files
node_data_paths = []
# Loop over files
for filename in directory_list:
    # Check if is node data time step file
    is_node_data_file = bool(re.search(r'^' + simulation_name + r'(.*)'
                                       + r'\.nodedata$', filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append node data time step file
    if is_node_data_file:
        node_data_paths.append(
            os.path.join(os.path.normpath(node_data_dir), filename))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sort node data time step files
node_data_paths = tuple(
    sorted(node_data_paths,
           key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get time history length
n_time = len(node_data_paths)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set '.csv' files directory
csv_dir = os.path.join(node_data_dir, 'csv_files')
# Check existing directory
if os.path.isdir(csv_dir):
    # Remove existing directory
    shutil.rmtree(csv_dir)
# Create directory
os.makedirs(csv_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over discrete time
for t in range(n_time):
    # Load data from node data files
    df_node = pandas.read_csv(node_data_paths[t], sep='\s+', header=None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node data
    if df_node.shape[1] != 18:
        raise RuntimeError(f'Expecting data frame to have 18 columns, but '
                            f'{df_node.shape[1]} were found. \n\n'
                            f'Expected columns: NODE | LOAD FACTOR | '
                            f'TOTAL TIME | X1 X2 X3 | U1 U2 U3 | '
                            f'INTF1 INTF2 INTF3 | EXTF1 EXTF2 EXTF3 | '
                            f'RF1 RF2 RF3')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract node data
    df_csv = df_node.iloc[:, [0, 3, 4, 5, 6, 7, 8, 15, 16, 17]]
    # Set node data headers
    df_csv.columns = ['NODE', 'X1', 'X2', 'X3', 'U1', 'U2', 'U3',
                      'RF1', 'RF2', 'RF3']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set node data '.csv' file path
    csv_file_path = os.path.join(os.path.normpath(csv_dir),
                                 f'{simulation_name}_tstep_{t}.csv')
    # Store data into '.csv' file format
    df_csv.to_csv(csv_file_path, encoding='utf-8', index=False)