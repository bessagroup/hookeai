"""Generate ADiMU model discovery data from Links simulation data.

Functions
---------
gen_adimu_data_from_links
    Generate ADiMU model discovery data from Links simulation data.
extract_element_type_from_links_data
    Extract element type from Links input data file.
links_to_abaqus_elem_type
    Convert Links element type to ABAQUS element type.
extract_mesh_coordinates_from_links_data
    Extract mesh initial coordinates from Links input data file.
extract_mesh_connectivities_from_links_data
    Extract mesh connectivities from Links input data file.
extract_mesh_dirichlet_from_links_data
    Extract mesh Dirichlet boundary conditions from Links input data file.
collect_node_data_files_from_links_data
    Collect node data files from Links output directory.
write_adimu_mesh_data_file
    Write ADiMU mesh data file.
write_adimu_mesh_hist_data_files
    Write ADiMU mesh history data files.
generate_dirichlet_sets
    Generate Dirichlet sets data for ADiMU mesh history data files.
update_adimu_mesh_hist_data_files_dirichlet_bcs
    Update ADiMU mesh history data files Dirichlet boundary conditions.
"""
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
import shutil
# Third-party
import numpy as np
import pandas
# Local
from user_scripts.global_model_update.material_finder.gen_specimen_data \
    import get_specimen_history_paths
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def gen_adimu_data_from_links(links_input_file_path, links_output_dir,
                              save_dir):
    """Generate ADiMU model discovery data from Links simulation data.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
    links_output_dir : str
        Links simulation output directory.
    save_dir : str
        Directory where the generated data is stored.
        
    Returns
    -------
    adimu_mesh_file_path : str
        ADiMU mesh data file path.
    adimu_mesh_hist_data_dir : str
        ADiMU mesh history data directory.
    """
    # Check Links input data file
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # Check Links simulation output directory
    if not os.path.exists(links_output_dir):
        raise RuntimeError(f'Links simulation output directory has not been '
                           f'found:\n\n{links_output_dir}')
    # Check storage directory
    if not os.path.exists(save_dir):
        raise RuntimeError(f'Data storage directory has not been '
                           f'found:\n\n{links_output_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(links_input_file_path))[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract Links element type from input data file
    links_elem_type, n_gauss = \
        extract_element_type_from_links_data(links_input_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert Links element type to ABAQUS element type
    abaqus_elem_type = links_to_abaqus_elem_type(links_elem_type, n_gauss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract mesh initial coordinates
    nodes_coords_mesh_init = \
        extract_mesh_coordinates_from_links_data(links_input_file_path)
    # Extract mesh connectivities
    connectivities_mesh = \
        extract_mesh_connectivities_from_links_data(links_input_file_path)
    # Extract mesh Dirichlet boundary conditions
    dirichlet_bcs = \
        extract_mesh_dirichlet_from_links_data(links_input_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect node data files from Links output directory
    node_data_files_paths = \
        collect_node_data_files_from_links_data(links_output_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set ADiMU mesh data file path
    adimu_mesh_file_path = os.path.join(save_dir, f'{simulation_name}.inp')
    # Write ADiMU mesh data file
    write_adimu_mesh_data_file(adimu_mesh_file_path,
                               nodes_coords_mesh_init, connectivities_mesh,
                               abaqus_elem_type, dirichlet_bcs=dirichlet_bcs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set ADiMU mesh history data directory
    adimu_mesh_hist_data_dir = os.path.join(save_dir, f'specimen_history_data')
    # Create ADiMU mesh history data directory
    if os.path.isdir(adimu_mesh_hist_data_dir):
        # Remove existing directory
        shutil.rmtree(adimu_mesh_hist_data_dir)
    # Create directory
    os.makedirs(adimu_mesh_hist_data_dir)
    # Set force equilibrium loss type
    force_equilibrium_loss_type = ('pointwise', 'dirichlet_sets')[0]
    # Write ADiMU mesh history data files
    write_adimu_mesh_hist_data_files(
        adimu_mesh_file_path, adimu_mesh_hist_data_dir, node_data_files_paths,
        dirichlet_bcs=dirichlet_bcs,
        force_equilibrium_loss_type=force_equilibrium_loss_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return adimu_mesh_file_path, adimu_mesh_hist_data_dir
# =============================================================================
def extract_element_type_from_links_data(links_input_file_path):
    """Extract element type from Links input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    elem_type : str
        Element type.
    n_gauss : int
        Number of Gauss integration points.
    """
    # Check Links input data file
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for ELEMENT_TYPES keyword and collect element type
    for line in links_input_file:
        if bool(re.search(r'ELEMENT_TYPES\b', line, re.IGNORECASE)):        
            # Start processing ELEMENT_TYPES section
            is_keyword_found = True
            # Collect number of element types
            n_elem_types = int(line.split()[1])
            # Check number of element types
            if n_elem_types > 1:
                raise RuntimeError('This function can only process meshes '
                                   'with a single element type.')
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing ELEMENT_TYPES section
            break
        elif is_keyword_found:
            # Get element type or number of Gauss integration points
            if bool(re.search(r'\bGP\b', line, re.IGNORECASE)):
                n_gauss = int(line.split()[0])
            else:
                elem_type = line.split()[1].upper()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close Links input data file
    links_input_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return elem_type, n_gauss
# =============================================================================
def links_to_abaqus_elem_type(links_elem_type, n_gauss):
    """Convert Links element type to ABAQUS element type.
    
    Parameters
    ----------
    links_elem_type : str
        Links element type.
    n_gauss : int
        Number of Gauss integration points.

    Returns
    -------
    abaqus_elem_type : str
        ABAQUS element type.
    """

    # Set Links to ABAQUS element type conversion
    elem_type_converter = {'TRI_3_1': 'CPE3', 'TRI3_1': 'CPE3',
                           'QUAD_4_4': 'CPE4', 'QUAD4_4': 'CPE4',
                           'TETRA_4_4': 'C3D4', 'TETRA4_4': 'C3D4',
                           'HEXA_8_8': 'C3D8', 'HEXA8_8': 'C3D8',
                           'HEXA_20_8': 'C3D20R', 'HEXA20_8': 'C3D20R'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get matching ABAQUS element type
    if f'{links_elem_type}_{n_gauss}' in elem_type_converter.keys():
        abaqus_elem_type = elem_type_converter[f'{links_elem_type}_{n_gauss}']
    else:
        raise RuntimeError(f'Unknown Links ({links_elem_type}) to ABAQUS '
                           f'element type conversion.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return abaqus_elem_type
# =============================================================================
def extract_mesh_coordinates_from_links_data(links_input_file_path):
    """Extract mesh initial coordinates from Links input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    nodes_coords_mesh_init : numpy.ndarray(2d)
        Initial coordinates of finite element mesh nodes stored as
        numpy.ndarray of shape (n_node_mesh, n_dim).
    """
    # Check Links input data file
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for NODE_COORDINATES keyword and collect nodes coordinates
    for line in links_input_file:
        if bool(re.search(r'NODE_COORDINATES\b', line, re.IGNORECASE)):
            # Start processing NODE_COORDINATES section
            is_keyword_found = True
            # Collect number of nodes
            n_node = int(line.split()[1])
            # Initialize list of nodes coordinates
            nodes_coords_list = []
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing NODE_COORDINATES section
            break
        elif is_keyword_found:
            # Get node coordinates
            node_coords = np.array([float(x) for x in line.split()[1:]])
            # Append node coordinates
            nodes_coords_list.append(node_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build mesh coordinates
    nodes_coords_mesh_init = np.array(nodes_coords_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close Links input data file
    links_input_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_mesh_init
# =============================================================================
def extract_mesh_connectivities_from_links_data(links_input_file_path):
    """Extract mesh connectivities from Links input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    connectivities_mesh : numpy.ndarray(2d)
       Connectivities of finite element mesh elements stored as numpy.ndarray
       of shape (n_elem, n_node_elem).
    """
    # Check Links input data file
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for ELEMENTS keyword and collect elements connectivities
    for line in links_input_file:
        if bool(re.search(r'ELEMENTS\b', line, re.IGNORECASE)):
            # Start processing ELEMENTS section
            is_keyword_found = True
            # Collect number of elements
            n_elem = int(line.split()[1])
            # Initialize list of elements connectivities
            connectivities_list = []
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing ELEMENTS section
            break
        elif is_keyword_found:
            # Get element connectivities
            element_nodes = [int(x) for x in line.split()[2:]]
            # Assemble element connectivities
            connectivities_list.append(element_nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build mesh connectivities
    connectivities_mesh = np.array(connectivities_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close Links input data file
    links_input_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return connectivities_mesh
# =============================================================================
def extract_mesh_dirichlet_from_links_data(links_input_file_path):
    """Extract mesh Dirichlet boundary conditions from Links input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    dirichlet_bcs : dict
        Dirichlet boundary conditions (item, dict) for each node with
        prescribed displacements (key, str[int]). The boundary conditions of
        each node consist of a prescribed displacement value (item, float) for
        the associated degree of freedom (key, str[int]).
    """
    # Check Links input data file
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for NODES_WITH_PRESCRIBED_DISPLACEMENTS keyword and collect
    # Dirichlet boundary conditions
    for line in links_input_file:
        if bool(re.search(r'NODES_WITH_PRESCRIBED_DISPLACEMENTS\b', line,
                          re.IGNORECASE)):
            # Start processing NODES_WITH_PRESCRIBED_DISPLACEMENTS section
            is_keyword_found = True
            # Initialize Dirichlet boundary conditions
            dirichlet_bcs = {}
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing NODES_WITH_PRESCRIBED_DISPLACEMENTS section
            break
        elif is_keyword_found:
            # Get node label
            node_label = int(line.split()[0])
            # Get node prescription code and values
            node_code = tuple(int(x) for x in line.split()[1])
            node_vals = tuple(float(x) for x in line.split()[2:5])
            # Initialize node Dirichlet boundary conditions
            dirichlet_bcs[str(node_label)] = {}
            # Assemble node Dirichlet boundary conditions
            for i_dof in range(3):
                # Assemble node degree of freedom boundary condition
                if node_code[i_dof] == 1:
                    dirichlet_bcs[str(node_label)][str(i_dof + 1)] = \
                        node_vals[i_dof]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close Links input data file
    links_input_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dirichlet_bcs
# =============================================================================
def collect_node_data_files_from_links_data(links_output_dir):
    """Collect node data files from Links output directory.
    
    Parameters
    ----------
    links_output_dir : str
        Links simulation output directory.
        
    Returns
    -------
    node_data_files_paths : tuple
        Links node data files paths. Data files are sorted according with the
        file name time step.
    """
    # Check Links simulation output directory
    if not os.path.exists(links_output_dir):
        raise RuntimeError(f'Links simulation output directory has not been '
                           f'found:\n\n{links_output_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = os.path.basename(links_output_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get node data output directory
    node_data_dir = os.path.join(links_output_dir, 'NODE_DATA')
    # Get files in node data output directory
    directory_list = os.listdir(node_data_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize node data files
    node_data_files_paths = []
    # Loop over files
    for filename in directory_list:
        # Check if is node data time step file
        is_node_data_file = bool(re.search(r'^' + simulation_name + r'(.*)'
                                           + r'\.nodedata$', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append node data time step file
        if is_node_data_file:
            node_data_files_paths.append(
                os.path.join(os.path.normpath(node_data_dir), filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort node data time step files
    node_data_files_paths = tuple(
        sorted(node_data_files_paths,
            key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return node_data_files_paths
# =============================================================================
def write_adimu_mesh_data_file(adimu_mesh_file_path, nodes_coords_mesh_init,
                               connectivities_mesh, abaqus_elem_type,
                               dirichlet_bcs=None):
    """Write ADiMU mesh data file.
    
    Parameters
    ----------
    adimu_mesh_file_path : str
        ADiMU mesh data file path.
    nodes_coords_mesh_init : numpy.ndarray(2d)
        Initial coordinates of finite element mesh nodes stored as
        numpy.ndarray of shape (n_node_mesh, n_dim).
    connectivities_mesh : numpy.ndarray(2d)
        Connectivities of finite element mesh elements stored as
        numpy.ndarray of shape (n_elem, n_node_elem).
    abaqus_elem_type : str
        ABAQUS element type.
    dirichlet_bcs : dict, default=None
        Dirichlet boundary conditions (item, dict) for each node with
        prescribed displacements (key, str[int]). The boundary conditions of
        each node consist of a prescribed displacement value (item, float) for
        the associated degree of freedom (key, str[int]).
    """
    # Get number of nodes and elements
    n_node = nodes_coords_mesh_init.shape[0]
    n_elem = connectivities_mesh.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize ADiMU mesh data file content
    write_lines = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set header
    write_lines += ['ADiMU mesh data file (generated from Links data)' + '\n']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set mesh coordinates
    write_lines += ['\n' + '*Node' + '\n']
    write_lines += \
        [f'{i:>3d}, '
         + ', '.join(f'{coord:16.8e}'
                     for coord in nodes_coords_mesh_init[i - 1, :])
         + '\n' for i in range(1, n_node + 1)]
    # Set element connectivities
    write_lines += ['\n' + f'*Element, type={abaqus_elem_type}' + '\n']
    write_lines += \
        [f'{i:>3d}, '
         + ', '.join(f'{node:d}'
                     for node in connectivities_mesh[i - 1, :]) + '\n'
         for i in range(1, n_elem + 1)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Dirichlet boundary conditions
    if dirichlet_bcs is not None:
        write_lines += ['\n' + '*Boundary' + '\n']
        write_lines += \
            [f'{int(i):>3d}, {j}, {j}, {dirichlet_bcs[i][j]}\n'
             for i in dirichlet_bcs.keys() for j in dirichlet_bcs[i].keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open ADiMU mesh data file
    with open(adimu_mesh_file_path, 'w') as adimu_mesh_file:
        # Write mesh data file content
        adimu_mesh_file.writelines(write_lines)
# =============================================================================
def write_adimu_mesh_hist_data_files(
    adimu_mesh_file_path, adimu_mesh_hist_data_dir, node_data_files_paths,
    dirichlet_bcs=None, force_equilibrium_loss_type='pointwise'):
    """Write ADiMU mesh history data files.
    
    Parameters
    ----------
    adimu_mesh_file_path : str
        ADiMU mesh data file path.
    adimu_mesh_hist_data_dir : str
        ADiMU mesh history data directory.
    node_data_files_paths : tuple
        Links node data files paths. Data files are sorted according with the
        file name time step.
    dirichlet_bcs : dict, default=None
        Dirichlet boundary conditions (item, dict) for each node with
        prescribed displacements (key, str[int]). The boundary conditions of
        each node consist of a prescribed displacement value (item, float) for
        the associated degree of freedom (key, str[int]).
    force_equilibrium_loss_type : str, default='pointwise'
        Force equilibrium loss type.
    """
    # Check ADiMU mesh history data directory
    if not os.path.exists(adimu_mesh_hist_data_dir):
        raise RuntimeError(f'ADiMU mesh history data directory has not been '
                           f'found:\n\n{adimu_mesh_hist_data_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(adimu_mesh_file_path))[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time history length
    n_time = len(node_data_files_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes from first node data file
    n_node = pandas.read_csv(node_data_files_paths[0], sep='\s+',
                             header=None).shape[0]
    # Initialize Dirichlet boundary conditions data
    dirichlet_bcs_mesh = np.zeros((n_node, 3), dtype=int)
    # Build Dirichlet boundary conditions data
    if (force_equilibrium_loss_type == 'pointwise'
            and dirichlet_bcs is not None):
        dirichlet_bcs_mesh = generate_pointwise_sets(n_node, dirichlet_bcs)
    elif force_equilibrium_loss_type == 'dirichlet_sets':
        dirichlet_bcs_mesh = generate_dirichlet_sets(n_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for t in range(n_time):
        # Load data from node data file
        df_node = pandas.read_csv(node_data_files_paths[t], sep='\s+',
                                  header=None)
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
        # Initialize data frame from headers
        df_csv = pandas.DataFrame(
            columns=['NODE', 'X1', 'X2', 'X3', 'U1', 'U2', 'U3',
                     'RF1', 'RF2', 'RF3', 'DBC1', 'DBC2', 'DBC3'])
        # Assemble node data
        df_csv[['NODE', 'X1', 'X2', 'X3', 'U1', 'U2', 'U3',
                'RF1', 'RF2', 'RF3']] = \
            df_node.iloc[:, [0, 3, 4, 5, 6, 7, 8, 15, 16, 17]].values
        # Assemble Dirichlet boundary conditions
        df_csv[['DBC1', 'DBC2', 'DBC3']] = dirichlet_bcs_mesh
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set time step data file path
        time_step_file_path = os.path.join(
            os.path.normpath(adimu_mesh_hist_data_dir),
                             f'{simulation_name}_tstep_{t}.csv')
        # Store time step data into '.csv' file format
        df_csv.to_csv(time_step_file_path, encoding='utf-8', index=False)
# =============================================================================
def generate_pointwise_sets(n_node_mesh, dirichlet_bcs):
    """Generate Dirichlet sets data for ADiMU mesh history data files.
    
    This function generates Dirichlet sets data consistent with the force
    force equilibrium loss type \'pointwise\' in ADiMU.
    
    Parameters
    ----------
    n_node_mesh : int
        Number of nodes of finite element mesh.
    dirichlet_bcs : dict
        Dirichlet boundary conditions (item, dict) for each node with
        prescribed displacements (key, str[int]). The boundary conditions of
        each node consist of a prescribed displacement value (item, float) for
        the associated degree of freedom (key, str[int]).
    
    Returns
    -------
    dirichlet_bcs_mesh : numpy.ndarray(2d)
        Dirichlet boundary conditions sets of finite element mesh nodes stored
        as numpy.ndarray(2d) of shape (n_node_mesh, 3).
    """
    # Initialize Dirichlet boundary conditions data
    dirichlet_bcs_mesh = np.zeros((n_node_mesh, 3), dtype=int)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes with Dirichlet boundary conditions
    for node_id, node_dofs_bcs in dirichlet_bcs.items():
        # Get node index
        node_idx = int(node_id) - 1
        # Loop over node degrees of freedom
        for dof_str in node_dofs_bcs.keys():
            # Get degree of freedom index
            dof_idx = int(dof_str) - 1
            # Set Dirichlet boundary condition
            dirichlet_bcs_mesh[node_idx, dof_idx] = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dirichlet_bcs_mesh
# =============================================================================
def generate_dirichlet_sets(n_node_mesh):
    """Generate Dirichlet sets data for ADiMU mesh history data files.
    
    This function generates Dirichlet sets data consistent with the force
    force equilibrium loss type \'dirichlet_sets\' in ADiMU.
    
    Attention: Constrained node sets must be explicitly defined in this
    function according to the specific specimen geometry and boundary
    conditions.
    
    Parameters
    ----------
    n_node_mesh : int
        Number of nodes of finite element mesh.
    
    Returns
    -------
    dirichlet_bcs_mesh : numpy.ndarray(2d)
        Dirichlet boundary conditions sets of finite element mesh nodes stored
        as numpy.ndarray(2d) of shape (n_node_mesh, 3).
    """
    # Set constrained node sets
    node_sets = {}
    node_sets['1'] = {
        'nodes': [21, 22, 23, 24,
                  317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
                  329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                  341, 342, 343, 344, 1673, 1674, 1675, 1676, 1677, 1678,
                  1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688,
                  1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698,
                  1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
                  1709, 1710, 1711, 1712],
        'code': [1, 1, 1]}
    node_sets['2'] = {
        'nodes': [2, 3, 5, 6,
                  41, 42, 43, 44, 81, 82, 83, 84,
                  125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                  137, 138, 139, 140, 141, 142, 143, 144, 577, 578, 579, 580,
                  581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592,
                  593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604,
                  605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616],
        'code': [1, 1, 1]}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet constrained sets label
    set_label = 1
    # Initialize Dirichlet constrained sets data
    dirichlet_sets = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over node sets
    for _, node_set in node_sets.items():
        # Initialize node set Dirichlet set labels
        set_labels = []
        # Assign Dirichlet set labels
        for val in node_set['code']:
            if val == 1:
                # Assign Dirichlet constrained set label
                set_labels.append(set_label)
                # Increment Dirichlet constrained set label
                set_label += 1
            else:
                # Assign free set label
                set_labels.append(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over node set nodes
        for node in node_set['nodes']:
            # Check if node already has Dirichlet set assigned
            if str(node) in dirichlet_sets.keys():
                raise RuntimeError(f'Node {node} has been assigned to more '
                                   f'than one Dirichlet set.')
            # Assign node Dirichlet set
            dirichlet_sets[str(node)] = tuple(set_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet boundary constraints data
    dirichlet_bcs_mesh = np.zeros((n_node_mesh, 3), dtype=int)
    # Build Dirichlet boundary constraints data
    for node_label, node_dofs in dirichlet_sets.items():
        # Get node index
        node_idx = int(node_label) - 1
        # Assemble node Dirichlet boundary constraints
        dirichlet_bcs_mesh[node_idx, :] = node_dofs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dirichlet_bcs_mesh
# =============================================================================
def update_adimu_mesh_hist_data_files_dirichlet(specimen_history_paths,
                                                dirichlet_bcs_mesh):
    """Update ADiMU mesh history data files Dirichlet boundary conditions.
    
    This function updates the Dirichlet boundary conditions in the ADiMU mesh
    history data files with the new provided Dirichlet boundary conditions.
    
    Parameters
    ----------
    specimen_history_paths : tuple
        Specimen history time step files paths (.csv). Files paths must be
        sorted according to history time.
    dirichlet_bcs_mesh : numpy.ndarray(2d)
        Dirichlet boundary conditions sets of finite element mesh nodes stored
        as numpy.ndarray(2d) of shape (n_node_mesh, 3).
    """
    # Get number of nodes from first node data file
    n_node_mesh = pandas.read_csv(specimen_history_paths[0], sep='\s+',
                                  header=0).shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time step files
    for time_step_file_path in specimen_history_paths:
        # Load data
        df = pandas.read_csv(time_step_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data
        if df.shape[0] != n_node_mesh:
            raise RuntimeError(f'Mismatch between expected number of nodes '
                               f'({n_node_mesh}) and number of nodes in the '
                               f'time step data file ({df.shape[0]}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update Dirichlet boundary conditions
        df[['DBC1', 'DBC2', 'DBC3']] = dirichlet_bcs_mesh
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data
        if df.shape[1] != 13:
            raise RuntimeError(f'Expecting data frame to have 13 columns, but '
                               f'{df.shape[1]} were found. \n\n'
                               f'Expected columns: NODE | X1 X2 X3 | '
                               f'U1 U2 U3 | RF1 RF2 RF3 | DBC1 DBC2 DBC3')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store time step data into '.csv' file format
        df.to_csv(time_step_file_path, encoding='utf-8', index=False)
# =============================================================================
if __name__ == '__main__':
    # Set computation process
    process = ('generate_adimu_data_from_links',
               'update_adimu_mesh_hist_data_files_dirichlet')[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate ADiMU model discovery data from Links simulation data
    if process == 'generate_adimu_data_from_links':
        # Set Links input data file path
        links_input_file_path = (
            '/home/username/Documents/brown/projects/'
            'colaboration_antonios/dtp_validation/2_dtp1_j2_validation/'
            '2_debug_new_loss_type/links_boundary_conditions/links_simulation/'
            'force_equilibrium_loss_dirichlet_sets/0_links_simulation/'
            'Ti6242_HIP2_UT_Specimen2_J2.dat')
        # Set Links simulation output directory
        links_output_dir = (
            '/home/username/Documents/brown/projects/'
            'colaboration_antonios/dtp_validation/2_dtp1_j2_validation/'
            '2_debug_new_loss_type/links_boundary_conditions/links_simulation/'
            'force_equilibrium_loss_dirichlet_sets/0_links_simulation/'
            'Ti6242_HIP2_UT_Specimen2_J2')
        # Set data storage directory
        save_dir = (
            '/home/username/Documents/brown/projects/'
            'colaboration_antonios/dtp_validation/2_dtp1_j2_validation/'
            '2_debug_new_loss_type/links_boundary_conditions/links_simulation/'
            'force_equilibrium_loss_dirichlet_sets/0_links_simulation')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set ADiMU data storage directory
        adimu_save_dir = os.path.join(save_dir, f'adimu_data')
        # Create ADiMU mesh history data directory
        if os.path.isdir(adimu_save_dir):
            # Remove existing directory
            shutil.rmtree(adimu_save_dir)
        # Create directory
        os.makedirs(adimu_save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate ADiMU model discovery data from Links simulation data
        adimu_mesh_file_path, adimu_mesh_hist_data_dir = \
            gen_adimu_data_from_links(links_input_file_path, links_output_dir,
                                      adimu_save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif process == 'update_adimu_mesh_hist_data_files_dirichlet':
        # Set specimen name
        specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
        # Set force equilibrium loss type
        force_equilibrium_loss_type = ('pointwise', 'dirichlet_sets')[0]
        # Set ADiMU mesh history data directory
        specimen_history_dir = (
            '/home/username/Documents/brown/projects/'
            'colaboration_antonios/dtp_validation/2_dtp1_j2_validation/'
            '2_debug_new_loss_type/links_boundary_conditions/'
            'abaqus_simulation/force_equilibrium_pointwise/'
            '1_discover_rc_von_mises/material_model_finder/0_simulation/'
            'specimen_history_data')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get specimen history time step files paths
        specimen_history_paths = \
            get_specimen_history_paths(specimen_history_dir, specimen_name)
        # Get number of nodes from first node data file
        n_node_mesh = pandas.read_csv(specimen_history_paths[0], sep='\s+',
                                      header=0).shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Dirichlet boundary conditions sets data
        if force_equilibrium_loss_type == 'pointwise':
            # Set reference Links input data file path
            links_input_file_path = ''
            # Extract Dirichlet boundary conditions from Links input data file
            dirichlet_bcs = \
                extract_mesh_dirichlet_from_links_data(links_input_file_path)
            # Build Dirichlet boundary conditions sets
            dirichlet_bcs_mesh = \
                generate_pointwise_sets(n_node_mesh, dirichlet_bcs)
        elif force_equilibrium_loss_type == 'dirichlet_sets':
            # Build Dirichlet boundary conditions sets
            dirichlet_bcs_mesh = generate_dirichlet_sets(n_node_mesh, n_dim=3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update ADiMU mesh history data files Dirichlet boundary conditions
        update_adimu_mesh_hist_data_files_dirichlet(specimen_history_paths,
                                                    dirichlet_bcs_mesh)
    