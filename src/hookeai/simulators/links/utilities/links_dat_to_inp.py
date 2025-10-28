"""Convert Links '.dat' file to Abaqus-like '.inp' file.

Functions
---------
links_dat_to_abaqus_inp
    Convert Links input data file to Abaqus data input data file.
abaqus_inp_to_links_dat
    Convert Abaqus data input data file to Links input data file.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def links_dat_to_abaqus_inp(links_input_file_path):
    """Convert Links input data file to Abaqus data input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    abaqus_inp_file_path : str
        Abaqus input data file path.
    """
    # Check if Links simulation input data file exists
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(links_input_file_path))[0]
    # Get Links simulation input data file directory
    links_input_file_dir = os.path.dirname(links_input_file_path)
    # Set ABAQUS '.inp' data file path
    abaqus_inp_file_path = \
        os.path.join(links_input_file_dir, f'{simulation_name}.inp')
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
                raise RuntimeError('This script can only process meshes with '
                                   'a single element type.')
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing ELEMENT_TYPES section
            break
        elif is_keyword_found:
            # Get element type or number of Gauss integration points
            if bool(re.search(r'\bGP\b', line, re.IGNORECASE)):
                n_gauss = int(line.split()[0])
            else:
                links_elem_type = line.split()[1].upper()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of spatial dimensions
    if links_elem_type in ('TRI_3', 'TRI_6', 'QUAD_4', 'QUAD_8'):
        n_dim = 2
    else:
        n_dim = 3
    # Set Links to ABAQUS element type conversion
    elem_type_converter = {'TRI_3_1': 'CPE3', 'TRI3_1': 'CPE3',
                           'QUAD_4_4': 'CPE4', 'QUAD4_4': 'CPE4',
                           'TETRA_4_4': 'C3D4', 'TETRA4_4': 'C3D4',
                           'HEXA_8_8': 'C3D8', 'HEXA8_8': 'C3D8',
                           'HEXA_20_8': 'C3D20R', 'HEXA20_8': 'C3D20R'}
    # Get matching ABAQUS element type
    if f'{links_elem_type}_{n_gauss}' in elem_type_converter.keys():
        abaqus_elem_type = elem_type_converter[f'{links_elem_type}_{n_gauss}']
    else:
        raise RuntimeError('Unknown Links to ABAQUS element type conversion.')
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
    # Build node coordinates
    nodes_coords = np.array(nodes_coords_list)
    # Set number of nodes
    n_node = nodes_coords.shape[0]
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
    # Build element connectivities
    connectivities = np.array(connectivities_list)
    # Set number of elements
    n_elem = connectivities.shape[0]
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
            dirichlet_bc = {}
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
            dirichlet_bc[str(node_label)] = {}
            # Assemble node Dirichlet boundary conditions
            for i_dof in range(3):
                if node_code[i_dof] == 1:
                    dirichlet_bc[str(node_label)][str(i_dof + 1)] = \
                        node_vals[i_dof]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize ABAQUS '.inp' data file content
    write_lines = []
    # Set header
    write_lines += ['Automatically generated ABAQUS format input file' + '\n']
    # Set node coordinates
    write_lines += ['\n' + '*Node' + '\n']
    write_lines += \
        [f'{i:>3d}, '
         + ', '.join(f'{coord:16.8e}' for coord in nodes_coords[i - 1, :])
         + '\n' for i in range(1, n_node + 1)]
    # Set element connectivities
    write_lines += ['\n' + f'*Element, type={abaqus_elem_type}' + '\n']
    write_lines += \
        [f'{i:>3d}, '
         + ', '.join(f'{node:d}' for node in connectivities[i - 1, :]) + '\n'
         for i in range(1, n_elem + 1)]
    # Set Dirichlet boundary conditions
    write_lines += ['\n' + '*Boundary' + '\n']
    write_lines += \
        [f'{int(i):>3d}, {j}, {j}, {dirichlet_bc[i][j]}\n'
         for i in dirichlet_bc.keys() for j in dirichlet_bc[i].keys()]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open ABAQUS '.inp' data file
    abaqus_file = open(abaqus_inp_file_path, 'w')
    # Write ABAQUS '.inp' data file
    abaqus_file.writelines(write_lines)
    # Close ABAQUS '.inp' data file
    abaqus_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return abaqus_inp_file_path
# =============================================================================
def abaqus_inp_to_links_dat(specimen_inp_path, n_dim=3):
    """Convert Abaqus data input data file to Links input data file.

    Parameters
    ----------
    specimen_inp_path : str
        Specimen mesh input file path (.inp).
    n_dim : int, default=3
        Number of spatial dimensions.
        
    Returns
    -------
    abaqus_inp_file_path : str
        Abaqus input data file path.
    """
    # Set unique Links group ID (element type + material type)
    group_id = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get specimen name
    specimen_name = os.path.splitext(os.path.basename(specimen_inp_path))[0]
    # Open specimen mesh input file
    input_file = open(specimen_inp_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize nodes coordinates
    nodes_coords = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for NODE keyword and collect nodes coordinates
    for line in input_file:
        if bool(re.search(r'\*node\b', line, re.IGNORECASE)):
            # Start processing NODE section
            is_keyword_found = True
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing NODE section
            break
        elif is_keyword_found:
            # Get node coordinates
            nodes_coords.append(
                [float(x) for x in line.split(sep=',')[1:1+n_dim]])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check mesh nodes initial coordinates
    if len(nodes_coords) == 0:
        raise RuntimeError('The *NODE keyword has not been found in the '
                           'specimen mesh input file (.inp).')
    # Get total number of nodes
    n_node_mesh = len(nodes_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize elements connectivities
    connectivities = {}
    # Initialize connectivities nodes (validation only)
    connect_nodes = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for ELEMENT keyword and collect nodes connectivities
    for line in input_file:
        if bool(re.search(r'\*element, type\b', line, re.IGNORECASE)):
            # Start processing ELEMENT section
            is_keyword_found = True
            # Get ABAQUS element type
            elem_abaqus_type = \
                re.search(r'type=([A-Z0-9]+)', line, re.IGNORECASE).groups()[0]
            # Get Links element type
            if elem_abaqus_type is None:
                raise RuntimeError('The *ELEMENT keyword TYPE parameter has '
                                   'not been found in the specimen mesh input '
                                   'file (.inp).')
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing ELEMENT section
            is_keyword_found = False
        elif is_keyword_found:
            # Get element label and nodes
            elem_id = int(line.split(sep=',')[0])
            elem_nodes = tuple([int(x) for x in line.split(sep=',')[1:]])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check element nodes
            for node in elem_nodes:
                if node not in range(1, n_node_mesh + 1):
                    raise RuntimeError(f'Invalid node in element {elem_id} '
                                       f'connectivities. Node labels must be '
                                       f'within the range 1 to {n_node_mesh} '
                                       f'for the provided mesh.')
            # Store element nodes
            connectivities[str(elem_id)] = elem_nodes
            connect_nodes += list(elem_nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check nodes connectivities
    if set(range(1, n_node_mesh + 1)) != set(connect_nodes):
        raise RuntimeError(f'Invalid mesh connectivities. Mesh has '
                           f'{(n_node_mesh)} nodes, but only '
                           f'{len(set(connect_nodes))} nodes are part of the '
                           f'elements connectivities.')
    # Get total number of elements
    n_element = len(connectivities.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet boundary conditions
    nodes_dirichlet_bc = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for BOUNDARY keyword and collect nodes constraints
    for line in input_file:
        if bool(re.search(r'\*boundary\b', line, re.IGNORECASE)):
            # Start processing BOUNDARY section
            is_keyword_found = True
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing BOUNDARY section
            is_keyword_found = False
        elif is_keyword_found:
            # Get Dirichlet boundary condition data
            dirichlet_bc = line.split(sep=',')
            # Get node label
            node_id = int(dirichlet_bc[0])
            # Get prescribed degrees of freedom
            node_dofs = tuple([int(x) for x in dirichlet_bc[1:3]])
            # Get prescribed displacement
            dirichlet_disp = float(dirichlet_bc[3])
            # Initialize node Dirichlet boundary condition data
            if str(node_id) not in nodes_dirichlet_bc.keys():
                nodes_dirichlet_bc[str(node_id)] = {}
            # Store node Dirichlet boundary condition data
            for dim in range(node_dofs[0] - 1, node_dofs[1]):
                nodes_dirichlet_bc[str(node_id)][str(dim)] = dirichlet_disp
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get total number of nodes with Dirichlet boundary conditions
    n_node_dirichlet = len(nodes_dirichlet_bc.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links input data file path
    links_file_path = os.path.join(os.path.dirname(specimen_inp_path),
                                   f'{specimen_name}_links.dat')
    # Open Links input data file
    links_file = open(links_file_path, 'w')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Links input file data
    write_lines = []
    # Append elements connectivities to Links input file data
    write_lines += \
        ['\nELEMENTS ' + str(n_element) + '\n'] \
        + ['{:>3s}'.format(str(elem))
            + '{:^5d}'.format(group_id)
            + ' '.join([str(node) for node in connectivities[str(elem)]])
            + '\n' for elem in range(1, n_element + 1)] 
    # Append node coordinates to Links input file data
    write_lines += \
        ['\nNODE_COORDINATES ' + str(n_node_mesh) + ' CARTESIAN' + '\n'] \
        + ['{:>3s}'.format(str(i + 1)) + ' '
           + ' '.join([str('{:16.8e}'.format(coord)) \
                       for coord in nodes_coords[i]]) + '\n' \
            for i in range(n_node_mesh)]
    # Append nodes Dirichlet boundary conditions
    write_lines += ['\nNODES_WITH_PRESCRIBED_DISPLACEMENTS ' \
                    + str(n_node_dirichlet) + '\n']
    nodes_id_sorted = sorted([int(x) for x in nodes_dirichlet_bc.keys()])
    for node_id in nodes_id_sorted:
        # Get node Dirichlet boundary conditions
        node_dirichlet_bc = nodes_dirichlet_bc[str(node_id)]
        # Get Dirichlet code
        dirichlet_code = [1 if str(dim) in node_dirichlet_bc.keys() else 0
                          for dim in range(3)]
        # Get Dirichlet displacements
        dirichlet_disps = [node_dirichlet_bc[str(dim)] if str(dim) in
                           node_dirichlet_bc.keys() else 0.0
                           for dim in range(3)]
        # Add required angle (2D only)
        if n_dim == 2:
            dirichlet_disps.append(0.0)
        # Append node Dirichlet boundary conditions
        write_lines += \
            [' ' + '{:>3s}'.format(str(node_id)) + ' '
             + ''.join(['{:^1d}'.format(i) for i in dirichlet_code]) \
             + ' '.join(['{:16.8e}'.format(val) for val in dirichlet_disps])
             + '\n']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    # Write Links input data file
    links_file.writelines(write_lines)
    # Close Links input data file
    links_file.close()
# =============================================================================
if __name__ == "__main__":
    # Set Links simulation input data file
    links_input_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                            'darpa_project/8_global_random_specimen/von_mises/'
                            '2_random_specimen_hexa8/'
                            'voids_lattice/'
                            '0_links_simulation/random_specimen/simulations/'
                            'random_specimen.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert Links input data file to Abaqus data input data file
    _ = links_dat_to_abaqus_inp(links_input_file_path)



