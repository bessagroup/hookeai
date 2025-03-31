# Standard
import os
import re
# Third-party
import numpy as np
# =============================================================================
# Summary: Generate specimen Dirichlet boundary conditions from Links mesh
# =============================================================================
def build_specimen_dbc(specimen_label, specimen_top_coord, specimen_top_disp,
                       nodes_coords_mesh):
    """Build specimen Dirichlet boundary conditions.
    
    Parameters
    ----------
    specimen_label : str
        Specimen name.
    specimen_top_coord : float
        Coordinate of the specimen top along uniaxial loading dimension.
    specimen_top_disp : float
        Prescribed displacement applied to the top of the specimen along
        uniaxial loading dimension.
    nodes_coords_mesh : np.ndarray(2d)
        Coordinates of finite element mesh nodes stored as numpy.ndarray(2d) of
        shape (n_node_mesh, n_dim).

    Returns
    -------
    dbc_array : np.ndarray(2d)
        Finite element mesh nodes Dirichlet boundary conditions stored as
        numpy.ndarray(2d) of shape (n_node_dbc, 7), where n_node_dbc is the
        number of nodes with Dirichlet boundary conditions, array[:, 0] is the
        node label, array[:, 1:4] is the Dirichlet boundary conditions
        prescription code, and array[:, 5:8] are the Dirichlet boundary
        conditions prescribed displacements.
    """
    # Get number of nodes
    n_node = nodes_coords_mesh.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet boundary conditions
    dbc_list = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for i in range(n_node):
        # Set node label
        node_label = i + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get node Dirichlet boundary conditions
        node_dbc_code, node_dbc_disps = get_specimen_node_dbc(
            specimen_label, specimen_top_coord, specimen_top_disp,
            nodes_coords_mesh[i, :])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Skip if node without Dirichlet boundary conditions
        if np.all(node_dbc_code == 0):
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store node Dirichlet boundary conditions
        dbc_list.append(np.concatenate(
            (np.array(node_label).reshape(-1), node_dbc_code, node_dbc_disps)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Dirichlet boundary conditions array
    dbc_array = np.array(dbc_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dbc_array
# =============================================================================
def get_specimen_node_dbc(specimen_label, top_coord, top_disp, coords):
    """Get node Dirichlet boundary conditions for given uniaxial specimen.
    
    This function is highly specific for particular specimens models and
    uniaxial loading by means of Dirichlet boundary conditions.
    
    Parameters
    ----------
    specimen_label : str
        Specimen name.
    coords : np.ndarray(1d)
        Node coordinates.
    top_coord : float
        Coordinate of the specimen top along uniaxial loading dimension.
    top_disp : float
        Prescribed uniaxial displacement applied to the top of the specimen.
        
    Returns
    -------
    node_dbc_code : np.ndarray(1d)
        Node Dirichlet boundary conditions prescription code stored as
        numpy.ndarray[int](1d) with shape (3,).
    node_dbc_disps : np.ndarray(1d)
        Node Dirichlet boundary conditions prescribed displacements stored as
        numpy.ndarray[float](1d) with shape (3,).
    """
    # Set minimum threshold to handle values close or equal to zero
    small = 1e-8
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set available specimens
    available_specimens = ('3d_tensile_double_notched',)
    # Check specimen
    if specimen_label not in available_specimens:
        raise RuntimeError('Unknown specimen.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check number of dimensions
    if not isinstance(coords, np.ndarray) or len(coords.shape) != 1:
        raise RuntimeError('Invalid node coordinates array.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize node Dirichlet boundary conditions
    node_dbc_code = np.array([0, 0, 0])
    node_dbc_disps = build_node_disps()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build node Dirichlet boundary conditions
    if specimen_label in ('3d_tensile_double_notched',):
        # Set loading dimension
        loading_dim = 1
        # Get node Dirichlet boundary conditions from coordinates:
        # Corners (2)
        if np.all(np.abs(coords) < small):
            node_dbc_code = np.array([1, 1, 1])
            node_dbc_disps = build_node_disps()
        elif (np.abs(coords[0]) < small
              and np.abs(coords[1] - top_coord) < small
              and np.abs(coords[2]) < small):
            node_dbc_code = np.array([1, 1, 1])
            node_dbc_disps = build_node_disps(value=top_disp, dim=loading_dim)
        # Edges (5)
        elif np.all(np.abs(coords[[0, 1]]) < small):
            node_dbc_code = np.array([1, 1, 0])
            node_dbc_disps = build_node_disps()
        elif np.all(np.abs(coords[[1, 2]]) < small):
            node_dbc_code = np.array([0, 1, 1])
            node_dbc_disps = build_node_disps()
        elif np.all(np.abs(coords[[0, 2]]) < small):
            node_dbc_code = np.array([1, 0, 1])
            node_dbc_disps = build_node_disps()
        elif (np.abs(coords[0]) < small
              and np.abs(coords[1] - top_coord) < small):
            node_dbc_code = np.array([1, 1, 0])
            node_dbc_disps = build_node_disps(value=top_disp, dim=loading_dim)
        elif (np.abs(coords[1] - top_coord) < small
              and np.abs(coords[2]) < small):
            node_dbc_code = np.array([0, 1, 1])
            node_dbc_disps = build_node_disps(value=top_disp, dim=loading_dim)
        # Faces (4)
        elif np.abs(coords[1]) < small:
            node_dbc_code = np.array([0, 1, 0])
            node_dbc_disps = build_node_disps()
        elif np.abs(coords[0]) < small:
            node_dbc_code = np.array([1, 0, 0])
            node_dbc_disps = build_node_disps()
        elif np.abs(coords[2]) < small:
            node_dbc_code = np.array([0, 0, 1])
            node_dbc_disps = build_node_disps()
        elif np.abs(coords[1] - top_coord) < small:
            node_dbc_code = np.array([0, 1, 0])
            node_dbc_disps = build_node_disps(value=top_disp, dim=loading_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return node_dbc_code, node_dbc_disps
# =============================================================================
def build_node_disps(value=None, dim=1):
    """Build node displacements array.
    
    Parameters
    ----------
    value : float, default=None
        Prescribed displacement along loading dimension. If None, then defaults
        to null displacement.
    dim : int, default=1
        Dimension where prescribed displacement is applied.
    
    Returns
    -------
    node_disps : np.ndarray(1d)
        Node displacements stored as numpy.ndarray(1d) of shape(3,).
    """
    # Initialize node displacements
    node_dbc_disps = np.zeros(3)
    # Enforce non-null prescribed displacement along loading dimension
    if value is not None:
        node_dbc_disps[dim] = value
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return node_dbc_disps
# =============================================================================
def write_links_dirichlet_bc_file(dbc_file_dir, dbc_array,
                                  is_reaction_group=True):
    """Write Dirichlet boundary conditions file (Links format).
    
    Parameters
    ----------
    dbc_file_dir : str
        Directory where Dirichlet boundary conditions file is stored.
    dbc_array : np.ndarray(2d)
        Finite element mesh nodes Dirichlet boundary conditions stored as
        numpy.ndarray(2d) of shape (n_node_dbc, 7), where n_node_dbc is the
        number of nodes with Dirichlet boundary conditions, array[:, 0] is the
        node label, array[:, 1:4] is the Dirichlet boundary conditions
        prescription code, and array[:, 5:8] are the Dirichlet boundary
        conditions prescribed displacements.
    is_reaction_group : bool, default=True
        If True, then write reaction node group with prescribed non-null
        Dirichlet boundary conditions.
    """
    # Check directory
    if not os.path.isdir(dbc_file_dir):
        raise RuntimeError('The directory where the Dirichlet boundary '
                           'conditions file is to be stored has not been '
                           'found:\n\n', dbc_file_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes with Dirichlet boundary conditions
    n_node_dbc = dbc_array.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize file data
    write_lines = [f'NODES_WITH_PRESCRIBED_DISPLACEMENTS {n_node_dbc}' + '\n']
    # Loop over Dirichlet boundary conditions nodes
    for i in range(n_node_dbc):
        # Get node label, prescription code and prescribed displacements
        node_label = '{:>4d}'.format(int(dbc_array[i, 0]))
        node_dbc_code = \
            ''.join(['{:>1d}'.format(int(x)) for x in dbc_array[i, 1:4]])
        node_dbc_disps = \
            ''.join(['{:11.4e}'.format(x) for x in dbc_array[i, 4:7]])
        # Append node to file data
        write_lines += \
            [f'  {node_label} {node_dbc_code} {node_dbc_disps}' + '\n']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reaction node group associated with prescribed non-null Dirichlet
    # boundary conditions
    if is_reaction_group:
        # Set minimum threshold to handle values close or equal to zero
        small = 1e-8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize reaction node group
        reaction_nodes = []
        # Loop over Dirichlet boundary conditions nodes
        for i in range(n_node_dbc):
            # Collect node label if non-null Dirichlet boundary condition
            if np.any(dbc_array[i, 4:7] > small):
                reaction_nodes.append(int(dbc_array[i, 0]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append reaction node group to file data
        write_lines += ([f'\nNODE_GROUPS 1' + '\n',
                         f'1 {len(reaction_nodes)}' + '\n']
                        + ['{:d}'.format(x) + '\n' for x in reaction_nodes])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Dirichlet boundary conditions file path
    dbc_file_path = os.path.join(dbc_file_dir, 'specimen_dirichlet_bc.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write Dirichlet boundary conditions data file
    with open(dbc_file_path, 'w') as dbc_file:
        # Write Dirichlet boundary conditions data file
        dbc_file.writelines(write_lines)
        # Close data file
        dbc_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dbc_file
# =============================================================================
def read_links_nodes_coords_mesh(links_input_file_path):
    """Read finite element mesh nodes coordinates from Links input data file.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
        
    Returns
    -------
    nodes_coords_mesh : np.ndarray(2d)
        Coordinates of finite element mesh nodes stored as numpy.ndarray(2d) of
        shape (n_node_mesh, n_dim).
    """
    # Check if Links simulation input data file exists
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flag
    is_keyword_found = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Search for NODE_COORDINATES keyword and collect nodes coordinates
    for line in links_input_file:
        if bool(re.search(r'NODE_COORDINATES\b', line, re.IGNORECASE)):
            # Start processing NODE_COORDINATES section
            is_keyword_found = True
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
    # Build finite element mesh nodes coordinates
    nodes_coords_mesh = np.array(nodes_coords_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_mesh
# =============================================================================
if __name__ == "__main__":
    # Set specimen label
    specimen_label = '3d_tensile_double_notched'
    # Set storage directory
    save_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'test_specimen_bc_auto')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen parameters
    if specimen_label == '3d_tensile_double_notched':
        # Set specimen Links input data file
        specimen_links_data_file = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'test_specimen_bc_auto/'
             'cambridge_specimen_plane_stress_notched_short.dat')
        # Set specimen top coordinate
        specimen_top_coord = 12.0
        # Set prescribed displacement
        specimen_top_disp = 1.0
    else:
        raise RuntimeError('Unknown specimen.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read specimen finite element mesh nodes coordinates
    nodes_coords_mesh = read_links_nodes_coords_mesh(specimen_links_data_file)
    # Build specimen Dirichlet boundary conditions
    dbc_array = build_specimen_dbc(specimen_label, specimen_top_coord,
                                   specimen_top_disp, nodes_coords_mesh)
    # Write Dirichlet boundary conditions file
    write_links_dirichlet_bc_file(save_dir, dbc_array, is_reaction_group=True)
    