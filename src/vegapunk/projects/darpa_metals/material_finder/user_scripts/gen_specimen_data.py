"""DARPA METALS PROJECT: Generate specimen data and material state files.

Functions
---------
gen_specimen_dataset
    Generate specimen data and material state files (training data set).
get_specimen_mesh_from_inp_file
    Get specimen finite element mesh data from mesh input file (.inp).
elem_type_abaqus_to_fetorch
    Convert ABAQUS element type to FETorch element type object.
get_specimen_history_paths
    Get specimen history time step file paths from directory.
get_specimen_numerical_data
    Get specimen numerical data from history data files (.csv).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import math
import random
# Third-party
import torch
import pandas
import numpy as np
# Local
from simulators.fetorch.element.type.tri3 import FETri3
from simulators.fetorch.element.type.quad4 import FEQuad4
from simulators.fetorch.element.type.tetra4 import FETetra4
from simulators.fetorch.element.type.hexa8 import FEHexa8
from simulators.fetorch.element.type.hexa20 import FEHexa20
from simulators.fetorch.structure.structure_state import StructureMaterialState
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from material_model_finder.data.specimen_data import SpecimenNumericalData
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def gen_specimen_dataset(specimen_name, specimen_raw_dir, specimen_inp_path,
                         specimen_history_paths, strain_formulation,
                         problem_type, n_dim, model_name, model_parameters,
                         model_kwargs, training_dataset_dir,
                         is_save_specimen_data=True,
                         is_save_specimen_material_state=True):
    """Generate specimen data and material state files (training data set).
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    specimen_raw_dir : str
        Specimen raw data directory.
    specimen_inp_path : str
        Specimen mesh input file path (.inp).
    specimen_history_paths : tuple
        Specimen history time step files paths (.csv). Files paths must be
        sorted according to history time.
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    n_dim : int
        Number of spatial dimensions.
    model_name : str
        Material constitutive model name.
    model_parameters : dict
        Material constitutive model parameters.
    model_kwargs : dict, default={}
        Other parameters required to initialize constitutive model.
    training_dataset_dir : str
        Specimen training data set directory.
    is_save_specimen_data : bool, default=True
        If True, then save the specimen data file.
    is_save_specimen_material_state : bool, default=True
        If True, then save the specimen material state file.
        
    Returns
    -------
    specimen_data_path : str
        Path of file containing the specimen numerical data translated from
        experimental results.
    specimen_material_state_path : str
        Path of file containing the FETorch specimen material state.
    """
    # Initialize specimen numerical data
    specimen_data = SpecimenNumericalData()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get specimen finite element mesh data
    nodes_coords_mesh_init, elements_type, connectivities, \
        dirichlet_bool_mesh = \
            get_specimen_mesh_from_inp_file(specimen_inp_path, n_dim)
    # Set specimen finite element mesh
    specimen_data.set_specimen_mesh(nodes_coords_mesh_init, elements_type,
                                    connectivities, dirichlet_bool_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get finite element mesh
    specimen_mesh = specimen_data.specimen_mesh
    # Get number of nodes and elements
    n_node_mesh = specimen_mesh.get_n_node_mesh()
    n_elem = specimen_mesh.get_n_elem()
    # Get type of elements of finite element mesh
    elements_type = specimen_mesh.get_elements_type()
    # Get elements labels
    elements_ids = tuple([elem_id for elem_id in range(1, n_elem + 1)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get specimen numerical data
    nodes_disps_mesh_hist, reaction_forces_mesh_hist, time_hist = \
        get_specimen_numerical_data(specimen_history_paths, n_dim, n_node_mesh)
    # Set specimen numerical data
    specimen_data.set_specimen_data(nodes_disps_mesh_hist,
                                    reaction_forces_mesh_hist, time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save specimen data
    if is_save_specimen_data:
        # Set specimen numerical data file path
        specimen_data_path = os.path.join(
            os.path.normpath(training_dataset_dir), 'specimen_data.pt')
        # Save specimen data
        torch.save(specimen_data, specimen_data_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize specimen material state
    if is_save_specimen_material_state:
        # Initialize specimen material state
        specimen_material_state = StructureMaterialState(
            strain_formulation, problem_type, n_elem)
        # Initialize elements constitutive model
        specimen_material_state.init_elements_model(
            model_name, model_parameters, elements_ids, elements_type,
            model_kwargs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen material state file path
        specimen_material_state_path = \
            os.path.join(os.path.normpath(training_dataset_dir),
                         'specimen_material_state.pt')
        # Save specimen material state
        torch.save(specimen_material_state, specimen_material_state_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return specimen_data_path, specimen_material_state_path
# =============================================================================
def get_specimen_mesh_from_inp_file(specimen_inp_path, n_dim):
    """Get specimen finite element mesh data from mesh input file (.inp).
    
    Parameters
    ----------
    specimen_inp_path : str
        Specimen mesh input file path (.inp).
    n_dim : int
        Number of spatial dimensions.
        
    Returns
    -------
    nodes_coords_mesh_init : torch.Tensor(2d)
        Initial coordinates of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    elements_type : dict
        FETorch element type (item, ElementType) of each finite element
        mesh element (str[int]). Elements labels must be within the range
        of 1 to n_elem (included).
    connectivities : dict
        Nodes (item, tuple[int]) of each finite element mesh element
        (key, str[int]). Nodes must be within the range of 1 to n_node_mesh
        (included). Elements labels must be within the range of 1 to n_elem
        (included).
    dirichlet_bool_mesh : torch.Tensor(2d)
        Degrees of freedom of finite element mesh subject to Dirichlet
        boundary conditions. Stored as torch.Tensor(2d) of shape
        (n_node_mesh, n_dim) where constrained degrees of freedom are
        labeled 1, otherwise 0.
    """
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
    # Build mesh nodes initial coordinates
    if len(nodes_coords) == 0:
        raise RuntimeError('The *NODE keyword has not been found in the '
                           'specimen mesh input file (.inp).')
    else:
        nodes_coords_mesh_init = torch.tensor(nodes_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes
    n_node_mesh = nodes_coords_mesh_init.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize elements types
    elements_type = {}
    # Initialize elements connectivities
    connectivities = {}
    # Initialize connectivities nodes
    connect_nodes = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for ELEMENT keyword and collect nodes connectivities
    for line in input_file:
        #if '*ELEMENT, TYPE' in line or '*Element, type' in line:
        if bool(re.search(r'\*element, type\b', line, re.IGNORECASE)):
            # Start processing ELEMENT section
            is_keyword_found = True
            # Get ABAQUS element type
            elem_abaqus_type = \
                re.search(r'type=([A-Z0-9]+)', line, re.IGNORECASE).groups()[0]
            # Get FETorch element type
            if elem_abaqus_type is None:
                raise RuntimeError('The *ELEMENT keyword TYPE parameter has '
                                   'not been found in the specimen mesh input '
                                   'file (.inp).')
            else:
                elem_type = elem_type_abaqus_to_fetorch(elem_abaqus_type)
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing ELEMENT section
            is_keyword_found = False
        elif is_keyword_found:
            # Get element label and nodes
            elem_id = int(line.split(sep=',')[0])
            elem_nodes = tuple([int(x) for x in line.split(sep=',')[1:]])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store element type
            elements_type[str(elem_id)] = elem_type
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet bounds conditions
    dirichlet_bool_mesh = torch.zeros((n_node_mesh, n_dim))
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
            # Get node label and constraints
            node_id = int(line.split(sep=',')[0])
            node_dofs = tuple([int(x) for x in line.split(sep=',')[1:3]])
            # Store node constrained degrees of freedom
            for dim in range(node_dofs[0] - 1, node_dofs[1]):
                if dim in range(n_dim):
                    dirichlet_bool_mesh[node_id - 1, dim] = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_mesh_init, elements_type, connectivities, \
        dirichlet_bool_mesh
# =============================================================================
def elem_type_abaqus_to_fetorch(elem_abaqus_type):
    """Convert ABAQUS element type to FETorch element type object.
    
    Parameters
    ----------
    elem_abaqus_type : str
        ABAQUS element type.
        
    Returns
    -------
    elem_fetorch_type : ElementType
        FETorch element type.
    """
    if elem_abaqus_type in ('CPE3',):
        elem_fetorch_type = FETri3(n_gauss=1)
    elif elem_abaqus_type in ('CPE4',):
        elem_fetorch_type = FEQuad4(n_gauss=4)
    elif elem_abaqus_type in ('C3D4',):
        elem_fetorch_type = FETetra4(n_gauss=4)
    elif elem_abaqus_type in ('C3D8',):
        elem_fetorch_type = FEHexa8(n_gauss=8)
    elif elem_abaqus_type in ('C3D20R',):
        elem_fetorch_type = FEHexa20(n_gauss=8)
    else:
        raise RuntimeError(f'The ABAQUS element type {elem_abaqus_type} '
                           f'is unknown or cannot be converted to a FETorch '
                           f'element type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return elem_fetorch_type
# =============================================================================
def get_specimen_history_paths(specimen_history_dir, specimen_name):
    """Get specimen history time step file paths from directory.
    
    Parameters
    ----------
    specimen_history_dir : str
        Specimen history data directory.
    specimen_name : str
        Specimen name.

    Returns
    -------
    specimen_history_paths : tuple
        Specimen history time step files paths (.csv). Files paths are sorted
        according to history time.
    """
    # Check specimen history data directory
    if not os.path.isdir(specimen_history_dir):
        raise RuntimeError('The specimen history data directory has not been '
                           'found:\n\n' + specimen_history_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize specimen history data file paths (.csv files)
    specimen_history_paths = []
    # Get files in specimen history data directory
    directory_list = os.listdir(specimen_history_dir)
    # Loop over files
    for filename in directory_list:
        # Check if file is specimen history time step file
        is_time_step_file = bool(
            re.search(r'^' + specimen_name + r'_tstep_[0-9]+' + r'\.csv',
                      filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append specimen history time step file
        if is_time_step_file:
            specimen_history_paths.append(
                os.path.join(os.path.normpath(specimen_history_dir), filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check specimen history time step files
    if len(specimen_history_paths) == 0:
        raise RuntimeError('No specimen history time step files (.csv) have '
                           'been found in the specimen history data '
                           'directory:\n\n' + specimen_history_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort specimen history time step files
    specimen_history_paths = tuple(
        sorted(specimen_history_paths,
               key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return specimen_history_paths
# =============================================================================
def get_specimen_numerical_data(specimen_history_paths, n_dim, n_node_mesh):
    """Get specimen numerical data from history data files (.csv).
    
    Parameters
    ----------
    specimen_history_paths : tuple
        Specimen history time step files paths (.csv). Files paths must be
        sorted according to history time.
    n_dim : int
        Number of spatial dimensions.
    n_node_mesh : int
        Number of nodes of finite element mesh.
        
    Returns
    -------
    nodes_disps_mesh_hist : torch.Tensor(3d)
        Displacements history of finite element mesh nodes stored as
        torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
    reaction_forces_mesh_hist : torch.Tensor(3d)
        Reaction forces (Dirichlet boundary conditions) history of finite
        element mesh nodes stored as torch.Tensor(3d) of shape
        (n_node_mesh, n_dim, n_time).
    time_hist : torch.Tensor(1d)
        Discrete time history.
    """
    # Get number of time steps
    n_time = len(specimen_history_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize node data
    node_data = torch.zeros((n_node_mesh, 10, n_time))
    # Loop over time step files
    for i, data_file_path in enumerate(specimen_history_paths):
        # Load data
        df = pandas.read_csv(data_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data
        if df.shape[0] != n_node_mesh:
            raise RuntimeError(f'Mismatch between expected number of nodes '
                               f'({n_node_mesh}) and number of nodes in the '
                               f'time step data file ({df.shape[0]}).')
        if df.shape[1] != 10:
            raise RuntimeError(f'Expecting data frame to have 10 columns, but '
                               f'{df.shape[1]} were found. \n\n'
                               f'Expected columns: NODE | X1 X2 X3 | '
                               f'U1 U2 U3 | U1 U2 U3 | RF1 RF2 RF3')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store time step data
        node_data[:, :, i] = torch.tensor(df.values)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract displacements history
    nodes_disps_mesh_hist = node_data[:, 4:4+n_dim, :]
    # Extract reaction forces history
    reaction_forces_mesh_hist = node_data[:, 7:7+n_dim, :]
    # Build time discrete history
    time_hist = torch.linspace(0, n_time - 1, steps=n_time)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_disps_mesh_hist, reaction_forces_mesh_hist, time_hist
# =============================================================================
def set_material_model_parameters():
    """Set material model parameters.

    Returns
    -------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    n_dim : int
        Number of spatial dimensions.
    model_name : str
        Material constitutive model name.
    model_parameters : dict
        Material constitutive model parameters.
    model_kwargs : dict, default={}
        Other parameters required to initialize constitutive model.
    """
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Get problem type parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model name
    model_name = 'rc_lou_zhang_yoon_vmap'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model name parameters
    if bool(re.search(r'^rc_.*$', model_name)):
        # Set constitutive model specific parameters
        if model_name == 'rc_elastic':
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 100, 'v': 0.3,
                'euler_angles': (0.0, 0.0, 0.0)}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set learnable parameters
            learnable_parameters = {}
            learnable_parameters['E'] = {'initial_value': 70.0,
                                         'bounds': (50, 150)}
            #learnable_parameters['v'] = {'initial_value': 0.25,
            #                             'bounds': (0.2, 0.4)}
            # Set material constitutive model name
            material_model_name = 'elastic'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name in ('rc_von_mises', 'rc_von_mises_vmap'):            
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 110e3, 'v': 0.33,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),
                'hardening_parameters':
                    {'s0': 900,
                     'a': 700,
                     'b': 0.5,
                     'ep0': 1e-5}}
            # Set constitutive model parameters (Rowan data)
            #model_parameters = {
            #    'elastic_symmetry': 'isotropic',
            #    'E': 118640, 'v': 0.334,
            #    'euler_angles': (0.0, 0.0, 0.0),
            #    'hardening_law': get_hardening_law('nadai_ludwik'),
            #    'hardening_parameters':
            #        {'s0': 855.89,
            #         'a': 1246.9,
            #         'b': 0.0795,
            #         'ep0': 1e-5}}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set learnable parameters
            learnable_parameters = {}
            learnable_parameters['E'] = {
                'initial_value': random.uniform(80e3, 140e3),
                'bounds': (80e3, 140e3)}
            learnable_parameters['v'] = {
                'initial_value': random.uniform(0.2, 0.4),
                'bounds': (0.2, 0.4)}
            learnable_parameters['s0'] = {
                'initial_value': random.uniform(500, 1500),
                'bounds': (500, 1500)}
            learnable_parameters['a'] = {
                'initial_value': random.uniform(500, 1500),
                'bounds': (500, 1500)}
            # Set material constitutive model name
            if model_name == 'rc_von_mises_vmap':
                material_model_name = 'von_mises_vmap'
            else:
                material_model_name = 'von_mises'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name in ('rc_drucker_prager', 'rc_drucker_prager_vmap'):            
            # Set frictional angle
            friction_angle = math.radians(5)
            # Set dilatancy angle
            dilatancy_angle = friction_angle
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute angle-related material parameters
            # (matching with Mohr-Coulomb under uniaxial tension and
            # compression)
            # Set yield surface cohesion parameter
            yield_cohesion_parameter = \
                (2.0/math.sqrt(3))*math.cos(friction_angle)
            # Set yield pressure parameter
            yield_pressure_parameter = \
                (3.0/math.sqrt(3))*math.sin(friction_angle)
            # Set plastic flow pressure parameter
            flow_pressure_parameter = \
                (3.0/math.sqrt(3))*math.sin(dilatancy_angle)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 110000, 'v': 0.33,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),
                'hardening_parameters':
                    {'s0': 900/(math.sqrt(3)*yield_cohesion_parameter),
                     'a': 700/(math.sqrt(3)*yield_cohesion_parameter),
                     'b': 0.5,
                     'ep0': 1e-5},
                'yield_cohesion_parameter': yield_cohesion_parameter,
                'yield_pressure_parameter': yield_pressure_parameter,
                'flow_pressure_parameter': flow_pressure_parameter,
                'friction_angle': friction_angle}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set learnable parameters
            learnable_parameters = {}
            #learnable_parameters['E'] = {
            #    'initial_value': random.uniform(80e3, 140e3),
            #    'bounds': (80e3, 140e3)}
            #learnable_parameters['v'] = {
            #    'initial_value': random.uniform(0.2, 0.4),
            #    'bounds': (0.2, 0.4)}
            #learnable_parameters['s0'] = {
            #    'initial_value': random.uniform(500, 1500),
            #    'bounds': (500, 1500)}
            #learnable_parameters['a'] = {
            #    'initial_value': random.uniform(500, 1500),
            #    'bounds': (500, 1500)}
            learnable_parameters['friction_angle'] = \
                {'initial_value': math.radians(random.uniform(0.1, 10.0)),
                 'bounds': (math.radians(1.0), math.radians(10.0))}
            # Set material constitutive model name
            if model_name == 'rc_drucker_prager_vmap':
                material_model_name = 'drucker_prager_vmap'
            else:
                material_model_name = 'drucker_prager'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name in ('rc_lou_zhang_yoon', 'rc_lou_zhang_yoon_vmap'):
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 110e3, 'v': 0.33,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),
                'hardening_parameters': {'s0': 900,
                                         'a': 700,
                                         'b': 0.5,
                                         'ep0': 1e-5},
                'a_hardening_law': get_hardening_law('linear'),
                'a_hardening_parameters': {'s0': 1.0,
                                           'a': 0},
                'b_hardening_law': get_hardening_law('linear'),
                'b_hardening_parameters': {'s0': 0.05,
                                           'a': 0},
                'c_hardening_law': get_hardening_law('linear'),
                'c_hardening_parameters': {'s0': 1.0,
                                           'a': 0},
                'd_hardening_law': get_hardening_law('linear'),
                'd_hardening_parameters': {'s0': 0.25,
                                           'a': 0},
                'is_associative_hardening': True}
            # Set learnable parameters
            learnable_parameters = {}
            learnable_parameters['yield_a_s0'] = \
                {'initial_value': random.uniform(0.5, 1.5),
                 'bounds': (0.5, 1.5)}
            learnable_parameters['yield_b_s0'] = \
                {'initial_value': random.uniform(0, 0.1),
                 'bounds': (0, 0.1)}
            learnable_parameters['yield_c_s0'] = \
                {'initial_value': random.uniform(-3.5, 2.5),
                 'bounds': (-3.5, 2.5)}
            learnable_parameters['yield_d_s0'] = \
                {'initial_value': random.uniform(-1.5, 1.5),
                 'bounds': (-1.5, 1.5)}
            # Set material constitutive model name
            if model_name == 'rc_lou_zhang_yoon_vmap':
                material_model_name = 'lou_zhang_yoon_vmap'
            else:
                material_model_name = 'lou_zhang_yoon'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown recurrent constitutive model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set automatic synchronization of material model parameters
        if bool(re.search(r'_vmap$', model_name)):
            is_auto_sync_parameters = False
        else:
            is_auto_sync_parameters = True
        # Set state update failure checking flag
        is_check_su_fail = False
        # Set parameters normalization
        is_normalized_parameters = True
        # Set model input and output features normalization
        is_model_in_normalized = False
        is_model_out_normalized = False
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set other parameters required to initialize constitutive model
        model_kwargs = {
            'n_features_in': len(comp_order_sym),
            'n_features_out': len(comp_order_sym),
            'learnable_parameters': learnable_parameters,
            'strain_formulation': strain_formulation,
            'problem_type': problem_type,
            'material_model_name': material_model_name,
            'material_model_parameters': model_parameters,
            'state_features_out': state_features_out,
            'is_auto_sync_parameters': is_auto_sync_parameters,
            'is_check_su_fail': is_check_su_fail,
            'model_directory': None,
            'model_name': model_name,
            'is_normalized_parameters': is_normalized_parameters,
            'is_model_in_normalized': is_model_in_normalized,
            'is_model_out_normalized': is_model_out_normalized,
            'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'gru_material_model':
        # Set model parameters
        model_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of input features
        n_features_in = 6
        # Set number of output features
        n_features_out = 6
        # Set hidden layer size
        hidden_layer_size = 200
        # Set number of recurrent layers (stacked RNN)
        n_recurrent_layers = 2
        # Set dropout probability
        dropout = 0
        # Set model input and output features normalization
        is_model_in_normalized = True
        is_model_out_normalized = True
        # Set GRU model source
        gru_model_source = 'custom'
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set other parameters required to initialize constitutive model
        model_kwargs = {'n_features_in': n_features_in,
                        'n_features_out': n_features_out,
                        'hidden_layer_size': hidden_layer_size,
                        'n_recurrent_layers': n_recurrent_layers,
                        'dropout': dropout,
                        'model_directory': None,
                        'model_name': model_name,
                        'is_model_in_normalized': is_model_in_normalized,
                        'is_model_out_normalized': is_model_out_normalized,
                        'gru_model_source': gru_model_source,
                        'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_formulation, problem_type, n_dim, model_name, \
        model_parameters, model_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set float precision
    is_double_precision = True
    if is_double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set computation processes
    is_save_specimen_data = True
    is_save_specimen_material_state = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case study base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/8_global_random_specimen/von_mises/'
                '2_random_specimen_hexa8/solid/'
                '4_discover_gru_material_model')
    # Set case study directory
    case_study_name = 'material_model_finder'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen name
    specimen_name = 'random_specimen'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Get problem type parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model name and parameters
    strain_formulation, problem_type, n_dim, model_name, \
        model_parameters, model_kwargs = set_material_model_parameters()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen raw data directory
    specimen_raw_dir = os.path.join(os.path.normpath(case_study_dir),
                                    '0_simulation')
    # Check specimen raw data directory
    if not os.path.isdir(specimen_raw_dir):
        raise RuntimeError('The specimen raw data directory has not been '
                           'found:\n\n' + specimen_raw_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen mesh input file path (.inp file)
    specimen_inp_path = os.path.join(os.path.normpath(specimen_raw_dir),
                                    f'{specimen_name}.inp')
    # Check specimen mesh input file path
    if not os.path.isfile(specimen_inp_path):
        raise RuntimeError(f'The specimen mesh input file '
                           f'({specimen_name}.inp) has not been found in '
                           f'specimen raw data directory:\n\n'
                           f'{specimen_raw_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen history data directory
    specimen_history_dir = os.path.join(os.path.normpath(specimen_raw_dir),
                                        'specimen_history_data')
    # Get specimen history time step file paths
    specimen_history_paths = \
        get_specimen_history_paths(specimen_history_dir, specimen_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set training data set directory
    training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                        '1_training_dataset')
    # Create training data set directory
    if not os.path.isdir(training_dataset_dir):
        make_directory(training_dataset_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create material models initialization subdirectory
    if is_save_specimen_material_state:
        # Set material models initialization subdirectory
        material_models_dir = os.path.join(
            os.path.normpath(training_dataset_dir), 'material_models_init')
        # Create material models initialization subdirectory
        make_directory(material_models_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material model directory
        material_model_dir = os.path.join(
            os.path.normpath(material_models_dir), 'model_1')
        # Create material model directory
        make_directory(material_model_dir, is_overwrite=True)
        # Assign material model directory
        model_kwargs['model_directory'] = material_model_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate specimen data and material state files (training data set)
    specimen_data_path, specimen_material_state_path = gen_specimen_dataset(
        specimen_name, specimen_raw_dir, specimen_inp_path,
        specimen_history_paths, strain_formulation, problem_type, n_dim,
        model_name, model_parameters, model_kwargs, training_dataset_dir,
        is_save_specimen_data, is_save_specimen_material_state)