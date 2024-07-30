"""DARPA METALS PROJECT: Pipeline validation with toy uniaxial specimen."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
import re
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import torch
import numpy as np
import pandas
# Local
from simulators.fetorch.element.type.tri3 import FETri3
from simulators.fetorch.element.type.tri6 import FETri6
from simulators.fetorch.element.type.quad4 import FEQuad4
from simulators.fetorch.element.type.quad8 import FEQuad8
from simulators.fetorch.element.type.tetra4 import FETetra4
from simulators.fetorch.element.type.hexa8 import FEHexa8
from simulators.fetorch.structure.structure_state import StructureMaterialState
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from material_model_finder.data.specimen_data import SpecimenNumericalData
from material_model_finder.model.material_discovery import MaterialModelFinder
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def validate_force_equilibrium_loss(specimen_name, strain_formulation,
                                    problem_type, model_name, model_parameters,
                                    mesh_type, nodes_disps_mesh_hist,
                                    reaction_forces_mesh_hist, time_hist,
                                    model_kwargs={}):
    """Validate computation of force equilibrium history loss.
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    model_name : str
        Material constitutive model name.
    model_parameters : dict
        Material constitutive model parameters.
    mesh_type : {'quad4',}
        Specimen finite element mesh type.
    nodes_disps_mesh_hist : torch.Tensor(3d)
        Displacements history of finite element mesh nodes stored as
        torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
    reaction_forces_mesh_hist : torch.Tensor(3d)
        Reaction forces (Dirichlet boundary conditions) history of finite
        element mesh nodes stored as torch.Tensor(3d) of shape
        (n_node_mesh, n_dim, n_time).
    time_hist : torch.Tensor(1d)
        Discrete time history.
    model_kwargs : dict, default={}
        Other parameters required to initialize constitutive model.
    """
    # Initialize specimen numerical data
    specimen_data = SpecimenNumericalData()
    # Set specimen finite element mesh
    specimen_data.set_specimen_mesh(*get_toy_uniaxial_specimen_mesh(mesh_type))
    # Set specimen numerical data translated from experimental results
    specimen_data.set_specimen_data(
        nodes_disps_mesh_hist, reaction_forces_mesh_hist, time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get finite element mesh
    specimen_mesh = specimen_data.specimen_mesh
    # Get number of spatial dimensions
    n_dim = specimen_data.get_n_dim()
    # Get number of elements
    n_elem = specimen_mesh.get_n_elem()
    # Get type of elements of finite element mesh
    elements_type = specimen_mesh.get_elements_type()
    # Get elements labels
    elements_ids = tuple([elem_id for elem_id in range(1, n_elem + 1)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize specimen material state
    specimen_material_state = StructureMaterialState(
        strain_formulation, problem_type, n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set available sequential modes
    if bool(re.search(r'^rc_.*$', model_name)):
        available_sequential_modes = ('sequential_element',)
    else:
        available_sequential_modes = ('sequential_time', 'sequential_element')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model finder directory
    model_directory = os.path.dirname(os.path.abspath(__file__))
    # Set material model finder name
    model_finder_name = 'material_model_finder'
    # Set force normalization
    is_force_normalization = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material model finder
    material_finder = MaterialModelFinder(
        model_directory, model_name=model_finder_name,
        is_force_normalization=is_force_normalization)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain and stress components
    if strain_formulation == 'infinitesimal':
        n_strain_comp = int(n_dim*(n_dim + 1)/2)
        n_stress_comp = int(n_dim*(n_dim + 1)/2)
    else:
        raise RuntimeError('Not implemented.')
    # Set strain minimum and maximum (data normalization)
    strain_minimum = -10*torch.ones(n_strain_comp)
    strain_maximum = 10*torch.ones(n_strain_comp)
    # Set stress minimum and maximum (data normalization)
    stress_minimum = -10*torch.ones(n_stress_comp)
    stress_maximum = 10*torch.ones(n_stress_comp)
    # Set material models data scaling type
    models_scaling_type = {}
    models_scaling_type['1'] = 'min-max'
    # Set material models data scalers parameters
    models_scaling_parameters = {'1': {}}
    models_scaling_parameters['1']['features_in'] = \
        (strain_minimum, strain_maximum)
    models_scaling_parameters['1']['features_out'] = \
        (stress_minimum, stress_maximum)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of force components
    n_force_comp = n_dim
    # Set force minimum and maximum (data normalization)
    force_minimum = -5*torch.ones(n_force_comp)
    force_maximum = 5*torch.ones(n_force_comp)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over sequential modes
    for sequential_mode in available_sequential_modes:
        # Initialize elements constitutive model
        specimen_material_state.init_elements_model(
            model_name, model_parameters, elements_ids, elements_type,
            model_kwargs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen data and material state
        material_finder.set_specimen_data(specimen_data,
                                          specimen_material_state,
                                          force_minimum=force_minimum,
                                          force_maximum=force_maximum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model fitted data scalers
        material_finder.set_material_models_fitted_data_scalers(
            models_scaling_type, models_scaling_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium history loss
        force_equilibrium_hist_loss = \
            material_finder(sequential_mode=sequential_mode)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display results
        print('')
        print('DARPA METALS PROJECT: Validate force equilibrium history loss')
        print('-------------------------------------------------------------')
        print(f'Specimen:           {specimen_name}')
        print(f'Data source:        FEM numerical simulation')
        print(f'\nSequential mode:    {sequential_mode}')
        print(f'\nElement type:       {mesh_type}')
        print(f'Material model:     {model_name}')
        print(f'Strain formulation: {strain_formulation}')
        print(f'\nForce equilibrium history loss: '
              f'{force_equilibrium_hist_loss:.4e}')
        print(f'Force normalization: {str(is_force_normalization):>15s}')
        print('-------------------------------------------------------------')
        print('')
# =============================================================================
def get_toy_uniaxial_specimen_mesh(mesh_type):
    """Get toy uniaxial specimen finite element mesh data.
    
    Parameters
    ----------
    mesh_type : {'quad4',}
        Specimen finite element mesh type.
        
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
    if mesh_type == 'tri3':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.00, 0.00],
                          [1.00, 0.00],
                          [2.00, 0.25],
                          [3.00, 0.00],
                          [4.00, 0.00],
                          [0.00, 1.00],
                          [1.00, 1.00],
                          [2.00, 0.75],
                          [3.00, 1.00],
                          [4.00, 1.00]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FETri3(n_gauss=1) for i in range(1, 9)}
        # Set elements connectivies
        connectivities = {'1': (1, 7, 6),
                          '2': (1, 2, 7),
                          '3': (2, 8, 7),
                          '4': (2, 3, 8),
                          '5': (3, 9, 8),
                          '6': (3, 4, 9),
                          '7': (4, 10, 9),
                          '8': (4, 5, 10)}
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[4, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[5, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'tri6':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.000, 0.000],
                          [0.500, 0.000],
                          [1.000, 0.000],
                          [1.500, 0.125],
                          [2.000, 0.250],
                          [2.500, 0.125],
                          [3.000, 0.000],
                          [3.500, 0.000],
                          [4.000, 0.000],
                          [0.000, 0.500],
                          [0.500, 0.500],
                          [1.000, 0.500],
                          [1.500, 0.375],
                          [2.000, 0.500],
                          [2.500, 0.625],
                          [3.000, 0.500],
                          [3.500, 0.500],
                          [4.000, 0.500],
                          [0.000, 1.000],
                          [0.500, 1.000],
                          [1.000, 1.000],
                          [1.500, 0.875],
                          [2.000, 0.750],
                          [2.500, 0.875],
                          [3.000, 1.000],
                          [3.500, 1.000],
                          [4.000, 1.000]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FETri6(n_gauss=3) for i in range(1, 9)}
        # Set elements connectivies
        connectivities = {'1': (1, 21, 19, 11, 20, 10),
                          '2': (1, 3, 21, 2, 12, 11),
                          '3': (3, 23, 21, 13, 22, 12),
                          '4': (3, 5, 23, 4, 14, 13),
                          '5': (5, 25, 23, 15, 24, 14),
                          '6': (5, 7, 25, 6, 16, 15),
                          '7': (7, 27, 25, 17, 26, 16),
                          '8': (7, 9, 27, 8, 18, 17)}
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[8, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[17, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[18, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[26, :] = torch.tensor([1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'quad4':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.00, 0.00],
                          [1.00, 0.00],
                          [2.00, 0.25],
                          [3.00, 0.00],
                          [4.00, 0.00],
                          [0.00, 1.00],
                          [1.00, 1.00],
                          [2.00, 0.75],
                          [3.00, 1.00],
                          [4.00, 1.00]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FEQuad4(n_gauss=4) for i in range(1, 5)}
        # Set elements connectivies
        connectivities = {'1': (1, 2, 7, 6),
                          '2': (2, 3, 8, 7),
                          '3': (3, 4, 9, 8),
                          '4': (4, 5, 10, 9)}
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[4, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[5, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'quad8':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.000, 0.000],
                          [0.500, 0.000],
                          [1.000, 0.000],
                          [1.500, 0.125],
                          [2.000, 0.250],
                          [2.500, 0.125],
                          [3.000, 0.000],
                          [3.500, 0.000],
                          [4.000, 0.000],
                          [0.000, 0.500],
                          [1.000, 0.500],
                          [2.000, 0.500],
                          [3.000, 0.500],
                          [4.000, 0.500],
                          [0.000, 1.000],
                          [0.500, 1.000],
                          [1.000, 1.000],
                          [1.500, 0.875],
                          [2.000, 0.750],
                          [2.500, 0.875],
                          [3.000, 1.000],
                          [3.500, 1.000],
                          [4.000, 1.000]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FEQuad8(n_gauss=4) for i in range(1, 5)}
        # Set elements connectivies
        connectivities = {'1': (1, 3, 17, 15, 2, 11, 16, 10),
                          '2': (3, 5, 19, 17, 4, 12, 18, 11),
                          '3': (5, 7, 21, 19, 6, 13, 20, 12),
                          '4': (7, 9, 23, 21, 8, 14, 22, 13)}
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[8, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[13, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[14, :] = torch.tensor([1, 1])
        dirichlet_bool_mesh[22, :] = torch.tensor([1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'tetra4':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.00, 0.00, 0.0],
                          [1.00, 0.00, 0.0],
                          [2.00, 0.25, 0.0],
                          [3.00, 0.00, 0.0],
                          [4.00, 0.00, 0.0],
                          [0.00, 1.00, 0.0],
                          [1.00, 1.00, 0.0],
                          [2.00, 0.75, 0.0],
                          [3.00, 1.00, 0.0],
                          [4.00, 1.00, 0.0],
                          [0.00, 0.00, 1.0],
                          [1.00, 0.00, 1.0],
                          [2.00, 0.25, 1.0],
                          [3.00, 0.00, 1.0],
                          [4.00, 0.00, 1.0],
                          [0.00, 1.00, 1.0],
                          [1.00, 1.00, 1.0],
                          [2.00, 0.75, 1.0],
                          [3.00, 1.00, 1.0],
                          [4.00, 1.00, 1.0]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FETetra4(n_gauss=4) for i in range(1, 25)}
        # Set elements connectivies
        connectivities = {'1': (1, 6, 16, 17),
                          '2': (1, 7, 6, 17),
                          '3': (1, 16, 11, 17),
                          '4': (1, 11, 12, 17),
                          '5': (1, 12, 2, 17),
                          '6': (1, 2, 7, 17),
                          '7': (2, 7, 17, 18),
                          '8': (2, 8, 7, 18),
                          '9': (2, 17, 12, 18),
                          '10': (2, 12, 13, 18),
                          '11': (2, 13, 3, 18),
                          '12': (2, 3, 8, 18),
                          '13': (3, 8, 18, 19),
                          '14': (3, 9, 8, 19),
                          '15': (3, 18, 13, 19),
                          '16': (3, 13, 14, 19),
                          '17': (3, 14, 4, 19),
                          '18': (3, 4, 9, 19),
                          '19': (4, 9, 19, 20),
                          '20': (4, 10, 9, 20),
                          '21': (4, 19, 14, 20),
                          '22': (4, 14, 15, 20),
                          '23': (4, 15, 5, 20),
                          '24': (4, 5, 10, 20),
                          }
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[4, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[5, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[10, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[14, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[15, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[19, :] = torch.tensor([1, 1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'tetra10':
        raise RuntimeError('Tetra10 finite element mesh type has not been '
                           'created.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'hexa8':
        # Set finite element mesh initial nodes coordinates
        nodes_coords_mesh_init = \
            torch.tensor([[0.00, 0.00, 0.0],
                          [1.00, 0.00, 0.0],
                          [2.00, 0.25, 0.0],
                          [3.00, 0.00, 0.0],
                          [4.00, 0.00, 0.0],
                          [0.00, 1.00, 0.0],
                          [1.00, 1.00, 0.0],
                          [2.00, 0.75, 0.0],
                          [3.00, 1.00, 0.0],
                          [4.00, 1.00, 0.0],
                          [0.00, 0.00, 1.0],
                          [1.00, 0.00, 1.0],
                          [2.00, 0.25, 1.0],
                          [3.00, 0.00, 1.0],
                          [4.00, 0.00, 1.0],
                          [0.00, 1.00, 1.0],
                          [1.00, 1.00, 1.0],
                          [2.00, 0.75, 1.0],
                          [3.00, 1.00, 1.0],
                          [4.00, 1.00, 1.0]], dtype=torch.float)
        # Set finite element mesh elements types
        elements_type = {str(i): FEHexa8(n_gauss=8) for i in range(1, 5)}
        # Set elements connectivies
        connectivities = {'1': (1, 11, 12, 2, 6, 16, 17, 7),
                          '2': (2, 12, 13, 3, 7, 17, 18, 8),
                          '3': (3, 13, 14, 4, 8, 18, 19, 9),
                          '4': (4, 14, 15, 5, 9, 19, 20, 10)}
        # Set Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            torch.zeros_like(nodes_coords_mesh_init, dtype=torch.int)
        dirichlet_bool_mesh[0, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[4, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[5, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[9, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[10, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[14, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[15, :] = torch.tensor([1, 1, 1])
        dirichlet_bool_mesh[19, :] = torch.tensor([1, 1, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif mesh_type == 'hexa20':
        raise RuntimeError('Hexa20 finite element mesh type has not been '
                           'created.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown finite element mesh type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_mesh_init, elements_type, connectivities, \
        dirichlet_bool_mesh
# =============================================================================
def get_specimen_numerical_data(links_output_directory, n_dim):
    """Read specimen numerical data from Links simulation directory.
    
    Assumes that the node data file is structured in tabular format with
    shape (n_node, n_features), where:
    
    j = 0     : Node ID (integer)
    j = 1     : Total load factor (float)
    j = 2     : Total (pseudo or real) time (float)
    j = 3-5   : Node coordinates (float)
    j = 6-8   : Node displacements (float)
    j = 9-11  : Node internal forces (float)
    j = 12-14 : Node external forces (including reaction forces) (float)
    j = 15-17 : Node reaction forces (float)
    
    Parameters
    ----------
    links_output_directory : str
        Links simulation output directory.
    n_dim : int
        Number of spatial dimensions.

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
    # Check if simulation output directory exists
    if not os.path.exists(links_output_directory):
        raise RuntimeError('Links simulation output directory has not been '
                           'found.')
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(links_output_directory))[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get node data output directory
    directory = os.path.join(links_output_directory, 'NODE_DATA')
    # Get files in node data output directory
    directory_list = os.listdir(directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize node data files
    node_data_files = []
    node_data_files_time_steps = []
    # Loop over files
    for filename in directory_list:
        # Check if is node data file
        is_node_data_file = bool(re.search(r'^' + simulation_name + r'(.*)'
                                           + r'\.nodedata$', filename))
        if is_node_data_file:
            # Get node data file time step
            time_step = int(os.path.splitext(filename)[0].split('_')[-1])
            # Store node data file and corresponding time step
            node_data_files.append(filename)
            node_data_files_time_steps.append(time_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort node data files by time step
    order = np.argsort(node_data_files_time_steps)
    node_data_files[:] = [node_data_files[i] for i in order]
    node_data_files_time_steps[:] = \
        [node_data_files_time_steps[i] for i in order]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of time steps
    n_time_steps = len(node_data_files_time_steps)
    # Loop over node data files
    for i, file in enumerate(node_data_files):
        # Get node data file path
        file_path = os.path.join(directory, file)
        # Get nodes features data
        data_array = torch.tensor(np.genfromtxt(file_path), dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node features data
        if i == 0:
            # Get number of nodes and features
            n_nodes = data_array.shape[0]
            n_features = data_array.shape[1]
            # Initialize node features data
            node_data = torch.zeros((n_nodes, n_features, n_time_steps),
                                    dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble node features data
        node_data[:, :, i] = data_array
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract nodes displacement history
    nodes_disps_mesh_hist = node_data[:, 6:6+n_dim, :]
    # Extract nodes reaction history
    reaction_forces_mesh_hist = node_data[:, 15:15+n_dim, :]
    # Extract discrete time history
    time_hist = node_data[0, 2, :]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_disps_mesh_hist, reaction_forces_mesh_hist, time_hist
# =============================================================================
def gen_specimen_history_csv_files(specimen_history_dir, specimen_name,
                                   nodes_coords_mesh_init,
                                   nodes_disps_mesh_hist,
                                   reaction_forces_mesh_hist):
    """Generate specimen history data files (.csv).
    
    Irrespective of the strain formulation, specimen nodes coordinates history
    is computed from the initial coordinates and the nodes displacements
    history.

    Parameters
    ----------
    specimen_history_dir : str
        Specimen history data directory.
    specimen_name : str
        Specimen code name.
    nodes_coords_mesh_init : torch.Tensor(2d)
        Initial coordinates of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    nodes_disps_mesh_hist : torch.Tensor(3d)
        Displacements history of finite element mesh nodes stored as
        torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
    reaction_forces_mesh_hist : torch.Tensor(3d)
        Reaction forces (Dirichlet boundary conditions) history of finite
        element mesh nodes stored as torch.Tensor(3d) of shape
        (n_node_mesh, n_dim, n_time).
    """
    # Get number of nodes
    n_node_mesh = nodes_coords_mesh_init.shape[0]
    # Get number of spatial dimensions
    n_dim = nodes_coords_mesh_init.shape[1]
    # Get number of time steps
    n_time = nodes_disps_mesh_hist.shape[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for t in range(n_time):
        # Initialize dataframe
        df = pandas.DataFrame(torch.zeros((n_node_mesh, 10)),
                              columns=['NODE', 'X1', 'X2', 'X3',
                                       'U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble nodes labels
        df['NODE'] = torch.arange(1, n_node_mesh + 1)
        # Assemble nodes coordinates, displacements and reaction forces
        if n_dim == 2:
            df[['X1', 'X2']] = (nodes_coords_mesh_init[:, :]
                                + nodes_disps_mesh_hist[:, :, t])
            df[['U1', 'U2']] = nodes_disps_mesh_hist[:, :, t]
            df[['RF1', 'RF2']] = reaction_forces_mesh_hist[:, :, t]
        else:
            df[['X1', 'X2', 'X3']] = (nodes_coords_mesh_init[:, :]
                                      + nodes_disps_mesh_hist[:, :, t])
            df[['U1', 'U2', 'U3']] = nodes_disps_mesh_hist[:, :, t]
            df[['RF1', 'RF2', 'RF3']] = reaction_forces_mesh_hist[:, :, t]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set time step '.csv' file path
        csv_file_path = os.path.join(os.path.normpath(specimen_history_dir),
                                     f'{specimen_name}_tstep_{t}.csv')
        # Store data into '.csv' file format
        df.to_csv(csv_file_path, encoding='utf-8', index=False)
# =============================================================================
if __name__ == '__main__':
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model name and parameters
    model_name = 'drucker_prager'
    # Set constitutive model parameters
    if model_name == 'elastic':
        # Set constitutive model parameters
        model_parameters = {
            'elastic_symmetry': 'isotropic',
            'E': 100, 'v': 0.3,
            'euler_angles': (0.0, 0.0, 0.0)}
        # Set other parameters required to initialize constitutive model
        model_kwargs = {}
        # Set model validation data directory name
        model_data_name = 'elastic'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'von_mises':
        # Set constitutive model parameters
        model_parameters = {
            'elastic_symmetry': 'isotropic',
            'E': 100, 'v': 0.3,
            'euler_angles': (0.0, 0.0, 0.0),
            'hardening_law': get_hardening_law('piecewise_linear'),
            'hardening_parameters':
                {'hardening_points': torch.tensor([[0.0, 2.0],
                                                   [1.0, 4.0]])}}
        # Set other parameters required to initialize constitutive model
        model_kwargs = {}
        # Set model validation data directory name
        model_data_name = 'von_mises'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'drucker_prager':
        # Set frictional angle
        friction_angle = np.deg2rad(10)
        # Set dilatancy angle
        dilatancy_angle = friction_angle
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute angle-related material parameters
        # (matching with Mohr-Coulomb under uniaxial tension and compression)
        # Set yield surface cohesion parameter
        yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
        # Set yield pressure parameter
        yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
        # Set plastic flow pressure parameter
        flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model parameters
        model_parameters = {
            'elastic_symmetry': 'isotropic',
            'E': 100, 'v': 0.3,
            'euler_angles': (0.0, 0.0, 0.0),
            'hardening_law': get_hardening_law('piecewise_linear'),
            'hardening_parameters':
                {'hardening_points':
                    torch.tensor([[0.0, 2.0/yield_cohesion_parameter],
                                  [1.0, 4.0/yield_cohesion_parameter]],
                                 dtype=torch.float)},
            'yield_cohesion_parameter': yield_cohesion_parameter,
            'yield_pressure_parameter': yield_pressure_parameter,
            'flow_pressure_parameter': flow_pressure_parameter}
        # Set other parameters required to initialize constitutive model
        model_kwargs = {}
        # Set model validation data directory name
        model_data_name = 'drucker_prager'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif bool(re.search(r'^rc_.*$', model_name)):
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
            learnable_parameters['E'] = {'initial_value': 100.0,
                                         'bounds': (80, 120)}
            learnable_parameters['v'] = {'initial_value': 0.3,
                                         'bounds': (0.2, 0.4)}
            # Set material constitutive model name
            material_model_name = 'elastic'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model validation data directory name
            model_data_name = 'elastic'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'rc_von_mises':
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 100, 'v': 0.3,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('piecewise_linear'),
                'hardening_parameters':
                    {'hardening_points': torch.tensor([[0.0, 2.0],
                                                       [1.0, 4.0]])}}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set learnable parameters
            learnable_parameters = {}
            learnable_parameters['E'] = {'initial_value': 100.0,
                                         'bounds': (80, 120)}
            learnable_parameters['v'] = {'initial_value': 0.3,
                                         'bounds': (0.2, 0.4)}
            # Set material constitutive model name
            material_model_name = 'von_mises'
            # Set material constitutive state variables (prediction)
            state_features_out = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model validation data directory name
            model_data_name = 'von_mises'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown recurrent constitutive model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = ('/home/bernardoferreira/Documents/brown/projects/'
                           'darpa_project/1_pipeline_validation/'
                           'toy_uniaxial_specimen/temp')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set parameters normalization
        is_normalized_parameters = True
        # Set data normalization
        is_data_normalization = False
        # Set device type
        device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set other parameters required to initialize constitutive model
        model_kwargs = {
            'n_features_in': None,
            'n_features_out': None,
            'learnable_parameters': learnable_parameters,
            'strain_formulation': strain_formulation,
            'problem_type': None,
            'material_model_name': material_model_name,
            'material_model_parameters': model_parameters,
            'state_features_out': state_features_out,
            'model_directory': model_directory,
            'model_name': model_name,
            'is_normalized_parameters': is_normalized_parameters,
            'is_data_normalization': is_data_normalization,
            'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown constitutive model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set toy uniaxial specimen finite element mesh types
    mesh_types = ('tri3', 'tri6', 'quad4', 'quad8', 'tetra4', 'hexa8')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over finite element mesh types
    for mesh_type in mesh_types:
        # Set problem type
        if ('tri' in mesh_type or 'quad' in mesh_type):
            problem_type = 1
        else:
            problem_type = 4
        # Get problem type parameters
        n_dim, comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model parameters according with problem dimensions
        if model_kwargs:
            # Set number of input and output features
            model_kwargs['n_features_in'] = len(comp_order_sym)
            model_kwargs['n_features_out'] = len(comp_order_sym)
            # Set problem type
            model_kwargs['problem_type'] = problem_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links simulation output directory
        links_output_directory = \
            (f'/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             f'1_pipeline_validation/toy_uniaxial_specimen/'
             f'{model_data_name}/{n_dim}D_toy_uniaxial_specimen_{mesh_type}')
        # Get specimen name
        specimen_name = os.path.basename(links_output_directory)
        # Read specimen numerical data from Links simulation directory
        nodes_disps_mesh_hist, reaction_forces_mesh_hist, time_hist = \
            get_specimen_numerical_data(links_output_directory, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validate computation of force equilibrium history loss
        validate_force_equilibrium_loss(specimen_name, strain_formulation,
                                        problem_type, model_name,
                                        model_parameters, mesh_type,
                                        nodes_disps_mesh_hist,
                                        reaction_forces_mesh_hist, time_hist,
                                        model_kwargs)