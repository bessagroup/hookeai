"""DARPA METALS PROJECT: Generate and simulate random material patches.

Functions
---------
generate_random_patches
    Generate and simulate material patches.
set_material_patch_dir
    Setup material patch directory.
generate_material_patch
    Generate material patch.
set_patch_geometric_params
    Set material patch geometric parameters.
set_patch_mesh_params
    Set material patch mesh parameters.
set_patch_material_params
    Set material patch material parameters.
perform_links_simulation
    Perform finite element simulation: Links.
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
import time
import datetime
import pickle
import math
import shutil
# Third-party
import numpy as np
# Local
from simulators.links.links import LinksSimulator
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from projects.gnn_material_patch.material_patch.patch_generator import \
    FiniteElementPatchGenerator
from testing_utilities.links_plot_tfact import plot_links_tfact_hist
from testing_utilities.links_dat_to_inp import links_dat_to_abaqus_inp
from testing_utilities.links_nodedata_to_csv import links_node_output_to_csv
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
def generate_random_patches(base_dir, patch_name='material_patch', n_patch=1,
                            is_save_material_patch=False,
                            is_load_material_patch=False, is_verbose=False):
    """Generate and simulate material patches.
    
    Parameters
    ----------
    base_dir : str
        Base directory where material patch and simulation data is stored.
    patch_name : str, default='material_patch'
        Material patch name.
    n_patch : int, default=1
        Number of material patches.
    is_save_material_patch : bool, default=False
        If True, then save material patch file in material patch directory.
    is_load_material_patch : bool, default=False
        If True, then load material patch file from material patch directory.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    links_file_paths : dict
        Links input data file path (item, str) for each material patch
        (key, str[int]).
    links_output_dirs : dict
        Links simulation output directory (item, str) for each material patch
        (key, str[int]).
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGeneration and simulation of material patches'
              '\n---------------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plot deformed material patch flag
    is_save_plot_patch = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize simulations files and directories
    links_file_paths = {}
    links_output_dirs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material patches
    for i in range(n_patch):
        if is_verbose:
            print(f'\n> Material patch {i} ({i + 1}/{n_patch}):')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch directory name
        if n_patch > 1:
            patch_dir_name = f'{patch_name}_{i}'
        else:
            patch_dir_name = patch_name
        # Set directory overwrite flag
        is_overwrite_data = not is_load_material_patch
        # Setup material patch directory 
        _, patch_data_dir, simulations_dir, plots_dir = \
            set_material_patch_dir(patch_dir_name, base_dir,
                                   is_overwrite_data=is_overwrite_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate or load material patch
        if is_load_material_patch:
            if is_verbose:
                print('\n  > Loading material patch...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material patch data file path
            patch_file_path = \
                os.path.join(patch_data_dir, 'material_patch.pkl')
            # Check material patch data file
            if not os.path.isfile(patch_file_path):
                raise RuntimeError(f'The material patch data file has not '
                                   f'been found:\n\n{patch_file_path}')
            # Load material patch
            with open(patch_file_path, 'rb') as patch_file:
                patch = pickle.load(patch_file)
        else:
            if is_verbose:
                print('\n  > Setting material patch parameters...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material patch geometric parameters
            patch_geometric_params = set_patch_geometric_params(n_dim)
            # Get material patch mesh parameters
            patch_mesh_params = set_patch_mesh_params(n_dim)
            # Get material patch material parameters
            patch_material_params = set_patch_material_params(
                patch_mesh_params['n_elems_per_dim'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print(f'\n  > Number of dimensions: {n_dim}')
                print(f'\n  > Patch dimensions: '
                    f'{patch_geometric_params["patch_dims"]}')
                print(f'\n  > Mesh element type: '
                    f'{patch_mesh_params["elem_type"]}')
                print(f'\n  > Number of elements: '
                    f'{patch_mesh_params["n_elems_per_dim"]}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print('\n  > Generating material patch...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate material patch
            patch = generate_material_patch(patch_geometric_params,
                                            patch_mesh_params)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch material parameters
        patch_material_params = set_patch_material_params(
            patch.get_n_elems_per_dim())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            model_name = \
                patch_material_params["mat_phases_descriptors"]["1"]["name"]
            print(f'\n    > Material model: {model_name}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save material patch
        if is_save_material_patch:
            # Set material patch data file path
            patch_file_path = \
                os.path.join(patch_data_dir, 'material_patch.pkl')
            # Save material patch data file
            with open(patch_file_path, 'wb') as patch_file:
                pickle.dump(patch, patch_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save plot of deformed material patch
        if is_save_plot_patch:
            if is_verbose:
                print('\n    > Plotting material patch...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot deformed material patch
            patch.plot_deformed_patch(
                is_hide_axes=False, is_show_fixed_dof=True,
                is_save_plot=is_save_plot_patch, save_directory=plots_dir,
                plot_name='specimen_deformed_configuration',
                is_overwrite_file=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n  > Performing material patch FEM simulation...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform Links simulation
        links_file_path, links_output_directory = perform_links_simulation(
            simulations_dir, patch, patch_material_params)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store Links input data file and simulation directory
        links_file_paths[str(i)] = links_file_path
        links_output_dirs[str(i)] = links_output_directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total time
    total_time_sec = time.time() - start_time_sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Total time: '
            f'{str(datetime.timedelta(seconds=int(total_time_sec)))}')
        print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return links_file_paths, links_output_dirs
# =============================================================================
def set_material_patch_dir(dir_name, parent_dir, is_overwrite_data=True):
    """Setup material patch directory.
    
    Parameters
    ----------
    dir_name : str
        Material patch directory name.
    parent_dir : str
        Directory where material patch directory is created.
    is_overwrite_data : bool, default=True
        If False, then preserve material patch data directory if it exists.

    Returns
    -------
    material_patch_dir : str
        Material patch directory.
    patch_data_dir : str
        Material patch data directory.
    simulations_dir : str
        Simulations directory.
    plots_dir : str
        Plots directory.
    """
    # Check parent directory
    if not os.path.isdir(parent_dir):
        raise RuntimeError('The parent directory has not been found:\n\n'
                           + parent_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch directory
    material_patch_dir = \
        os.path.join(os.path.normpath(parent_dir), dir_name)
    # Create material patch directory
    if not os.path.isdir(material_patch_dir) or is_overwrite_data:
        make_directory(material_patch_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch data directory
    patch_data_dir = \
        os.path.join(os.path.normpath(material_patch_dir), 'material_patch')
    # Create material patch data directory
    if not os.path.isdir(patch_data_dir) or is_overwrite_data:
        make_directory(patch_data_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set simulations directory
    simulations_dir = \
        os.path.join(os.path.normpath(material_patch_dir), 'simulations')
    # Create simulations directory
    make_directory(simulations_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = os.path.join(os.path.normpath(material_patch_dir), 'plots')
    # Create plots directory
    make_directory(plots_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return material_patch_dir, patch_data_dir, simulations_dir, plots_dir
# =============================================================================
def generate_material_patch(patch_geometric_params, patch_mesh_params):
    """Generate material patch.
    
    Parameters
    ----------
    patch_geometric_params : dict
        Material patch geometric parameters.
    patch_mesh_params : dict
        Material patch mesh parameters.
    
    Returns
    -------
    patch : FiniteElementPatch
        Finite element patch.
    """
    # Get number of spatial dimensions                                         
    n_dim = patch_geometric_params['n_dim']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get patch dimensions
    patch_dims = patch_geometric_params['patch_dims']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set corners displacements
    corners_lab_disp_range = patch_geometric_params['avg_deformation_ranges']
    # Set edges polynomial deformation order
    edges_lab_def_order = patch_geometric_params['edge_deformation_order']
    # Set edges displacement range
    edges_lab_disp_range = \
        patch_geometric_params['edge_deformation_magnitude_ranges']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set rigid body translation
    translation_range = patch_geometric_params['translation_range']
    # Set rigid body rotation
    rotation_angles_range = patch_geometric_params['rotation_angles_range']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get deformation noise
    deformation_noise = patch_geometric_params['deformation_noise']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get finite element type
    elem_type = patch_mesh_params['elem_type']
    # Get number of finite elements per dimension
    n_elems_per_dim = patch_mesh_params['n_elems_per_dim']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patch generator
    patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
    # Generate deformed material patch
    is_admissible, patch = patch_generator.generate_deformed_patch(
        elem_type, n_elems_per_dim,
        corners_lab_disp_range=corners_lab_disp_range,
        edges_lab_def_order=edges_lab_def_order,
        edges_lab_disp_range=edges_lab_disp_range,
        translation_range=translation_range,
        rotation_angles_range=rotation_angles_range,
        deformation_noise=deformation_noise)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check material patch deformation geometric admissibility
    if not is_admissible:
        raise RuntimeError('Material patch deformation is not geometrically '
                           'admissible.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch
# =============================================================================
def set_patch_geometric_params(n_dim):
    """Set material patch geometric parameters.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
        
    Returns
    -------
    patch_geometric_params : dict
        Material patch geometric parameters.
    """
    # Initialize material patch geometric parameters
    patch_geometric_params = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch geometric parameters
    if n_dim == 2:
        # Set material patch size along each dimension
        patch_dims = (1.0, 1.0)
        # Set range of average deformation along each dimension for each corner
        avg_deformation_ranges = {'1': ((-0.05, 0.05), (-0.05, 0.05)),
                                  '2': ((-0.05, 0.05), (-0.05, 0.05)),
                                  '3': ((-0.05, 0.05), (-0.05, 0.05)),
                                  '4': ((-0.05, 0.05), (-0.05, 0.05))}
        # Set polynomial deformation order for each edge label
        edge_deformation_order = {'1': 3, '2': 3, '3': 3, '4': 3}
        # Set range of polynomial deformation for each edge label
        edge_deformation_magnitude_ranges = {'1': (-0.05, 0.05),
                                             '2': (-0.05, 0.05),
                                             '3': (-0.05, 0.05),
                                             '4': (-0.05, 0.05)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set rigid body translation
        translation_range = {'1': (0.0, 0.0), '2': (0.0, 0.0)}
        # Set rigid body rotation
        rotation_angles_range = {'alpha': (0.0, 0.0), 'beta': (0.0, 0.0),
                                 'gamma': (0.0, 0.0)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set boundary deformation noise
        deformation_noise = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build material patch geometric parameters
        patch_geometric_params['n_dim'] = n_dim
        patch_geometric_params['patch_dims'] = patch_dims
        patch_geometric_params['avg_deformation_ranges'] = \
            avg_deformation_ranges
        patch_geometric_params['edge_deformation_order'] = \
            edge_deformation_order
        patch_geometric_params['edge_deformation_magnitude_ranges'] = \
            edge_deformation_magnitude_ranges
        patch_geometric_params['translation_range'] = translation_range
        patch_geometric_params['rotation_angles_range'] = \
            rotation_angles_range
        patch_geometric_params['deformation_noise'] = deformation_noise
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif n_dim == 3:
        # Set material patch size along each dimension
        patch_dims = (1.0, 1.0, 1.0)
        # Set range of average deformation along each dimension for each corner
        avg_deformation_ranges = \
            {str(i): ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))
             for i in range(1, 9)}
        # Set polynomial deformation order for each edge label
        edge_deformation_order = \
            {str(i): 3 for i in range(1, 25)}
        # Set range of polynomial deformation for each edge label
        edge_deformation_magnitude_ranges = \
            {str(i): (-0.05, 0.05) for i in range(1, 25)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set rigid body translation
        translation_range = {'1': (0.0, 0.0), '2': (0.0, 0.0), '3': (0.0, 0.0)}
        # Set rigid body rotation
        rotation_angles_range = {'alpha': (0.0, 0.0), 'beta': (0.0, 0.0),
                                 'gamma': (0.0, 0.0)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set boundary deformation noise
        deformation_noise = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build material patch geometric parameters
        patch_geometric_params['n_dim'] = n_dim
        patch_geometric_params['patch_dims'] = patch_dims
        patch_geometric_params['avg_deformation_ranges'] = \
            avg_deformation_ranges
        patch_geometric_params['edge_deformation_order'] = \
            edge_deformation_order
        patch_geometric_params['edge_deformation_magnitude_ranges'] = \
            edge_deformation_magnitude_ranges
        patch_geometric_params['translation_range'] = translation_range
        patch_geometric_params['rotation_angles_range'] = \
            rotation_angles_range
        patch_geometric_params['deformation_noise'] = deformation_noise
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Invalid number of spatial dimensions.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_geometric_params
# =============================================================================
def set_patch_mesh_params(n_dim):
    """Set material patch mesh parameters.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
        
    Returns
    -------
    patch_mesh_params : dict
        Material patch mesh parameters.
    """
    # Initialize material patch mesh parameters
    patch_mesh_params = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch mesh parameters
    if n_dim == 2:
        # Set finite element type
        elem_type = 'SQUAD4'
        # Set number of finite elements per dimension
        n_elems_per_dim = (5, 5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif n_dim == 3:
        # Set finite element type
        elem_type = 'SHEXA8'
        # Set number of finite elements per dimension
        n_elems_per_dim = (9, 9, 9)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Invalid number of spatial dimensions.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build material patch mesh parameters
    patch_mesh_params['elem_type'] = elem_type
    patch_mesh_params['n_elems_per_dim'] = n_elems_per_dim
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_mesh_params
# =============================================================================
def set_patch_material_params(n_elems_per_dim):
    """Set material patch material parameters.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    n_elems_per_dim : tuple[int]
        Number of finite elements per dimension that completely defines the
        regular finite element mesh by assuming equal-sized elements.

    Returns
    -------
    patch_material_params : dict
        Material patch material parameters.
    """
    # Initialize material patch material parameters
    patch_material_params = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material phases spatial domains
    mesh_elem_material = np.ones(n_elems_per_dim, dtype=int)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material model
    model_name = 'VON_MISES'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material phase descriptors
    mat_phases_descriptors = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material phase descriptors
    if model_name == 'ELASTIC':
        mat_phases_descriptors['1'] = {
            'name': 'ELASTIC',
            'density': 0.0,
            'young': 110e3,
            'poisson': 0.33}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'VON_MISES':
        mat_phases_descriptors['1'] = {
            'name': 'VON_MISES',
            'density': 0.0,
            'young': 110e3,
            'poisson': 0.33,
            'n_hard_point': 10000,
            'hardening_law': get_hardening_law('nadai_ludwik'),
            'hardening_parameters': {'s0': math.sqrt(3)*900,
                                     'a': math.sqrt(3)*700,
                                     'b': 0.5,
                                     'ep0': 1e-5}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'DRUCKER_PRAGER':
        mat_phases_descriptors['1'] = {
            'name': 'DRUCKER_PRAGER',
            'density': 0.0,
            'young': 110e3,
            'poisson': 0.33,
            'friction_angle_deg': 2,
            'dilatancy_angle_deg': 2,
            'n_hard_point': 200,
            'hardening_law': get_hardening_law('nadai_ludwik'),
            'hardening_parameters': {'s0': 900,
                                     'a': 700,
                                     'b': 0.5,
                                     'ep0': 1e-5}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_name == 'LOU':
        mat_phases_descriptors['1'] = {
            'name': 'LOU',
            'density': 0.0,
            'young': 110e3,
            'poisson': 0.33,
            'yield_a_init': 1.0,
            'a_hard_slope': 0.0,
            'yield_b_init': 0.05,
            'b_hard_slope': 0.0,
            'yield_c_init': 1.0,
            'c_hard_slope': 0.0,
            'yield_d_init': 0.25,
            'd_hard_slope': 0.0,
            'n_hard_point': 200,
            'hardening_law': get_hardening_law('nadai_ludwik'),
            'hardening_parameters': {'s0': 900,
                                     'a': 700,
                                     'b': 0.5,
                                     'ep0': 1e-5}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown material constitutive model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build material patch material parameters
    patch_material_params['mesh_elem_material'] = mesh_elem_material
    patch_material_params['mat_phases_descriptors'] = mat_phases_descriptors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_material_params
# =============================================================================
def perform_links_simulation(simulations_dir, patch, patch_material_params):
    """Perform finite element simulation: Links.
    
    Parameters
    ----------
    simulations_dir : str
        Simulations directory.
    patch : FiniteElementPatch
        Finite element patch.
    patch_material_params : dict
        Material patch material parameters.
        
    Returns
    -------
    links_file_path : str
        Links input data file path.
    links_output_directory : str
        Links simulation output directory.
    """
    # Set Links binary absolute path
    links_bin_path = ('/home/bernardoferreira/Documents/repositories/external/'
                      'LINKS/bin/LINKS')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links strain formulation
    strain_formulation = 'infinitesimal'
    # Set Links analysis type
    if patch.get_n_dim() == 2:
        analysis_type = 'plane_strain'
    else:
        analysis_type = 'tridimensional'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links input data file parameters
    links_input_params = {}
    links_input_params['number_of_increments'] = 1
    links_input_params['vtk_output'] = 'ASCII'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Links simulator
    links_simulator = LinksSimulator(links_bin_path, strain_formulation,
                                     analysis_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set configuration
    configuration = 'voids_lattice'
    # Get number of finite elements per dimension
    n_elems_per_dim = patch.get_n_elems_per_dim()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set elements to be removed (create structure internal voids)
    if configuration == 'voids_lattice':
        # Check number of elements along dimensions
        if any(x % 2 == 0 or x < 3 for x in n_elems_per_dim):
            raise RuntimeError('In order to generate a lattice configuration '
                               'with internal voids, the number of elements '
                               'along each dimension must be odd (avoid '
                               'removing boundary elements) and greater or '
                               'equal than 3.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element number
        elem_label = 1
        # Initialize removed elements
        remove_elements_labels = []
        # Select removed elements
        if patch.get_n_dim() == 2:
            # Loop over elements
            for j in range(n_elems_per_dim[1]):
                # Loop over elements
                for i in range(n_elems_per_dim[0]):
                    # Check void position
                    if all(x % 2 != 0 for x in (i, j)):
                        remove_elements_labels.append(elem_label)
                    # Update element number
                    elem_label += 1
        else:
            # Loop over elements
            for k in range(n_elems_per_dim[2]):
                # Loop over elements
                for j in range(n_elems_per_dim[1]):
                    # Loop over elements
                    for i in range(n_elems_per_dim[0]):
                        # Check void position
                        if all(x % 2 != 0 for x in (i, j, k)):
                            remove_elements_labels.append(elem_label)
                        # Update element number
                        elem_label += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        remove_elements_labels = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate Links input data file
    links_file_path = links_simulator.generate_input_data_file(
        'random_specimen', simulations_dir, patch, patch_material_params,
        remove_elements_labels=remove_elements_labels,
        links_input_params=links_input_params, is_overwrite_file=True,
        is_save_increm_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Links total loading factor history
    plot_links_tfact_hist(links_file_path, is_save_fig=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform Links simulation
    is_success, links_output_directory = \
        links_simulator.run_links_simulation(links_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check simulation status
    if not is_success:
        raise RuntimeError('Links simulation was not successful.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return links_file_path, links_output_directory
# =============================================================================
if __name__ == "__main__":
    # Set base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_paper_examples/global/specimens/random_material_patch/'
                'meshes')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch name
    patch_name = 'random_specimen'
    # Set number of material patches
    n_patch = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch storage flag
    is_save_material_patch = True
    # Set material patch loading flag
    is_load_material_patch = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check base directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patches directory
    if n_patch > 1:
        # Set (multiple) material patches directory
        material_patches_dir = os.path.join(os.path.normpath(base_dir),
                                            'material_patches_generation')
        # Create main material patch directory
        if not is_load_material_patch:
            make_directory(material_patches_dir, is_overwrite=True)
    else:
        # Set (single) material patch directory
        material_patches_dir = base_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patches
    links_file_paths, links_output_dirs = \
        generate_random_patches(material_patches_dir, patch_name=patch_name,
                                n_patch=n_patch,
                                is_save_material_patch=is_save_material_patch,
                                is_load_material_patch=is_load_material_patch,
                                is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material patches
    for patch_key in links_file_paths.keys():
        # Convert Links input data file to Abaqus data input data file
        abaqus_inp_file_path = \
            links_dat_to_abaqus_inp(links_file_paths[patch_key])
        # Convert Links '.nodedata' output files to '.csv' files
        csv_output_dir = links_node_output_to_csv(links_output_dirs[patch_key])   
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch directory
        material_patch_dir = \
            os.path.dirname(os.path.dirname(abaqus_inp_file_path))
        # Set material model finder directory
        material_model_finder_dir = os.path.join(
            os.path.normpath(material_patch_dir), 'material_model_finder')
        # Create material model finder directory
        make_directory(material_model_finder_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set simulation data directory
        simulation_dir = os.path.join(
            os.path.normpath(material_model_finder_dir), '0_simulation')
        # Create simulation data directory
        make_directory(simulation_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Copy Abaqus data input data file
        shutil.copy(abaqus_inp_file_path,
                    os.path.join(os.path.normpath(simulation_dir),
                                 os.path.basename(abaqus_inp_file_path)))
        # Copy node data directory
        shutil.copytree(csv_output_dir,
                        os.path.join(os.path.normpath(simulation_dir),
                                     os.path.basename(csv_output_dir)))
        
    