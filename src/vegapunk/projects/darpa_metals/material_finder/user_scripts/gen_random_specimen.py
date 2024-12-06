"""DARPA METALS PROJECT: Generate random specimen.

Functions
---------
generate_random_specimen
    Generate random specimen.
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
import re
# Third-party
import numpy as np
# Local
from simulators.links.links import LinksSimulator
from projects.gnn_material_patch.material_patch.patch_generator import \
    FiniteElementPatchGenerator
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
def generate_random_specimen(simulations_dir, plots_dir):
    """Generate random specimen.
    
    Parameters
    ----------
    simulations_dir : str
        Simulations directory.
    plots_dir : str
        Plots directory.
    """
    # Set plot deformed material patch flag
    is_save_plot_patch = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material patch geometric parameters
    patch_geometric_params = set_patch_geometric_params(n_dim)
    # Get material patch mesh parameters
    patch_mesh_params = set_patch_mesh_params(n_dim)
    # Get material patch material parameters
    patch_material_params = set_patch_material_params(
        patch_mesh_params['n_elems_per_dim'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch
    patch = generate_material_patch(patch_geometric_params, patch_mesh_params)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save plot of deformed material patch
    if is_save_plot_patch:
        patch.plot_deformed_patch(
            is_hide_axes=False, is_show_fixed_dof=True,
            is_save_plot=is_save_plot_patch, save_directory=plots_dir,
            plot_name='specimen_deformed_configuration',
            is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform Links simulation
    perform_links_simulation(simulations_dir, patch, patch_material_params)
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
        avg_deformation_ranges = {'1': ((-0.1, 0.1), (-0.1, 0.1)),
                                  '2': ((-0.1, 0.1), (-0.1, 0.1)),
                                  '3': ((-0.1, 0.1), (-0.1, 0.1)),
                                  '4': ((-0.1, 0.1), (-0.1, 0.1))}
        # Set polynomial deformation order for each edge label
        edge_deformation_order = {'1': 2, '2': 3, '3': 2, '4': 3}
        # Set range of polynomial deformation for each edge label
        edge_deformation_magnitude_ranges = {'1': (-0.2, 0.2),
                                             '2': (-0.2, 0.2),
                                             '3': (-0.2, 0.2),
                                             '4': (-0.2, 0.2)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set rigid body translation
        translation_range = {'1': (-0.5, 0.5), '2': (-0.5, 0.5)}
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
            {str(i): ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
             for i in range(1, 9)}
        # Set polynomial deformation order for each edge label
        edge_deformation_order = \
            {str(i): 3 for i in range(1, 25)}
        # Set range of polynomial deformation for each edge label
        edge_deformation_magnitude_ranges = \
            {str(i): (-0.1, 0.1) for i in range(1, 25)}
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
        n_elems_per_dim = (6, 3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif n_dim == 3:
        # Set finite element type
        elem_type = 'SHEXA8'
        # Set number of finite elements per dimension
        n_elems_per_dim = (4, 3, 2)
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
    # Set material phase descriptors
    mat_phases_descriptors = {}
    mat_phases_descriptors['1'] = {'name': 'ELASTIC',
                                   'density': 0.0,
                                   'young': 110e3,
                                   'poisson': 0.33}
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
    # Set elements to be removed (create structure internal voids)
    remove_elements_labels = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate Links input data file
    links_file_path = links_simulator.generate_input_data_file(
        'random_specimen', simulations_dir, patch, patch_material_params,
        remove_elements_labels=remove_elements_labels,
        links_input_params=links_input_params, is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform Links simulation
    is_success, _ = \
        links_simulator.run_links_simulation(links_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check simulation status
    if not is_success:
        raise RuntimeError('Links simulation was not successful.')
# =============================================================================
if __name__ == "__main__":
    # Set base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/test_patches')
    # Set specimen name
    specimen_name = 'random_specimen'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check base directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set specimen directory
    specimen_dir = os.path.join(os.path.normpath(base_dir), specimen_name)
    # Create simulations directory (overwrite)
    make_directory(specimen_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set simulations directory
    simulations_dir = \
        os.path.join(os.path.normpath(specimen_dir), 'simulations')
    # Create simulations directory
    if not os.path.isdir(simulations_dir):
        make_directory(simulations_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = os.path.join(os.path.normpath(specimen_dir), 'plots')
    # Create plots directory
    if not os.path.isdir(plots_dir):
        make_directory(plots_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random specimen
    generate_random_specimen(simulations_dir, plots_dir)