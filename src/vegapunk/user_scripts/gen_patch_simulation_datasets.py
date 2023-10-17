"""User script: Generate material patch simulation data sets."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import numpy as np
# Local
from material_patch.patch_dataset import generate_material_patch_dataset
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def generate_dataset(case_study_name, simulation_directory, n_sample,
                     is_verbose=False):
    """Generate material patch simulation data sets.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    simulation_directory : str
        Directory where files associated with the generation of the material
        patch finite element simulations dataset are written. All existent
        files are overridden when saving new data files.
    n_sample : int
        Number of material patch samples.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Set default Links simulation parameters
    links_bin_path, strain_formulation, analysis_type, links_input_params = \
        set_default_links_parameters()
    # Set default files and directories storage options
    is_save_simulation_dataset, is_append_n_sample, is_save_simulation_files, \
        is_remove_failed_samples, is_save_plot_patch = \
            set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch simulation data set
    if case_study_name == '2d_elastic':
        # Set number of spatial dimensions
        n_dim = 2
        # Set finite element discretization
        elem_type = 'SQUAD4'
        n_elems_per_dim = (1, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set patch material data
        patch_material_data = set_patch_material_data(
            mode='homogeneous_elastic', n_elems_per_dim=n_elems_per_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set range of material patch size along each dimension
        patch_dims_ranges = {'1': (1.0, 2.0), '2': (1.0, 2.0)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set range of average deformation along each dimension for each corner
        avg_deformation_ranges = {'1': ((-0.2, 0.2), (-0.2, 0.2)),
                                  '2': ((-0.2, 0.2), (-0.2, 0.2)),
                                  '3': ((-0.2, 0.2), (-0.2, 0.2)),
                                  '4': ((-0.2, 0.2), (-0.2, 0.2))}
        # Set range of polynomial deformation order for each edge label
        edge_deformation_order_ranges = {'1': (1, 1),
                                         '2': (1, 1),
                                         '3': (1, 1),
                                         '4': (1, 1)}
        # Set range of polynomial deformation for each edge label
        edge_deformation_magnitude_ranges = {'1': (-0.2, 0.2),
                                             '2': (-0.2, 0.2),
                                             '3': (-0.2, 0.2),
                                             '4': (-0.2, 0.2)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save patch plot
        is_save_plot_patch = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch simulation data set
    dataset_simulation_data = generate_material_patch_dataset(
        n_dim, links_bin_path, strain_formulation, analysis_type,
        elem_type, n_elems_per_dim, patch_material_data,
        simulation_directory, n_sample=n_sample,
        patch_dims_ranges=patch_dims_ranges,
        avg_deformation_ranges=avg_deformation_ranges,
        edge_deformation_order_ranges=edge_deformation_order_ranges,
        edge_deformation_magnitude_ranges=
        edge_deformation_magnitude_ranges,
        is_remove_failed_samples = is_remove_failed_samples,
        links_input_params=links_input_params,
        is_save_simulation_dataset=is_save_simulation_dataset,
        is_append_n_sample=is_append_n_sample, 
        is_save_plot_patch=is_save_plot_patch,
        is_save_simulation_files=is_save_simulation_files,
        is_verbose=is_verbose)
# =============================================================================
def set_patch_material_data(mode, n_elems_per_dim):
    """Set patch material data.
    
    Parameters
    ----------
    mode : {'homogeneous_elastic',}
        Patch material identifier.
    n_elems_per_dim : tuple[int]
        Number of finite elements per dimension that completely defines the
        regular finite element mesh by assuming equal-sized elements. If not
        specified, a single finite element is assumed.

    Returns
    -------
    patch_material_data : dict
        Finite element patch material data. Expecting
        'mesh_elem_material': numpy.ndarray [int](n_elems_per_dim) (finite
        element mesh elements material matrix where each element
        corresponds to a given finite element position and whose value is
        the corresponding material phase (int)) and
        'mat_phases_descriptors': dict (constitutive model descriptors
        (item, dict) for each material phase (key, str[int])).
    """
    # Initialize finite element patch material data
    patch_material_data = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch data
    if mode == 'homogeneous_elastic':
        # Set material phases spatial domains
        patch_material_data['mesh_elem_material'] = \
            np.ones(n_elems_per_dim, dtype=int)
        # Set material phase descriptors
        patch_material_data['mat_phases_descriptors'] = {}
        patch_material_data['mat_phases_descriptors']['1'] = \
            {'name': 'ELASTIC',
             'density': 0.0,
             'young': 210000,
             'poisson': 0.3}
    else:
        raise RuntimeError('Unknown patch material identifier.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return patch_material_data
# =============================================================================
def set_default_links_parameters():
    """Set default Links simulation parameters.
    
    Returns
    -------
    links_bin_path : str
        Links binary absolute path.
    strain_formulation: {'infinitesimal', 'finite'}
        Links strain formulation.
    analysis_type : {'plane_stress', 'plane_strain', 'axisymmetric', \
                     'tridimensional'}
        Links analysis type.
    links_input_params : dict
        Links input data file parameters.
    """
    # Set Links binary absolute path
    links_bin_path = '/home/bernardoferreira/Documents/repositories/' \
        + 'external/CM2S/LINKS/bin/LINKS_debug'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links strain formulation and analysis type
    strain_formulation = 'finite'
    analysis_type = 'plane_strain'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links input data file parameters
    links_input_params = {}
    links_input_params['number_of_increments'] = 10
    links_input_params['vtk_output'] = 'ASCII'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return links_bin_path, strain_formulation, analysis_type, \
        links_input_params
# =============================================================================
def set_default_saving_options():
    """Set default files and directories storage options.
    
    Returns
    -------
    is_save_simulation_dataset : bool
        Save material patch finite element simulations data set in simulation
        directory.
    is_append_n_sample : bool
        If True, then data set size (number of samples) is appended to
        material patch finite element simulations data set filename.
    is_save_simulation_files : bool
        Save material patch simulation files in simulation directory.
    is_remove_failed_samples : bool
        Remove failed material patches from data set. Size of resulting data
        set is lower or equal to prescribed number of material patch samples.
    is_save_plot_patch : bool
        Save plot of material patch design sample in simulation directory.
    """
    is_save_simulation_dataset = True
    is_append_n_sample = True
    is_save_simulation_files = False
    is_remove_failed_samples = True
    is_save_plot_patch = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_save_simulation_dataset, is_append_n_sample, \
        is_save_simulation_files, is_remove_failed_samples, is_save_plot_patch
# =============================================================================
if __name__ == "__main__":
    # Set case study name
    case_study_name = '2d_elastic'
    # Set case study directory
    case_study_base_dirs = {
        '2d_elastic': f'/home/bernardoferreira/Documents/temp',}
    case_study_dir = \
        os.path.join(os.path.normpath(case_study_base_dirs[case_study_name]),
                     f'cs_{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch simulation data set size
    n_sample = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set simulation directory
    simulation_directory = os.path.join(os.path.normpath(case_study_dir),
                                        '0_simulation')
    # Create simulation directory
    if not os.path.isdir(simulation_directory):
        make_directory(simulation_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch simulation data set
    generate_dataset(case_study_name, simulation_directory, n_sample,
                     is_verbose=True)