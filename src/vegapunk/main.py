"""Test script: GNN-based finite element material patch."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from patch_generator import FiniteElementPatchGenerator
from simulators.links.links import LinksSimulator
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
#
#                                                      OUTPUT WORKING DIRECTORY
# -----------------------------------------------------------------------------
# Set output working directory
directory = '/home/bernardoferreira/Documents/temp'
#
#                                                       GENERATE MATERIAL PATCH
# -----------------------------------------------------------------------------
# Set number of dimensions
n_dim = 2
# Set material patch dimensions
patch_dims = (1.0, 1.0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize material patch generator
patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set finite element discretization
elem_type = 'SQUAD8'
n_elems_per_dim = (4, 4)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set corners boundary conditions
corners_lab_bc = None
# Set corners displacement range
corners_lab_disp_range = {'1': ((-0.2, 0.2), (-0.2, 0.2)),
                          '2': ((-0.2, 0.2), (-0.2, 0.2)),
                          '3': ((-0.2, 0.2), (-0.2, 0.2)),
                          '4': ((-0.2, 0.2), (-0.2, 0.2))}
# Set edges polynomial deformation order and displacement range
edges_lab_def_order = {'1': 2, '2': 2, '3': 2, '4': 2}
edges_lab_disp_range = {'1': (-0.2, 0.2),
                        '2': (-0.2, 0.2),
                        '3': (-0.2, 0.2),
                        '4': (-0.2, 0.2)}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate randomly deformed material patch
is_admissible, patch = patch_generator.generate_deformed_patch(
    elem_type, n_elems_per_dim, corners_lab_bc=corners_lab_bc,
    corners_lab_disp_range=corners_lab_disp_range,
    edges_lab_def_order=edges_lab_def_order,
    edges_lab_disp_range=edges_lab_disp_range,
    is_verbose=True)
# Plot randomly deformed material patch
patch.plot_deformed_patch(is_save_plot=True, save_directory=directory)
#
#                                               SET SIMULATOR AND MATERIAL DATA
# -----------------------------------------------------------------------------
# Set simulator
links_bin_path = '/home/bernardoferreira/Documents/repositories/external/' \
    'CM2S/LINKS/bin/LINKS_debug'
strain_formulation = 'finite'
analysis_type = 'plane_strain'
links_simulator = LinksSimulator(links_bin_path, strain_formulation,
                                 analysis_type)
# Set simulation filename
filename = 'gnn_patch_test'
# Set simulator parameters
links_input_params = {}
links_input_params['number_of_increments'] = 10
links_input_params['vtk_output'] = 'ASCII'
# Set finite element patch material data
patch_material_data = {}
patch_material_data['mesh_elem_material'] = \
    np.ones(patch.get_n_elems_per_dim(), dtype=int)
patch_material_data['mat_phases_descriptors'] = {}
patch_material_data['mat_phases_descriptors']['1'] = \
    {'name': 'ELASTIC', 'density': 0.0, 'young': 210000, 'poisson': 0.3}
#
#                                                                RUN SIMULATION
# -----------------------------------------------------------------------------
# Generate simulator input data file
links_file_path = links_simulator.generate_input_data_file(
    filename, directory, patch, patch_material_data, links_input_params,
    is_overwrite_file=True, is_verbose=True)
# Run simulation
is_success, links_output_directory = \
    links_simulator.run_links_simulation(links_file_path, is_verbose=True)
# Read simulation results
results_keywords = ('node_data', 'global_data')
results = links_simulator.read_links_simulation_results(links_output_directory,
                                                        results_keywords,
                                                        is_verbose=True)
