"""Generate and simulate set of deformed finite element material patches.

Classes
-------
MaterialPatchSimulator(f3dasm.datageneration.DataGenerator)
    Material patch generator and simulator.

Functions
---------
generate_material_patch_dataset
    Generate and simulate a set of deformed finite element material patches.
get_default_design_parameters
    Generate finite element material patch design space default parameters.
read_simulation_dataset_from_file
    Read material patch finite element data set from file.
write_patch_dataset_summary_file
    Write summary data file for material patch data set generation.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
import pickle
import time
import datetime
import random
# Third-party
import numpy as np
import f3dasm
import f3dasm.datageneration
# Local
from gnn_material_patch.material_patch.patch_generator import \
    FiniteElementPatchGenerator
from simulators.links.links import LinksSimulator
from ioput.iostandard import write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Pre-Alpha'
# =============================================================================
#
# =============================================================================
def generate_material_patch_dataset(
    n_dim, links_bin_path, strain_formulation, analysis_type, elem_type,
    n_elems_per_dim, patch_material_data, simulation_directory, n_sample=1,
    patch_dims_ranges=None, avg_deformation_ranges=None,
    edge_deformation_order_ranges=None, edge_deformation_magnitude_ranges=None,
    translation_range=None, rotation_angles_range=None,
    max_iter_per_patch=10, is_remove_failed_samples=False,
    links_input_params=None, is_save_simulation_dataset=False,
    is_append_n_sample=True, is_save_simulation_files=False,
    is_save_plot_patch=False, is_verbose=False):
    """Generate and simulate a set of deformed finite element material patches.
    
    Material patch is assumed quadrilateral (2d) or parallelepipedic (3D)
    and discretized in a regular finite element mesh of quadrilateral (2d) /
    hexahedral (3d) finite elements.
    
    Simulations are performed with Links (Large Strain Implicit Nonlinear
    Analysis of Solids Linking Scales), a finite element code developed by the
    CM2S research group at the Faculty of Engineering, University of Porto.
    
    Outputs for each sample include nodal and global features data across
    multiple time steps of the simulation deformation path.
    
    Material patch finite element simulation data set is stored in
    simulation_directory as a pickle file named material_patch_fem_dataset.pkl
    or material_patch_fem_dataset_n< n_sample >.pkl
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    links_bin_path : str
        Links binary absolute path.
    strain_formulation: {'infinitesimal', 'finite'}
        Links strain formulation.
    analysis_type : {'plane_stress', 'plane_strain', 'axisymmetric', \
                     'tridimensional'}
        Links analysis type.
    elem_type : str, default=None
        Finite element type employed to discretize the material patch in a
        regular finite element mesh. Available quadrilateral (2d) /
        hexahedral (3d) finite elements:
        
        'SQUAD4'  : 4-node linear element of Serendipity family (2d)
        
        'SQUAD8'  : 8-node quadratic element of Serendipity family (2d)
        
        'SQUAD12' : 12-node cubic element of Serendipity family (2d)
        
        'LQUAD4'  : 4-node linear element of Lagrangian family (2d)
        
        'LQUAD9'  : 9-node quadratic element of Lagrangian family (2d)
        
        'LQUAD16' : 16-node cubic element of Lagrangian family (2d)
        
        If not specified, 'SQUAD4' (2d) is assumed.
    n_elems_per_dim : tuple[int], default=None
        Number of finite elements per dimension that completely defines the
        regular finite element mesh by assuming equal-sized elements. If not
        specified, a single finite element is assumed.
    patch_material_data : dict
        Finite element patch material data. Expecting
        'mesh_elem_material': numpy.ndarray [int](n_elems_per_dim) (finite
        element mesh elements material matrix where each element
        corresponds to a given finite element position and whose value is
        the corresponding material phase (int)) and
        'mat_phases_descriptors': dict (constitutive model descriptors
        (item, dict) for each material phase (key, str[int])).
    simulation_directory : str
        Directory where files associated with the generation of the material
        patch finite element simulations dataset are written. All existent
        files are overridden when saving new data files.
    n_sample : int, default=1
        Number of material patch samples.
    patch_dims_ranges : dict, default=None
        Range of material patch size (item, tuple[float](2)) along each
        dimension (key, str). The range is specified as a
        tuple(lower_bound, upper_bound). Range defaults to (1.0, 1.0) if not
        specified.
    avg_deformation_ranges : dict, default=None
        Range of average deformation along each dimension
        (item, tuple[tuple(2)]) for each corner label (key, str[int]). Corners
        are labeled from 1 to number of corners. The deformation is relative to
        the material patch size along the corresponding dimension. The range
        for each dimension is specified as a tuple(lower_bound, upper_bound)
        and where positive/negative values are associated with
        tension/compression.
    edge_deformation_order_ranges : dict, default=None
        Range of polynomial deformation order (item, tuple[int](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The range is specified as a tuple(lower_bound, upper_bound),
        where the minimum allowed is zero order.
    edge_deformation_magnitude_ranges : dict, default=None
        Range of polynomial deformation (item, tuple[float](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The edge deformation is orthogonal to the corresponding
        direction (defined by limiting corner nodes in the deformed
        configuration) and its magnitude is relative to the material patch size
        orthogonal to its dimension in the reference configuration. The range
        is specified as a tuple(lower_bound, upper_bound) and where
        positive/negative values are associated with tension/compression.
    translation_range : dict, default=None
        Translational displacement range (item, tuple[float](2)) along each
        dimension (key, str[int]). Range is specified as tuple(min, max)
        for each dimension. Null range is assumed for unspecified
        dimensions. If None, then there is no translational motion.
    rotation_angles_range : dict, default=None
        Rotational angle range (item, tuple[float](2)) for each Euler angle
        (key, str). Euler angles follow Bunge convention (Z1-X2-Z3) and are
        labelled ('alpha', 'beta', 'gamma'), respectively. Null range is
        assumed for unspecified angles. If None, then there is no
        rotational motion.
    max_iter_per_patch : int, default=10
        Maximum number of iterations to get a geometrically admissible
        deformed patch configuration.
    is_remove_failed_samples : bool, default=False
        Remove failed material patches from data set. Size of resulting data
        set is lower or equal to prescribed number of material patch samples.
    links_input_params : dict, default=None
        Links input data file parameters. If None, default parameters are set.
    is_save_simulation_dataset : bool, default=False
        Save material patch finite element simulations data set in simulation
        directory. Existing data set file is overriden.
    is_append_n_sample : bool, default=True
        If True, then data set size (number of samples) is appended to
        material patch finite element simulations data set filename.
    is_save_simulation_files : bool, default=False
        Save material patch simulation files in simulation directory.
    is_save_plot_patch : bool, default=False
        Save plot of material patch design sample in simulation directory.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset_simulation_data : list[dict]
        Material patches finite element simulations data set. Each material
        patch data is stored in a dict, where:
        
        'patch' : Instance of FiniteElementPatch, the simulated finite \
                  element material patch.
        
        'node_data' : numpy.ndarray(3d) of shape \
                      (n_nodes, n_data_dim, n_time_steps), where the i-th \
                      node output data at the k-th time step is stored in \
                      indexes [i, :, k].
        
        'global_data' : numpy.ndarray(3d) of shape \
                        (1, n_data_dim, n_time_steps) where the global output \
                        data at the k-th time step is stored in [0, :, k].
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGenerate material patch simulation data set'
              '\n-------------------------------------------')
        print('\n> Setting default design space parameters...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check simulation directory
    if not os.path.isdir(simulation_directory):
        raise RuntimeError('The simulation directory has not been found:\n\n'
                           + simulation_directory)
    else:
        simulation_directory = os.path.normpath(simulation_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get default design space parameters
    default_parameters = get_default_design_parameters(n_dim)
    # Set default parameters
    if patch_dims_ranges is None:
        patch_dims_ranges = default_parameters['patch_dims_ranges']
    if avg_deformation_ranges is None:
        avg_deformation_ranges = default_parameters['avg_deformation_ranges']
    if edge_deformation_order_ranges is None:
        edge_deformation_order_ranges = \
            default_parameters['edge_deformation_order_ranges']
    if edge_deformation_magnitude_ranges is None:
        edge_deformation_magnitude_ranges = \
            default_parameters['edge_deformation_magnitude_ranges']
    if translation_range is None:
        translation_range = \
            default_parameters['translation_range']
    if rotation_angles_range is None:
        rotation_angles_range = \
            default_parameters['rotation_angles_range']
    if elem_type is None:
        elem_type = default_parameters['elem_type']
    if n_elems_per_dim is None:
        n_elems_per_dim = default_parameters['n_elems_per_dim']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Setting input constant parameters...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input constant parameters
    constant_parameters = {
        'n_dim': n_dim,
        'elem_type': elem_type,
        'n_elems_per_dim': n_elems_per_dim,
        'edge_deformation_order_ranges': edge_deformation_order_ranges,
        'max_iter': max_iter_per_patch,
        'links_bin_path': links_bin_path,
        'strain_formulation': strain_formulation,
        'analysis_type': analysis_type,
        'sample_basename': 'material_patch',
        'directory': simulation_directory,
        'patch_material_data': patch_material_data,
        'links_input_params': links_input_params,
        'is_save_simulation_files': is_save_simulation_files,
        'is_save_plot_patch': is_save_plot_patch} 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Building design space:')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize design space
    domain = f3dasm.design.Domain()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('  > [ input parameter] Setting material patch size...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch size
    #
    # Loop over dimensions
    for i in range(n_dim):
        # Set name
        name = 'patch_size_' + str(i + 1)
        # Set bounds
        lower_bound = patch_dims_ranges[str(i + 1)][0]
        upper_bound = patch_dims_ranges[str(i + 1)][1]
        # Add design input parameter
        if np.isclose(lower_bound, upper_bound):
            domain.add_constant(name=name, value=lower_bound)
        else:
            domain.add_float(name=name, low=lower_bound, high=upper_bound)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('  > [ input parameter] Setting material patch average '
              'deformation...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch average deformation
    #
    # Set number of corners
    if n_dim == 2:
        n_corners = 4
    else:
        raise RuntimeError('Missing 3D implementation.')
    # Loop over dimensions
    for i in range(n_dim):
        # Loop over corners
        for j in range(n_corners):
            # Set corner label
            label = str(j + 1)
            # Set name
            name = 'corner_' + label + '_deformation_' + str(i + 1)
            # Set bounds
            lower_bound = avg_deformation_ranges[label][i][0]
            upper_bound = avg_deformation_ranges[label][i][1]
            # Add design input parameter
            if np.isclose(lower_bound, upper_bound):
                domain.add_constant(name=name, value=lower_bound) 
            else:
                domain.add_float(name=name, low=lower_bound, high=upper_bound)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('  > [ input parameter] Setting material patch edge polynomial '
              'deformation order...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch polynomial deformation order
    #
    # Set number of edges
    if n_dim == 2:
        n_edges = 4
    else:
        raise RuntimeError('Missing 3D implementation.')
    # Loop over edges
    for i in range(n_edges):
        # Set edge label
        label = str(i + 1)
        # Set name
        name = 'edge_' + label + '_deformation_order'
        # Set bounds
        lower_bound = edge_deformation_order_ranges[label][0]
        upper_bound = edge_deformation_order_ranges[label][1]
        # Add design input parameter
        if np.isclose(lower_bound, upper_bound):
            domain.add_constant(name=name, value=lower_bound)
        else:
            domain.add_int(name=name, low=lower_bound, high=upper_bound)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch rigid body translation
    #
    # Loop over dimensions
    for i in range(n_dim):
        # Set name
        name = f'translation_{i}'
        # Set bounds
        lower_bound = translation_range[str(i + 1)][0]
        upper_bound = translation_range[str(i + 1)][1]
        # Add design input parameter
        if np.isclose(lower_bound, upper_bound):
            domain.add_constant(name=name, value=lower_bound)
        else:
            domain.add_float(name=name, low=lower_bound, high=upper_bound)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch rigid body rotation
    #
    # Loop over dimensions
    for euler_angle in ('alpha', 'beta', 'gamma'):
        # Set name
        name = f'rotation_angle_{euler_angle}'
        # Set bounds
        lower_bound = rotation_angles_range[euler_angle][0]
        upper_bound = rotation_angles_range[euler_angle][1]
        # Add design input parameter
        if np.isclose(lower_bound, upper_bound):
            domain.add_constant(name=name, value=lower_bound)
        else:
            domain.add_float(name=name, low=lower_bound, high=upper_bound)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set experiment data directory
    experiment_data_dir = os.path.join(simulation_directory, 'experiment_data')
    # Initialize experiment data
    experiment_data = f3dasm.ExperimentData(domain,
                                            project_dir=experiment_data_dir)  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Sampling design space input data...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate samples
    experiment_data.sample(sampler='sobol', n_samples=n_sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting samples generation process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patch simulator
    material_patch_simulator = MaterialPatchSimulator()
    # Generate samples simulation data
    experiment_data.evaluate(data_generator=material_patch_simulator,
                             mode='parallel',
                             kwargs=constant_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material patches finite element simulations data set
    dataset_simulation_data = []
    # Build material patches finite element simulations data set
    for i in experiment_data.index:
        # Get material patch sample output data
        sample_output_data = \
            experiment_data.get_experiment_sample(i).output_data
        # Append sample output data
        dataset_simulation_data.append(sample_output_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove failed material patch samples from data set
    if is_remove_failed_samples:
        # Initialize failed samples indexes
        n_fail_ids = []
        # Loop over data set samples
        for i in range(len(dataset_simulation_data)):
            # Get material patch and simulation results
            patch = dataset_simulation_data[i]['patch']
            node_data = dataset_simulation_data[i]['node_data']
            # Check if failed sample
            if patch is None or node_data is None:
                n_fail_ids.append(i)
        # Remove failed samples from data set
        for i in sorted(n_fail_ids, reverse=True):
            del dataset_simulation_data[i]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize samples status counters
    n_success = 0
    n_fail_patch = 0
    n_fail_simulation = 0
    # Loop over data set samples
    for i in range(len(dataset_simulation_data)):
        # Get material patch and simulation results
        patch = dataset_simulation_data[i]['patch']
        node_data = dataset_simulation_data[i]['node_data']
        # Increment counter according to sample status
        if patch is None:
            n_fail_patch += 1
        elif patch is not None and node_data is None:
            n_fail_simulation += 1
        else:
            n_success += 1
    # Compute total number of failed samples
    n_failure = n_fail_patch + n_fail_simulation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Material patch simulation data set status:')
        print('\n  > Prescribed number of material patch samples: ', n_sample)
        print('\n  > Successful material patch samples: ',
              f'{n_success:d}/{n_sample:d} ({100*n_success/n_sample:>.1f}%)')
        if n_failure > 0:
            print('\n  > Failed material patch samples: ',
                  (f'{n_failure:d}/{n_sample:d} '
                   f'({100*n_failure/n_sample:>.1f}%)'))
            print('\n    > Non-admissible deformed configuration: ',
                  (f'{n_fail_patch:d}/{n_sample:d} '
                   f'({100*n_fail_patch/n_sample:>.1f}%)'))
            print('    > Failed finite element simulation: ',
                  (f'{n_fail_simulation:d}/{n_sample:d} '
                   f'({100*n_fail_simulation/n_sample:>.1f}%)'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished material patches generation process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_save_simulation_dataset:
        if is_verbose:
            print('\n> Saving material patch simulation data set...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set file
        dataset_file_name = 'material_patch_fem_dataset'
        # Append data set size
        if is_append_n_sample:
            dataset_file_name += f'_n{len(dataset_simulation_data)}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set file path
        dataset_file_path = os.path.join(
            os.path.normpath(simulation_directory), dataset_file_name + '.pkl')
        # Save data set
        with open(dataset_file_path, 'wb') as dataset_file:
            pickle.dump(dataset_simulation_data, dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total generation time and average generation time per patch
    total_time_sec = time.time() - start_time_sec
    avg_time_sec = total_time_sec/n_sample
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Data set directory: ', simulation_directory)
        print(f'\n> Total generation time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. generation time per patch: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for material patch data set generation
    write_patch_dataset_summary_file(
        simulation_directory, n_sample, n_dim, strain_formulation,
        analysis_type, patch_dims_ranges, elem_type, n_elems_per_dim,
        avg_deformation_ranges, edge_deformation_order_ranges,
        edge_deformation_magnitude_ranges, translation_range,
        rotation_angles_range, patch_material_data,
        n_success, n_failure, n_fail_patch, n_fail_simulation, total_time_sec,
        avg_time_sec)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_simulation_data
# =============================================================================
class MaterialPatchSimulator(f3dasm.datageneration.DataGenerator):
    """Material patch generator and simulator.
    
    Attributes
    ----------
    experiment_sample : f3dasm.ExperimentSample
        Material patch sample.
            
    Methods
    -------
    execute(self, **kwargs)
        Core data generation.
    simulate_material_patch(self, experiment_sample, **kwargs)
        Generate and simulate finite element material patch sample.
    """
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    def execute(self, **kwargs):
        """Core data generation.
        
        Parameters
        ----------
        kwargs : dict
            Any arguments required to generate and simulate finite element
            material patch sample that are not available from the sample
            object itself.
        """
        self.simulate_material_patch(self.experiment_sample, **kwargs)
    # -------------------------------------------------------------------------
    def simulate_material_patch(self, experiment_sample, **kwargs):
        """Generate and simulate finite element material patch sample.
        
        Parameters
        ----------
        experiment_sample : f3dasm.ExperimentSample
            Material patch sample.
        kwargs : dict
            Any arguments required to generate and simulate finite element
            material patch sample that are now available from the sample
            object itself.
        """
        # Get material patch design sample ID
        sample_id = experiment_sample.job_number
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reseed random generators
        random.seed()
        np.random.seed()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of spatial dimensions                                         
        n_dim = kwargs['n_dim']
        # Get finite element type
        elem_type = kwargs['elem_type']
        # Get number of finite elements per dimension
        n_elems_per_dim = kwargs['n_elems_per_dim']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch dimensions 
        patch_dims = tuple([experiment_sample.get('patch_size_' + str(i + 1))
                            for i in range(n_dim)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of corners
        if n_dim == 2:
            n_corners = 4
        else:
            raise RuntimeError('Missing 3D implementation.')
        # Set corners labels
        corners_labels = tuple([str(i + 1) for i in range(n_corners)])
        # Set corners tensile/compression displacements mapping
        if n_dim == 2:
            corners_disp_sign = {'1': (-1, -1), '2': (1, -1),
                                 '3': (1, 1), '4': (-1, 1)}
        else:
            raise RuntimeError('Missing 3D implementation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize corners displacements
        corners_lab_disp_range = {}
        # Loop over corners
        for label in corners_labels:
            # Initialize corner displacement
            corner_disp = []
            # Loop over dimensions
            for i in range(n_dim):
                # Set parameter name
                name = 'corner_' + label + '_deformation_' + str(i + 1)
                # Get average deformation
                avg_deformation = experiment_sample.get(name)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute corner displacement along dimension
                disp = (corners_disp_sign[label][i]
                        *0.5*avg_deformation*patch_dims[i])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Append corner displacement
                corner_disp.append(disp)
            # Set corner displacement
            corners_lab_disp_range[label] = tuple([n_dim*(corner_disp[i],)
                                                   for i in range(n_dim)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of edges and deformation dimension
        if n_dim == 2:
            n_edges = 4
            edges_def_dimension = {'1': '2', '2': '2', '3': '1', '4': '1'}
        else:
            raise RuntimeError('Missing 3D implementation.')
        # Set edges labels
        edges_labels = tuple([str(i + 1) for i in range(n_edges)])    
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize edges polynomial deformation order
        edges_lab_def_order = {}
        # Loop over edges
        for label in edges_labels:
            # Set parameter name
            name = 'edge_' + label + '_deformation_order'
            # Get polynomial deformation order
            order = experiment_sample.get(name)
            # Set edge polynomial deformation order
            edges_lab_def_order[label] = int(order)                            # Unexpected behavior: int was converted to float
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get edges polynomial deformation magnitude range
        edge_deformation_order_ranges = kwargs['edge_deformation_order_ranges']
        # Initialize edges polynomial deformation displacement range
        edges_lab_disp_range = {}
        # Loop over edges
        for label in edges_labels:
            # Get edge polynomial deformation magnitude range
            avg_deformation_range = edge_deformation_order_ranges[label]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get orthogonal (deformation) dimension index
            orth_dim = int(edges_def_dimension[label]) - 1
            # Compute edge displacement range
            min_disp = 0.5*avg_deformation_range[0]*patch_dims[orth_dim]
            max_disp = 0.5*avg_deformation_range[1]*patch_dims[orth_dim]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set edge displacement range
            edges_lab_disp_range[label] = (min_disp, max_disp)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize rigid body translation
        translation_range = {}
        # Loop over dimensions
        for i in range(n_dim):
            # Set parameter name
            name = f'translation_{i}'
            # Get translation along dimension
            disp = experiment_sample.get(name)
            # Set translation along dimension
            translation_range[str(i + 1)] = 2*(disp,)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize rigid body rotation
        rotation_angles_range = {}
        # Loop over Euler angles
        for euler_angle in ('alpha', 'beta', 'gamma'):
            # Set name
            name = f'rotation_angle_{euler_angle}'
            # Get rotation angle
            angle = experiment_sample.get(name)
            # Set translation along dimension
            rotation_angles_range[euler_angle] = 2*(angle,)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Initialize material patch generator
        patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
        # Generate deformed material patch
        is_admissible, patch = patch_generator.generate_deformed_patch(
            elem_type, n_elems_per_dim,
            corners_lab_disp_range=corners_lab_disp_range,
            edges_lab_def_order=edges_lab_def_order,
            edges_lab_disp_range=edges_lab_disp_range,
            translation_range=translation_range,
            rotation_angles_range=rotation_angles_range)
        # Save plot of deformed material patch
        is_save_plot_patch = kwargs['is_save_plot_patch']
        if is_save_plot_patch and is_admissible:
            patch.plot_deformed_patch(
                is_save_plot=is_save_plot_patch,
                save_directory=kwargs['directory'],
                plot_name=kwargs['sample_basename'] + f'_{sample_id}_plot',
                is_overwrite_file=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize results
        results_keywords = ('node_data', 'global_data')
        results = {key: None for key in results_keywords}
        # Simulate material patch
        if is_admissible:
            # Initialize simulator
            links_simulator = LinksSimulator(kwargs['links_bin_path'],
                                             kwargs['strain_formulation'],
                                             kwargs['analysis_type'])
            # Generate simulator input data file
            links_file_path = links_simulator.generate_input_data_file(
                kwargs['sample_basename'] + f'_{sample_id}',
                kwargs['directory'], patch, kwargs['patch_material_data'],
                kwargs['links_input_params'], is_overwrite_file=True)
            # Run simulation
            is_success, links_output_directory = \
                links_simulator.run_links_simulation(links_file_path)
            # Read simulation results
            if is_success:
                results = links_simulator.read_links_simulation_results(
                    links_output_directory, results_keywords)
            # Remove simulation files
            if not kwargs['is_save_simulation_files']:
                # Remove simulator input data file
                if os.path.isfile(links_file_path):
                    os.remove(links_file_path)
                # Remove simulator output directory
                if os.path.isdir(links_output_directory):
                    shutil.rmtree(links_output_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch sample output data
        experiment_sample.store(name='patch', object=patch, to_disk=True)
        experiment_sample.store(name='node_data', object=results['node_data'],
                                to_disk=True)
        experiment_sample.store(name='global_data',
                                object=results['global_data'],
                                to_disk=True)
# =============================================================================
def get_default_design_parameters(n_dim):
    """Generate finite element material patch design space default parameters.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.

    Returns
    -------
    default_parameters : dict
        Design space default parameters for finite element material patch.
    """
    # Set number of corners and edges
    if n_dim == 2:
        n_corners = 4
        n_edges = 4
    else:
        n_corners = 8
        n_edges = 12
    # Set corners and edges labels
    corners_labels = tuple([str(i + 1) for i in range(n_corners)])
    edges_labels = tuple([str(i + 1) for i in range(n_edges)]) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default finite element type and number of finite elements per
    # dimension
    if n_dim == 2:
        elem_type = 'SQUAD4'
        n_elems_per_dim = (1, 1)
    else:
        raise RuntimeError('Missing 3D implementation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default patch size
    patch_dims_ranges = {str(i + 1): (1.0, 1.0) for i in range(n_dim)}
    # Set default range of average deformation (corners)
    avg_deformation_ranges = {label: n_dim*((0, 0),)
                              for label in corners_labels}
    # Set default polynomial deformation order (edges)
    edge_deformation_order_ranges = {label: (0, 0) for label in edges_labels}
    # Set default polynomial deformation magnitude (edges)
    edge_deformation_magnitude_ranges = {label: (0, 0)
                                         for label in edges_labels}
    # Set default rigid body translation
    translation_range = {str(i + 1): (0.0, 0.0) for i in range(n_dim)}
    # Set default rigid body rotation
    rotation_angles_range = {angle: (0.0, 0.0)
                             for angle in ('alpha', 'beta', 'gamma')}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store default parameters
    default_parameters = {
        'elem_type': elem_type,
        'n_elems_per_dim': n_elems_per_dim,
        'patch_dims_ranges': patch_dims_ranges,
        'avg_deformation_ranges': avg_deformation_ranges,
        'edge_deformation_order_ranges': edge_deformation_order_ranges,
        'edge_deformation_magnitude_ranges': edge_deformation_magnitude_ranges,
        'translation_range': translation_range,
        'rotation_angles_range': rotation_angles_range}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return default_parameters
# =============================================================================
def read_simulation_dataset_from_file(dataset_file_path):
    """Read material patch finite element data set from file.

    Parameters
    ----------
    dataset_file_path : str
        Material patches finite element simulations data set file path.
    
    Returns
    -------
    dataset_simulation_data : list[dict]
        Material patches finite element simulations data set. Each material
        patch data is stored in a dict, where:
        
        'patch' : Instance of FiniteElementPatch.
        
        'node_data' : numpy.ndarray(3d) of shape \
                      (n_nodes, n_data_dim, n_time_steps), where the i-th \
                      node output data at the k-th time step is stored in \
                      indexes [i, :, k].
        
        'global_data' : numpy.ndarray(3d) of shape \
                        (1, n_data_dim, n_time_steps) where the global output \
                        data at the k-th time step is stored in [0, :, k].
    """
    # Check data set file path
    if not os.path.isfile(dataset_file_path):
        raise RuntimeError('Material patch finite element data set file '
                           'has not been found:\n\n', dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material patch finite element data set
    with open(dataset_file_path, 'rb') as dataset_file:
        dataset_simulation_data = pickle.load(dataset_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_simulation_data
# =============================================================================
def write_patch_dataset_summary_file(
    simulation_directory, n_sample, n_dim, strain_formulation, analysis_type,
    patch_dims_ranges, elem_type, n_elems_per_dim, avg_deformation_ranges,
    edge_deformation_order_ranges, edge_deformation_magnitude_ranges,
    translation_range, rotation_angles_range, patch_material_data, n_success,
    n_failure, n_fail_patch, n_fail_simulation, total_time_sec,
    avg_time_sample):
    """Write summary data file for material patch data set generation.
    
    Parameters
    ----------
    simulation_directory : str
        Directory where files associated with the generation of the material
        patch finite element simulations dataset are written. All existent
        files are overridden when saving new data files.
    n_dim : int
        Number of spatial dimensions.
    strain_formulation: {'infinitesimal', 'finite'}
        Links strain formulation.
    analysis_type : {'plane_stress', 'plane_strain', 'axisymmetric', \
                     'tridimensional'}
        Links analysis type.
    patch_dims_ranges : dict
        Range of material patch size (item, tuple[float](2)) along each
        dimension (key, str). The range is specified as a
        tuple(lower_bound, upper_bound). Range defaults to (1.0, 1.0) if not
        specified.
    elem_type : str
        Finite element type employed to discretize the material patch in a
        regular finite element mesh.
    n_elems_per_dim : tuple[int]
        Number of finite elements per dimension that completely defines the
        regular finite element mesh by assuming equal-sized elements. If not
        specified, a single finite element is assumed.
    avg_deformation_ranges : dict
        Range of average deformation along each dimension
        (item, tuple[tuple(2)]) for each corner label (key, str[int]). Corners
        are labeled from 1 to number of corners. The deformation is relative to
        the material patch size along the corresponding dimension. The range
        for each dimension is specified as a tuple(lower_bound, upper_bound)
        and where positive/negative values are associated with
        tension/compression.
    edge_deformation_order_ranges : dict
        Range of polynomial deformation order (item, tuple[int](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The range is specified as a tuple(lower_bound, upper_bound),
        where the minimum allowed is zero order. Range defaults to (0, 0) if
        not specified along a given dimension.
    edge_deformation_magnitude_ranges : dict
        Range of polynomial deformation (item, tuple[float](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The edge deformation is orthogonal to the corresponding
        direction (defined by limiting corner nodes in the deformed
        configuration) and its magnitude is relative to the material patch size
        orthogonal to its dimension in the reference configuration. The range
        is specified as a tuple(lower_bound, upper_bound) and where
        positive/negative values are associated with tension/compression. Range
        defaults to (0, 0) if not specified along a given dimension.
    translation_range : dict
        Translational displacement range (item, tuple[float](2)) along each
        dimension (key, str[int]). Range is specified as tuple(min, max)
        for each dimension. Null range is assumed for unspecified
        dimensions. If None, then there is no translational motion.
    rotation_angles_range : dict
        Rotational angle range (item, tuple[float](2)) for each Euler angle
        (key, str). Euler angles follow Bunge convention (Z1-X2-Z3) and are
        labelled ('alpha', 'beta', 'gamma'), respectively. Null range is
        assumed for unspecified angles. If None, then there is no
        rotational motion.
    patch_material_data : dict
        Finite element patch material data. Expecting
        'mesh_elem_material': numpy.ndarray [int](n_elems_per_dim) (finite
        element mesh elements material matrix where each element
        corresponds to a given finite element position and whose value is
        the corresponding material phase (int)) and
        'mat_phases_descriptors': dict (constitutive model descriptors
        (item, dict) for each material phase (key, str[int])).
    n_success : int
        Number of successfully generated material patch samples.
    n_failure : int
        Number of failed material patch samples.
    n_fail_patch : int
        Number of failed material patch samples (non-admissible deformed
        configuration).
    n_fail_simulation : int
        Number of failed material patch samples (failed finite element
        simulation).
    total_time_sec : int
        Total generation time in seconds.
    avg_time_sample : float
        Average generation time per patch.
    """    
    # Set summary data
    summary_data = {}
    summary_data['n_dim'] = n_dim
    summary_data['strain_formulation'] = strain_formulation
    summary_data['analysis_type'] = analysis_type
    summary_data['patch_dims_ranges'] = patch_dims_ranges
    summary_data['elem_type'] = elem_type
    summary_data['n_elems_per_dim'] = n_elems_per_dim
    summary_data['avg_deformation_ranges'] = avg_deformation_ranges
    summary_data['edge_deformation_order_ranges'] = \
        edge_deformation_order_ranges
    summary_data['edge_deformation_magnitude_ranges'] = \
        edge_deformation_magnitude_ranges
    summary_data['translation_range'] = translation_range
    summary_data['rotation_angles_range'] = rotation_angles_range
    summary_data['patch_material_data'] = patch_material_data
    summary_data['Prescribed number of material patch samples'] = n_sample
    summary_data['Successful material patch samples'] = \
        f'{n_success:d}/{n_sample:d} ({100*n_success/n_sample:>.1f}%)'
    if n_failure > 0:
        summary_data['Failed material patch samples'] = \
            f'{n_failure:d}/{n_sample:d} ({100*n_failure/n_sample:>.1f}%)'
        summary_data['Non-admissible deformed configuration'] = \
            (f'{n_fail_patch:d}/{n_sample:d} '
             f'({100*n_fail_patch/n_sample:>.1f}%)')
        summary_data['Failed finite element simulation'] = \
            (f'{n_fail_simulation:d}/{n_sample:d} '
             f'({100*n_fail_simulation/n_sample:>.1f}%)')
    summary_data['Total generation time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. generation time per sample'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=simulation_directory,
        summary_title='Summary: Material patch data set generation',
        **summary_data)
    