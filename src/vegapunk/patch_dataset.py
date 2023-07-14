"""Generate and simulate set of deformed finite element material patches.

Functions
---------
generate_material_patch_dataset(n_dim, links_bin_path, strain_formulation, \
                                analysis_type, elem_type, n_elems_per_dim, \
                                patch_material_data, simulation_directory, \
                                n_sample=1, patch_dims_ranges=None, \
                                avg_deformation_ranges=None, \
                                edge_deformation_order_ranges=None, \
                                edge_deformation_magnitude_ranges=None,
                                max_iter_per_patch=10, links_input_params=None)
    Generate and simulate a set of deformed finite element material patches.
generate_dataset_output_data(dataset_input_data, constant_parameters={})
    Generate material patches simulations output data.   
simulate_material_patch(design)
    Generate and simulate finite element material patch design sample.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import os
import shutil
# Third-party
import numpy as np
#sys.stdout, sys.stderr = os.devnull, os.devnull
import f3dasm
#sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
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
def generate_material_patch_dataset(
    n_dim, links_bin_path, strain_formulation, analysis_type, elem_type,
    n_elems_per_dim, patch_material_data, simulation_directory, n_sample=1,
    patch_dims_ranges=None, avg_deformation_ranges=None,
    edge_deformation_order_ranges=None, edge_deformation_magnitude_ranges=None,
    max_iter_per_patch=10, links_input_params=None,
    is_save_simulation_data=False, is_save_plot_patch=False):
    """Generate and simulate a set of deformed finite element material patches.
    
    Material patch is assumed quadrilateral (2d) or parallelepipedic (3D)
    and discretized in a regular finite element mesh of quadrilateral (2d) /
    hexahedral (3d) finite elements.
    
    Simulations are performed with Links (Large Strain Implicit Nonlinear
    Analysis of Solids Linking Scales), a finite element code developed by the
    CM2S research group at the Faculty of Engineering, University of Porto.
    
    Outputs for each sample include nodal and global features data across
    multiple time steps of the simulation deformation path.
    
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
        Directory where the Links simulations input and output data files are
        written.
    n_sample : int, default=1
        Number of material patch samples.
    patch_dims_ranges : dict, default=None
        Range of material patch size (item, tuple[float](2)) along each
        dimension. The range is specified as a tuple(lower_bound, upper_bound).
        Range defaults to (1.0, 1.0) if not specified.
    avg_deformation_ranges : dict, default=None
        Range of average deformation along each dimension
        (item, tuple[tuple(2)]) for each corner label (key, str[int]). Corners
        are labeled from 1 to number of corners. The deformation is relative to
        the material patch size along the corresponding dimension. The range
        for each dimension is specified as a tuple(lower_bound, upper_bound)
        and where positive/negative values are associated with
        tension/compression. Range defaults to (0, 0) if not specified along a
        given dimension.
    edge_deformation_order_ranges : dict, default=None
        Range of polynomial deformation order (item, tuple[int](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The range is specified as a tuple(lower_bound, upper_bound),
        where the minimum allowed is zero order. Range defaults to (0, 0) if
        not specified along a given dimension.
    edge_deformation_magnitude_ranges : dict, default=None
        Range of polynomial deformation (item, tuple[float](2)) prescribed
        for each edge label (key, str[int]). Edges are labeled from 1 to number
        of edges. The edge deformation is orthogonal to the corresponding
        dimension and its magnitude is relative to the material patch size
        along that deformation dimension, measured from the midplane between
        the two limiting corner nodes. The range is specified as a
        tuple(lower_bound, upper_bound) and where positive/negative values are
        associated with tension/compression. Range defaults to (0, 0) if not
        specified along a given dimension.
    max_iter_per_patch : int, default=10
        Maximum number of iterations to get a geometrically admissible
        deformed patch configuration.
    links_input_params : dict, default=None
        Links input data file parameters. If None, default parameters are set.
    is_save_simulation_data : bool, default=False
        Save material patch simulation files.
    is_save_plot_patch : bool, default=False
        Save plot of material patch design sample in simulation directory.
        
    Returns
    -------
    dataset_output_data : list[tuple(2)]
        Material patches simulations output data. Output data of each material
        patch is stored in a tuple(2), where:
        
        'node_data' : numpy.ndarray(3d) where each i-th row has the node \
                      label (i, 0, :) and the corresponding nodal \
                      features (i, 1:, :). The last index is the time step.
        
        'global_data' : numpy.ndarray(3d) where the single row has the
                        global features (0, :, :). The last index is the
                        time step.
    """
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
    if elem_type is None:
        elem_type = default_parameters['elem_type']
    if n_elems_per_dim is None:
        n_elems_per_dim = default_parameters['n_elems_per_dim']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize design space
    design_space = f3dasm.Domain()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch size
    #
    # Loop over dimensions
    for i in range(n_dim):
        # Set name
        name = 'patch_size_' + str(i + 1)
        # Set bounds
        lower_bound=patch_dims_ranges[str(i + 1)][0]
        upper_bound=patch_dims_ranges[str(i + 1)][1]
        # Set parameter
        if np.isclose(lower_bound, upper_bound):
            parameter = f3dasm.ConstantParameter(value=lower_bound) 
        else:
            parameter = f3dasm.ContinuousParameter(lower_bound=lower_bound,
                                                   upper_bound=upper_bound)
        # Add design input parameter
        design_space.add_input_space(name=name, space=parameter)
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
            lower_bound=avg_deformation_ranges[label][i][0]
            upper_bound=avg_deformation_ranges[label][i][1]
            # Set parameter
            if np.isclose(lower_bound, upper_bound):
                parameter = f3dasm.ConstantParameter(value=lower_bound)     
            else:
                parameter = f3dasm.ContinuousParameter(
                    lower_bound=lower_bound, upper_bound=upper_bound)
            # Add design input parameter
            design_space.add_input_space(name=name, space=parameter)
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
        lower_bound=edge_deformation_order_ranges[label][0]
        upper_bound=edge_deformation_order_ranges[label][1]
        # Set parameter
        if np.isclose(lower_bound, upper_bound):
            parameter = f3dasm.ConstantParameter(value=lower_bound) 
        else:
            parameter = f3dasm.DiscreteParameter(lower_bound=lower_bound,
                                                 upper_bound=upper_bound)
        # Add design input parameter
        design_space.add_input_space(name=name, space=parameter)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input parameter: Material patch edge polynomial deformation magnitude
    #
    # Loop over edges
    for i in range(n_edges):
        # Set edge label
        label = str(i + 1)
        # Set name
        name = 'edge_' + label + '_deformation_magnitude'
        # Set bounds
        lower_bound=edge_deformation_magnitude_ranges[label][0]
        upper_bound=edge_deformation_magnitude_ranges[label][1]
        # Set parameter
        if np.isclose(lower_bound, upper_bound):
            parameter = f3dasm.ConstantParameter(value=lower_bound)
        else:
            parameter = f3dasm.ContinuousParameter(lower_bound=lower_bound,
                                                   upper_bound=upper_bound)
        # Add design input parameter
        design_space.add_input_space(name=name, space=parameter)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input constant parameters: Material patch generator
    #
    # Set number of spatial dimensions
    design_space.add_input_space(name='n_dim',
                           space=f3dasm.ConstantParameter(value=n_dim))
    # Set finite element type
    design_space.add_input_space(
        name='elem_type',
        space=f3dasm.CategoricalParameter(categories=[elem_type,]))
    # Set number of finite elements per dimension
    for i in range(n_dim):
        design_space.add_input_space(
            name='n_elems_' + str(i + 1),
            space=f3dasm.ConstantParameter(value=n_elems_per_dim[i]))
    # Set maximum number of iterations per patch
    design_space.add_input_space(
        name='max_iter',
        space=f3dasm.ConstantParameter(value=max_iter_per_patch))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input constant parameters: Material patch material data              
    #
    # Set material patch data                                                  # Error: Non-hashable Parameter is not accepted
    #design_space.add_input_space(
    #    name='patch_material_data',
    #    space=f3dasm.ConstantParameter(value=patch_material_data))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set input constant parameters: Links simulator
    #
    # Set Links binary path
    design_space.add_input_space(
        name='links_bin_path',
        space=f3dasm.CategoricalParameter(categories=[links_bin_path,]))
    # Set strain formulation
    design_space.add_input_space(
        name='strain_formulation',
        space=f3dasm.CategoricalParameter(categories=[strain_formulation,]))
    # Set analysis type
    design_space.add_input_space(
        name='analysis_type',
        space=f3dasm.CategoricalParameter(categories=[analysis_type,]))
    # Set simulation filename
    design_space.add_input_space(
        name='filename',
        space=f3dasm.CategoricalParameter(categories=['material_patch',]))
    # Set simulation filename
    design_space.add_input_space(
        name='directory',
        space=f3dasm.CategoricalParameter(categories=[simulation_directory,]))
    # Set Links parameters                                                     # Error: Non-hashable Parameter is not accepted
    #design_space.add_input_space(
    #    name='links_input_params',
    #    space=f3dasm.ConstantParameter(value=links_input_params))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set output parameters:
    #
    # Set node features data                                                   # Question: How to hande a numpy.ndarray output?
    design_space.add_output_space(name='node_data',
                                  space=f3dasm.design.parameter.Parameter())
    # Set global features data                                                 # Question: How to hande a numpy.ndarray output?
    design_space.add_output_space(name='global_data',
                                  space=f3dasm.design.parameter.Parameter())        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Initialize sampler
    sampler = f3dasm.sampling.RandomUniform(design_space)
    # Generate samples (class ExperimentData)
    dataset = sampler.get_samples(numsamples=n_sample)
    # Get samples input data (class pd.DataFrame)
    dataset_input_data = dataset.get_input_data()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate samples output data                                             
    option = ('no_f3dasm', 'f3dasm')[0]
    if option == 'no_f3dasm':
        # Group constant parameters
        constant_parameters = {
            'patch_material_data': patch_material_data,
            'links_input_params': links_input_params,
            'is_save_simulation_data': is_save_simulation_data,
            'is_save_plot_patch': is_save_plot_patch}
        # Generate samples output data
        dataset_output_data = generate_dataset_output_data(dataset_input_data,
                                                           constant_parameters)
    elif option == 'f3dasm':
        dataset.run(simulate_material_patch)                                   # Error: AttributeError: 'NoneType' object has no attribute '_dict_output'
    else:
        raise RuntimeError('Unavailable option')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_output_data
# =============================================================================
def generate_dataset_output_data(dataset_input_data, constant_parameters={}):
    """Generate material patches simulations output data.
    
    Parameters
    ----------
    dataset_input_data : pandas.DataFrame
        Material patches dataset input data.
    constant_parameters : dict, default={}
        Constant parameters required for data generation process.
    
    Returns
    -------
    dataset_output_data : list[tuple(2)]
        Material patches simulations output data. Output data of each material
        patch is stored in a tuple(2), where:
        
        'node_data' : numpy.ndarray(3d) where each i-th row has the node \
                      label (i, 0, :) and the corresponding nodal \
                      features (i, 1:, :). The last index is the time step.
        
        'global_data' : numpy.ndarray(3d) where the single row has the
                        global features (0, :, :). The last index is the
                        time step.
    """
    # Initialize material patches simulations output data
    dataset_output_data = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of material patches
    n_sample = dataset_input_data.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material patches
    for i in range(n_sample):
        # Get material patch design sample
        design = dataset_input_data.iloc[i].to_dict()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add constant parameters required for data generation process
        design.update(constant_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convert to Design object
        design = f3dasm.design.design.Design(design, {}, jobnumber=0)
        # Generate and simulate material patch
        simulate_material_patch(design)
        # Get material patch sample simulation output data
        output_data = design.output_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store material patch sample simulation output data
        dataset_output_data.append((design.output_data['node_data'],
                                    design.output_data['global_data']))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_output_data
# =============================================================================
def simulate_material_patch(design):
    """Generate and simulate finite element material patch design sample.
    
    Parameters
    ----------
    design : f3dasm.Design
        Material patch design sample.
    """
    # Get number of spatial dimensions                                         # Unexpected behavior: int was converted to float
    n_dim = int(design.get('n_dim'))
    # Get finite element type
    elem_type = design.get('elem_type')
    # Get number of finite elements per dimension                              # Unexpected behavior: int was converted to float
    n_elems_per_dim = tuple([int(design.get('n_elems_' + str(i + 1)))
                             for i in range(n_dim)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch dimensions 
    patch_dims = tuple([design.get('patch_size_' + str(i + 1))
                        for i in range(n_dim)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                             '3': (1, -1), '4': (-1, 1)}
    else:
        raise RuntimeError('Missing 3D implementation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            avg_deformation = design.get(name)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute corner displacement along dimension
            disp = \
                corners_disp_sign[label][i]*0.5*avg_deformation*patch_dims[i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Append corner displacement
            corner_disp.append(disp)
        # Set corner displacement
        corners_lab_disp_range[label] = tuple([n_dim*(corner_disp[i],)
                                               for i in range(n_dim)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of edges and deformation dimension
    if n_dim == 2:
        n_edges = 4
        edges_def_dimension = {'1': '2', '2': '2', '3': '1', '4': '1'}
    else:
        raise RuntimeError('Missing 3D implementation.')
    # Set edges labels
    edges_labels = tuple([str(i + 1) for i in range(n_edges)])    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize edges polynomial deformation order
    edges_lab_def_order = {}
    # Loop over edges
    for label in edges_labels:
        # Set parameter name
        name = 'edge_' + label + '_deformation_order'
        # Get polynomial deformation order
        order = design.get(name)
        # Set edge polynomial deformation order
        edges_lab_def_order[label] = int(order)                                # Unexpected behavior: int was converted to float
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize edges polynomial deformation magnitude
    edges_lab_disp_range = {}
    # Loop over edges
    for label in edges_labels:
        # Set parameter name
        name = 'edge_' + label + '_deformation_magnitude'
        # Get average deformation
        avg_deformation = design.get(name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get orthogonal (deformation) dimension index
        orth_dim = int(edges_def_dimension[label]) - 1
        # Compute edge displacement
        disp = 0.5*avg_deformation*patch_dims[orth_dim]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edge displacement
        edges_lab_disp_range[label] = (disp, disp)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Initialize material patch generator
    patch_generator = FiniteElementPatchGenerator(n_dim, patch_dims)
    # Generate deformed material patch
    is_admissible, patch = patch_generator.generate_deformed_patch(
        elem_type, n_elems_per_dim,
        corners_lab_disp_range=corners_lab_disp_range,
        edges_lab_def_order=edges_lab_def_order,
        edges_lab_disp_range=edges_lab_disp_range)
    # Save plot of deformed material patch
    is_save_plot_patch = design.get('is_save_plot_patch')
    if is_save_plot_patch and is_admissible:
        patch.plot_deformed_patch(
            is_save_plot=is_save_plot_patch,
            save_directory=design.get('directory'),
            plot_name=design.get('filename'),
            is_overwrite_file=not design.get('is_save_plot_patch'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize results
    results_keywords = ('node_data', 'global_data')
    results = {key: None for key in results_keywords}
    # Simulate material patch
    if is_admissible:
        # Initialize simulator
        links_simulator = LinksSimulator(design.get('links_bin_path'),
                                         design.get('strain_formulation'),
                                         design.get('analysis_type'))
        # Generate simulator input data file
        links_file_path = links_simulator.generate_input_data_file(
            design.get('filename'), design.get('directory'), patch,
            design.get('patch_material_data'),
            design.get('links_input_params'),
            is_overwrite_file=not design.get('is_save_simulation_data'))
        # Run simulation
        is_success, links_output_directory = \
            links_simulator.run_links_simulation(links_file_path)
        # Read simulation results
        if is_success:
            results = links_simulator.read_links_simulation_results(
                links_output_directory, results_keywords)
        # Remove simulation files
        if not design.get('is_save_simulation_data'):
            # Remove simulator input data file
            if os.path.isfile(links_file_path):
                os.remove(links_file_path)
            # Remove simulator output directory
            if os.path.isdir(links_output_directory):
                shutil.rmtree(links_output_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch design sample output data
    design.set('node_data', results['node_data'])                              
    design.set('global_data', results['global_data'])                          
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
        raise RuntimeError('Missing 3D implementation.')
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store default parameters
    default_parameters = {
        'elem_type': elem_type,
        'n_elems_per_dim': n_elems_per_dim,
        'patch_dims_ranges': patch_dims_ranges,
        'avg_deformation_ranges': avg_deformation_ranges,
        'edge_deformation_order_ranges': edge_deformation_order_ranges,
        'edge_deformation_magnitude_ranges': edge_deformation_magnitude_ranges}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return default_parameters
# =============================================================================
if __name__ == "__main__":
    # Mandatory parameters:
    #
    # Set number of spatial dimensions
    n_dim = 2
    # Set Links binary absolute path
    links_bin_path = '/home/bernardoferreira/Documents/repositories/external/CM2S/LINKS/bin/LINKS_debug'
    # Set Links simulation directory
    simulation_directory = '/home/bernardoferreira/Documents/temp'
    # Set Links strain formulation and analysis type
    strain_formulation = 'finite'
    analysis_type = 'plane_strain'
    # Set finite element discretization
    elem_type = 'SQUAD4'
    n_elems_per_dim = (4, 4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Optional parameters:
    #
    # Set range of material patch size along each dimension
    patch_dims_ranges = {'1': (1.0, 2.0), '2': (1.0, 2.0)}
    # Set range of average deformation along each dimension for each corner
    avg_deformation_ranges = {'1': ((-0.4, 0.4), (-0.4, 0.4)),
                              '2': ((-0.4, 0.4), (-0.4, 0.4)),
                              '3': ((-0.4, 0.4), (-0.4, 0.4)),
                              '4': ((-0.4, 0.4), (-0.4, 0.4))}
    # Set range of polynomial deformation order prescribed for each edge label
    edge_deformation_order_ranges = {'1': (1, 3),
                                     '2': (1, 3),
                                     '3': (1, 3),
                                     '4': (1, 3)}
    # Set range of polynomial deformation prescribed for each edge label
    edge_deformation_magnitude_ranges = {'1': (-0.2, 0.2),
                                         '2': (-0.2, 0.2),
                                         '3': (-0.2, 0.2),
                                         '4': (-0.2, 0.2)}
    # Set finite element patch material data
    patch_material_data = {}
    patch_material_data['mesh_elem_material'] = \
        np.ones(n_elems_per_dim, dtype=int)
    patch_material_data['mat_phases_descriptors'] = {}
    patch_material_data['mat_phases_descriptors']['1'] = \
        {'name': 'ELASTIC', 'density': 0.0, 'young': 210000, 'poisson': 0.3}
    # Set Links parameters
    links_input_params = {}
    links_input_params['number_of_increments'] = 10
    links_input_params['vtk_output'] = 'ASCII'
    # Save material patch simulation files
    is_save_simulation_data = False
    # Save plot of material patch
    is_save_plot_patch = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of samples
    n_sample = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset_output_data = generate_material_patch_dataset(
        n_dim, links_bin_path, strain_formulation, analysis_type, elem_type,
        n_elems_per_dim, patch_material_data, simulation_directory,
        n_sample=n_sample, patch_dims_ranges=patch_dims_ranges,
        avg_deformation_ranges=avg_deformation_ranges,
        edge_deformation_order_ranges=edge_deformation_order_ranges,
        edge_deformation_magnitude_ranges=edge_deformation_magnitude_ranges,
        links_input_params=links_input_params,
        is_save_plot_patch=is_save_plot_patch,
        is_save_simulation_data=is_save_simulation_data)
    
    
    