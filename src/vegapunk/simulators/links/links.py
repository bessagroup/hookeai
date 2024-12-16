"""Finite element method simulator: Links.

Links (Large Strain Implicit Nonlinear Analysis of Solids Linking Scales) is
a finite element code developed by the CM2S research group at the Faculty of
Engineering, University of Porto.

Classes
-------
LinksSimulator
    Finite element method simulator: Links.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import subprocess
import re
import copy
import itertools
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.iostandard import make_directory, new_file_path_with_int
from ioput.plots import plot_xy_data
from simulators.links.discretization.finite_element import FiniteElement
from simulators.links.models.links_elastic import LinksElastic
from simulators.links.models.links_von_mises import LinksVonMises
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================
class LinksSimulator:
    """Finite element method simulator: Links.
    
    Attributes
    ----------
    _links_bin_path : str
        Links binary absolute path.
    _strain_formulation: {'infinitesimal', 'finite'}
        Links strain formulation.
    _analysis_type : {'plane_stress', 'plane_strain', 'axisymmetric', \
                        'tridimensional'}
        Links analysis type.
            
    Methods
    -------
    _get_n_dim(self)
        Get number of spatial dimensions from analysis type.
    generate_input_data_file(self, filename, directory, patch, \
                             patch_material_data, links_input_params=None)
        Generate Links input data file.
    run_links_simulation(self, links_file_path)
        Run Links simulation.
    read_links_simulation_results(self, links_output_directory, \
                                  results_keywords)
        Read Links simulation results.
    _write_links_input_data_file(self, links_file_path, input_params, \
                                 node_coords, elements, elements_mat_phase, \
                                 element_type, n_gauss_points, \
                                 mesh_elem_material, mat_phases_descriptors, \
                                 node_displacements)
        Write Links input data file.
    _get_links_default_input_params(self)
        Get Links default input data file parameters.
    _get_links_mesh_data(self, patch, mesh_elem_material)
        Get Links finite element mesh data.
    _get_links_elem_type_data(self, patch)
        Get Links finite element type data.
    _get_links_loading_incrementation(self)
        Get Links loading incrementation.
    remove_mesh_elements(cls, remove_elements_labels, node_coords, elements, \
                         elements_mat_phase, node_disps=None, \
                         boundary_nodes_labels=None)
        Remove elements from finite element mesh.
    """
    def __init__(self, links_bin_path, strain_formulation, analysis_type):
        """Constructor.
        
        Parameters
        ----------
        links_bin_path : str
            Links binary absolute path.
        strain_formulation: {'infinitesimal', 'finite'}
            Links strain formulation.
        analysis_type : {'plane_stress', 'plane_strain', 'axisymmetric', \
                         'tridimensional'}
            Links analysis type.
        """
        self._links_bin_path = links_bin_path
        self._strain_formulation = strain_formulation
        self._analysis_type = analysis_type
    # -------------------------------------------------------------------------
    def _get_n_dim(self):
        """Get number of spatial dimensions from analysis type.
        
        Returns
        -------
        n_dim : int
            Number of spatial dimensions.
        """
        if self._analysis_type == 'tridimensional':
            n_dim = 3
        else:
            n_dim = 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return n_dim
    # -------------------------------------------------------------------------
    def generate_input_data_file(self, filename, directory, patch,
                                 patch_material_data,
                                 remove_elements_labels=None,
                                 links_input_params=None,
                                 is_overwrite_file=False, is_verbose=False):
        """Generate Links input data file.
        
        Parameters
        ----------
        filename : str
            Links input data file name.
        directory : str
            Directory where Links input data file is stored.
        patch : FiniteElementPatch
            Finite element patch. If `is_admissible` is False, then returns
            None.
        patch_material_data : dict
            Finite element patch material data. Expecting
            'mesh_elem_material': numpy.ndarray [int](n_elems_per_dim) (finite
            element mesh elements material matrix where each element
            corresponds to a given finite element position and whose value is
            the corresponding material phase (int)) and
            'mat_phases_descriptors': dict (constitutive model descriptors
            (item, dict) for each material phase (key, str[int])).
        remove_elements_labels : tuple[int], default=None
            Finite elements to be removed from finite element mesh.
        links_input_params : dict, default=None
            Links input data file parameters. If None, default parameters are
            set.
        is_overwrite_file : bool, default=False
            Overwrite existing Links input data file if True, otherwise
            generate non-existent file path by extending the original file path
            with an integer.
        is_verbose : bool, default=False
            If True, enable verbose output.
            
        Returns
        -------
        links_file_path : str
            Links input data file path.
        """
        if is_verbose:
            print('\nGenerating Links simulation input data file'
                  '\n-------------------------------------------')
            print('\n> Setting input data file path...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create directory if it does not exist
        if not os.path.exists(directory):
            make_directory(directory)
        # Set Links input data file path
        links_file_path = \
            os.path.join(os.path.normpath(directory), filename) + '.dat'
        if os.path.isfile(links_file_path) and not is_overwrite_file:
            links_file_path = new_file_path_with_int(links_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Setting Links input data file parameters...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links default input data file parameters
        input_params = self._get_links_default_input_params()
        # Set Links input data file parameters
        if links_input_params is not None:
            for param in links_input_params.keys():
                if param in input_params.keys():
                    input_params[param] = links_input_params[param]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Generating finite element mesh data...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links finite element mesh elements material matrix
        mesh_elem_material = patch_material_data['mesh_elem_material']        
        # Get Links finite element mesh data
        node_coords, elements, elements_mat_phase = \
            self._get_links_mesh_data(patch, mesh_elem_material)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Getting prescribed node displacements...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links prescribed node displacements
        node_displacements = patch.get_mesh_boundary_nodes_disps()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove elements from Links finite elements mesh
        if remove_elements_labels is not None:
            # Get mesh boundary nodes labels
            boundary_nodes_labels = patch.get_boundary_nodes_labels()
            # Remove elements from Links finite elements mesh
            node_coords, elements, elements_mat_phase, node_displacements = \
                self.remove_mesh_elements(
                    remove_elements_labels, node_coords, elements,
                    elements_mat_phase, node_disps=node_displacements,
                    boundary_nodes_labels=boundary_nodes_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links finite element type
        element_type, n_gauss_points = self._get_links_elem_type_data(patch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links material phases descriptors
        mat_phases_descriptors = patch_material_data['mat_phases_descriptors']
        # Check if descriptors were provided for all material phases
        if not set(mesh_elem_material.flatten()).issubset(
                set([int(id) for id in mat_phases_descriptors.keys()])):
            raise RuntimeError('Material descriptors were not provided '
                               'for all material phases in the material '
                               'patch.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Writing Links simulation input data file...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write Links input data file
        self._write_links_input_data_file(
            links_file_path, input_params, node_coords, elements,
            elements_mat_phase, element_type, n_gauss_points,
            mesh_elem_material, mat_phases_descriptors, node_displacements)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> File: ' + links_file_path + '\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return links_file_path
    # -------------------------------------------------------------------------  
    def run_links_simulation(self, links_file_path, is_verbose=False):
        """Run Links simulation.
        
        Parameters
        ----------
        links_file_path : str
            Links input data file path.
        
        Returns
        -------
        is_success : bool
            True if Links simulation is successfully solved.
        links_output_directory : str
            Links simulation output directory.
        """
        if is_verbose:
            print('\nRunning Links simulation'
                  '\n------------------------')
            print('\n> File: ' + links_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run Links simulation
        subprocess.run([self._links_bin_path, links_file_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Searching for simulation output directory...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get simulation output directory
        links_output_directory = os.path.normpath(
            os.path.splitext(links_file_path)[0])
        # Check if simulation output directory exists
        if not os.path.exists(links_output_directory):
            raise RuntimeError('Links simulation output directory has not '
                               'been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input data file name
        filename = os.path.splitext(os.path.basename(links_file_path))[0]
        # Get '.screen' file path
        screen_file_path = os.path.join(links_output_directory, filename) \
            + '.screen'
        # Check if simulation was successfully solved
        if not os.path.isfile(screen_file_path):
            raise RuntimeError('Links simulation \'.screen\' file has not '
                               'been found.')
        else:
            is_success = False
            # Open '.screen' file
            screen_file = open(screen_file_path, 'r')
            screen_file.seek(0)
            # Look for succesful completion message
            line_number = 0
            for line in screen_file:
                line_number = line_number + 1
                if 'Program L I N K S successfully completed.' in line:
                    is_success = True
                    break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            if is_success:
                print('\n> Simulation status: Success')
            else:
                print('\n> Simulation status: Failure')
            print('\n> Output directory: ' + links_output_directory + '\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_success, links_output_directory
    # -------------------------------------------------------------------------  
    def read_links_simulation_results(self, links_output_directory,
                                      results_keywords, is_verbose=False):
        """Read Links simulation results.
        
        Parameters
        ----------
        links_output_directory : str
            Links simulation output directory.
        results_keywords : tuple[str]
            A list of results to be output among:
            
            'node_data' : numpy.ndarray(3d) of shape \
                        (n_nodes, n_data_dim, n_time_steps), where the i-th \
                        node output data at the k-th time step is stored in \
                        indexes [i, :, k].
            
            'global_data' : numpy.ndarray(3d) of shape \
                            (1, n_data_dim, n_time_steps) where the global \
                            output data at the k-th time step is stored in \
                            [0, :, k].
        is_verbose : bool, default=False
            If True, enable verbose output.
        
        Returns
        -------
        results : dict
            Links simulation results for each requested results keyword
            (key, str).
        """
        if is_verbose:
            print('\nRead Links simulation results'
                  '\n-----------------------------')
            print('\n> Output directory: ' + links_output_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if simulation output directory exists
        if not os.path.exists(links_output_directory):
            raise RuntimeError('Links simulation output directory has not '
                               'been found.')
        # Get simulation name
        simulation_name = \
            os.path.splitext(os.path.basename(links_output_directory))[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize simulation results
        results = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get nodes features data
        if 'node_data' in results_keywords:
            if is_verbose:
                print('\n> Reading nodes features data...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize node data
            node_data = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get node data output directory
            directory = os.path.join(links_output_directory, 'NODE_DATA')
            # Get files in node data output directory
            directory_list = os.listdir(directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize node data files
            node_data_files = []
            node_data_files_time_steps = []
            # Loop over files
            for filename in directory_list:
                # Check if is node data file
                is_node_data_file = \
                    bool(re.search(r'^' + simulation_name + r'(.*)'
                                   + r'\.nodedata$', filename))
                if is_node_data_file:
                    # Get node data file time step
                    time_step = \
                        int(os.path.splitext(filename)[0].split('_')[-1])
                    # Store node data file and corresponding time step
                    node_data_files.append(filename)
                    node_data_files_time_steps.append(time_step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Sort node data files by time step
            order = np.argsort(node_data_files_time_steps)
            node_data_files[:] = [node_data_files[i] for i in order]
            node_data_files_time_steps[:] = [node_data_files_time_steps[i]
                                             for i in order]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get number of time steps
            n_time_steps = len(node_data_files_time_steps)
            # Loop over node data files
            for i, file in enumerate(node_data_files):
                # Get node data file path
                file_path = os.path.join(directory, file)
                # Get nodes features data
                data_array = np.genfromtxt(file_path)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize node features data
                if i == 0:
                    # Get number of nodes and features
                    n_nodes = data_array.shape[0]
                    n_features = data_array.shape[1]
                    # Initialize node features data
                    node_data = np.zeros((n_nodes, n_features, n_time_steps))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble node features data
                node_data[:, :, i] = data_array
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            # Store node features data
            results['node_data'] = node_data   
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if 'global_data' in results_keywords:
            if is_verbose:
                print('\n> Reading global features data...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize global data
            global_data = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get global data output directory
            directory = os.path.join(links_output_directory, 'GLOBAL_DATA')
            # Get files in global data output directory
            directory_list = os.listdir(directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize global data files
            global_data_files = []
            global_data_files_time_steps = []
            # Loop over files
            for filename in directory_list:
                # Check if is global data file
                is_global_data_file = \
                    bool(re.search(r'^' + simulation_name + r'(.*)'
                                   + r'\.globaldata$', filename))
                if is_global_data_file:
                    # Get global data file time step
                    time_step = \
                        int(os.path.splitext(filename)[0].split('_')[-1])
                    # Store global data file and corresponding time step
                    global_data_files.append(filename)
                    global_data_files_time_steps.append(time_step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Sort global data files by time step
            order = np.argsort(global_data_files_time_steps)
            global_data_files[:] = [global_data_files[i] for i in order]
            global_data_files_time_steps[:] = [global_data_files_time_steps[i]
                                             for i in order]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get number of time steps
            n_time_steps = len(global_data_files_time_steps)
            # Loop over global data files
            for i, file in enumerate(global_data_files):
                # Get global data file path
                file_path = os.path.join(directory, file)
                # Get nodes features data
                data_array = np.atleast_1d(np.genfromtxt(file_path))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize global features data
                if i == 0:
                    # Get number of features
                    n_features = len(data_array)
                    # Initialize global features data
                    global_data = np.zeros((1, n_features, n_time_steps))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble global features data
                global_data[:, :, i] = data_array
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            # Store global features data
            results['global_data'] = global_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return results
    # -------------------------------------------------------------------------
    def _write_links_input_data_file(self, links_file_path, input_params,
                                     node_coords, elements,
                                     elements_mat_phase, element_type,
                                     n_gauss_points, mesh_elem_material,
                                     mat_phases_descriptors,
                                     node_displacements):
        """Write Links input data file.
        
        Parameters
        ----------
        links_file_path : str
            Links input data file path.
        input_params : dict
            Links input data file parameters.
        node_coords : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]).
        elements : dict
            Nodes (item, tuple[int]) of each finite element (key, str[int]).
        elements_mat_phase : dict
            Material phase (item, int) of each finite element (key, str).
        element_type : str
            Finite element type.
        n_gauss_points : int
            Number of Gauss integration points per element.
        mesh_elem_material : numpy.ndarray(2d or 3d)
            Finite element mesh elements material matrix
            (numpy.ndarray[int](n_elems_per_dim) where each element
            corresponds to a given finite element position and whose value is
            the corresponding material phase (int).
        mat_phases_descriptors : dict 
            Constitutive model descriptors (item, dict) for each material phase
            (key, str[int])).
        node_displacements : dict
            Displacements (item, numpy.ndarray(n_dim)) prescribed on each
            finite element mesh node (key, str[int]). Free degrees of
            freedom must be set as None. 
        """
        # Open Links input data file
        links_file = open(links_file_path, 'w')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Short name for input parameters
        ip = input_params
        # Get number of spatial dimensions
        n_dim = self._get_n_dim()
        # Get material phases
        mat_phases = set(mesh_elem_material.flatten())
        # Get number of material phases
        n_mat_phases = len(mat_phases)
        # Get number of nodes
        n_nodes = len(node_coords.keys())
        # Get number of elements
        n_elements = len(elements.keys())
        # Get number of nodes with prescribed displacements
        n_nodes_disps = len(node_displacements.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize input file data
        write_lines = \
            ['\nTITLE ' + '\n' + ip['title'] + '\n'] \
            + ['\nANALYSIS_TYPE ' + ip['analysis_type'] + '\n'] \
            + ['\nLARGE_STRAIN_FORMULATION ' + ip['large_strain_formulation'] \
               + '\n'] \
            + ['\nSOLUTION_ALGORITHM ' + ip['solution_algorithm'] + '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append loading increment
        write_lines += self._get_links_loading_incrementation()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Continue input file data
        write_lines += \
            ['\n\nSOLVER ' + ip['solver'] + '\n'] \
            + ['\nPARALLEL_SOLVER ' + str(ip['parallel_solver']) + '\n'] \
            + ['\nVTK_OUTPUT ' + ip['vtk_output'] + '\n'] \
            + ['\n' + ip['Node_Data_Output'] + '\n'] \
            + ['\n' + ip['Global_Data_Output'] + '\n'] \
            + ['\nVTK_OUTPUT ' + ip['vtk_output'] + '\n'] \
            + ['\nELEMENT_GROUPS ' + str(n_mat_phases) + '\n'] \
            + [str(mat) + ' 1 ' + str(mat) + '\n' for mat in mat_phases] \
            + ['\nELEMENT_TYPES 1' + '\n'] \
            + ['1 ' + element_type + '\n'] \
            + ['  ' + str(n_gauss_points) + ' GP' + '\n'] \
            + ['\nMATERIALS ' + str(n_mat_phases) + '\n'] \
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in mat_phases:
            # Get constitutive model descriptors
            descriptors = mat_phases_descriptors[str(mat_phase)]
            # Get constitutive model name
            name = descriptors['name']
            # Initialize constitutive model
            if name == 'ELASTIC':
                model = LinksElastic()
            elif name == 'VON_MISES':
                model = LinksVonMises()
            else:
                raise RuntimeError('Unknown Links constitutive model.')
            # Append constitutive model descriptors to input file data
            write_lines += model._format_material_descriptors(mat_phase,
                                                              descriptors)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~               
        # Append elements connectivities to input file data
        write_lines += \
            ['\nELEMENTS ' + str(n_elements) + '\n'] \
            + ['{:>3s}'.format(str(elem))
               + '{:^5d}'.format(elements_mat_phase[str(elem)])
               + ' '.join([str(node) for node in elements[str(elem)]]) + '\n' \
               for elem in range(1, n_elements + 1)]
        # Append node coordinates to input file data
        write_lines += \
            ['\nNODE_COORDINATES ' + str(n_nodes) + ' CARTESIAN' + '\n'] \
            + ['{:>3s}'.format(str(node)) + ' '
               + ' '.join([str('{:16.8e}'.format(coord)) \
                           for coord in node_coords[str(node)]]) + '\n' \
               for node in range(1, n_nodes + 1)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append prescribed node displacements
        write_lines += ['\nNODES_WITH_PRESCRIBED_DISPLACEMENTS ' \
                        + str(n_nodes_disps) + '\n']
        nodes_sorted = [str(node) for node in
                        np.sort([int(x) for x in node_displacements.keys()])]
        for node in nodes_sorted:
            # Get prescribed displacement
            disp = node_displacements[node]
            # Add angle
            if n_dim == 2:
                disp = np.append(disp, 0.0)
            # Generate prescription code
            code = [1 if x != None else 0 for x in disp]
            # Append node displacement
            write_lines += \
                [' ' + '{:>3s}'.format(str(node)) + ' ' \
                 + ''.join(['{:^1d}'.format(i) for i in code]) \
                 + ' '.join(['{:16.8e}'.format(val) for val in disp]) + '\n']         
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
        # Write Links input data file
        links_file.writelines(write_lines)
        # Close Links input data file
        links_file.close()        
    # -------------------------------------------------------------------------
    def _get_links_default_input_params(self):
        """Get Links default input data file parameters.
        
        Returns
        -------
        default_parameters : dict
            Links input data file default parameters.
        """
        # Initialize Links default input parameters
        default_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set title
        title = 'Links input data file generated automatically.'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self._strain_formulation == 'infinitesimal':
            # Set finite strains flag
            large_strain_formulation = 'OFF'
        else:
            # Set finite strains flag
            large_strain_formulation = 'ON'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set analysis type
        if self._analysis_type == 'plane_stress':
            analysis_type = '1'
        elif self._analysis_type == 'plane_strain':
            analysis_type = '2'
        elif self._analysis_type == 'axisymmetric':
            analysis_type = '3'
        elif self._analysis_type == 'tridimensional':
            analysis_type = '6'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set solution algorithm
        solution_algorithm = '2'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set solver
        solver = 'PARDISO'
        # Set number of threads (solver parallelization)
        parallel_solver = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set VTK output
        vtk_output = 'None'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node and global features data output
        node_data_output = 'Node_Data_Output'
        global_data_output = 'Global_Data_Output'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links default input parameters
        default_parameters['title'] = title
        default_parameters['large_strain_formulation'] = \
            large_strain_formulation
        default_parameters['analysis_type'] = analysis_type
        default_parameters['solution_algorithm'] = solution_algorithm
        default_parameters['solver'] = solver
        default_parameters['parallel_solver'] = parallel_solver
        default_parameters['vtk_output'] = vtk_output
        default_parameters['Node_Data_Output'] = node_data_output
        default_parameters['Global_Data_Output'] = global_data_output
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return default_parameters
    # -------------------------------------------------------------------------
    def _get_links_mesh_data(self, patch, mesh_elem_material):
        """Get Links finite element mesh data.
        
        Parameters
        ----------
        patch : FiniteElementPatch
            Finite element patch.
        mesh_elem_material : numpy.ndarray(2d or 3d)
            Finite element mesh elements material matrix
            (numpy.ndarray[int](n_elems_per_dim) where each element
            corresponds to a given finite element position and whose value is
            the corresponding material phase (int).

        Returns
        -------
        node_coords : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]).
        elements : dict
            Nodes (item, tuple[int]) of each finite element (key, str[int]).
        elements_mat_phase : dict
            Material phase (item, int) of each finite element (key, str).
        """
        # Get number of spatial dimensions
        n_dim = self._get_n_dim()
        # Get patch finite element type
        elem_type = patch.get_elem_type()
        # Get patch number of finite elements per dimension
        n_elems_per_dim = patch.get_n_elems_per_dim()
        # Get patch number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = patch.get_n_edge_nodes_per_dim()
        # Get finite element mesh nodes matrix
        mesh_nodes_matrix = patch.get_mesh_nodes_matrix()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get finite element
        finite_element = FiniteElement(elem_type)
        # Get number of nodes per finite element
        n_nodes_elem = finite_element.get_n_nodes()
        # Get number of edge nodes per finite element
        n_edge_nodes_elem = finite_element.get_n_edge_nodes()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node coordinates
        node_coords = patch.get_mesh_nodes_coords_ref()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element number
        elem_label = 1
        # Initialize elements connectivities
        elements = {}
        # Initialize elements material phase
        elements_mat_phase = {}
        # Build elements data
        if n_dim == 2:
            # Loop over nodes
            for j in range(0, n_edge_nodes_per_dim[1]
                           - n_edge_nodes_elem + 1,
                           n_edge_nodes_elem - 1):
                # Loop over nodes
                for i in range(0, n_edge_nodes_per_dim[0]
                               - n_edge_nodes_elem + 1,
                               n_edge_nodes_elem - 1):                   
                    # Initialize element nodes
                    elem_nodes = []
                    # Loop over element nodes
                    for p in range(1, n_nodes_elem + 1):
                        # Get node local index
                        local_index = finite_element.get_node_label_index(p)
                        # Get mesh node index and label
                        node_index = tuple(np.add((i, j), local_index))
                        node_label = mesh_nodes_matrix[node_index]
                        # Store element node
                        elem_nodes.append(node_label)
                    # Store element nodes
                    elements[str(elem_label)] = tuple(elem_nodes)
                    # Increment element number
                    elem_label += 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize element number
            elem_label = 1
            # Loop over elements
            for j in range(n_elems_per_dim[1]):
                # Loop over elements
                for i in range(n_elems_per_dim[0]):        
                    # Store element material phase
                    elements_mat_phase[str(elem_label)] = \
                        mesh_elem_material[i, j]
                    # Increment element number
                    elem_label += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Loop over nodes
            for k in range(0, n_edge_nodes_per_dim[2]
                           - n_edge_nodes_elem + 1,
                           n_edge_nodes_elem - 1):
                # Loop over nodes
                for j in range(0, n_edge_nodes_per_dim[1]
                               - n_edge_nodes_elem + 1,
                               n_edge_nodes_elem - 1):
                    # Loop over nodes
                    for i in range(0, n_edge_nodes_per_dim[0]
                                   - n_edge_nodes_elem + 1,
                                   n_edge_nodes_elem - 1):                   
                        # Initialize element nodes
                        elem_nodes = []
                        # Loop over element nodes
                        for p in range(1, n_nodes_elem + 1):
                            # Get node local index
                            local_index = \
                                finite_element.get_node_label_index(p)
                            # Get mesh node index and label
                            node_index = tuple(np.add((i, j, k), local_index))
                            node_label = mesh_nodes_matrix[node_index]
                            # Store element node
                            elem_nodes.append(node_label)
                        # Store element nodes
                        elements[str(elem_label)] = tuple(elem_nodes)
                        # Increment element number
                        elem_label += 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize element number
            elem_label = 1
            # Loop over elements
            for k in range(n_elems_per_dim[2]):
                # Loop over elements
                for j in range(n_elems_per_dim[1]):
                    # Loop over elements
                    for i in range(n_elems_per_dim[0]):        
                        # Store element material phase
                        elements_mat_phase[str(elem_label)] = \
                            mesh_elem_material[i, j, k]
                        # Increment element number
                        elem_label += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_coords, elements, elements_mat_phase        
    # -------------------------------------------------------------------------
    def _get_links_elem_type_data(self, patch):
        """Get Links finite element type data.
        
        Parameters
        ----------
        patch : FiniteElementPatch
            Finite element patch.
            
        Returns
        -------
        element_type : str
            Finite element type.
        n_gauss_points : int
            Number of Gauss integration points per element.
        """
        # Get number of spatial dimensions
        n_dim = self._get_n_dim()
        # Get patch finite element type
        elem_type = patch.get_elem_type()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set element type and number of Gauss integration points
        if n_dim == 2:
            if elem_type == 'SQUAD4':
                element_type = 'QUAD4'
                n_gauss_points = 4
            elif elem_type == 'SQUAD8':
                element_type = 'QUAD8'
                n_gauss_points = 4
            else:
                raise RuntimeError(f'Unavailable 2D Links element type '
                                   f'data ({elem_type}).')
        else:
            if elem_type == 'SHEXA8':
                element_type = 'HEXA8'
                n_gauss_points = 8
            elif elem_type == 'SHEXA20':
                element_type = 'HEXA20'
                n_gauss_points = 8
            else:
                raise RuntimeError(f'Unavailable 3D Links element type '
                                   f'data ({elem_type}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_type, n_gauss_points
    # -------------------------------------------------------------------------
    def _get_links_loading_incrementation(self):
        """Get Links loading incrementation.
        
        Returns
        -------
        increm_input_lines : list[str]
            Loading incrementation formatted to Links input data file.
        """
        # Set available incrementation modes
        available_incrementation_modes = ('n_increments', 'increment_list')
        # Set available loading types
        available_loading_type = (
            'proportional', 'cyclic_plus', 'cyclic_sym', 'polynomial')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set incrementation mode
        incrementation_mode = available_incrementation_modes[1]
        # Set loading type
        loading_type = available_loading_type[3]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of loading reversal (effective for cyclic loadings only)
        n_reverse = 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of increments
        n_inc = 100
        # Set convergence tolerance
        conv_tol = '{:<16.8e}'.format(1e-6)
        # Set maximum number of iterations
        max_n_iter = 20
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize formatted loading incrementation
        increm_input_lines = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set formatted loading incrementation
        if incrementation_mode == 'n_increments':
            # Check loading type
            if loading_type != 'proportional':
                raise RuntimeError(
                    f'The incrementation mode \'{incrementation_mode}\' only '
                    f'accepts the loading type \'proportional\'.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set formatted loading incrementation
            increm_input_lines += \
                [f'\nNumber_of_Increments {n_inc}\n'] \
                + [f'\nCONVERGENCE_TOLERANCE\n{conv_tol}\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif incrementation_mode == 'increment_list':
            # Set loading incrementation
            if loading_type in ('proportional', 'cyclic_plus', 'cyclic_sym'):
                # Override number of loading reversals if proportional loading
                if loading_type == 'proportional':
                    n_reverse = 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check minimum number of loading increments
                if loading_type == 'cyclic_plus' and (n_inc < n_reverse + 1):
                    raise RuntimeError(
                        f'In order to generate a loading path with '
                        f'{n_reverse} reversions for loading type '
                        f'{loading_type}, the minimum required number of '
                        f'loading increments is {n_reverse + 1}.')
                elif (loading_type == 'cyclic_sym'
                        and (n_inc < 2*n_reverse + 1)):
                    raise RuntimeError(
                        f'In order to generate a loading path with '
                        f'{n_reverse} reversions for loading type '
                        f'{loading_type}, the minimum required number of '
                        f'loading increments is {2*n_reverse + 1}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set number of proportional increments and switch frequency
                if loading_type in ('proportional', 'cyclic_plus'):
                    n_inc_mon = int(np.floor(n_inc/(n_reverse + 1)))
                    dfact_switch = 1
                elif loading_type == 'cyclic_sym':
                    n_inc_mon = int(np.floor(n_inc/(2*n_reverse + 1)))
                    dfact_switch = 2
                # Compute incremental load factor
                dfact = 1.0/n_inc_mon
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute effective number of increments
                n_inc_eff = n_inc_mon*(dfact_switch*n_reverse + 1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set initial proportional loading increments
                increm_input_lines += \
                    [f'\nINCREMENTS {n_inc}'] \
                    + n_inc_mon*[f'\n{dfact:>16.8e} {conv_tol} {max_n_iter}']
                # Loop over loading reversals
                for _ in range(n_reverse):
                    # Switch loading direction
                    dfact = -dfact
                    # Append proportional loading increments
                    increm_input_lines += dfact_switch*n_inc_mon*[
                        f'\n{dfact:>16.8e} {conv_tol} {max_n_iter}']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set null increments until prescribed number of increments
                # is met
                if n_inc > n_inc_eff:
                    # Compute number of null increments
                    n_inc_diff = n_inc - n_inc_eff
                    # Append null loading increments
                    increm_input_lines += n_inc_diff*[
                        f'\n1.0 {0.0:>16.8e} {conv_tol} {max_n_iter}']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add lineskip
                increm_input_lines += ['\n']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif loading_type == 'polynomial':
                # Set discrete time history
                time_hist = \
                    np.linspace(0.0, 1.0, n_inc + 1, endpoint=True,
                                dtype=float)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set number of control points sampling bounds
                n_control_bounds = (4, 7)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Sample number of polynomial control points
                n_control = np.random.randint(n_control_bounds[0],
                                              n_control_bounds[1] + 1)
                # Set polynomial degree
                polynomial_degree = n_control - 1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set strain control times
                control_time = \
                    np.linspace(0.0, 1.0, n_control, endpoint=True,
                                dtype=float)
                # Sample control total load factor
                control_tfact = \
                    np.random.uniform(low=-1.0, high=1.0, size=n_control)
                # Enforce initial null total load factor
                control_tfact[0] = 0.0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Fit polynomial model
                polynomial_coefficients = \
                    np.polyfit(control_time, control_tfact, polynomial_degree)
                # Get polynomial model
                polynomial_model = np.poly1d(polynomial_coefficients)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get total load factor history with polynomial model
                tfact_hist = np.array(
                    [polynomial_model(x) for x in time_hist])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set loading increments header
                increm_input_lines += [f'\nINCREMENTS {n_inc}']
                # Loop over increments
                for i in range(n_inc):
                    # Compute incremental load factor
                    dfact = tfact_hist[i + 1] - tfact_hist[i]
                    # Append loading increment
                    increm_input_lines += [
                        f'\n1.0 {dfact:>16.8e} {conv_tol} {max_n_iter}']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set plot flag
                is_display_tfact_hist = True
                # Display total load factor history
                if is_display_tfact_hist:
                    # Assemble total load factor history data
                    data_xy = np.column_stack((time_hist, tfact_hist))
                    # Plot total load factor history data
                    _, _ = plot_xy_data(data_xy, x_lims=(0.0, 1.0),
                                        x_label='Time',
                                        y_label='Total load factor',
                                        is_latex=True)
                    # Display total load factor history
                    plt.show()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unavailable loading type for '
                                   'incrementation mode '
                                   '\'{incrementation_mode}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unavailable loading incrementation mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return increm_input_lines
    # -------------------------------------------------------------------------
    @classmethod
    def remove_mesh_elements(cls, remove_elements_labels, node_coords,
                             elements, elements_mat_phase, node_disps=None,
                             boundary_nodes_labels=None):
        """Remove elements from finite element mesh.

        Parameters
        ----------
        remove_elements_labels : tuple[int]
            Finite elements to be removed from finite element mesh.
        node_coords : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]).
        elements : dict
            Nodes (item, tuple[int]) of each finite element (key, str[int]).
        elements_mat_phase : dict
            Material phase (item, int) of each finite element (key, str).
        node_disps : dict, default=None
            Displacements (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]).
        boundary_nodes_labels : tuple[int], default=None
            Finite element mesh boundary nodes labels. If provided, then
            removal of elements that would lead to the removal of a boundary
            node raises error.
            
        Returns
        -------
        node_coords : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]).
        elements : dict
            Nodes (item, tuple[int]) of each finite element (key, str[int]).
        elements_mat_phase : dict
            Material phase (item, int) of each finite element (key, str).
        node_disps : dict
            Displacements (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]). None if displacements are not provided.
        """
        # Store old nodes coordinates
        node_coords_old = copy.deepcopy(node_coords)
        # Store old elements
        elements_old = copy.deepcopy(elements)
        # Store old elements material phases
        elements_mat_phase_old = copy.deepcopy(elements_mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get old nodes labels
        nodes_old_labels = set(itertools.chain(*list(elements_old.values())))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element number
        elem_label = 1
        # Initialize elements
        elements = {}
        # Initialize elements mapping (old to new)
        elem_label_map = {}
        # Loop over old elements
        for elem_old_label, elem_old_nodes in elements_old.items():
            # Process remaining element
            if int(elem_old_label) not in remove_elements_labels:
                # Collect remaining element nodes
                elements[str(elem_label)] = elem_old_nodes
                # Assemble elements mapping (old to new)
                elem_label_map[str(elem_old_label)] = str(elem_label)
                # Increment element number
                elem_label += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get current nodes
        nodes_labels = set(itertools.chain(*list(elements.values())))
        # Get nodes to be removed
        remove_nodes_labels = nodes_old_labels - nodes_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check removal of boundary nodes
        if boundary_nodes_labels is not None:
            # Loop over nodes
            for node_label in remove_nodes_labels:
                # Check if removed node is boundary node
                if node_label in boundary_nodes_labels:
                    # Initialize invalid removed elements
                    inv_remove_elements_labels = []
                    # Loop over elements
                    for elem_label in remove_elements_labels:
                        # Collect invalid removed elements
                        if node_label in elements_old[str(elem_label)]:
                            inv_remove_elements_labels.append(elem_label)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    raise RuntimeError(
                        f'Removing elements that would lead to the removal of '
                        f'boundary nodes is not allowed. The following '
                        f'elements cannot be removed due to boundary node '
                        f'{node_label}: {inv_remove_elements_labels}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node number
        node_label = 1
        # Initialize nodes coordinates
        node_coords = {}
        # Initialize nodes mapping (old to new)
        node_label_map = {}
        # Loop over old nodes
        for node_old_label, node_old_coords in node_coords_old.items():
            # Process remaining node
            if int(node_old_label) not in remove_nodes_labels:
                # Collect remaining node coordinates
                node_coords[str(node_label)] = node_old_coords
                # Assemble nodes mapping (old to new)
                node_label_map[str(node_old_label)] = node_label
                # Increment node number
                node_label += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for elem_label in elements.keys():
            # Store old element nodes
            elem_old_nodes = elements[elem_label]
            # Build new element nodes
            elem_nodes = [node_label_map[str(node_old_label)]
                          for node_old_label in elem_old_nodes]
            # Store new element nodes
            elements[elem_label] = tuple(elem_nodes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements material phases
        elements_mat_phase = {}
        # Loop over elements
        for elem_old_label, mat_phase in elements_mat_phase_old.items():
            # Process remaining element
            if elem_old_label in elem_label_map.keys():
                # Collect remaining element material phase
                elements_mat_phase[str(elem_label_map[elem_old_label])] = \
                    mat_phase
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Map nodes displacements
        if isinstance(node_disps, dict):
            # Store old nodes displacements
            node_disps_old = copy.deepcopy(node_disps)
            # Initialize nodes displacements
            node_disps = {}
            # Loop over old nodes
            for node_old_label, node_old_disps in node_disps_old.items():
                # Get node label
                node_label = node_label_map[str(node_old_label)]
                # Collect node displacements
                node_disps[str(node_label)] = node_old_disps
        else:
            node_disps = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_coords, elements, elements_mat_phase, node_disps