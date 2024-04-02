"""DARPA METALS PROJECT: Generate strain-stress material response data set.

Classes
-------
MaterialResponseDatasetGenerator
    Strain-stress material response path data set generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import copy
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory, \
    get_time_series_data_loader
from projects.darpa_metals.rnn_material_model.strain_paths.interface import \
    StrainPathGenerator
from projects.darpa_metals.rnn_material_model.strain_paths.random_path import \
    RandomStrainPathGenerator
from projects.darpa_metals.rnn_material_model.strain_paths.proportional_path \
    import ProportionalStrainPathGenerator
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_from_mf
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from ioput.plots import plot_xy_data, plot_xyz_data, plot_histogram, \
    plot_boxplots, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class MaterialResponseDatasetGenerator():
    """Strain-stress material response path data set generator.
    
    Attributes
    ----------
    _strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    _problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : tuple[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : tuple[str]
        Strain/Stress components nonsymmetric order.
    
    Methods
    -------
    generate_response_dataset(self, n_path, strain_path_type, \
                              strain_path_kwargs, model_name, \
                              model_parameters)
        Generate strain-stress material response path data set.
    compute_stress_path(self, strain_comps, time_hist, strain_path, \
                        constitutive_model)
        Compute material stress response for given strain path.
    build_tensor_from_comps(n_dim, comps, comps_array, is_symmetric=False)
        Build strain/stress tensor from given components.
    store_tensor_comps(comps, tensor)
        Store strain/stress tensor components in array.
    plot_material_response_path(strain_comps, strain_path, stress_comps, \
                                stress_path, time_hist, \
                                strain_filename='strain_path', \
                                stress_filename='stress_path', \
                                strain_axis_lims=None, stress_axis_lims=None, \
                                save_dir=None, is_save_fig=False, \
                                is_stdout_display=False, is_latex=False)
        Plot strain-stress material response path.
    plot_stress_space_metrics(cls, strain_formulation, stress_comps_order, \
                              stress_path, time_hist, \
                              is_plot_principal_stress_path=False, \
                              is_plot_stress_invar_hist=False, \
                              is_plot_stress_invar_box=False, \
                              is_plot_stress_path_triax_lode=False, \
                              is_plot_stress_triax_lode_hist=False, \
                              is_plot_stress_triax_lode_box=False, \
                              stress_units='', \
                              filename='stress_path', \
                              save_dir=None, is_save_fig=False, \
                              is_stdout_display=False, is_latex=False)
        Plot strain-stress material response path in principal stress space.
    compute_stress_invariants(stress)
        Compute stress invariants.
    """
    def __init__(self, strain_formulation, problem_type):
        """Constructor.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        """
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
    # -------------------------------------------------------------------------
    def generate_response_dataset(self, n_path, strain_path_type,
                                  strain_path_kwargs, model_name,
                                  model_parameters, save_dir=None,
                                  is_save_fig=False):
        """Generate strain-stress material response path data set.

        Parameters
        ----------
        n_path : int
            Number of strain-stress paths.
        strain_path_type : {'random', 'proportional'}
            Strain path type that sets the corresponding generator. 
        strain_path_kwargs : dict
            Parameters of strain path generator method set by strain_path_type.
        model_name : str
            FETorch material constitutive model name.
        model_parameters : dict
            FETorch material constitutive model parameters.
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        # Initialize strain path generator
        if strain_path_type == 'random':
            strain_path_generator = RandomStrainPathGenerator(
                self._strain_formulation, self._n_dim)
        elif strain_path_type == 'proportional':
            strain_path_generator = ProportionalStrainPathGenerator(
                self._strain_formulation, self._n_dim)
        else:
            raise RuntimeError('Unknown strain path type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model
        if model_name == 'elastic':
            constitutive_model = Elastic(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'von_mises':
            constitutive_model = VonMises(
                self._strain_formulation, self._problem_type, model_parameters)
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize time series data set samples
        dataset_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples
        for i in range(n_path):
            # Generate strain path
            strain_comps_order, time_hist, strain_path = \
                strain_path_generator.generate_strain_path(
                    **strain_path_kwargs)
            # Compute material stress response
            stress_comps_order, stress_path = \
                self.compute_stress_path(strain_comps_order, time_hist,
                                         strain_path, constitutive_model)
            # Store strain-stress material response path
            dataset_samples.append(
                {'strain_comps_order': strain_comps_order,
                 'strain_path': torch.tensor(strain_path, dtype=torch.float),
                 'stress_comps_order': stress_comps_order,
                 'stress_path': torch.tensor(stress_path, dtype=torch.float),
                 'time_hist': torch.tensor(time_hist,
                                           dtype=torch.float).reshape(-1, 1)})
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot material response path
            self.plot_material_response_path(
                strain_comps_order, strain_path,
                stress_comps_order, stress_path,
                time_hist,
                strain_filename=f'strain_path_{i}',
                stress_filename=f'stress_path_{i}',
                save_dir=save_dir, is_save_fig=is_save_fig,
                is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create strain-stress material response path data set
        dataset = TimeSeriesDatasetInMemory(dataset_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dataset
    # -------------------------------------------------------------------------
    def compute_stress_path(self, strain_comps_order, time_hist, strain_path,
                            constitutive_model):
        """Compute material stress response for given strain path.

        Parameters
        ----------
        strain_comps_order : tuple[str]
            Strain components order.
        time_hist : numpy.ndarray(1d)
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        constitutive_model : ConstitutiveModel
            FETorch material constitutive model.

        Returns
        -------
        stress_comps : tuple
            Stress components order.
        stress_path : numpy.ndarray(2d)
            Stress path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_stress_comps).
        """
        # Set stress components order
        stress_comps = strain_comps_order
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress path history
        stress_path = np.zeros((n_time, len(stress_comps)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model state variables
        state_variables = constitutive_model.state_init()
        # Initialize last converged material constitutive state variables
        state_variables_old = copy.deepcopy(state_variables)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(1, n_time):
            # Get previous and current strain tensors
            strain_tensor_old = self.build_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx - 1, :],
                is_symmetric=self._strain_formulation == 'infinitesimal')
            strain_tensor = self.build_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx, :],
                is_symmetric=self._strain_formulation == 'infinitesimal')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute incremental strain tensor
            if self._strain_formulation == 'infinitesimal':
                # Compute incremental infinitesimal strain tensor
                inc_strain = strain_tensor - strain_tensor_old
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Material state update
            state_variables, _ = material_state_update(
                self._strain_formulation, self._problem_type,
                constitutive_model, inc_strain, state_variables_old,
                def_gradient_old=None)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check material state update convergence
            if state_variables['is_su_fail']:
                raise RuntimeError('Material state update convergence '
                                   'failure.')
            # Update last converged material constitutive state variables
            state_variables_old = copy.deepcopy(state_variables)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get stress tensor
            if self._strain_formulation == 'infinitesimal':
                # Get Cauchy stress tensor
                stress = get_tensor_from_mf(state_variables['stress_mf'],
                                            self._n_dim, self._comp_order_sym)
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store stress tensor
            stress_path[time_idx, :] = \
                self.store_tensor_comps(stress_comps, stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_comps, stress_path
    # -------------------------------------------------------------------------
    @classmethod
    def build_tensor_from_comps(cls, n_dim, comps, comps_array,
                                is_symmetric=False):
        """Build strain/stress tensor from given components.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comps : tuple[str]
            Strain/Stress components order.
        comps_array : numpy.ndarray(1d)
            Strain/Stress components array.
        is_symmetric : bool, default=False
            If True, then assembles off-diagonal strain components from
            symmetric component.
        
        Returns
        -------
        tensor : numpy.ndarray(2d)
            Strain/Stress tensor.
        """
        # Initialize tensor
        tensor = np.zeros((n_dim, n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over components
        for k, comp in enumerate(comps):
            # Get component indexes
            i, j = [int(x) - 1 for x in comp]
            # Assemble tensor component
            tensor[i, j] = comps_array[k]
            # Assemble symmetric tensor component
            if is_symmetric and i != j:
                tensor[j, i] = comps_array[k]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tensor
    # -------------------------------------------------------------------------
    @classmethod
    def store_tensor_comps(cls, comps, tensor):
        """Store strain/stress tensor components in array.
        
        Parameters
        ----------
        comps : tuple[str]
            Strain/Stress components order.
        tensor : numpy.ndarray(2d)
            Strain/Stress tensor.
        
        Returns
        -------
        comps_array : numpy.ndarray(1d)
            Strain/Stress components array.
        """
        # Initialize tensor components array
        comps_array = np.zeros(len(comps))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over components
        for k, comp in enumerate(comps):
            # Get component indexes
            i, j = [int(x) - 1 for x in comp]
            # Assemble tensor component
            comps_array[k] = tensor[i, j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return comps_array
    # -------------------------------------------------------------------------
    @classmethod
    def plot_material_response_path(cls, strain_comps_order, strain_path,
                                    stress_comps_order, stress_path,
                                    time_hist,
                                    strain_filename='strain_path',
                                    stress_filename='stress_path',
                                    strain_axis_lims=None,
                                    stress_axis_lims=None,
                                    save_dir=None, is_save_fig=False,
                                    is_stdout_display=False, is_latex=False):
        """Plot strain-stress material response path.
        
        Parameters
        ----------
        strain_comps_order : tuple[str]
            Strain components order.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        stress_comps_order : tuple
            Stress components order.
        stress_path : numpy.ndarray(2d)
            Stress path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_stress_comps).
        time_hist : numpy.ndarray(1d)
            Discrete time history.
        strain_filename : str, default='strain_path'
            Strain path figure name.
        stress_filename : str, default='stress_path'
            Stress path figure name.
        strain_axis_lims : tuple, default=None
            Enforce the limits of the plot strain axis, stored as
            tuple(min, max).
        stress_axis_lims : tuple, default=None
            Enforce the limits of the plot stress axis, stored as
            tuple(min, max).
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.
        is_stdout_display : bool, default=False
            True if displaying figure to standard output device, False
            otherwise.
        is_latex : bool, default=False
            If True, then render all strings in LaTeX. If LaTex is not
            available, then this option is silently set to False and all input
            strings are processed to remove $(...)$ enclosure.
        """
        # Loop over path types
        for path_type in ('strain', 'stress'):
            # Set path specific parameters
            if path_type == 'strain':
                # Set components
                comps = strain_comps_order
                # Set component label
                comp_label = 'Strain'
                # Set y-axis label
                y_label = 'Strain'
                # Set axis limits
                axis_lims = strain_axis_lims
                # Set path
                path = strain_path
                # Set filename
                filename = strain_filename
            else:
                # Set components
                comps = stress_comps_order
                # Set component label
                comp_label = 'Stress'
                # Set y-axis label
                y_label = 'Stress'
                # Set axis limits
                axis_lims = stress_axis_lims
                # Set path
                path = stress_path
                # Set filename
                filename = stress_filename
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize data array
            data_xy = np.zeros((len(time_hist), 2*len(comps)))
            # Set data array
            for j in range(len(comps)):
                data_xy[:, 2*j] = time_hist
                data_xy[:, 2*j + 1] = path[:, j]
            # Set data labels
            data_labels = [f'{comp_label} {x}' for x in comps]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set axis limits
            y_lims = (None, None)
            if isinstance(axis_lims, tuple):
                y_lims = axis_lims
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot path
            figure, _ = plot_xy_data(data_xy, data_labels=data_labels,
                                     x_lims=(time_hist[0], time_hist[-1]),
                                     y_lims=y_lims,
                                     x_label='Time', y_label=y_label,
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display figure
            if is_stdout_display:
                plt.show()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename, format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Close plot
            plt.close(figure)
    # -------------------------------------------------------------------------
    @classmethod
    def plot_stress_space_metrics(cls, strain_formulation, stress_comps_order,
                                  stress_path, time_hist,
                                  is_plot_principal_stress_path=False,
                                  is_plot_stress_invar_hist=False,
                                  is_plot_stress_invar_box=False,
                                  is_plot_stress_path_triax_lode=False,
                                  is_plot_stress_triax_lode_hist=False,
                                  is_plot_stress_triax_lode_box=False,
                                  stress_units='',
                                  filename='stress_path',
                                  save_dir=None, is_save_fig=False,
                                  is_stdout_display=False, is_latex=False):
        """Plot strain-stress material response path in principal stress space.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
        stress_comps_order : tuple
            Stress components order.
        stress_path : {numpy.ndarray(2d), list[numpy.ndarray(2d)]}
            Stress path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_stress_comps) or list of multiple stress path
            histories.
        time_hist : {numpy.ndarray(1d), list[numpy.ndarray(1d)]}
            Discrete time history or list of multiple discrete time histories.
        is_plot_principal_stress_path : bool, default=False
            Plot stress path in the principal stress space.
        is_plot_stress_invar_hist : bool, default=False
            Plot distribution of stress invariants.
        is_plot_stress_invar_box : bool, default=False
            Plot box plot with stress invariants.
        is_plot_stress_path_triax_lode : bool, default=False
            Plot stress triaxiality and Lode parameter paths.
        is_plot_stress_triax_lode_hist : bool, default=False
            Plot stress triaxiality and Lode parameter distributions.
        is_plot_stress_triax_lode_box : bool, default=False
            Plot box plot with stress triaxiality and Lodeb parameter.
        stress_units : str, default=''
            Stress units label.
        filename : str, default='stress_path'
            Figure name.
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.
        is_stdout_display : bool, default=False
            True if displaying figure to standard output device, False
            otherwise.
        is_latex : bool, default=False
            If True, then render all strings in LaTeX. If LaTex is not
            available, then this option is silently set to False and all input
            strings are processed to remove $(...)$ enclosure.
        """
        # Set number of spatial dimensions
        n_dim = 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check dimensionality
        if len(stress_comps_order) not in (6, 9):
            raise RuntimeError('Strain-stress material response plots in the '
                               'principal stress space are only available if '
                               'the complete 3D stress path is provided.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check stress paths data
        if isinstance(time_hist, list) or isinstance(stress_path, list):
            # Check multiple strain paths
            if (not isinstance(time_hist, list)
                    and not isinstance(stress_path, list)):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'stress path histories when providing '
                                   'multiple stress paths.')
            elif len(time_hist) != len(stress_path):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'stress path histories when providing '
                                   'multiple stress paths.')
            # Get number of strain paths
            n_path = len(stress_path)
            # Get longest strain path
            n_time_max = max([len(x) for x in time_hist])
            # Get minimum and maximum discrete times
            time_min = min([x[0] for x in time_hist])
            time_max = max([x[-1] for x in time_hist])
        else:
            # Set number of strain paths
            n_path = 1
            # Set longest strain path
            n_time_max = len(time_hist)
            # Set minimum and maximum discrete times
            time_min = time_hist[0]
            time_max = time_hist[-1]
            # Store discrete time history and stress path in list
            time_hist = [time_hist,]
            stress_path = [stress_path,]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize figure
        figure = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize principal stresses
        stress_paths_principal = []
        # Compute stress invariants
        for k in range(n_path):
            # Compute principal stresses
            stress_eigen = np.zeros((n_time_max, n_dim))
            # Loop over discrete time
            for i in range(len(time_hist[k])):
                # Get stress tensor
                stress = cls.build_tensor_from_comps(
                    n_dim, stress_comps_order, stress_path[k][i, :],
                    is_symmetric=strain_formulation == 'infinitesimal')
                # Compute principal stresses
                eigenvalues, _ = np.linalg.eig(stress)
                # Store principal stresses (sorted)
                stress_eigen[i, :] = np.sort(eigenvalues)
            # Store stress path principal stresses
            stress_paths_principal.append(stress_eigen)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress path in principal stress space
        if is_plot_principal_stress_path:
            # Initialize stress data array
            stress_data_xyz = np.full((n_time_max, 3*n_path),
                                      fill_value=np.nan)
            # Loop over stress paths
            for k in range(n_path):
                # Set stress data array
                stress_data_xyz[:len(time_hist[k]), 3*k:3*k+3] = \
                    stress_paths_principal[k]
            # Plot stress path in principal stress space
            figure, _ = plot_xyz_data(
                data_xyz=stress_data_xyz,
                x_label='Stress 1' + stress_units,
                y_label='Stress 2' + stress_units,
                z_label='Stress 3' + stress_units,
                view_angles_deg=(30, 30, 0),
                is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_principal',
                            format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress invariants
        stress_paths_invar = []
        stress_paths_invar_dev = []
        # Compute stress invariants
        for k in range(n_path):
            # Initialize stress invariants
            stress_invar = np.zeros((n_time_max, 3))
            stress_invar_dev = np.zeros((n_time_max, 3))
            # Loop over discrete time
            for i in range(len(time_hist[k])):
                # Get stress tensor
                stress = cls.build_tensor_from_comps(
                    n_dim, stress_comps_order, stress_path[k][i, :],
                    is_symmetric=strain_formulation=='infinitesimal')
                # Compute stress invariants
                invar, invar_dev = cls.compute_stress_invariants(stress)
                # Store stress invariants
                stress_invar[i, :] = invar
                stress_invar_dev[i, :] = invar_dev
            # Store stress path invariants
            stress_paths_invar.append(stress_invar)
            stress_paths_invar_dev.append(stress_invar_dev)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress invariants (distribution and box plot)
        if is_plot_stress_invar_hist or is_plot_stress_invar_box:
            # Set stress invariants data labels
            invar_labels = ('I1', 'I2', 'I3', 'J1', 'J2', 'J3')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate stress paths
            stress_invar_data = \
                np.hstack([np.vstack(stress_paths_invar),
                           np.vstack(stress_paths_invar_dev)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress invariants distribution
            if is_plot_stress_invar_hist:
                for i, invariant in enumerate(invar_labels):
                    figure, _ = plot_histogram(
                        (stress_invar_data[:, i],), bins=20, density=True,
                        x_label=(f'Stress invariant ({invariant})'
                                 + stress_units),
                        y_label='Probability density',
                        is_latex=is_latex)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save figure
                    if is_save_fig:
                        save_figure(figure, filename
                                    + f'_invariant_{invariant}_hist',
                                    format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress invariants box plot
            if is_plot_stress_invar_box:
                figure, _ = plot_boxplots(stress_invar_data, invar_labels,
                                          x_label=f'Stress invariants',
                                          y_label=f'Stress' + stress_units,
                                          y_scale='symlog',
                                          is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + 'invariants_boxplot',
                                format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize triaxiality and Lode parameter
        stress_paths_triax = []
        stress_paths_lode = []
        # Compute triaxiality and Lode parameter
        for k in range(n_path):
            # Initialize stress invariants
            stress_triax = np.zeros((n_time_max, 1))
            stress_lode = np.zeros((n_time_max, 1))
            # Get stress path stress invariants
            stress_invar = stress_paths_invar[k]
            stress_invar_dev = stress_paths_invar_dev[k]
            # Loop over discrete time
            for i in range(len(time_hist[k])):
                # Get stress invariants
                i1 = stress_invar[i, 0]
                j2 = stress_invar_dev[i, 1]
                j3 = stress_invar_dev[i, 2]
                # Compute triaxiality and Lode parameter
                tolerance = 1e-8
                if j2 < tolerance:
                    stress_triax[i, 0] = np.nan
                    stress_lode[i, 0] = np.nan
                else:                    
                    stress_triax[i, 0] = (((1.0/3.0)*i1)/(np.sqrt(3.0*j2)))
                    stress_lode[i, 0] = -(j3/2.0)*((3.0/j2)**(3/2))
            # Store stress path triaxiality and Lode parameter
            stress_paths_triax.append(stress_triax)
            stress_paths_lode.append(stress_lode)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress triaxiality and Lode parameter paths
        if is_plot_stress_path_triax_lode:
            # Set stress triaxiality data array
            triax_data_xy = np.zeros((n_time_max, 2*n_path))
            for k in range(n_path):
                triax_data_xy[:len(time_hist[k]), 2*k] = time_hist[k][:, 0]
                triax_data_xy[:len(time_hist[k]), 2*k+1] = \
                    stress_paths_triax[k][:, 0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress triaxility path
            figure, _ = plot_xy_data(data_xy=triax_data_xy,
                                     x_lims=(time_min, time_max),
                                     x_label='Time', y_label='Triaxiality',
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_triaxiality', format='pdf',
                            save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set stress Lode parameter data array
            lode_data_xy = np.zeros((n_time_max, 2*n_path))
            for k in range(n_path):
                lode_data_xy[:len(time_hist[k]), 2*k] = time_hist[k][:, 0]
                lode_data_xy[:len(time_hist[k]), 2*k+1] = \
                    stress_paths_lode[k][:, 0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress Lode parameter path
            figure, _ = plot_xy_data(data_xy=lode_data_xy,
                                     x_lims=(time_min, time_max),
                                     x_label='Time', y_label='Lode parameter',
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_lode', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress invariants (distribution and box plot)
        if is_plot_stress_triax_lode_hist or is_plot_stress_triax_lode_box:
            # Set stress triaxiality and Lode parameter data labels
            data_labels = ('Triaxiality', 'Lode parameter')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate stress paths
            stress_triax_lode_data = \
                np.hstack([np.vstack(stress_paths_triax),
                           np.vstack(stress_paths_lode)])
            stress_triax_lode_data = stress_triax_lode_data[
                ~np.isnan(stress_triax_lode_data).any(axis=1), :]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress triaxiality and Lode parameter distributions
            if is_plot_stress_triax_lode_hist:
                for i, metric in enumerate(data_labels):
                    figure, _ = plot_histogram(
                        (stress_triax_lode_data[:, i],), bins=20, density=True,
                        x_label=metric, y_label='Probability density',
                        is_latex=is_latex)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save figure
                    if is_save_fig:
                        save_figure(figure, filename
                                    + f'_{metric.lower().replace(" ", "_")}'
                                    + '_hist',
                                    format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress triaxiality and Lode parameter box plot
            if is_plot_stress_triax_lode_box:
                figure, _ = plot_boxplots(
                    stress_triax_lode_data, data_labels,
                    x_label='Stress invariants', y_label='Value',
                    is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + 'invariants_boxplot',
                                format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        if figure is not None:
            plt.close(figure)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def compute_stress_invariants(stress):
        """Compute stress invariants.
        
        Parameters
        ----------
        stress : np.ndarray(2d)
            Stress tensor.
            
        Returns
        -------
        stress_invar : np.ndarray(1d)
            Stress invariants.
        stress_invar_dev : np.ndarray(1d)
            Deviatoric stress invariants
        """
        # Initialize stress invariants
        stress_invar = np.zeros(3)
        # Compute stress invariants
        stress_invar[0] = np.trace(stress)
        stress_invar[1] = \
            0.5*(np.trace(stress)**2 - np.trace(np.matmul(stress, stress)))
        stress_invar[2] = np.linalg.det(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize deviatoric stress invariants
        stress_invar_dev = np.zeros(3)
        # Compute deviatoric stress invariants
        stress_invar_dev[0] = 0.0
        stress_invar_dev[1] = (1.0/3.0)*stress_invar[0]**2 - stress_invar[1]
        stress_invar_dev[2] = ((2.0/27.0)*stress_invar[0]**3
                               - (1.0/3.0)*stress_invar[0]*stress_invar[1]
                               + stress_invar[2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_invar, stress_invar_dev 
# =============================================================================
if __name__ == '__main__':
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of discrete times
    n_time = 10
    # Set initial and final time
    time_init = 0.0
    time_end = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain components bounds
    if n_dim == 2:
        strain_bounds = {x: (-1.0, 1.0) for x in ('11', '22', '12')}
    else:
        strain_bounds = \
            {x: (-1.0, 1.0) for x in ('11', '22', '33', '12', '23', '13')}
    # Set incremental strain norm
    inc_strain_norm = None
    # Set strain noise
    strain_noise_std = None
    # Set cyclic loading
    is_cyclic_loading = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model
    model_name = 'elastic'
    # Set constitutive model parameters
    model_parameters = {'elastic_symmetry': 'isotropic',
                        'E': 100, 'v': 0.3,
                        'euler_angles': (0.0, 0.0, 0.0)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain-stress paths of each type
    n_path_type = {'proportional': 0, 'random': 2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path generators parameters
    strain_path_kwargs_type = {}
    # Proportional strain path generator
    strain_path_kwargs_type['proportional'] = \
        {'strain_bounds': strain_bounds,
         'n_time': n_time,
         'time_init': time_init,
         'time_end': time_end,
         'inc_strain_norm': inc_strain_norm,
         'strain_noise_std': strain_noise_std,
         'is_cyclic_loading': is_cyclic_loading}
    # Random strain path generator 
    strain_path_kwargs_type['random'] = \
        {'n_control': 6,
         'strain_bounds': strain_bounds,
         'n_time': n_time,
         'generative_type': 'polynomial',
         'time_init': time_init,
         'time_end': time_end,
         'inc_strain_norm': inc_strain_norm,
         'strain_noise_std': strain_noise_std,
         'is_cyclic_loading': is_cyclic_loading}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain-stress material response path data set generator
    dataset_generator = \
        MaterialResponseDatasetGenerator(strain_formulation, problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Initialize data sets
    datasets = []
    # Loop over data set types
    for strain_path_type, n_path in n_path_type.items():
        # Check number of strain-stress paths
        if n_path < 1:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain path generators parameters
        strain_path_kwargs = strain_path_kwargs_type[strain_path_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate data set
        dataset = dataset_generator.generate_response_dataset(
            n_path, strain_path_type, strain_path_kwargs, model_name,
            model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store data set
        datasets.append(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get joint data set
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = torch.utils.data.ConcatDataset(datasets)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get data loader
    data_loader = get_time_series_data_loader(dataset, batch_size=1)
    # Initialize strain-stress paths data
    time_hists = []
    strain_paths = []
    stress_paths = []
    # Loop over strain-stress paths
    for i, path in enumerate(data_loader):
        # Collect strain-stress path data
        time_hists.append(np.array(path['time_hist'][:, 0, :]))
        strain_paths.append(np.array(path['strain_path'][:, 0, :]))
        stress_paths.append(np.array(path['stress_path'][:, 0, :]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get strain components order
    strain_comps_order = datasets[0][0]['strain_comps_order']
    # Plot strain paths data
    StrainPathGenerator.plot_strain_path(
        strain_formulation, n_dim,
        strain_comps_order, time_hists, strain_paths,
        is_plot_strain_path=False,
        is_plot_strain_comp_hist=False,
        is_plot_strain_norm=False,
        is_plot_strain_norm_hist=False,
        is_plot_inc_strain_norm=False,
        is_plot_inc_strain_norm_hist=False,
        is_plot_strain_path_pairs=False,
        is_plot_strain_pairs_hist=False,
        is_plot_strain_pairs_marginals=False,
        is_plot_strain_comp_box=False,
        strain_label='Strain',
        strain_units='',
        filename='strain_path',
        is_stdout_display=True,
        is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get stress components order
    stress_comps_order = datasets[0][0]['stress_comps_order']
    # Plot stress paths data
    dataset_generator.plot_stress_space_metrics(
                                  strain_formulation,
                                  stress_comps_order,
                                  stress_paths, time_hists,
                                  is_plot_principal_stress_path=False,
                                  is_plot_stress_invar_hist=False,
                                  is_plot_stress_invar_box=False,
                                  is_plot_stress_path_triax_lode=False,
                                  is_plot_stress_triax_lode_hist=False,
                                  is_plot_stress_triax_lode_box=False,
                                  stress_units='',
                                  filename='stress_path',
                                  save_dir=None, is_save_fig=False,
                                  is_stdout_display=True, is_latex=True)