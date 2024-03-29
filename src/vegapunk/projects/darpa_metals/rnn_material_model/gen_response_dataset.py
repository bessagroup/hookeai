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
from ioput.plots import plot_xy_data, save_figure
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
                is_symmetric=self._strain_formulation=='infinitesimal')
            strain_tensor = self.build_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx, :],
                is_symmetric=self._strain_formulation=='infinitesimal')
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
    @staticmethod
    def build_tensor_from_comps(n_dim, comps, comps_array, is_symmetric=False):
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
    @staticmethod
    def store_tensor_comps(comps, tensor):
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
    @staticmethod
    def plot_material_response_path(strain_comps, strain_path,
                                    stress_comps, stress_path,
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
        strain_comps : tuple[str]
            Strain components order.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        stress_comps : tuple
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
                comps = strain_comps
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
                comps = stress_comps
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
# =============================================================================
if __name__ == '__main__':
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain-stress paths
    n_path = 1
    # Set strain deformation path type
    strain_path_type = 'random'
    # Set strain deformation path generator parameters
    strain_path_kwargs = {'n_control': 6,
                          'strain_bounds': {'11': (-1.0, 1.0),
                                            '22': (-1.0, 1.0),
                                            '12': (-1.0, 1.0)},
                          'n_time': 100,
                          'time_init': 0.0,
                          'time_end': 1.0}
    # Set constitutive model
    model_name = 'elastic'
    # Set constitutive model parameters
    model_parameters = {'elastic_symmetry': 'isotropic',
                        'E': 100, 'v': 0.3,
                        'euler_angles': (0.0, 0.0, 0.0)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain-stress material response path data set generator
    dataset_generator = \
        MaterialResponseDatasetGenerator(strain_formulation, problem_type)
    # Generate dataset
    dataset = dataset_generator.generate_response_dataset(
        n_path, strain_path_type, strain_path_kwargs, model_name,
        model_parameters)