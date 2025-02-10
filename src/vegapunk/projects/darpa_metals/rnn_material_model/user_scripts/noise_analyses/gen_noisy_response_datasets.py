"""DARPA METALS PROJECT: Generate noisy strain-stress response data sets.

Classes
-------
NoisyMaterialResponseDatasetGenerator
    Noisy strain-stress material response path data set generator.
NoiseGenerator
    Noise generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[5])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import copy
import time
import datetime
# Third-party
import torch
import numpy as np
import tqdm
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset, load_dataset
from projects.darpa_metals.rnn_material_model.strain_paths.random_path import \
    RandomStrainPathGenerator
from projects.darpa_metals.rnn_material_model.strain_paths.proportional_path \
    import ProportionalStrainPathGenerator
from projects.darpa_metals.rnn_material_model.user_scripts \
    .gen_response_dataset import generate_dataset_plots
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_from_mf
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.von_mises_mixed import \
    VonMisesMixed
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from utilities.data_scalers import TorchMinMaxScaler
from utilities.data_denoisers import Denoiser
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
class NoisyMaterialResponseDatasetGenerator:
    """Noisy strain-stress material response path data set generator.
    
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
    build_response_path(cls, strain_comps_order, strain_path, \
                        stress_comps_order, stress_path, time_hist, \
                        state_path={}, strain_noise_path=None, \
                        stress_noise_path=None)
        Assemble material response path data.
    generate_noiseless_dataset(self, n_path, strain_path_type, \
                               strain_path_kwargs, constitutive_model, \
                               state_features={}, is_verbose=False)
        Generate noiseless strain-stress material response data set.
    generate_noisy_dataset(self, noiseless_dataset, noise_generator, \
                           constitutive_model, state_features={}, \
                           strain_data_scaler=None, \
                           noise_variability='homoscedastic', \
                           heteroscedastic_weights=None, \
                           is_verbose=False)
        Generate noisy strain-stress material response data set.
    compute_stress_path(self, strain_comps_order, time_hist, strain_path, \
                        constitutive_model, state_features={})
        Compute material stress response for given strain path.
    compute_norm_path(cls, comps_path, n_dim, comps_array, is_symmetric=False)
        Compute strain/stress tensor norm path.
    build_tensor_from_comps(cls, n_dim, comps, comps_array, is_symmetric=False)
        Build strain/stress tensor from given components.
    store_tensor_comps(cls, comps, tensor)
        Store strain/stress tensor components in array.
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
    @classmethod
    def build_response_path(cls, strain_comps_order, strain_path,
                            stress_comps_order, stress_path, time_hist,
                            state_path={}, strain_noise_path=None,
                            stress_noise_path=None):
        """Assemble material response path data.
        
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
        state_path : dict, default={}
            Store each requested constitutive model state variable (key, str)
            path history as numpy.ndarray(2d) of shape
            (sequence_length, n_features).
        strain_noise_path : numpy.ndarray(2d), default=None
            Strain noise path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        stress_noise_path : numpy.ndarray(2d), default=None
            Stress noise path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        
        Returns
        -------
        response_path : dict
            Material response path.
        """
        # Initialize material response path data
        response_path = {}
        # Assemble strain-stress material response path
        response_path['strain_comps_order'] = strain_comps_order
        response_path['strain_path'] = \
            torch.tensor(strain_path, dtype=torch.get_default_dtype())
        response_path['stress_comps_order'] = stress_comps_order
        response_path['stress_path'] = \
            torch.tensor(stress_path, dtype=torch.get_default_dtype())
        # Assemble time path
        response_path['time_hist'] = torch.tensor(
            time_hist, dtype=torch.get_default_dtype()).reshape(-1, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble state variables path
        for state_var in state_path.keys():
            response_path[state_var] = torch.tensor(
                state_path[state_var], dtype=torch.get_default_dtype())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble strain noise path
        if strain_noise_path is not None:
            response_path['strain_noise_path'] = torch.tensor(
                strain_noise_path, dtype=torch.get_default_dtype())
        # Assemble stress noise path
        if stress_noise_path is not None:
            response_path['stress_noise_path'] = torch.tensor(
                stress_noise_path, dtype=torch.get_default_dtype())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return response_path
    # -------------------------------------------------------------------------
    def generate_noiseless_dataset(self, n_path, strain_path_type,
                                   strain_path_kwargs, constitutive_model,
                                   state_features={}, is_verbose=False):
        """Generate noiseless strain-stress material response data set.
        
        Parameters
        ----------
        n_path : int
            Number of strain-stress paths.
        strain_path_type : {'random', 'proportional'}
            Strain path type that sets the corresponding generator. 
        strain_path_kwargs : dict
            Parameters of strain path generator method set by strain_path_type.
        constitutive_model : ConstitutiveModel
            FETorch material constitutive model.
        state_features : dict, default={}
            FETorch material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally included in the data set. Unavailable state
            variables are ignored.
        is_verbose : bool, default=False
            If True, enable verbose output.
            
        Returns
        -------
        dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        start_time_sec = time.time()
        if is_verbose:
            print('\nGenerate noiseless strain-stress response data set'
                  '\n--------------------------------------------------')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Set maximum number of sample trials
        max_path_trials = 10
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize time series data set samples
        dataset_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n> Strain paths type: {strain_path_type}\n')
            print('\n> Starting strain-stress paths generation process...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples
        for i in tqdm.tqdm(range(n_path),
                           desc='> Generating strain-stress paths: ',
                           disable=not is_verbose):
            # Initialize number of sample trials
            n_path_trials = 0
            # Initialize stress response path failure flag
            is_stress_path_fail = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Attempt to compute material response
            while is_stress_path_fail:
                # Increment number of sample trials
                n_path_trials += 1
                # Generate strain path
                strain_comps_order, time_hist, strain_path = \
                    strain_path_generator.generate_strain_path(
                        **strain_path_kwargs)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute material response
                stress_comps_order, stress_path, state_path, \
                    is_stress_path_fail = self.compute_stress_path(
                        strain_comps_order, time_hist, strain_path,
                        constitutive_model, state_features=state_features)
                # Check maximum number of sample trials
                if n_path_trials > max_path_trials:
                    raise RuntimeError(f'The maximum of number of trials '
                                       f'({max_path_trials}) to compute '
                                       f'a material response path sample was '
                                       f'reached without success.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build material response path
            response_path = self.build_response_path(
                strain_comps_order, strain_path, stress_comps_order,
                stress_path, time_hist, state_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store strain-stress material response path
            dataset_samples.append(response_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Finished strain-stress paths generation process!\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create strain-stress material response path data set (in memory)
        dataset = TimeSeriesDatasetInMemory(dataset_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute total generation time and average generation time per path
        total_time_sec = time.time() - start_time_sec
        avg_time_sec = total_time_sec/n_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n> Total generation time: '
                  f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
                  f'Avg. generation time per path: '
                  f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dataset
    # -------------------------------------------------------------------------
    def generate_noisy_dataset(self, noiseless_dataset, noise_generator,
                               constitutive_model, state_features={},
                               strain_data_scaler=None,
                               noise_variability='homoscedastic',
                               heteroscedastic_normalizer=None,
                               is_denoise_strain_path=False,
                               is_verbose=False):
        """Generate noisy strain-stress material response data set.
        
        Parameters
        ----------
        noiseless_dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        noise_generator : NoiseGenerator
            Noise generator.
        constitutive_model : ConstitutiveModel
            FETorch material constitutive model.
        state_features : dict, default={}
            FETorch material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally included in the data set. Unavailable state
            variables are ignored.
        strain_data_scaler : {TorchMinMaxScaler, TorchStandardScaler}, \
                             default=None
            Fitted data scaler for strain components. If provided, then noise
            is defined and processed in the normalized space.
        noise_variability: str, {'homoscedastic', 'heteroscedastic'}, \
                           default='homoscedastic'
            Variability of noise across the data. In 'homoscedastic' noise, the
            variance of the noise remains constant across the data points
            (uniform effect regardless of independent variable). In
            'heteroscedastic' noise, the variance of the noise depends on the
            data point.
        heteroscedastic_normalizer : float, default=None
            Factor used to normalize the strain norm paths and set the noise
            heteroscedasticity weights. Weights materialize noise
            heteroscedasticity by scaling the noise distribution variance for
            each data point. If None, then maximum strain norm of each path.
        is_denoise_strain_path : bool, default=False
            If True, then denoise strain path prior to computation of material
            response.
        is_verbose : bool, default=False
            If True, enable verbose output.
            
        Returns
        -------
        noisy_dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        """
        start_time_sec = time.time()
        if is_verbose:
            print('\nGenerate noisy strain-stress response data set'
                  '\n----------------------------------------------')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of paths
        n_path = len(noiseless_dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set maximum number of sample trials
        max_path_trials = 10
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize time series data set samples
        dataset_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Starting strain-stress paths generation process...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples
        for i in tqdm.tqdm(range(n_path),
                           desc='> Generating strain-stress paths: ',
                           disable=not is_verbose):
            # Initialize number of sample trials
            n_path_trials = 0
            # Initialize stress response path failure flag
            is_stress_path_fail = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Attempt to compute material response
            while is_stress_path_fail:
                # Increment number of sample trials
                n_path_trials += 1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get noiseless response path
                noiseless_response_path = noiseless_dataset[i]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get noiseless response path data
                strain_comps_order = \
                    noiseless_response_path['strain_comps_order']
                noiseless_strain_path = \
                    noiseless_response_path['strain_path'].numpy()
                stress_comps_order = \
                    noiseless_response_path['stress_comps_order']
                noiseless_stress_path = \
                    noiseless_response_path['stress_path'].numpy()
                time_hist = \
                    noiseless_response_path['time_hist'].numpy()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set heteroscedastic noise weights
                if noise_variability == 'heteroscedastic':
                    # Compute noiseless strain norm path
                    noiseless_strain_norm_path = \
                        dataset_generator.compute_norm_path(
                            noiseless_strain_path, n_dim, strain_comps_order,
                            is_symmetric=strain_formulation == 'infinitesimal')
                    # Set heteroscedastic normalization
                    if heteroscedastic_normalizer is None:
                        heteroscedastic_normalizer = \
                            np.max(noiseless_strain_norm_path)
                    # Compute heteroscedastic noise weights
                    heteroscedastic_weights = \
                        noiseless_strain_norm_path/heteroscedastic_normalizer
                else:
                    heteroscedastic_weights = None
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate strain noise path
                strain_noise_path = \
                    noise_generator.generate_noise_path(noiseless_strain_path, 
                            noise_variability=noise_variability,
                            heteroscedastic_weights=heteroscedastic_weights)
                # Enforce null initial noise
                strain_noise_path[0, :] = 0.0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Normalize strain path
                if strain_data_scaler:
                    noiseless_strain_path = strain_data_scaler.transform(
                        torch.tensor(noiseless_strain_path)).numpy()
                # Set noisy strain path
                strain_path = noiseless_strain_path + strain_noise_path
                # Denormalize strain path
                if strain_data_scaler:
                    strain_path = strain_data_scaler.inverse_transform(
                        torch.tensor(strain_path)).numpy()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Denoise strain path
                if is_denoise_strain_path:
                    # Initialize denoiser
                    denoiser = Denoiser()
                    # Set denoise method
                    denoise_method = 'moving_average'
                    # Set denoise method parameters
                    if denoise_method == 'moving_average':
                        denoise_parameters = {'window_size': 5}
                    # Set number of denoising cycles
                    n_denoise_cycle = 1
                    # Get denoised strain path
                    strain_path = denoiser.denoise(
                        torch.tensor(strain_path),
                        denoise_method, denoise_parameters=denoise_parameters,
                        n_denoise_cycle=n_denoise_cycle).numpy()
                    # Enforce null initial noise
                    strain_path[0, :] = noiseless_strain_path[0, :]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute material response
                stress_comps_order, stress_path, state_path, \
                    is_stress_path_fail = self.compute_stress_path(
                        strain_comps_order, time_hist, strain_path,
                        constitutive_model, state_features=state_features)
                # Check maximum number of sample trials
                if n_path_trials > max_path_trials:
                    raise RuntimeError(f'The maximum of number of trials '
                                       f'({max_path_trials}) to compute '
                                       f'a material response path sample was '
                                       f'reached without success.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute stress noise path
                stress_noise_path = stress_path - noiseless_stress_path
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build material response path
            response_path = self.build_response_path(
                strain_comps_order, strain_path, stress_comps_order,
                noiseless_stress_path, time_hist, state_path=state_path,
                strain_noise_path=strain_noise_path,
                stress_noise_path=stress_noise_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store strain-stress material response path
            dataset_samples.append(response_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Finished strain-stress paths generation process!\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create strain-stress material response path data set (in memory)
        noisy_dataset = TimeSeriesDatasetInMemory(dataset_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute total generation time and average generation time per path
        total_time_sec = time.time() - start_time_sec
        avg_time_sec = total_time_sec/n_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n> Total generation time: '
                  f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
                  f'Avg. generation time per path: '
                  f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return noisy_dataset
    # -------------------------------------------------------------------------
    def compute_stress_path(self, strain_comps_order, time_hist, strain_path,
                            constitutive_model, state_features={}):
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
        state_features : dict, default={}
            FETorch material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally output. Unavailable state variables are ignored.

        Returns
        -------
        stress_comps_order : tuple
            Stress components order.
        stress_path : numpy.ndarray(2d)
            Stress path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_stress_comps).
        state_path : dict
            Store each requested constitutive model state variable (key, str)
            path history as numpy.ndarray(2d) of shape
            (sequence_length, n_features).
        is_stress_path_fail : bool
            Stress response path failure flag.
        """
        # Set stress components order
        stress_comps_order = strain_comps_order
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress path history
        stress_path = np.zeros((n_time, len(stress_comps_order)))
        # Initialize state path history
        state_path = {}
        for state_var, n_features in state_features.items():
            state_path[state_var] = np.zeros((n_time, n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model state variables
        state_variables = constitutive_model.state_init()
        # Initialize last converged material constitutive state variables
        state_variables_old = copy.deepcopy(state_variables)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress response failure flag
        is_stress_path_fail = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in tqdm.tqdm(
                range(1, n_time), leave=False,
                desc='  > Computing time steps state update: '):    
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
            # Convert incremental strain tensor to Torch tensor
            inc_strain = \
                torch.tensor(inc_strain, dtype=torch.get_default_dtype())
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Material state update
            state_variables, _ = material_state_update(
                self._strain_formulation, self._problem_type,
                constitutive_model, inc_strain, state_variables_old,
                def_gradient_old=None)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check material state update convergence
            if state_variables['is_su_fail']:
                is_stress_path_fail = True
            # Stop computation of material response path
            if is_stress_path_fail:
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                self.store_tensor_comps(stress_comps_order, stress)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over state variables
            for state_var in state_features.keys():
                # Skip if state variable is not available
                if state_var not in state_variables.keys():
                    state_path.pop(state_var)
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store state variable
                if 'strain_mf' in state_var:
                    # Build strain tensor
                    if self._strain_formulation == 'infinitesimal':
                        strain = get_tensor_from_mf(
                            state_variables[state_var], self._n_dim,
                            self._comp_order_sym)
                    else:
                        raise RuntimeError('Not implemented.')
                    # Store strain tensor
                    state_path[state_var][time_idx, :] = \
                        self.store_tensor_comps(strain_comps_order, strain)
                else:
                    # Store generic state variable
                    state_path[state_var][time_idx, :] = \
                        state_variables[state_var]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_comps_order, stress_path, state_path, is_stress_path_fail
    # -------------------------------------------------------------------------
    @classmethod
    def compute_norm_path(cls, comps_path, n_dim, comps_order,
                          is_symmetric=False):
        """Compute strain/stress tensor norm path.
        
        Parameters
        ----------
        comps_path: numpy.ndarray(2d)
            Strain/Stress path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_comps).
        n_dim : int
            Problem number of spatial dimensions.
        comps_order : tuple[str]
            Strain/Stress components order.
        is_symmetric : bool, default=False
            If True, then assembles off-diagonal components from symmetric
            component.

        Return
        ------
        norm_path : numpy.ndarray(1d)
            Strain/Stress tensor norm path.
        """
        # Get number of discrete time steps
        n_time = comps_path.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress noise path
        norm_path = np.zeros(n_time)
        # Loop over time steps
        for t in range(n_time):
            # Build strain/stress tensor
            tensor = cls.build_tensor_from_comps(n_dim, comps_order,
                                                 comps_path[t, :],
                                                 is_symmetric=is_symmetric)
            # Compute tensor norm
            tensor_norm = np.linalg.norm(tensor)
            # Store tensor norm
            norm_path[t] = tensor_norm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return norm_path
    # -------------------------------------------------------------------------
    @classmethod
    def build_tensor_from_comps(cls, n_dim, comps_order, comps_array,
                                is_symmetric=False):
        """Build strain/stress tensor from given components.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comps_order : tuple[str]
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
        for k, comp in enumerate(comps_order):
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
    def store_tensor_comps(cls, comps_order, tensor):
        """Store strain/stress tensor components in array.
        
        Parameters
        ----------
        comps_order : tuple[str]
            Strain/Stress components order.
        tensor : numpy.ndarray(2d)
            Strain/Stress tensor.
        
        Returns
        -------
        comps_array : numpy.ndarray(1d)
            Strain/Stress components array.
        """
        # Initialize tensor components array
        comps_array = np.zeros(len(comps_order))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over components
        for k, comp in enumerate(comps_order):
            # Get component indexes
            i, j = [int(x) - 1 for x in comp]
            # Assemble tensor component
            comps_array[k] = tensor[i, j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return comps_array
# =============================================================================
class NoiseGenerator:
    """Noise generator.
    
    Attributes
    ----------
    _noise_distribution : str, {'uniform', 'gaussian', 'spiked_gaussian'}
        Noise distribution type.
    _noise_parameters : dict
        Noise distribution parameters.

    Methods
    -------
    set_noise_distribution(self, noise_distribution)
        Set noise distribution type.
    set_noise_parameters(self, noise_parameters)
        Set noise distribution parameters.
    get_required_parameters(cls, noise_distribution)
        Get required parameters for given noise distribution type.
    generate_noise_path(self, noiseless_path, \
                        noise_variability='homoscedastic', \
                        heteroscedastic_weights=None)
        Generate noise path.
    """
    def __init__(self):
        """Constructor."""
        # Initialize noise distribution
        self._noise_distribution = None
        # Initialize noise distribution parameters
        self._noise_parameters = None
    # -------------------------------------------------------------------------
    def set_noise_distribution(self, noise_distribution):
        """Set noise distribution type.
        
        Parameters
        ----------
        noise_distribution : str, {'uniform', 'gaussian', 'spiked_gaussian'}
            Noise distribution type.
        """
        # Check noise distribution type
        if noise_distribution not in ('uniform', 'gaussian',
                                      'spiked_gaussian'):
            raise RuntimeError('Unknown noise distribution type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set noise distribution type
        self._noise_distribution = noise_distribution
    # -------------------------------------------------------------------------
    def set_noise_parameters(self, noise_parameters):
        """Set noise distribution parameters.
        
        Parameters
        ----------
        noise_parameters : dict
            Noise distribution parameters.
        """
        # Get required parameters
        required_parameters = \
            self.get_required_parameters(self._noise_distribution)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check parameters
        for parameter in required_parameters:
            if parameter not in noise_parameters.keys():
                raise RuntimeError(f'Parameter {parameter} must be provided '
                                   f'for noise distribution of type '
                                   f'{self._noise_distribution}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set parameters
        self._noise_parameters = noise_parameters  
    # -------------------------------------------------------------------------
    @classmethod
    def get_required_parameters(cls, noise_distribution):
        """Get required parameters for given noise distribution type.
        
        Parameters
        ----------
        noise_distribution : str, {'uniform', 'gaussian'}
            Noise distribution type.

        Returns
        -------
        required_noise_parameters : tuple[str]
            Noise distribution required parameters.
        """
        if noise_distribution == 'uniform':
            required_parameters = ('amp',)
        elif noise_distribution == 'gaussian':
            required_parameters = ('std',)
        elif noise_distribution == 'spiked_gaussian':
            required_parameters = ('std', 'spike', 'p_spike')
        else:
            raise RuntimeError('Unknown noise distribution type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return required_parameters
    # -------------------------------------------------------------------------
    def generate_noise_path(self, noiseless_path,
                            noise_variability='homoscedastic',
                            heteroscedastic_weights=None):
        """Generate noise path.
        
        Noise is applied independently for each signal feature.
        
        Parameters
        ----------
        noiseless_path : numpy.ndarray(2d)
            Noiseless signal path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_features). 
        noise_variability: str, {'homoscedastic', 'heteroscedastic'}, \
                           default='homoscedastic'
            Variability of noise across the data. In 'homoscedastic' noise, the
            variance of the noise remains constant across the data points
            (uniform effect regardless of independent variable). In
            'heteroscedastic' noise, the variance of the noise depends on the
            data point.
        heteroscedastic_weights : numpy.ndarray(1d), default=None
            Weights that materialize noise heteroscedasticity by scaling the
            noise distribution variance for each data point. Stored as
            numpy.ndarray(1d) of shape (sequence_length). If None, then
            defaults to ones (homoscedastic noise).

        Returns
        -------
        noise_path : numpy.ndarray(2d)
            Noise path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_features).
        """
        # Set noise path shape
        noise_path_shape = noiseless_path.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate noise path
        if noise_variability == 'homoscedastic':
            # Sample noise path
            if self._noise_distribution == 'uniform':
                # Set bounds
                low = -0.5*abs(self._noise_parameters['amp'])
                high = -low
                # Sample noise
                noise_path = np.random.uniform(low, high,
                                               size=noise_path_shape)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._noise_distribution in ('gaussian', 'spiked_gaussian'):
                # Set standard deviation
                std = self._noise_parameters['std']
                # Sample noise
                noise_path = np.random.normal(loc=0.0, scale=std,
                                              size=noise_path_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif noise_variability == 'heteroscedastic':
            # Set heteroscedasticity weights
            if heteroscedastic_weights is None:
                heteroscedastic_weights = np.ones(noise_path_shape[0])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize noise path
            noise_path = np.zeros(noise_path_shape)
            # Loop over time steps
            for t in range(noise_path.shape[0]):
                # Get heteroscedasticity weight
                weight = heteroscedastic_weights[t]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Sample noise
                if self._noise_distribution == 'uniform':
                    # Set bounds
                    low = -0.5*abs(self._noise_parameters['amp'])*weight
                    high = -low
                    # Sample noise
                    noise = np.random.uniform(low, high,
                                              size=noise_path_shape[1])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif self._noise_distribution in ('gaussian',
                                                  'spiked_gaussian'):
                    # Set standard deviation
                    std = self._noise_parameters['std']*weight
                    # Sample noise
                    noise = np.random.normal(loc=0.0, scale=std,
                                             size=noise_path_shape[1])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble noise
                noise_path[t, :] = noise
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add noise spike
        if self._noise_distribution in ('spiked_gaussian',):
            # Set spike magnitude and probability
            spike = self._noise_parameters['spike']
            p_spike = self._noise_parameters['p_spike']
            # Sample noise spike
            spike_path = spike*np.random.binomial(n=1, p=p_spike,
                                                  size=noise_path_shape)
            # Add noise spike to noise path
            noise_path += spike_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return noise_path
# =============================================================================
if __name__ == '__main__':
    # Set data set type
    dataset_type = ('training', 'validation', 'testing_id', 'testing_od')[0]
    # Set save dataset plots flags
    is_save_dataset_plots = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set noiseless generation only
    is_only_noiseless = True
    # Set reference noiseless directory
    is_reference_noiseless = False
    # Set reference noiseless data set directory
    if is_reference_noiseless:
        reference_noiseless_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '6_local_rnn_training_noisy/von_mises/'
             'convergence_analyses_homoscedastic_gaussian/noiseless')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set sizes
    if dataset_type == 'training':
        n_paths_dirs = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        n_paths = n_paths_dirs
    elif dataset_type == 'validation':
        n_paths_dirs = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        n_paths = (2, 4, 8, 16, 32, 64, 128, 256, 512)
    elif dataset_type == 'testing_id':
        n_paths_dirs = (512,)
        n_paths = (512,)
    elif dataset_type == 'testing_od':
        n_paths_dirs = (512,)
        n_paths = (512,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data set sizes
    for i, n_path in enumerate(n_paths):
        # Set data sets base directory
        datasets_base_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '7_local_hybrid_training/case_learning_drucker_prager_pressure/'
             '0_datasets/datasets_base')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data sets directory
        if not os.path.isdir(datasets_base_dir):
            raise RuntimeError('The data sets base directory has not been '
                               'found:\n\n' + datasets_base_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data sets directory (current number of paths)
        datasets_dir = os.path.join(os.path.normpath(datasets_base_dir),
                                    f'n{n_paths_dirs[i]}')
        # Create data sets directory
        if not os.path.isdir(datasets_dir):
            make_directory(datasets_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set file basename
        dataset_basename = 'ss_paths_dataset'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set directory
        if dataset_type == 'training':
            dataset_type_basename = '1_training_dataset'
        elif dataset_type == 'validation':
            dataset_type_basename = '2_validation_dataset'
        elif dataset_type == 'testing_id':
            dataset_type_basename = '5_testing_id_dataset'
        elif dataset_type == 'testing_od':
            dataset_type_basename = '6_testing_od_dataset'
        else:
            raise RuntimeError('Unknown data set type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain formulation
        strain_formulation = 'infinitesimal'
        # Set problem type
        problem_type = 4
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            get_problem_type_parameters(problem_type)
        # Set strain components order
        if strain_formulation == 'infinitesimal':
            strain_comps_order = comp_order_sym
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of discrete times
        n_time = 100
        # Set initial and final time
        time_init = 0.0
        time_end = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain components bounds
        strain_bounds = {x: (-0.05, 0.05) for x in strain_comps_order}
        # Set incremental strain norm
        inc_strain_norm = None
        # Set strain noise
        strain_noise_std = None
        # Set number of loading cycles
        n_cycle = 4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model
        model_name = 'drucker_prager'
        # Set constitutive model parameters:
        if model_name == 'von_mises':
            # Set constitutive model parameters
            model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 110e3, 'v': 0.33,
                 'euler_angles': (0.0, 0.0, 0.0),
                 'hardening_law': get_hardening_law('nadai_ludwik'),
                 'hardening_parameters':
                     {'s0': 900,
                      'a': 700,
                      'b': 0.5,
                      'ep0': 1e-5}}
            # Set constitutive state variables to be additionally included in
            # the data set
            #state_features = {'acc_p_strain': 1}
            state_features = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = VonMises(strain_formulation, problem_type,
                                          model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'von_mises_mixed':
            # Set constitutive model parameters
            model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 110e3, 'v': 0.33,
                 'euler_angles': (0.0, 0.0, 0.0),
                 'hardening_law': get_hardening_law('nadai_ludwik'),
                 'hardening_parameters':
                     {'s0': 900,
                      'a': 700,
                      'b': 0.5,
                      'ep0': 1e-5},
                 'kinematic_hardening_law': get_hardening_law('linear'),
                 'kinematic_hardening_parameters':
                     {'s0': 0,
                      'a': 600}}
            # Set constitutive state variables to be additionally included in
            # the data set
            state_features = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = VonMisesMixed(strain_formulation,
                                               problem_type, model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'drucker_prager':
            # Set frictional angle
            friction_angle = np.deg2rad(10)
            # Set dilatancy angle
            dilatancy_angle = friction_angle
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute angle-related material parameters (matching with
            # Mohr-Coulomb under uniaxial tension and compression)
            # Set yield surface cohesion parameter
            yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
            # Set yield pressure parameter
            yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
            # Set plastic flow pressure parameter
            flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set constitutive model parameters
            # (matching Von Mises yield surface for null pressure)
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 110e3, 'v': 0.33,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),            # Fix: np.sqrt(3) matching factor!
                'hardening_parameters':
                    {'s0': 900/yield_cohesion_parameter,
                     'a': 700/yield_cohesion_parameter,
                     'b': 0.5,
                     'ep0': 1e-5},
                'yield_cohesion_parameter': yield_cohesion_parameter,
                'yield_pressure_parameter': yield_pressure_parameter,
                'flow_pressure_parameter': flow_pressure_parameter,
                'friction_angle': friction_angle}
            # Set constitutive state variables to be additionally included in
            # the data set
            state_features = {'e_strain_mf': len(strain_comps_order)}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = DruckerPrager(strain_formulation,
                                               problem_type, model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Set constitutive model parameters
            model_parameters = {'elastic_symmetry': 'isotropic',
                                'E': 110e3, 'v': 0.33,
                                'euler_angles': (0.0, 0.0, 0.0)}
            # Set constitutive state variables to include in data set
            state_features = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = Elastic(strain_formulation, problem_type,
                                        model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain path type
        strain_path_type = 'proportional'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain path generator parameters
        if strain_path_type == 'proportional':
            strain_path_kwargs = \
                {'strain_bounds': strain_bounds,
                'n_time': n_time,
                'time_init': time_init,
                'time_end': time_end,
                'inc_strain_norm': inc_strain_norm,
                'strain_noise_std': strain_noise_std,
                'n_cycle': n_cycle}
        elif strain_path_type == 'random':
            strain_path_kwargs = \
                {'n_control': (4, 7),
                'strain_bounds': strain_bounds,
                'n_time': n_time,
                'generative_type': 'polynomial',
                'time_init': time_init,
                'time_end': time_end,
                'inc_strain_norm': inc_strain_norm,
                'strain_noise_std': strain_noise_std,
                'n_cycle': 0}
        else:
            raise RuntimeError('Unknown strain path type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain-stress material response path data set generator
        dataset_generator = NoisyMaterialResponseDatasetGenerator(
            strain_formulation, problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate or load noiseless strain-stress material response data set
        if is_reference_noiseless:
            # Set reference noiseless data set path
            reference_noiseless_path = os.path.join(
                os.path.normpath(reference_noiseless_dir),
                f'n{n_paths_dirs[i]}',
                f'{dataset_type_basename}',
                f'{dataset_basename}_n{n_paths[i]}.pkl')
            # Load reference noiseless data set
            noiseless_dataset = load_dataset(reference_noiseless_path)
        else:
            # Generate reference noiseless data set
            noiseless_dataset = dataset_generator.generate_noiseless_dataset(
                n_path, strain_path_type,strain_path_kwargs,
                constitutive_model, state_features=state_features,
                is_verbose=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set noiseless data set directory
        noiseless_dataset_dir = os.path.join(datasets_dir, 'noiseless')
        # Create noiseless data set directory
        if not os.path.isdir(noiseless_dataset_dir):
            make_directory(noiseless_dataset_dir, is_overwrite=True)
        # Set noiseless data set type directory
        noiseless_dataset_type_dir = os.path.join(noiseless_dataset_dir,
                                                  dataset_type_basename)
        # Create noiseless data set directory
        make_directory(noiseless_dataset_type_dir, is_overwrite=True)
        # Save noiseless data set
        save_dataset(noiseless_dataset, dataset_basename,
                     noiseless_dataset_type_dir, is_append_n_sample=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate plots
        if is_save_dataset_plots:
            # Set data set plots directory
            plots_dir = os.path.join(noiseless_dataset_type_dir, 'plots')
            # Create plots directory
            plots_dir = make_directory(plots_dir)
            # Generate data set plots
            generate_dataset_plots(strain_formulation, n_dim,
                                   noiseless_dataset, save_dir=plots_dir,
                                   is_save_fig=True, is_stdout_display=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Proceed to generation of next noiseless data set
        if is_only_noiseless:
            continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of strain components
        n_strain_comps = len(strain_comps_order)
        # Set strain normalization bounds
        strain_minimum = torch.tensor([strain_bounds[key][0]
                                       for key in strain_bounds.keys()])
        strain_maximum = torch.tensor([strain_bounds[key][1]
                                       for key in strain_bounds.keys()])
        # Set strain components min-max fitted data scaler
        strain_data_scaler = TorchMinMaxScaler(n_features=n_strain_comps,
                                               minimum=strain_minimum,
                                               maximum=strain_maximum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set noise variability
        noise_variability = 'homoscedastic'
        # Set noise distribution type
        noise_distribution = 'gaussian'
        # Set strain path denoising
        is_denoise_strain_path = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize noise generator
        noise_generator = NoiseGenerator()
        noise_generator.set_noise_distribution(noise_distribution)
        # Set noise distribution parameters
        if noise_distribution == 'uniform':
            noise_parameters_cases = {'homuni_noise_4e-2': {'amp': 4e-2},
                                      'homuni_noise_1e-1': {'amp': 1e-1},
                                      'homuni_noise_2e-1': {'amp': 2e-1},
                                      'homuni_noise_4e-1': {'amp': 4e-1}}
        elif noise_distribution == 'gaussian':
            noise_parameters_cases = {'homgau_noise_1e-2': {'std': 1e-2},
                                      'homgau_noise_2d5e-2': {'std': 2.5e-2},
                                      'homgau_noise_5e-2': {'std': 5e-2},
                                      'homgau_noise_1e-1': {'std': 1e-1}}
        elif noise_distribution == 'spiked_gaussian':
            spike_magnitude = 0.2
            p_spike = 0.05
            noise_parameters_cases = {
                'homsgau_noise_1e-2': {
                    'std': 1e-2,
                    'spike': spike_magnitude, 'p_spike': p_spike},
                'homsgau_noise_2d5e-2': {
                    'std': 2.5e-2,
                    'spike': spike_magnitude, 'p_spike': p_spike},
                'homsgau_noise_5e-2': {
                    'std': 5e-2,
                    'spike': spike_magnitude, 'p_spike': p_spike},
                'homsgau_noise_1e-1': {
                    'std': 1e-1,
                    'spike': spike_magnitude, 'p_spike': p_spike}}
        else:
            raise RuntimeError('Unknown noise distribution.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set heteroscedastic noise weights
        if noise_variability == 'heteroscedastic':
            # Initialize heteroscedastic normalizer
            heteroscedastic_normalizer = 0.0
            # Set heteroscedastic normalizer as maximum strain norm among all
            # paths of noiseless data set
            for response_path in noiseless_dataset:
                # Get noiseless strain path
                noiseless_strain_path = response_path['strain_path']
                # Compute noiseless strain norm path
                noiseless_strain_norm_path = \
                    dataset_generator.compute_norm_path(
                        noiseless_strain_path, n_dim, strain_comps_order,
                        is_symmetric=strain_formulation == 'infinitesimal')
                # Update heteroscedastic normalizer
                if (np.max(noiseless_strain_norm_path) >
                        heteroscedastic_normalizer):
                    heteroscedastic_normalizer = \
                        np.max(noiseless_strain_norm_path)
        else:
            heteroscedastic_normalizer = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over noise cases
        for noise_label, noise_parameters in noise_parameters_cases.items():
            # Set noise distribution parameters
            noise_generator.set_noise_parameters(noise_parameters)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate noisy strain-stress material response data set
            noisy_dataset = dataset_generator.generate_noisy_dataset(
                noiseless_dataset, noise_generator, constitutive_model,
                state_features={}, strain_data_scaler=strain_data_scaler,
                noise_variability=noise_variability,
                heteroscedastic_normalizer=heteroscedastic_normalizer,
                is_denoise_strain_path=is_denoise_strain_path,
                is_verbose=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set noiseless data set directory
            noisy_dataset_dir = os.path.join(datasets_dir, noise_label)
            # Create noiseless data set directory
            if not os.path.isdir(noisy_dataset_dir):
                make_directory(noisy_dataset_dir, is_overwrite=True)
            # Set noiseless data set type directory
            noisy_dataset_type_dir = os.path.join(noisy_dataset_dir,
                                                dataset_type_basename)
            # Create noiseless data set directory
            make_directory(noisy_dataset_type_dir, is_overwrite=True)
            # Save noiseless data set
            save_dataset(noisy_dataset, dataset_basename,
                         noisy_dataset_type_dir, is_append_n_sample=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate plots
            if is_save_dataset_plots:
                # Set data set plots directory
                plots_dir = os.path.join(noisy_dataset_type_dir, 'plots')
                # Create plots directory
                plots_dir = make_directory(plots_dir)
                # Generate data set plots
                generate_dataset_plots(strain_formulation, n_dim,
                                       noisy_dataset, save_dir=plots_dir,
                                       is_save_fig=True,
                                       is_stdout_display=False)