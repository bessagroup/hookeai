"""DARPA METALS PROJECT: Generate strain-stress material response data set.

Classes
-------
MaterialResponseDatasetGenerator
    Strain-stress material response path data set generator.
    
Functions
---------
generate_dataset_plots
    Generate plots for strain-stress material response path data set.
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
import copy
import time
import datetime
import re
# Third-party
import torch
import numpy as np
import pandas
import tqdm
import matplotlib.pyplot as plt
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory, \
    TimeSeriesDataset, get_time_series_data_loader, save_dataset
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
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from simulators.fetorch.material.models.external.lou import LouZhangYoon
from simulators.fetorch.material.models.external.bazant_m7 import BazantM7
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from ioput.iostandard import make_directory
from ioput.plots import plot_xy_data, plot_xyz_data, scatter_xy_data, \
    plot_histogram, plot_boxplots, save_figure
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
    generate_response_dataset(self, n_path, strain_path_type,
                              strain_path_kwargs, model_name, \
                              model_parameters, state_features={}, \
                              is_in_memory_dataset=True, \
                              dataset_directory=None, \
                              dataset_basename=None, \
                              save_dir=None, is_save_fig=False, \
                              is_verbose=False)
        Generate strain-stress material response path data set.
    compute_stress_path(self, strain_comps, time_hist, strain_path, \
                        constitutive_model)
        Compute material stress response for given strain path.
    build_tensor_from_comps(n_dim, comps, comps_array, is_symmetric=False)
        Build strain/stress tensor from given components.
    store_tensor_comps(comps, tensor)
        Store strain/stress tensor components in array.
    gen_response_dataset_from_csv(self, response_file_paths, \
                                  is_in_memory_dataset=True, save_dir=None, \
                                  is_save_fig=False, is_verbose=False)
    plot_material_response_path(cls, strain_formulation, n_dim, \
                                strain_comps_order, strain_path, \
                                stress_comps_order, stress_path, \
                                time_hist, \
                                is_plot_strain_stress_paths=False, \
                                is_plot_eq_strain_stress=False, \
                                stress_units='', \
                                filename='response_path', \
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
                                  model_parameters, state_features={},
                                  is_in_memory_dataset=True,
                                  dataset_directory=None,
                                  dataset_basename=None,
                                  save_dir=None, is_save_fig=False,
                                  is_verbose=False):
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
        state_features : dict, default={}
            FETorch material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally included in the data set. Unavailable state
            variables are ignored.
        is_in_memory_dataset : bool, default=True
            If True, then generate in-memory time series data set, otherwise
            time series data set samples are stored in local directory.
        dataset_directory : str, default=None
            Directory where the time series data set is stored (all data set
            samples files). Required if is_in_memory_dataset=False.
        dataset_basename : str, default=None
            Data set file base name. Required if is_in_memory_dataset=False.
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.
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
            print('\nGenerate strain-stress material response path data set'
                  '\n------------------------------------------------------')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set parameters requirements
        if not is_in_memory_dataset:
            # Check data set directory
            if dataset_directory is None:
                raise RuntimeError('Time series data set directory must be '
                                   'provided when is_in_memory_dataset=False.')
            # Check data set file base name
            if dataset_basename is None:
                raise RuntimeError('Time series data set file base name must '
                                   'be provided when '
                                   'is_in_memory_dataset=False.')
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
        # Initialize constitutive model
        if model_name == 'elastic':
            constitutive_model = Elastic(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'von_mises':
            constitutive_model = VonMises(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'drucker_prager':
            constitutive_model = DruckerPrager(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'lou_zhang_yoon':
            constitutive_model = LouZhangYoon(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'bazant_m7':
            constitutive_model = BazantM7(
                self._strain_formulation, self._problem_type, model_parameters)
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set maximum number of sample trials
        max_path_trials = 10
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize time series data set samples
        if is_in_memory_dataset:
            dataset_samples = []
        else:
            dataset_sample_files = []
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
            # Initialize material response path data
            response_path = {}
            # Assemble strain-stress material response path
            response_path['strain_comps_order'] = strain_comps_order
            response_path['strain_path'] = torch.tensor(strain_path,
                                                        dtype=torch.float)
            response_path['stress_comps_order'] = stress_comps_order
            response_path['stress_path'] = torch.tensor(stress_path,
                                                        dtype=torch.float)
            # Assemble time path
            response_path['time_hist'] = \
                torch.tensor(time_hist, dtype=torch.float).reshape(-1, 1)
            # Assemble state variables path
            for state_var in state_path.keys():
                response_path[state_var] = torch.tensor(state_path[state_var],
                                                        dtype=torch.float)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store strain-stress material response path
            if is_in_memory_dataset:
                # Store material response path (in memory)
                dataset_samples.append(response_path)
            else:
                # Set material response path file path
                sample_file_path = os.path.join(dataset_directory,
                                                f'ss_response_path_{i}.pt')
                # Store material response path (local directory)
                torch.save(response_path, sample_file_path)
                # Append material response path file path
                dataset_sample_files.append(sample_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot material response path
            if is_save_fig and os.path.isdir(save_dir):
                self.plot_material_response_path(
                    self._strain_formulation, self._n_dim,
                    strain_comps_order, strain_path,
                    stress_comps_order, stress_path,
                    time_hist,
                    is_plot_strain_stress_paths=True,
                    is_plot_eq_strain_stress=True,
                    filename=f'response_path_{i}',
                    save_dir=save_dir, is_save_fig=is_save_fig,
                    is_stdout_display=True, is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Finished strain-stress paths generation process!\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create strain-stress material response path data set
        if is_in_memory_dataset:
            dataset = TimeSeriesDatasetInMemory(dataset_samples)
        else:
            dataset = TimeSeriesDataset(dataset_directory,
                                        dataset_sample_files,
                                        dataset_basename=dataset_basename)
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
            inc_strain = torch.tensor(inc_strain, dtype=torch.float)
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
    def gen_response_dataset_from_csv(self, response_file_paths,
                                      is_in_memory_dataset=True, save_dir=None,
                                      is_save_fig=False, is_verbose=False):
        """Generate strain-stress path data set from set of .csv files.
        
        Parameters
        ----------
        response_file_paths : tuple
            Strain-stress response paths data files paths (.csv).
        is_in_memory_dataset : bool, default=True
            If True, then generate in-memory time series data set, otherwise
            time series data set samples are stored in local directory.
        dataset_directory : str, default=None
            Directory where the time series data set is stored (all data set
            samples files). Required if is_in_memory_dataset=False.
        dataset_basename : str, default=None
            Data set file base name. Required if is_in_memory_dataset=False.
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.
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
            print('\nGenerate strain-stress material response path data set'
                  '\n------------------------------------------------------')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data set parameters requirements
        if not is_in_memory_dataset:
            # Check data set directory
            if dataset_directory is None:
                raise RuntimeError('Time series data set directory must be '
                                   'provided when is_in_memory_dataset=False.')
            # Check data set file base name
            if dataset_basename is None:
                raise RuntimeError('Time series data set file base name must '
                                   'be provided when '
                                   'is_in_memory_dataset=False.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain and stress components order
        if self._strain_formulation == 'infinitesimal':
            strain_comps_order = self._comp_order_sym
            stress_comps_order = self._comp_order_sym
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort response path files
        response_file_paths = tuple(
            sorted(response_file_paths,
                   key=lambda x: int(re.search(r'(\d+)\D*$', x).groups()[-1])))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set total number of response paths files
        n_path_files = len(response_file_paths)
        # Initialize number of invalid response paths files
        n_path_files_invalid = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ v SECTION TO BE REMOVED
        # Set volume fraction input flag 
        is_input_volume_fraction = False
        # Gather volume fraction data
        if is_input_volume_fraction:
            # Set volume fraction data file path
            vf_data_file_path = ('/home/bernardoferreira/Documents/brown/'
                                 'projects/darpa_project/2_local_rnn_training/'
                                 'composite_rve/dataset_07_2024/0_yaga_files/'
                                 'exp_Ti6Al4V_3D_input.csv')
            # Load volume fraction data
            df_vf = pandas.read_csv(vf_data_file_path, header=0)
            # Get volume fraction data
            volume_fractions_data = df_vf.values
            # Initialize valid response paths volume fractions
            vf_path_valid = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ SECTION TO BE REMOVED
        # Initialize time series data set samples
        if is_in_memory_dataset:
            dataset_samples = []
        else:
            dataset_sample_files = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Starting strain-stress paths reading process...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples
        for i, data_file_path in enumerate(tqdm.tqdm(
                response_file_paths, desc='> Reading strain-stress paths: ',
                disable=not is_verbose)):
            # Load data
            df = pandas.read_csv(data_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check data
            if df.shape[1] != 13:
                raise RuntimeError(f'Expecting data frame to have 13 columns, '
                                   f'but {df.shape[1]} were found. \n\n'
                                   f'Expected columns: TIME | E11 E22 E33 E12 '
                                   f'E23 E13 | S11 S22 S33 S12 S23 S13')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get response path data
            response_path_data = df.values
            # Get discrete time history
            time_hist = response_path_data[:, 0]
            # Get strain and stress paths
            if self._n_dim == 2:
                strain_path = response_path_data[:, [1, 2, 4]]
                stress_path = response_path_data[:, [7, 8, 10]]
            else:
                strain_path = response_path_data[:, 1:7]
                stress_path = response_path_data[:, 7:13]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize material response path data
            response_path = {}
            # Assemble strain-stress material response path
            response_path['strain_comps_order'] = strain_comps_order
            response_path['strain_path'] = torch.tensor(strain_path,
                                                        dtype=torch.float)
            response_path['stress_comps_order'] = stress_comps_order
            response_path['stress_path'] = torch.tensor(stress_path,
                                                        dtype=torch.float)
            # Assemble time path
            response_path['time_hist'] = \
                torch.tensor(time_hist, dtype=torch.float).reshape(-1, 1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ v SECTION TO BE REMOVED
            # Get volume fraction path
            if is_input_volume_fraction:
                # Get response path volume fraction
                vf = volume_fractions_data[i, 1]
                # Set volume fraction path
                vf_path = torch.tile(torch.tensor(vf, dtype=torch.float),
                                     (time_hist.shape[0], 1))
                # Assemble volume fraction path
                response_path['vf_path'] = vf_path
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ SECTION TO BE REMOVED
            # Initialize response path invalid flag
            is_invalid_path = False
            # Evaluate strain-stress material response path
            if torch.allclose(response_path['stress_path'],
                              torch.zeros_like(response_path['stress_path'])):
                n_path_files_invalid += 1
                is_invalid_path = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store or discard strain-stress material response path
            if is_invalid_path:
                continue
            else:
                if is_in_memory_dataset:
                    # Store material response path (in memory)
                    dataset_samples.append(response_path)
                else:
                    # Set material response path file path
                    sample_file_path = os.path.join(dataset_directory,
                                                    f'ss_response_path_{i}.pt')
                    # Store material response path (local directory)
                    torch.save(response_path, sample_file_path)
                    # Append material response path file path
                    dataset_sample_files.append(sample_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ v SECTION TO BE REMOVED
            # Store valid response path volume fraction
            if is_input_volume_fraction:
                vf_path_valid.append(vf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ SECTION TO BE REMOVED
            # Plot material response path
            if is_save_fig and os.path.isdir(save_dir):
                self.plot_material_response_path(
                    self._strain_formulation, self._n_dim,
                    strain_comps_order, strain_path,
                    stress_comps_order, stress_path,
                    time_hist,
                    is_plot_strain_stress_paths=True,
                    is_plot_eq_strain_stress=True,
                    filename=f'response_path_{i}',
                    save_dir=save_dir, is_save_fig=is_save_fig,
                    is_stdout_display=False, is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Finished strain-stress paths reading process!\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute valid response file paths
        n_path_files_valid = n_path_files - n_path_files_invalid
        # Compute ratio of valid and invalid response file paths
        ratio_valid = n_path_files_valid/n_path_files
        ratio_invalid = n_path_files_invalid/n_path_files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n> Number of response path files: {n_path_files}')
            print(f'\n  > Number of valid response path files: '
                  f'{n_path_files_valid:d}/{n_path_files:d} '
                  f'({100*ratio_valid:>.1f}%)')
            print(f'\n  > Number of invalid response path files: '
                  f'{n_path_files_invalid:d}/{n_path_files:d} '
                  f'({100*ratio_invalid:>.1f}%)\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create strain-stress material response path data set
        if is_in_memory_dataset:
            dataset = TimeSeriesDatasetInMemory(dataset_samples)
        else:
            dataset = TimeSeriesDataset(dataset_directory,
                                        dataset_sample_files,
                                        dataset_basename=dataset_basename)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of strain-stress paths
        n_path = len(dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute total generation time and average generation time per path
        total_time_sec = time.time() - start_time_sec
        avg_time_sec = total_time_sec/n_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n> Total generation time: '
                  f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
                  f'Avg. processing time per path: '
                  f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ v SECTION TO BE REMOVED      
        # Plot valid response paths volume fraction
        if is_input_volume_fraction:
            # Plot valid response paths volume fraction
            figure, _ = \
                plot_histogram((np.array(vf_path_valid),),
                                bins=20, x_lims=(0.0, 0.5),
                                x_label='Volume fraction',
                                y_label='Number of valid response paths',
                                is_latex=True)
            # Save figure
            save_figure(figure, f'ss_paths_valid_vf_hist_{n_path_files_valid}',
                        format='pdf', save_dir=save_dir)        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ SECTION TO BE REMOVED
        return dataset
    # -------------------------------------------------------------------------
    @classmethod
    def plot_material_response_path(cls, strain_formulation, n_dim,
                                    strain_comps_order, strain_path,
                                    stress_comps_order, stress_path,
                                    time_hist,
                                    is_plot_strain_stress_paths=False,
                                    is_plot_eq_strain_stress=False,
                                    stress_units='',
                                    filename='response_path',
                                    save_dir=None, is_save_fig=False,
                                    is_stdout_display=False, is_latex=False):
        """Plot strain-stress material response path.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
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
        is_plot_strain_stress_paths : bool, default=False
            Plot strain and stress components path (only available for single
            path).
        is_plot_eq_strain_stress : bool, default=False
            Plot equivalent strain-stress path.
        stress_units : str, default=''
            Stress units label.
        filename : str, default='response_path'
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
        # Check strain-stress paths data
        if isinstance(time_hist, list):
            # Get number of strain-stress paths
            n_path = len(time_hist)
            # Get longest path
            n_time_max = max([len(x) for x in time_hist])
            # Get minimum and maximum discrete times
            time_min = min([x[0] for x in time_hist])
            time_max = max([x[-1] for x in time_hist])
        else:
            # Set number of strain-stress paths
            n_path = 1
            # Set longest strain-stress path
            n_time_max = len(time_hist)
            # Set minimum and maximum discrete times
            time_min = time_hist[0]
            time_max = time_hist[-1]
            # Store discrete time history and strain-stress path in list
            time_hist = [time_hist,]
            strain_path = [strain_path,]
            stress_path = [stress_path,]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store file basename
        basename = filename
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain-stress path
        if is_plot_strain_stress_paths and n_path == 1:
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
                    # Set path
                    path = strain_path
                    # Set filename
                    filename = basename + '_strain'
                else:
                    # Set components
                    comps = stress_comps_order
                    # Set component label
                    comp_label = 'Stress'
                    # Set y-axis label
                    y_label = 'Stress' + stress_units
                    # Set path
                    path = stress_path
                    # Set filename
                    filename = basename + '_stress'
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize data array
                data_xy = np.zeros((n_time_max, 2*len(comps)))
                # Set data array
                for j in range(len(comps)):
                    data_xy[:, 2*j] = time_hist[0].reshape(-1)
                    data_xy[:, 2*j + 1] = path[0][:, j]
                # Set data labels
                data_labels = [f'{comp_label} {x}' for x in comps]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot path
                figure, _ = plot_xy_data(data_xy, data_labels=data_labels,
                                         x_lims=(time_min, time_max),
                                         x_label='Time', y_label=y_label,
                                         is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename, format='pdf',
                                save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize equivalent strain and stress paths
        eq_strain_paths = []
        eq_stress_paths = []
        # Loop over strain-stress paths
        for k in range(n_path):
            # Initialize equivalent strain and stress path
            eq_strain = np.zeros((n_time_max, 1))
            eq_stress = np.zeros((n_time_max, 1))
            # Loop over discrete time
            for i in range(len(time_hist[k])):
                # Get strain tensor
                strain = cls.build_tensor_from_comps(
                    n_dim, strain_comps_order, strain_path[k][i, :],
                    is_symmetric=strain_formulation == 'infinitesimal')
                # Get stress tensor
                stress = cls.build_tensor_from_comps(
                    n_dim, stress_comps_order, stress_path[k][i, :],
                    is_symmetric=strain_formulation == 'infinitesimal')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute equivalent strain and stress
                if strain_formulation == 'infinitesimal':
                    eq_strain[i, 0] = np.sqrt(2.0/3.0)*np.linalg.norm(
                        strain - (1.0/3.0)*np.trace(strain))
                    eq_stress[i, 0] = np.sqrt(3.0/2.0)*np.linalg.norm(
                        stress - (1.0/3.0)*np.trace(stress))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store equivalent strain and stress path
            eq_strain_paths.append(eq_strain)
            eq_stress_paths.append(eq_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot equivalent strain-stress path
        if n_dim == 3 and is_plot_eq_strain_stress:
            # Build equivalent strain-stress paths data array
            eq_strain_stress_data_xy = np.concatenate(
                [np.concatenate((eq_strain_paths[k], eq_stress_paths[k]),
                                axis=1) for k in range(n_path)], axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot equivalent strain-stress path
            figure, _ = plot_xy_data(
                data_xy=eq_strain_stress_data_xy,
                x_lims=(0, None),
                y_lims=(0, None),
                x_label='Equivalent strain',
                y_label='Equivalent stress' + stress_units,
                is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, basename + '_eq_strain_stress',
                            format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
    # -------------------------------------------------------------------------
    @classmethod
    def plot_stress_space_metrics(cls, strain_formulation, stress_comps_order,
                                  stress_path, time_hist,
                                  is_plot_principal_stress_path=False,
                                  is_plot_pi_stress_path_pairs=False,
                                  is_plot_stress_invar_hist=False,
                                  is_plot_stress_invar_box=False,
                                  is_plot_stress_path_triax_lode=False,
                                  is_plot_stress_triax_lode_space=False,
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
        is_plot_pi_stress_path_pairs : bool, default=False
            Plot the stress path for pairs of pi-stress components in the
            principal stress space.
        is_plot_stress_invar_hist : bool, default=False
            Plot distribution of stress invariants.
        is_plot_stress_invar_box : bool, default=False
            Plot box plot with stress invariants.
        is_plot_stress_path_triax_lode : bool, default=False
            Plot stress triaxiality and Lode parameter paths.
        is_plot_stress_triax_lode_space : bool, default=False
            Plot stress triaxiality and Lode parameter space.
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
        # Initialize principal stresses
        stress_paths_principal = []
        # Compute stress invariants
        for k in range(n_path):
            # Initialize principal stresses
            stress_eigen = np.zeros((n_time_max, n_dim))
            # Loop over discrete time
            for i in range(len(time_hist[k])):
                # Get stress tensor
                stress = cls.build_tensor_from_comps(
                    n_dim, stress_comps_order, stress_path[k][i, :],
                    is_symmetric=strain_formulation == 'infinitesimal')
                # Compute principal stresses
                eigenvalues, _ = np.linalg.eig(stress)
                # Store principal stresses
                is_sort_principal = True
                if is_sort_principal:
                    # Sort principal stresses in descending order
                    stress_eigen[i, :] = np.sort(eigenvalues)[::-1]
                else:
                    # Keep principal stresses order stemming from eigenvalues
                    # computation algorithm
                    stress_eigen[i, :] = eigenvalues
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
                view_angles_deg=(20, 15, 0),
                is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_principal',
                            format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Plot stress path in principal stress space (pi stress coordinates)
        if is_plot_pi_stress_path_pairs:
            # Set rotation matrix between principal stress coordinates to pi
            # stress coordinates
            rotation_matrix = \
                np.array([[np.sqrt(2/3), -np.sqrt(1/6), -np.sqrt(1/6)],
                          [0, np.sqrt(1/2), -np.sqrt(1/2)],
                          [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize pi stresses
            stress_paths_pi = []
            # Loop over stress paths
            for k in range(n_path):
                # Randomize path principal stresses order (same permutation is
                # applied for all time steps)
                stress_path_principal_random = \
                    stress_paths_principal[k][:, np.random.permutation(3)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize stresses in pi-stress space
                stress_pi = np.zeros((n_time_max, n_dim))
                # Loop over discrete time
                for i in range(len(time_hist[k])):
                    # Get principal stresses
                    stress_eigen = stress_path_principal_random[i, :]
                    # Compute pi stresses
                    stress_pi[i, :] = np.matmul(rotation_matrix, stress_eigen)
                # Store stress path pi stresses
                stress_paths_pi.append(stress_pi)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set pi stress pairs
            pi_stress_pairs = (('1', '2'), ('1', '3'), ('2', '3'))
            # Loop over pi stress pairs
            for pi_stress_pair in pi_stress_pairs:
                # Get pi stress components indexes
                j_x = int(pi_stress_pair[0]) - 1
                j_y = int(pi_stress_pair[1]) - 1
                # Initialize stress data array
                stress_data_xy = np.full((n_time_max, 2*n_path),
                                         fill_value=np.nan)
                # Loop over stress paths 
                for k in range(n_path):
                    # Set stress data array
                    stress_data_xy[:len(time_hist[k]), 2*k] = \
                        stress_paths_pi[k][:, j_x]
                    stress_data_xy[:len(time_hist[k]), 2*k + 1] = \
                        stress_paths_pi[k][:, j_y]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot stress path for pi stress components pair
                figure, _ = plot_xy_data(
                    data_xy=stress_data_xy,
                    x_label=(f'Pi-Stress {pi_stress_pair[0]}'
                             + stress_units),
                    y_label=(f'Pi-Stress {pi_stress_pair[1]}'
                             + stress_units),
                    marker='o', markersize=2, is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + '_pi_stress_'
                                + f'{pi_stress_pair[0]}v{pi_stress_pair[1]}',
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
                        (stress_invar_data[:, i],), bins=50, density=True,
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
                    stress_lode[i, 0] = (3.0*np.sqrt(3)/2.0)*(j3/(j2**(3/2)))
            # Store stress path triaxiality and Lode parameter
            stress_paths_triax.append(stress_triax)
            stress_paths_lode.append(stress_lode)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress triaxiality and Lode parameter paths
        if is_plot_stress_path_triax_lode:
            # Set stress triaxiality data array
            triax_data_xy = np.zeros((n_time_max, 2*n_path))
            for k in range(n_path):
                triax_data_xy[:len(time_hist[k]), 2*k] = time_hist[k]
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
                lode_data_xy[:len(time_hist[k]), 2*k] = time_hist[k]
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
                save_figure(figure, filename + '_lode_parameter', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress invariants (distribution and box plot)
        if (is_plot_stress_triax_lode_space or is_plot_stress_triax_lode_hist
                or is_plot_stress_triax_lode_box):
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
            # Plot stress triaxiality and Lode parameter space
            if is_plot_stress_triax_lode_space:
                # Set concatenation option
                is_concatenate_paths = True
                # Set scatter plot data
                if is_concatenate_paths:
                    stress_triax_lode_data_xy = stress_triax_lode_data[:, ::-1]
                else:
                    # Initialize data array
                    stress_triax_lode_data_xy = \
                        np.zeros((n_time_max, 2*n_path))
                    # Loop over stress paths
                    for k in range(n_path):
                        # Assemble stress path data
                        stress_triax_lode_data_xy[:, 2*k] = \
                            stress_paths_lode[k].reshape(-1)
                        stress_triax_lode_data_xy[:, 2*k+1] = \
                            stress_paths_triax[k].reshape(-1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                figure, _ = scatter_xy_data(
                    data_xy=stress_triax_lode_data_xy,
                    x_label=data_labels[1],
                    y_label=data_labels[0],
                    is_marginal_dists = True,
                    x_lims=(-1, 1),
                    is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename
                                + '_triaxiality_lode_marginals',
                                format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot stress triaxiality and Lode parameter distributions
            if is_plot_stress_triax_lode_hist:
                for i, metric in enumerate(data_labels):
                    figure, _ = plot_histogram(
                        (stress_triax_lode_data[:, i],), bins=50, density=True,
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
                    save_figure(figure, filename + '_triaxiality_lode_boxplot',
                                format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
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
    # -------------------------------------------------------------------------
    @classmethod
    def plot_state_history(cls, strain_formulation, strain_comps_order,
                           stress_comps_order, state_variable_name,
                           state_variable_label, state_path, time_hist,
                           state_units='',
                           filename='state_variable_path',
                           save_dir=None, is_save_fig=False,
                           is_stdout_display=False, is_latex=False):
        """Plot constitutive state variable path.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
        strain_comps_order : tuple
            Stress components order.
        stress_comps_order : tuple
            Stress components order.
        state_variable_name : str
            State variable name.
        state_variable_label : str
            State variable label.
        state_path : {numpy.ndarray(2d), list[numpy.ndarray(2d)]}
            State variable path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_features) or list of multiple state variable
            path histories.
        time_hist : {numpy.ndarray(1d), list[numpy.ndarray(1d)]}
            Discrete time history or list of multiple discrete time histories.
        state_units : str, default=''
            State variable units label.
        filename : str, default='state_variable_path'
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
        # Check state paths data
        if isinstance(time_hist, list) or isinstance(state_path, list):
            # Check multiple state paths
            if (not isinstance(time_hist, list)
                    and not isinstance(state_path, list)):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'state path histories when providing '
                                   'multiple state paths.')
            elif len(time_hist) != len(state_path):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'state path histories when providing '
                                   'multiple state paths.')
            # Get number of state paths
            n_path = len(state_path)
            # Get longest state path
            n_time_max = max([len(x) for x in time_hist])
            # Get minimum and maximum discrete times
            time_min = min([x[0] for x in time_hist])
            time_max = max([x[-1] for x in time_hist])
        else:
            # Set number of state paths
            n_path = 1
            # Set longest state path
            n_time_max = len(time_hist)
            # Set minimum and maximum discrete times
            time_min = time_hist[0]
            time_max = time_hist[-1]
            # Store discrete time history and stress path in list
            time_hist = [time_hist,]
            state_path = [state_path,]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize figure
        figure = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Probe dimensionality of state variable
        n_features = state_path[0].shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot state variable path
        if n_features == 1:
            # Initialize state variable data array
            state_data_xy = np.full((n_time_max, 2*n_path), fill_value=np.nan)
            # Loop over strain paths
            for k in range(n_path):
                state_data_xy[:len(time_hist[k]), 2*k] = time_hist[k]
                state_data_xy[:len(time_hist[k]), 2*k+1] = \
                    state_path[k].reshape(-1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot state variable path
            figure, _ = \
                plot_xy_data(data_xy=state_data_xy,
                             x_lims=(time_min, time_max),
                             x_label='Time',
                             y_label=state_variable_label + state_units,
                             is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename, format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate state variable paths
            state_data = np.vstack(state_path).reshape(-1)
            # Plot state variable distribution
            figure, _ = plot_histogram(
                (state_data,), bins=50, density=True,
                x_label=state_variable_label + state_units,
                y_label='Probability density',
                is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_hist', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
# =============================================================================
def generate_dataset_plots(strain_formulation, n_dim, dataset,
                           save_dir=None, is_save_fig=False,
                           is_stdout_display=False):
    """Generate plots for strain-stress material response path data set.
    
    The different plots options must be set manually by means of the
    corresponding plotting methods arguments.
    
    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    n_dim : int
        Problem number of spatial dimensions.
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    save_dir : str
        Directory where data set plots are saved. If None, then plots are
        saved in current working directory.
    is_save_fig : bool, default=False
        Save data set plots.
    is_stdout_display : bool, default=False
        True if displaying data set plots to standard output device, False
        otherwise.
    """
    # Get data loader
    data_loader = get_time_series_data_loader(dataset, batch_size=1)
    # Initialize strain-stress paths data
    time_hists = []
    strain_paths = []
    stress_paths = []
    # Loop over strain-stress paths
    for path in data_loader:
        # Collect strain-stress path data
        time_hists.append(np.array(path['time_hist'][:, 0, :]).reshape(-1))
        strain_paths.append(np.array(path['strain_path'][:, 0, :]))
        stress_paths.append(np.array(path['stress_path'][:, 0, :]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get strain components order
    strain_comps_order = dataset[0]['strain_comps_order']
    # Plot strain paths data
    StrainPathGenerator.plot_strain_path(
        strain_formulation, n_dim,
        strain_comps_order, time_hists, strain_paths,
        is_plot_strain_path=True,
        is_plot_strain_comp_hist=False,
        is_plot_strain_norm=False,
        is_plot_strain_norm_hist=False,
        is_plot_inc_strain_norm=False,
        is_plot_inc_strain_norm_hist=False,
        is_plot_strain_path_pairs=False,
        is_plot_strain_pairs_hist=False,
        is_plot_strain_pairs_marginals=False,
        is_plot_strain_comp_box=True,
        strain_label='Strain',
        strain_units='',
        filename='strain_path',
        save_dir=save_dir,
        is_save_fig=is_save_fig,
        is_stdout_display=is_stdout_display,
        is_latex=True)
    # Plot stress paths data
    StrainPathGenerator.plot_strain_path(
        strain_formulation, n_dim,
        strain_comps_order, time_hists, stress_paths,
        is_plot_strain_path=True,
        is_plot_strain_comp_hist=False,
        is_plot_strain_norm=False,
        is_plot_strain_norm_hist=False,
        is_plot_inc_strain_norm=False,
        is_plot_inc_strain_norm_hist=False,
        is_plot_strain_path_pairs=False,
        is_plot_strain_pairs_hist=False,
        is_plot_strain_pairs_marginals=False,
        is_plot_strain_comp_box=True,
        strain_label='Stress',
        strain_units=' (MPa)',
        filename='stress_path',
        save_dir=save_dir,
        is_save_fig=is_save_fig,
        is_stdout_display=is_stdout_display,
        is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get stress components order
    stress_comps_order = dataset[0]['stress_comps_order']
    # Plot stress paths data
    if n_dim == 3:
        MaterialResponseDatasetGenerator.plot_stress_space_metrics(
            strain_formulation, stress_comps_order,
            stress_paths, time_hists,
            is_plot_principal_stress_path=False,
            is_plot_pi_stress_path_pairs=True,
            is_plot_stress_invar_hist=False,
            is_plot_stress_invar_box=False,
            is_plot_stress_path_triax_lode=False,
            is_plot_stress_triax_lode_space=False,
            is_plot_stress_triax_lode_hist=False,
            is_plot_stress_triax_lode_box=True,
            stress_units=' (MPa)',
            filename='stress_path',
            save_dir=save_dir,
            is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display, is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot state variable paths data
    is_plot_state_variable_paths = True
    if is_plot_state_variable_paths:
        # Initialize state variable path data
        state_paths = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state variable name and label
        state_variable_name = 'acc_p_strain'
        state_variable_label = 'Accumulated plastic strain'
        # Loop over strain-stress paths
        for path in data_loader:
            # Check state variable
            if state_variable_name not in path.keys():
                continue
            # Collect state variable path data
            state_paths.append(np.array(path[state_variable_name][:, 0, :]))
        # Plot state variable paths data
        if state_paths:
            MaterialResponseDatasetGenerator.plot_state_history(
                strain_formulation, strain_comps_order, stress_comps_order,
                state_variable_name, state_variable_label, state_paths,
                time_hists, state_units='', filename='acc_p_strain_path',
                save_dir=save_dir, is_save_fig=is_save_fig,
                is_stdout_display=is_stdout_display, is_latex=True)
# =============================================================================
if __name__ == '__main__':
    # Set data set type
    dataset_type = ('training', 'validation', 'testing_id', 'testing_od')[0]
    # Set data set storage type
    is_in_memory_dataset = True
    # Set save dataset plots flags
    is_save_dataset_plots = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/3_local_rc_training/')
    # Set case study directory
    case_study_name = 'lou'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # Set data set file basename
    dataset_basename = 'ss_paths_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set directory
    if dataset_type == 'training':
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '1_training_dataset')
    elif dataset_type == 'validation':
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '2_validation_dataset')
    elif dataset_type == 'testing_id':
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '5_testing_id_dataset')
    elif dataset_type == 'testing_od':
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '6_testing_od_dataset')
    else:
        raise RuntimeError('Unknown data set type.')
    # Create data set directory (overwrite existing directory)
    make_directory(dataset_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of discrete times
    n_time = 100
    # Set initial and final time
    time_init = 0.0
    time_end = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain components bounds
    if n_dim == 2:
        strain_bounds = {x: (-0.05, 0.05) for x in ('11', '22', '12')}
    else:
        strain_bounds = \
            {x: (-0.05, 0.05) for x in ('11', '22', '33', '12', '23', '13')}
    # Set incremental strain norm
    inc_strain_norm = None
    # Set strain noise
    strain_noise_std = None
    # Set cyclic loading
    is_cyclic_loading = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model
    model_name = 'lou_zhang_yoon'
    # Set constitutive model parameters:
    if model_name == 'von_mises':
        # Set constitutive model parameters
        model_parameters = {'elastic_symmetry': 'isotropic',
                            'E': 110e3, 'v': 0.33,
                            'euler_angles': (0.0, 0.0, 0.0),
                            'hardening_law': get_hardening_law('nadai_ludwik'),
                            'hardening_parameters':
                                {'s0': 900,
                                 'a': 700,
                                 'b': 0.5,
                                 'ep0': 1e-5}}
        # Set constitutive state variables to be additionally included in the
        # data set
        state_features = {'acc_p_strain': 1}
    elif model_name == 'drucker_prager':
        # Set frictional angle
        friction_angle = np.deg2rad(10)
        # Set dilatancy angle
        dilatancy_angle = friction_angle
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute angle-related material parameters
        # (matching with Mohr-Coulomb under uniaxial tension and compression)
        # Set yield surface cohesion parameter
        yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
        # Set yield pressure parameter
        yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
        # Set plastic flow pressure parameter
        flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model parameters
        # (matching Von Mises yield surface for null pressure)
        model_parameters = {
            'elastic_symmetry': 'isotropic',
            'E': 110e3, 'v': 0.33,
            'euler_angles': (0.0, 0.0, 0.0),
            'hardening_law': get_hardening_law('nadai_ludwik'),
            'hardening_parameters':
                {'s0': 900/yield_cohesion_parameter,
                 'a': 700/yield_cohesion_parameter,
                 'b': 0.5,
                 'ep0': 1e-5},
            'yield_cohesion_parameter': yield_cohesion_parameter,
            'yield_pressure_parameter': yield_pressure_parameter,
            'flow_pressure_parameter': flow_pressure_parameter,
            'friction_angle': friction_angle}
        # Set constitutive state variables to be additionally included in the
        # data set
        state_features = {}
    elif model_name == 'lou_zhang_yoon':
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
             'a_hardening_law': get_hardening_law('linear'),
             'a_hardening_parameters':
                 {'s0': np.sqrt(3),
                   'a': 0},
             'b_hardening_law': get_hardening_law('linear'),
             'b_hardening_parameters':
                 {'s0': 0,
                   'a': 0},
             'c_hardening_law': get_hardening_law('linear'),
             'c_hardening_parameters':
                 {'s0': 0,
                   'a': 0},
             'd_hardening_law': get_hardening_law('linear'),
             'd_hardening_parameters':
                 {'s0': 0,
                   'a': 0}}
        # Set constitutive state variables to be additionally included in the
        # data set
        state_features = {}
    elif model_name == 'bazant_m7':
        # Set constitutive model parameters
        model_parameters = {}
        # Set constitutive state variables to include in data set
        state_features = {}
    else:
        # Set constitutive model parameters
        model_parameters = {'elastic_symmetry': 'isotropic',
                            'E': 100, 'v': 0.3,
                            'euler_angles': (0.0, 0.0, 0.0)}
        # Set constitutive state variables to include in data set
        state_features = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain-stress paths of each type
    n_path_type = {'proportional': 1, 'random': 0}
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
        {'n_control': (4, 7),
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
            model_parameters, state_features=state_features,
            is_in_memory_dataset=is_in_memory_dataset,
            dataset_directory=dataset_directory,
            dataset_basename=dataset_basename, is_verbose=True)
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
    # Save data set
    save_dataset(dataset, dataset_basename, dataset_directory,
                 is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots
    if is_save_dataset_plots:
        # Set data set plots directory
        plots_dir = os.path.join(dataset_directory, 'plots')
        # Create plots directory
        plots_dir = make_directory(plots_dir)
        # Generate data set plots
        generate_dataset_plots(strain_formulation, n_dim, dataset,
                               save_dir=plots_dir, is_save_fig=True,
                               is_stdout_display=False)