"""Recurrent constitutive model (wrapper of known constitutive model).

Classes
-------
RecurrentConstitutiveModel(torch.nn.Module)
    Recurrent constitutive model.
    
Functions
---------
standard_partial_fit(dataset, features_type, n_features, is_verbose=False)
    Perform batch fitting of standardization data scalers.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
import re
import pickle
# Third-party
import torch
import tqdm
import sklearn.preprocessing
# Local
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_from_mf
from simulators.fetorch.material.material_su import material_state_update
from rnn_base_model.data.time_dataset import get_time_series_data_loader
from gnn_base_model.model.gnn_model import TorchStandardScaler
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class RecurrentConstitutiveModel(torch.nn.Module):
    """Recurrent constitutive model.
    
    While preserving the structure of a standard recurrent neural network
    model, the model core is a wrapper around a known constitutive model.
    
    The learnable parameters are parameters from the wrapped constitutive
    model and are set as initialization parameters of the recurrent model.
    
    Attributes
    ----------
    model_directory : str
        Directory where model is stored.
    model_name : str, default='gru_rnn_model'
        Name of model.
    _n_features_in : int
        Number of input features.
    _n_features_out : int
        Number of output features.
    _learnable_parameters : tuple[str]
        Learnable material constitutive model parameters.
    _strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    _problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    _material_model_name : str
        Material constitutive model name.
    _material_model_parameters : torch.nn.ParameterDict
        Material constitutive model parameters with learnable parameters.
    _state_features_out : dict, default={}
        Material constitutive model state variables (key, str) and
        corresponding dimensionality (item, int) for which the path history
        is additionally predicted in the output features besides the stress
        path history. State variables are sorted as output features
        according with the insertion order.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_data_normalization : bool
        If True, then input and output features are normalized for training
        False otherwise. Data scalers need to be fitted with fit_data_scalers()
        and are stored as model attributes.
    _data_scalers : dict
        Data scaler (item, sklearn.preprocessing.StandardScaler) for each
        feature data (key, str).

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self)
        Forward propagation.
    _recurrent_constitutive_model(self, strain_paths)
        Compute material response.
    _compute_stress_path(self, strain_path)
        Compute material response for given strain path.
    build_tensor_from_comps(cls, n_dim, comps, comps_array, \
                            is_symmetric=False)
        Build strain/stress tensor from given components.
    store_tensor_comps(cls, comps, tensor)
        Store strain/stress tensor components in array.
    save_model_init_file(self)
        Save model class initialization attributes.
    save_model_state(self)
        Save model state to file.
    load_model_state(self)
        Load model state from file.
    _check_state_file(self, filename)
        Check if file is model training epoch state file.
    _check_best_state_file(self, filename)
        Check if file is model training epoch best state file.
    _remove_posterior_state_files(self, epoch)
        Delete model training epoch state files posterior to given epoch.
    _remove_best_state_files(self)
        Delete existent model best state files.
    _init_data_scalers(self)
        Initialize model data scalers.
    fit_data_scalers(self, dataset, is_verbose=False)
        Fit model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    load_model_data_scalers_from_file(self)
        Load data scalers from model initialization file.
    check_normalized_return(self)
        Check if model data normalization is available.
    """
    def __init__(self, n_features_in, n_features_out, learnable_parameters,
                 strain_formulation, problem_type, material_model_name,
                 material_model_parameters, model_directory, model_name,
                 state_features_out={}, is_data_normalization=False,
                 is_save_model_init_file=True, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features_in : int
            Number of input features.
        n_features_out : int
            Number of output features.
        learnable_parameters : tuple[str]
            Learnable material constitutive model parameters.
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        material_model_name : str
            Material constitutive model name.
        material_model_parameters : dict
            Material constitutive model parameters.
        model_directory : str
            Directory where model is stored.
        model_name : str, default='wrapper_recurrent_model'
            Name of model.
        state_features_out : dict, default={}
            Material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally predicted in the output features besides the stress
            path history. State variables are sorted as output features
            according with the insertion order.
        is_data_normalization : bool, default=False
            If True, then input and output features are normalized for
            training, False otherwise. Data scalers need to be fitted with
            fit_data_scalers() and are stored as model attributes.
        is_save_model_init_file: bool, default=True
            If True, saves model initialization file when model is initialized
            (overwritting existent initialization file), False otherwise. When
            initializing model from initialization file this option should be
            set to False to avoid updating the initialization file and preserve
            fitted data scalers.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(RecurrentConstitutiveModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_features_in = n_features_in
        self._n_features_out = n_features_out
        # Set additional state variable output features
        self._state_features_out = state_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material constitutive model name
        self._material_model_name = material_model_name
        # Set learnable parameters
        self._learnable_parameters = learnable_parameters
        # Setup constitutive model learnable parameters
        self._material_model_parameters = self._setup_learnable_parameters(
            learnable_parameters, material_model_name,
            material_model_parameters)        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory and name
        if os.path.isdir(model_directory):
            self.model_directory = str(model_directory)
        else:
            raise RuntimeError('The model directory has not been found.')
        if not isinstance(model_name, str):
            raise RuntimeError('The model name must be a string.')
        else:
            self.model_name = model_name
        # Set normalization flag
        self.is_data_normalization = is_data_normalization
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model
        if material_model_name == 'elastic':
            self._constitutive_model = \
                Elastic(self._strain_formulation, self._problem_type,
                        self._material_model_parameters,
                        device_type=self._device_type)
        elif material_model_name == 'von_mises':
            self._constitutive_model = \
                VonMises(self._strain_formulation, self._problem_type,
                         self._material_model_parameters)
        elif material_model_name == 'drucker_prager':
            self._constitutive_model = \
                DruckerPrager(self._strain_formulation, self._problem_type,
                              self._material_model_parameters)
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{material_model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        self._data_scalers = None
        if self.is_data_normalization:
            self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model initialization file
        if is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    @staticmethod
    def init_model_from_file(model_directory):
        """Initialize model from initialization file.
        
        Initialization file is assumed to be stored in the model directory
        under the name model_init_file.pkl.
        
        Parameters
        ----------
        model_directory : str
            Directory where model is stored.
        """
        # Check model directory
        if not os.path.isdir(model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The model initialization file has not been '
                               'found:\n\n' + model_init_file_path)
        else:
            with open(model_init_file_path, 'rb') as model_init_file:
                model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization attributes
        model_init_args = model_init_attributes['model_init_args']
        # Update model directory
        model_init_args['model_directory'] = model_directory
        # Initialize model
        model = RecurrentConstitutiveModel(**model_init_args,
                                           is_save_model_init_file=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        model._data_scalers = model_data_scalers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Returns
        -------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def _setup_learnable_parameters(self, learnable_parameters,
                                    material_model_name,
                                    material_model_parameters):
        """Setup constitutive model learnable parameters requiring gradients.
        
        Parameters
        ----------
        learnable_parameters : tuple[str]
            Learnable material constitutive model parameters.
        material_model_name : str
            Material constitutive model name.
        material_model_parameters : dict
            Material constitutive model parameters.
        
        Returns
        -------
        material_model_parameters_grad : torch.nn.ParameterDict
            Material constitutive model parameters with learnable parameters.
        """
        # Initialize material constitutive model parameters
        material_model_parameters_grad = \
            torch.nn.ParameterDict(material_model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup constitutive model learnable parameters
        if material_model_name == 'von_mises':
            # Set valid learnable parameters
            valid_learnable_parameters = ('E', 'v', 's0', 'a', 'b')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make hardening parameters accessible as learnable parameters
            material_model_parameters_grad['hardening'] = \
                torch.nn.ParameterDict(
                    material_model_parameters_grad['hardening'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Activate computation graphs for learnable parameters
            for param in learnable_parameters:
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set learnable parameter                
                if param in ('s0', 'a', 'b'):
                    # Pick initialization value
                    init_value = float(material_model_parameters \
                        ['hardening_parameters'][param])
                    # Set learnable parameter
                    material_model_parameters['hardening_parameters'][param] =\
                        torch.nn.Parameter(
                            torch.tensor(init_value, requires_grad=True))
                else:
                    # Pick initialization value
                    init_value = float(material_model_parameters[param])
                    # Set learnable parameter
                    material_model_parameters_grad[param] = torch.nn.Parameter(
                        torch.tensor(init_value, requires_grad=True))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif material_model_name == 'drucker_prager':
            # Set valid learnable parameters
            valid_learnable_parameters = ('E', 'v', 's0', 'a', 'b',
                                          'yield_cohesion_parameter',
                                          'yield_pressure_parameter',
                                          'flow_pressure_parameter')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Make hardening parameters accessible as learnable parameters
            material_model_parameters_grad['hardening'] = \
                torch.nn.ParameterDict(
                    material_model_parameters_grad['hardening'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Activate computation graphs for learnable parameters
            for param in learnable_parameters:
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set learnable parameter                
                if param in ('s0', 'a', 'b'):
                    # Pick initialization value
                    init_value = material_model_parameters \
                        ['hardening_parameters'][param]
                    # Set learnable parameter
                    material_model_parameters['hardening_parameters'][param] =\
                        torch.nn.Parameter(
                            torch.tensor(init_value, requires_grad=True))
                else:
                    # Get initialization value
                    init_value = float(material_model_parameters[param])
                    # Set learnable parameter
                    material_model_parameters_grad[param] = torch.nn.Parameter(
                        torch.tensor(init_value, requires_grad=True))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif material_model_name == 'elastic':
            # Set valid learnable parameters
            valid_learnable_parameters = ('E', 'v')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Activate computation graphs for learnable parameters
            for param in learnable_parameters:
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~              
                # Pick initialization value
                init_value = float(material_model_parameters[param])
                # Set learnable parameter
                material_model_parameters_grad[param] = torch.nn.Parameter(
                    torch.tensor(init_value, requires_grad=True))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return material_model_parameters_grad
    # -------------------------------------------------------------------------
    def forward(self, features_in, is_normalized=False):
        """Forward propagation.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        is_normalized : bool, default=False
            If True, get normalized output features, False otherwise.
            
        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Check input state features
        if not isinstance(features_in, torch.Tensor):
            raise RuntimeError('Input features were not provided as '
                               'torch.Tensor.')
        # Check model data normalization
        if is_normalized:
            self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize input features data
        if self.is_data_normalization:
            features_in = \
                self.data_scaler_transform(tensor=features_in,
                                           features_type='features_in',
                                           mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Forward propagation: Material constitutive model
        features_out = self._recurrent_constitutive_model(features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features data
        if self.is_data_normalization and not is_normalized:
            features_out = \
                self.data_scaler_transform(tensor=features_out,
                                           features_type='features_out',
                                           mode='denormalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
    # -------------------------------------------------------------------------
    def _recurrent_constitutive_model(self, strain_paths):
        """Compute material response.
        
        Parameters
        ----------
        strain_paths : torch.Tensor
            Tensor of strain paths stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        
        Returns
        -------
        response_paths : torch.Tensor
            Tensor of material response paths (stress and state variables)
            stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Check input data
        if len(strain_paths.shape) == 3:
            # Set batched input flag
            is_batched = True
            # Get number of paths
            n_path = strain_paths.shape[1]
        elif len(strain_paths.shape) == 2:
            # Set batched input flag
            is_batched = False
            # Set number of paths
            n_path = 1
        else:
            raise RuntimeError('Tensor of strain paths must be stored as'
                               'torch.Tensor (2d) of shape '
                               '(sequence_length, n_features_in) for '
                               'unbatched input or torch.Tensor (2d) of shape '
                               '(sequence_length, batch_size, n_features_out) '
                               'for batched input.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = strain_paths.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor of material response
        response_paths = torch.full((n_time, n_path, self._n_features_out),
                                    fill_value=float('nan'),
                                    device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over paths
        for i in range(n_path):
            # Get strain path
            if is_batched:
                strain_path = strain_paths[:, i, :]
            else:
                strain_path = strain_paths[:, :]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute material response
            stress_path, state_path = self._compute_stress_path(strain_path)            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate stress and state variables
            response_path = stress_path
            for state_var in state_path.keys():
                response_path = \
                    torch.cat((response_path, state_path[state_var]), dim=1)           
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check number of output features
            if response_path.shape[1] != self._n_features_out:
                raise RuntimeError(f'Material response number of dimensions '
                                   f'({response_path.shape[1]}) does not '
                                   f'match the model number of output '
                                   f'features ({self._n_features_out}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material response
            response_paths[:, i, :] = response_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batch dimension
        if not is_batched:
            response_paths = response_paths[:, 0, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return response_paths
    # -------------------------------------------------------------------------
    def _compute_stress_path(self, strain_path):
        """Compute material response for given strain path.

        Parameters
        ----------
        strain_path : torch.Tensor(2d)
            Strain path history stored as torch.Tensor(2d) of shape
            (sequence_length, n_strain_comps).

        Returns
        -------
        stress_path : torch.Tensor(2d)
            Stress path history stored as torch.Tensor(2d) of shape
            (sequence_length, n_stress_comps).
        state_path : dict
            Store each constitutive model state variable (key, str) path
            history as torch.Tensor(2d) of shape (sequence_length, n_features).
        """
        # Get strain and stress components order
        if self._strain_formulation == 'infinitesimal':
            strain_comps_order = self._comp_order_sym
            stress_comps_order = self._comp_order_sym
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = strain_path.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize stress path history
        stress_path = torch.zeros((n_time, len(stress_comps_order)),
                                  device=self._device)
        # Initialize state path history
        state_path = {}
        for state_var, n_features in self._state_features_out.items():
            state_path[state_var] = torch.zeros((n_time, n_features),
                                                device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model state variables
        state_variables = self._constitutive_model.state_init()
        # Initialize last converged material constitutive state variables      # Warning: Removed copy.deepcopy()
        state_variables_old = state_variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(1, n_time):
            # Get previous and current strain tensors
            strain_tensor_old = self.build_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx - 1, :],
                is_symmetric=self._strain_formulation == 'infinitesimal',
                device=self._device)
            strain_tensor = self.build_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx, :],
                is_symmetric=self._strain_formulation == 'infinitesimal',
                device=self._device)
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
                self._constitutive_model, inc_strain, state_variables_old,
                def_gradient_old=None)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check material state update convergence
            if state_variables['is_su_fail']:
                raise RuntimeError('Material state update convergence '
                                    'failure.')
            # Update last converged material constitutive state variables      # Warning: Removed copy.deepcopy()
            state_variables_old = state_variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get stress tensor
            if self._strain_formulation == 'infinitesimal':
                # Get Cauchy stress tensor
                stress = get_tensor_from_mf(state_variables['stress_mf'],
                                            self._n_dim, self._comp_order_sym,
                                            device=self._device)
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store stress tensor
            stress_path[time_idx, :] = \
                self.store_tensor_comps(stress_comps_order, stress,
                                        device=self._device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over state variables
            for state_var in self._state_features_out.keys():
                # Skip if state variable is not available
                if state_var not in state_variables.keys():
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store state variable
                if 'strain_mf' in state_var:
                    # Build strain tensor
                    if self._strain_formulation == 'infinitesimal':
                        strain = get_tensor_from_mf(
                            state_variables[state_var], self._n_dim,
                            self._comp_order_sym, device=self._device)
                    else:
                        raise RuntimeError('Not implemented.')
                    # Store strain tensor
                    state_path[state_var][time_idx, :] = \
                        self.store_tensor_comps(strain_comps_order, strain,
                                                device=self._device)
                else:
                    # Store generic state variable
                    state_path[state_var][time_idx, :] = \
                        state_variables[state_var]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_path, state_path
    # -------------------------------------------------------------------------
    @classmethod
    def build_tensor_from_comps(cls, n_dim, comps, comps_array,
                                is_symmetric=False, device=None):
        """Build strain/stress tensor from given components.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comps : tuple[str]
            Strain/Stress components order.
        comps_array : torch.Tensor(1d)
            Strain/Stress components array.
        is_symmetric : bool, default=False
            If True, then assembles off-diagonal strain components from
            symmetric component.
        device : torch.device, default=None
            Device on which torch.Tensor is allocated.
        
        Returns
        -------
        tensor : torch.Tensor(2d)
            Strain/Stress tensor.
        """
        # Initialize tensor
        tensor = torch.zeros((n_dim, n_dim), device=device)
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
    def store_tensor_comps(self, comps, tensor, device=None):
        """Store strain/stress tensor components in array.
        
        Parameters
        ----------
        comps : tuple[str]
            Strain/Stress components order.
        tensor : torch.Tensor(2d)
            Strain/Stress tensor.
        device : torch.device, default=None
            Device on which torch.Tensor is allocated.
        
        Returns
        -------
        comps_array : torch.Tensor(1d)
            Strain/Stress components array.
        """
        # Initialize tensor components array
        comps_array = torch.zeros(len(comps), device=device)
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
    def move_state_tensors_to_device(self, state_variables):
        """
        
        Parameters
        ----------
        state_variables : dict
            Material constitutive model state variables.
        
        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        """
        # Move state variables tensors to device
        for key, val in state_variables.items():
            if isinstance(val, torch.Tensor):
                state_variables[key] = val.to(self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables
    # -------------------------------------------------------------------------
    def save_model_init_file(self):
        """Save model initialization file.
        
        Initialization file is stored in the model directory under the name
        model_init_file.pkl.
        
        Initialization file contains a dictionary model_init_attributes that
        includes:
        
        'model_init_args' - Model initialization parameters
        
        'model_data_scalers' - Model fitted data scalers
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model initialization attributes
        model_init_attributes = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initialization parameters
        model_init_args = {}
        model_init_args['n_features_in'] = self._n_features_in
        model_init_args['n_features_out'] = self._n_features_out
        model_init_args['learnable_parameters'] = self._learnable_parameters
        model_init_args['strain_formulation'] = self._strain_formulation
        model_init_args['problem_type'] = self._problem_type
        model_init_args['material_model_name'] = self._material_model_name
        model_init_args['material_model_parameters'] = \
            self._material_model_parameters
        model_init_args['model_directory'] = self.model_directory
        model_init_args['model_name'] = self.model_name
        model_init_args['state_features_out'] = self._state_features_out
        model_init_args['is_data_normalization'] = self.is_data_normalization
        model_init_args['device_type'] = self._device_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble initialization parameters
        model_init_attributes['model_init_args'] = model_init_args
        # Assemble model data scalers
        model_init_attributes['model_data_scalers'] = self._data_scalers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model initialization file path
        model_init_file_path = os.path.join(self.model_directory,
                                            'model_init_file' + '.pkl')
        # Save model initialization file
        with open(model_init_file_path, 'wb') as init_file:
            pickle.dump(model_init_attributes, init_file)
    # -------------------------------------------------------------------------
    def save_model_state(self, epoch=None, is_best_state=False,
                         is_remove_posterior=True):
        """Save model state to file.
        
        Model state file is stored in model_directory under the name
        < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt or
        < model_name >-< epoch >-best.pt if epoch is known.
        
        Parameters
        ----------
        epoch : int, default=None
            Training epoch corresponding to current model state.
        is_best_state : bool, default=False
            If True, save model state file corresponding to the best
            performance instead of regular state file.
        is_remove_posterior : bool, default=True
            Remove model and optimizer state files corresponding to training
            epochs posterior to the saved state file. Effective only if saved
            training epoch is known.
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        model_state_file = self.model_name
        # Append epoch
        if isinstance(epoch, int):
            model_state_file += '-' + str(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state corresponding to the best performance
        if is_best_state:
            # Append best performance
            model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove any existent best model state file
            self._remove_best_state_files()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Save model state
        torch.save(self.state_dict(), model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model epoch state files posterior to saved epoch
        if isinstance(epoch, int) and is_remove_posterior:
            self._remove_posterior_state_files(epoch)
    # -------------------------------------------------------------------------
    def load_model_state(self, load_model_state=None,
                         is_remove_posterior=True):
        """Load model state from file.
        
        Model state file is stored in model_directory under the name
        < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt or
        < model_name >-< epoch >-best.pt if epoch if known.
        
        Parameters
        ----------
        load_model_state : {'best', 'last', int, None}, default=None
            Load available model state from the model directory.
            Options:
            
            'best'      : Model state corresponding to best performance
            
            'last'      : Model state corresponding to highest training epoch
            
            int         : Model state corresponding to given training epoch
            
            None        : Model default state file
        
        is_remove_posterior : bool, default=True
            Remove model state files corresponding to training epochs posterior
            to the loaded state file. Effective only if loaded training epoch
            is known.
            
        Returns
        -------
        epoch : int
            Loaded model state training epoch. Defaults to None if training
            epoch is unknown.
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if load_model_state == 'best':
            # Get state files in model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model best state files epochs
            best_state_epochs = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in model directory
            for filename in directory_list:
                # Check if file is model epoch best state file
                is_best_state_file, best_state_epoch = \
                    self._check_best_state_file(filename)
                # Store model best state file training epoch
                if is_best_state_file:
                    best_state_epochs.append(best_state_epoch)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model best state file
            if not best_state_epochs:
                raise RuntimeError('Model best state file has not been found '
                                   'in directory:\n\n' + self.model_directory)
            elif len(best_state_epochs) > 1:
                raise RuntimeError('Two or more model best state files have '
                                   'been found in directory:'
                                   '\n\n' + self.model_directory)
            else:
                # Set best state epoch
                epoch = best_state_epochs[0]
                # Set model best state file
                model_state_file = self.model_name
                if isinstance(epoch, int):
                    model_state_file += '-' + str(epoch)      
                model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model epoch state files posterior to loaded epoch
            if isinstance(epoch, int) and is_remove_posterior:
                self._remove_posterior_state_files(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif load_model_state == 'last':
            # Get state files in model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model state files training epochs
            epochs = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in model directory
            for filename in directory_list:
                # Check if file is model epoch state file
                is_state_file, epoch = self._check_state_file(filename)
                # Store model state file training epoch
                if is_state_file:
                    epochs.append(epoch)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set highest epoch model state file
            if epochs:
                # Set highest epoch
                epoch = max(epochs)
                # Set highest epoch model state file
                model_state_file = self.model_name + '-' + str(epoch)
            else:
                raise RuntimeError('Model state files corresponding to epochs '
                                   'have not been found in directory:\n\n'
                                   + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        elif isinstance(load_model_state, int):
            # Get epoch
            epoch = load_model_state
            # Set model state filename with epoch
            model_state_file = self.model_name + '-' + str(int(epoch))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model epoch state files posterior to loaded epoch
            if is_remove_posterior:
                self._remove_posterior_state_files(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Set model state filename
            model_state_file = self.model_name
            # Set epoch as unknown
            epoch = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Check model state file
        if not os.path.isfile(model_path):
            raise RuntimeError('Model state file has not been found:\n\n'
                               + model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model state
        self.load_state_dict(torch.load(model_path,
                                        map_location=torch.device('cpu')))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return epoch
    # -------------------------------------------------------------------------
    def _check_state_file(self, filename):
        """Check if file is model training epoch state file.
        
        Model training epoch state file is stored in model_directory under the
        name < model_name >-< epoch >.pt.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_state_file : bool
            True if model training epoch state file, False otherwise.
        epoch : {None, int}
            Training epoch corresponding to model state file if
            is_state_file=True, None otherwise.
        """
        # Check if file is model epoch state file
        is_state_file = bool(re.search(r'^' + self.model_name + r'-[0-9]+'
                                       + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epoch = None
        if is_state_file:
            # Get model state epoch
            epoch = int(os.path.splitext(filename)[0].split('-')[-1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_state_file, epoch
    # -------------------------------------------------------------------------
    def _check_best_state_file(self, filename):
        """Check if file is model best state file.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt. or
        < model_name >-< epoch >-best.pt if the training epoch is known.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_best_state_file : bool
            True if model training epoch state file, False otherwise.
        epoch : {None, int}
            Training epoch corresponding to model state file if
            is_best_state_file=True and training epoch is known, None
            otherwise.
        """
        # Check if file is model epoch best state file
        is_best_state_file = bool(re.search(r'^' + self.model_name
                                            + r'-?[0-9]*' + r'-best' + r'\.pt',
                                            filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epoch = None
        if is_best_state_file:
            # Get model state epoch
            epoch = int(os.path.splitext(filename)[0].split('-')[-2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_best_state_file, epoch
    # -------------------------------------------------------------------------
    def _remove_posterior_state_files(self, epoch):
        """Delete model training epoch state files posterior to given epoch.
        
        Parameters
        ----------
        epoch : int
            Training epoch.
        """
        # Get files in model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model epoch state file
            is_state_file, file_epoch = self._check_state_file(filename)
            # Delete model epoch state file posterior to given epoch
            if is_state_file and file_epoch > epoch:
                os.remove(os.path.join(self.model_directory, filename))
    # -------------------------------------------------------------------------
    def _remove_best_state_files(self):
        """Delete existent model best state files."""
        # Get files in model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model best state file
            is_best_state_file, _ = self._check_best_state_file(filename)
            # Delete state file
            if is_best_state_file:
                os.remove(os.path.join(self.model_directory, filename))
    # -------------------------------------------------------------------------
    def _init_data_scalers(self):
        """Initialize model data scalers."""
        self._data_scalers = {}
        self._data_scalers['features_in'] = None
        self._data_scalers['features_out'] = None
    # -------------------------------------------------------------------------
    def fit_data_scalers(self, dataset, is_verbose=False):
        """Fit model data scalers.
        
        Data scalers are set a standard scalers where features are normalized
        by removing the mean and scaling to unit variance.
        
        Calling this method turns on model data normalization.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
        is_verbose : bool, default=False
            If True, enable verbose output.
        """
        if is_verbose:
            print('\nFitting model data scalers'
                  '\n--------------------------\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data normalization
        self.is_data_normalization = True
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate data scalers
        scaler_features_in = TorchStandardScaler(
            n_features=self._n_features_in, device_type=self._device_type)
        scaler_features_out = TorchStandardScaler(
            n_features=self._n_features_out, device_type=self._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get scaling parameters and fit data scalers: input features
        mean, std = standard_partial_fit(dataset, features_type='features_in',
                                         n_features=self._n_features_in)
        scaler_features_in.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: output features
        mean, std = standard_partial_fit(dataset, features_type='features_out',
                                         n_features=self._n_features_out)
        scaler_features_out.set_mean_and_std(mean, std)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Setting fitted standard scalers...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['features_in'] = scaler_features_in
        self._data_scalers['features_out'] = scaler_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model initialization file with fitted data scalers
        self.save_model_init_file()
    # -------------------------------------------------------------------------
    def get_fitted_data_scaler(self, features_type):
        """Get fitted model data scalers.
        
        Parameters
        ----------
        features_type : str
            Features for which data scaler is required:
            
            'features_in'  : Input features

            'features_out' : Output features

        Returns
        -------
        data_scaler : sklearn.preprocessing.StandardScaler
            Fitted data scaler.
        """
        # Get fitted data scaler
        if features_type not in self._data_scalers.keys():
            raise RuntimeError(f'Unknown data scaler for {features_type}.')
        elif self._data_scalers[features_type] is None:
            raise RuntimeError(f'Data scaler for {features_type} has not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')
        else:
            data_scaler = self._data_scalers[features_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_scaler
    # -------------------------------------------------------------------------
    def data_scaler_transform(self, tensor, features_type, mode='normalize'):
        """Perform data scaling operation on features PyTorch tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor.
        features_type : str
            Features for which data scaler is required:
            
            'features_in'  : Input features

            'features_out' : Output features

        mode : {'normalize', 'denormalize'}, default=normalize
            Data scaling transformation type.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Transformed features PyTorch tensor.
        """
        # Check model data normalization
        self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input features tensor
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Input tensor is not torch.Tensor.')
        # Get input features tensor data type
        input_dtype = tensor.dtype
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fitted data scaler for input features
        data_scaler = self.get_fitted_data_scaler(features_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform data scaling normalization/denormalization
        if mode == 'normalize':
            transformed_tensor = data_scaler.transform(tensor)
        elif mode == 'denormalize':
            transformed_tensor = data_scaler.inverse_transform(tensor)
        else:
            raise RuntimeError('Invalid data scaling transformation type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce same data type of input features tensor 
        transformed_tensor = transformed_tensor.to(input_dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed features tensor
        if not isinstance(transformed_tensor, torch.Tensor):
            raise RuntimeError('Transformed tensor is not torch.Tensor.') 
        elif not torch.equal(torch.tensor(transformed_tensor.size()),
                             torch.tensor(tensor.size())):
            raise RuntimeError('Input and transformed tensors do not have '
                               'the same shape.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # -------------------------------------------------------------------------
    def load_model_data_scalers_from_file(self):
        """Load data scalers from model initialization file."""
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(self.model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The model initialization file '
                               'has not been found:\n\n'
                                + model_init_file_path)
        else:
            with open(model_init_file_path, 'rb') as model_init_file:
                model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        self._data_scalers = model_data_scalers
    # -------------------------------------------------------------------------
    def check_normalized_return(self):
        """Check if model data normalization is available."""
        if not self.is_data_normalization or self._data_scalers is None:
            raise RuntimeError('Data scalers for model features have not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')
        if all([x is None for x in self._data_scalers.values()]):
            raise RuntimeError('Data scalers for model features have not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')
# =============================================================================
def standard_partial_fit(dataset, features_type, n_features, is_verbose=False):
    """Perform batch fitting of standardization data scalers.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_type : str
        Features for which data scaler is required:
        
        'features_in'  : Input features

        'features_out' : Output features
    
    n_features : int
        Number of features to standardize.
    is_verbose : bool, default=False
        If True, enable verbose output.
    
    Returns
    -------
    mean : torch.Tensor
        Features standardization mean tensor stored as a torch.Tensor with
        shape (n_features,).
    std : torch.Tensor
        Features standardization standard deviation tensor stored as a
        torch.Tensor with shape (n_features,).
        
    Notes
    -----
    A biased estimator is used to compute the standard deviation according with
    scikit-learn 1.3.2 documentation (sklearn.preprocessing.StandardScaler).
    """
    # Instantiate data scaler
    data_scaler = sklearn.preprocessing.StandardScaler()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    data_loader = get_time_series_data_loader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample in tqdm.tqdm(data_loader,
                            desc='> Processing data samples: ',
                            disable=not is_verbose):        
        # Check sample
        if not isinstance(sample, dict):
            raise RuntimeError('Time series sample must be dictionary where '
                               'each feature (key, str) data is a '
                               'torch.Tensor(2d) of shape '
                               '(sequence_length, n_featues).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check sample features tensor
        if features_type not in sample.keys():
            raise RuntimeError(f'Unavailable feature from sample.')
        else:
            features_tensor = sample[features_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process sample to fit data scaler
        if isinstance(features_tensor, torch.Tensor):
            # Check number of features
            if features_tensor.shape[-1] != n_features:
                raise RuntimeError(f'Mismatch between input graph '
                                   f'({features_tensor.shape[-1]}) and '
                                   f'model ({n_features}) number of '
                                   f'features for features type: '
                                   f'{features_type}')
            # Process sample
            data_scaler.partial_fit(features_tensor[:, 0, :].clone())
        else:
            raise RuntimeError('Sample features tensor is not torch.Tensor.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get fitted mean and standard deviation tensors
    mean = torch.tensor(data_scaler.mean_)
    std = torch.sqrt(torch.tensor(data_scaler.var_))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check features standardization mean tensor
    if not isinstance(mean, torch.Tensor):
        raise RuntimeError('Features standardization mean tensor is not a '
                           'torch.Tensor.')
    elif len(mean) != features_tensor.shape[-1]:
        raise RuntimeError('Features standardization mean tensor is not a '
                           'torch.Tensor(1d) with shape (n_features,).')
    # Check features standardization standard deviation tensor
    if not isinstance(std, torch.Tensor):
        raise RuntimeError('Features standardization standard deviation '
                           'tensor is not a torch.Tensor.')
    elif len(std) != features_tensor.shape[-1]:
        raise RuntimeError('Features standardization standard deviation '
                           'tensor is not a torch.Tensor(1d) with shape '
                           '(n_features,).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mean, std