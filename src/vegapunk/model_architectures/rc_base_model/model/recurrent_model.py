"""Recurrent constitutive model (wrapper of known constitutive model).

Classes
-------
RecurrentConstitutiveModel(torch.nn.Module)
    Recurrent constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
import re
import pickle
import itertools
import math
# Third-party
import torch
# Local
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.von_mises_mixed import \
    VonMisesMixed
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from simulators.fetorch.material.models.standard.lou import LouZhangYoon
from simulators.fetorch.material.models.vmap.von_mises import VonMisesVMAP
from simulators.fetorch.material.models.vmap.von_mises_mixed import \
    VonMisesMixedVMAP
from simulators.fetorch.material.models.vmap.drucker_prager import \
    DruckerPragerVMAP
from simulators.fetorch.material.models.vmap.lou import LouZhangYoonVMAP
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_from_mf
from simulators.fetorch.material.material_su import material_state_update
from model_architectures.procedures.model_data_scaling import init_data_scalers
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
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
    model_directory : {str, None}
        Directory where model is stored. If None, then all methods that depend
        on an existent model directory become unavailable.
    model_name : {str, None}
        Name of model. If None, then all methods that depend on a valid model
        name become unavailable.
    _n_features_in : int
        Number of input features.
    _n_features_out : int
        Number of output features.
    _learnable_parameters : dict
        Learnable material constitutive model parameters. For each parameter
        (key, str), stores a dictionary with the parameter initial value
        (key, 'initial_value', item, float) and the parameter bounds
        (key, 'bounds', item, tuple(lower_bound, upper_bound)).
    _strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    _problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    _material_model_name : str
        Material constitutive model name.
    _material_model_parameters : torch.nn.ParameterDict
        Material constitutive model parameters with learnable parameters.
    _constitutive_model : ConstitutiveModel
        FETorch material constitutive model.
    _state_features_out : dict, default={}
        Material constitutive model state variables (key, str) and
        corresponding dimensionality (item, int) for which the path history
        is additionally predicted in the output features besides the stress
        path history. State variables are sorted as output features
        according with the insertion order.
    _model_parameters : torch.nn.ParameterDict
        Model parameters.
    _model_parameters_bounds : dict
        Model learnable parameters bounds. For each parameter (key, str),
        the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    _model_parameters_norm_bounds : dict
        Model learnable parameters normalized bounds. For each parameter
        (key, str), the corresponding bounds are stored as a
        tuple(lower_bound, upper_bound) (item, tuple).
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_normalized_parameters : bool
        If True, then learnable parameters are normalized for optimization,
        False otherwise. The initial values and bounds of each parameter
        are normalized accordingly.
    is_explicit_parameters : bool
        True if model learnable parameters are explicit, False otherwise.
    is_model_in_normalized : bool
        If True, then model expects normalized input features (normalized
        input data has been seen during model training).
    is_model_out_normalized : bool
        If True, then model expects normalized output features (normalized
        output data has been seen during model training).
    _is_save_model_init_file: bool, default=True
        If True, saves model initialization file when model is initialized
        (overwritting existent initialization file), False otherwise.
    _is_auto_sync_parameters : bool, default=True
        If True, then automatically synchronize material model parameters
        with learnable parameters in forward propagation.
    _is_check_su_fail : bool, default=True
        If True, then check if material constitutive model state update failed.
    _data_scalers : dict
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str).

    Methods
    -------
    init_model_from_file(model_directory=None, model_init_file_path=None)
        Initialize model from initialization file.
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    _set_model_parameters(self, learnable_parameters)
        Set recurrent constitutive model learnable parameters.
    sync_material_model_parameters(self)
        Synchronize material model parameters with learnable parameters.
    get_material_model_name(self)
        Get material constitutive model name.
    transform_parameter(self, name, value, mode='normalize')
        Transform model parameter by means of min-max scaling.
    freeze_parameter(self, name)
        Freeze model parameter (disable gradient computation).
    get_model_parameters(self)
        Get model parameters.
    get_model_parameters_bounds(self)
        Get model parameters bounds.
    get_model_parameters_norm_bounds(self)
        Get model parameters normalized bounds.
    enforce_parameters_bounds(self)
        Enforce bounds in model parameters.
    enforce_parameters_constraints(self)
        Enforce material model-dependent parameters constraints.
    get_detached_model_parameters(self, is_normalized_out=False)
        Get model parameters detached of gradients.
    get_material_model_parameters(self)
        Get current material constitutive model parameters.
    forward(self, features_in)
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
    _vrecurrent_constitutive_model(self, strain_paths)
        Compute material response.
    _vcompute_stress_path(self, strain_path)
        Compute material response for given strain path.
    vbuild_tensor_from_comps(cls, n_dim, comps, comps_array, device=None)
        Build strain/stress tensor from given components.
    vstore_tensor_comps(cls, comps, tensor, device=None)
        Store strain/stress tensor components in array.  
    move_state_tensors_to_device(self, state_variables)
        Move state variables tensors to model device.
    save_model_init_file(self)
        Save model class initialization attributes.
    """
    def __init__(self, n_features_in, n_features_out, learnable_parameters,
                 strain_formulation, problem_type, material_model_name,
                 material_model_parameters, model_directory,
                 model_name='wrapper_recurrent_model',
                 is_auto_sync_parameters=True, is_check_su_fail=True,
                 state_features_out={}, is_model_in_normalized=False,
                 is_model_out_normalized=False, is_normalized_parameters=False,
                 is_save_model_init_file=True, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features_in : int
            Number of input features.
        n_features_out : int
            Number of output features.
        learnable_parameters : dict
            Learnable material constitutive model parameters. For each
            parameter (key, str), stores a dictionary with the parameter
            initial value (key, 'initial_value', item, float) and the parameter
            bounds (key, 'bounds', item, tuple(lower_bound, upper_bound)).
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        material_model_name : str
            Material constitutive model name.
        material_model_parameters : dict
            Material constitutive model parameters.
        model_directory : {str, None}
            Directory where model is stored. If None, then all methods that
            depend on an existent model directory become unavailable.
        model_name : {str, None}, default='wrapper_recurrent_model'
            Name of model. If None, then all methods that depend on a valid
            model name become unavailable.
        is_auto_sync_parameters : bool, default=True
            If True, then automatically synchronize material model parameters
            with learnable parameters in forward propagation.
        is_check_su_fail : bool, default=True
            If True, then check if material constitutive model state update
            failed.
        state_features_out : dict, default={}
            Material constitutive model state variables (key, str) and
            corresponding dimensionality (item, int) for which the path history
            is additionally predicted in the output features besides the stress
            path history. State variables are sorted as output features
            according with the insertion order.
        is_normalized_parameters : bool, default=False
            If True, then learnable parameters are normalized for optimization,
            False otherwise. The initial values and bounds of each parameter
            are normalized accordingly.
        is_model_in_normalized : bool, default=False
            If True, then model expects normalized input features (normalized
            input data has been seen during model training).
        is_model_out_normalized : bool, default=False
            If True, then model expects normalized output features (normalized
            output data has been seen during model training).
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
        # Set automatic synchronization of material model parameters
        self._is_auto_sync_parameters = is_auto_sync_parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_features_in = n_features_in
        self._n_features_out = n_features_out
        # Set additional state variable output features
        self._state_features_out = state_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material constitutive model name
        self._material_model_name = material_model_name
        # Set material constitutive model parameters
        self._material_model_parameters = material_model_parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set learnable parameters
        self._learnable_parameters = learnable_parameters
        # Set model parameters normalization flag
        self.is_normalized_parameters = is_normalized_parameters
        # Set model parameters
        self._set_model_parameters(learnable_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        if model_directory is None:
            self.model_directory = model_directory
        elif os.path.isdir(model_directory):
            self.model_directory = str(model_directory)
        else:
            raise RuntimeError('The model directory has not been found.')
        # Set model name
        if model_name is None:
            self.model_name = model_name
        elif isinstance(model_name, str):
            self.model_name = model_name
        else:
            raise RuntimeError('The model name must be a string.')
        # Set model input and output features normalization
        self.is_model_in_normalized = is_model_in_normalized
        self.is_model_out_normalized = is_model_out_normalized
        # Set save initialization file flag
        self._is_save_model_init_file = is_save_model_init_file
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
                         self._material_model_parameters,
                         device_type=self._device_type)
        elif material_model_name == 'von_mises_mixed':
            self._constitutive_model = \
                VonMisesMixed(self._strain_formulation, self._problem_type,
                              self._material_model_parameters,
                              device_type=self._device_type)
        elif material_model_name == 'drucker_prager':
            self._constitutive_model = \
                DruckerPrager(self._strain_formulation, self._problem_type,
                              self._material_model_parameters,
                              device_type=self._device_type)
        elif material_model_name == 'lou_zhang_yoon':
            self._constitutive_model = \
                LouZhangYoon(self._strain_formulation, self._problem_type,
                             self._material_model_parameters,
                             device_type=self._device_type)
        elif material_model_name == 'von_mises_vmap':
            self._constitutive_model = \
                VonMisesVMAP(self._strain_formulation, self._problem_type,
                             self._material_model_parameters,
                             device_type=self._device_type)
        elif material_model_name == 'von_mises_mixed_vmap':
            self._constitutive_model = \
                VonMisesMixedVMAP(self._strain_formulation, self._problem_type,
                                  self._material_model_parameters,
                                  device_type=self._device_type)
        elif material_model_name == 'drucker_prager_vmap':
            self._constitutive_model = \
                DruckerPragerVMAP(self._strain_formulation, self._problem_type,
                                  self._material_model_parameters,
                                  device_type=self._device_type)
        elif material_model_name == 'lou_zhang_yoon_vmap':
            self._constitutive_model = \
                LouZhangYoonVMAP(self._strain_formulation, self._problem_type,
                                 self._material_model_parameters,
                                 device_type=self._device_type)
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{material_model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update failure checking flag
        self._is_check_su_fail = is_check_su_fail
        # Force state update failure checking flag
        if bool(re.search(r'_vmap$', self._material_model_name)):
            self._is_check_su_fail = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set learnable parameters nature
        self.is_explicit_parameters = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        init_data_scalers(self)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Save model initialization file
        if self._is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    @staticmethod
    def init_model_from_file(model_directory=None, model_init_file_path=None):
        """Initialize model from initialization file.
        
        If model directory is provided, then (1) model initialization file is
        assumed to be stored in the model directory under the name
        model_init_file.pkl and (2) model initialization attributes are read
        from the stored model_init_file.pkl file.
        
        In model initialization file path is provided, then (1) model
        initialization attributes are read from the provided
        model_init_file.pkl file and (2) model directory is set as the
        corresponding directory.
        
        Parameters
        ----------
        model_directory : str, default=None
            Directory where model is stored.
        model_init_file_path : str, default=None
            Model initialization file path. Ignored if model_directory is
            provided.
        """
        # Get model directory or model initialization file path
        if model_directory is not None:
            # Check model directory
            if not os.path.isdir(model_directory):
                raise RuntimeError('The model directory has not been found:'
                                   '\n\n' + model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model initialization file path from model directory
            model_init_file_path = os.path.join(model_directory,
                                                'model_init_file' + '.pkl')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check model initialization file
            if not os.path.isfile(model_init_file_path):
                raise RuntimeError('The model initialization file has not '
                                   'been found:\n\n' + model_init_file_path)
        elif model_init_file_path is not None:
            # Check model initialization file
            if not os.path.isfile(model_init_file_path):
                raise RuntimeError('The model initialization file has not '
                                   'been found:\n\n' + model_init_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model directory from model initialization file path
            model_directory = os.path.dirname(model_init_file_path)
        else:
            raise RuntimeError('Either the model directory or the model '
                               'initialization file path must be provided in '
                               'order to initialize the model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
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
        # Set model data scalers device
        if model._data_scalers is not None:
            # Loop over model data scalers
            for _, data_scaler in model._data_scalers.items():
                if data_scaler is not None:
                    data_scaler.set_device(model._device_type)
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Consistent update of material constitutive model device
            if hasattr(self, '_constitutive_model'):
                self._constitutive_model.set_device(self._device_type)
            # Consistent update of data scalers device
            if (hasattr(self, '_data_scalers')
                    and self._data_scalers is not None):
                # Loop over data scalers
                for _, data_scaler in self._data_scalers.items():
                    # Update data scaler device
                    if data_scaler is not None:
                        data_scaler.set_device(self._device_type)    
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
    def _set_model_parameters(self, learnable_parameters):
        """Set recurrent constitutive model learnable parameters.
        
        Parameters
        ----------
        learnable_parameters : dict
            Learnable material constitutive model parameters. For each
            parameter (key, str), stores a dictionary with the parameter
            initial value (key, 'initial_value', item, float) and the parameter
            bounds (key, 'bounds', item, tuple(lower_bound, upper_bound)).
        """
        # Initialize model parameters
        self._model_parameters = torch.nn.ParameterDict({})
        # Initialize model parameters bounds
        self._model_parameters_bounds = {}
        # Initialize model parameters normalized bounds
        self._model_parameters_norm_bounds = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over learnable parameters
        for param, param_dict in learnable_parameters.items():
            # Get initial value
            init_val = param_dict['initial_value']
            # Get lower and upper bounds
            if param_dict['bounds'][1] < param_dict['bounds'][0]:
                raise RuntimeError(f'Invalid bounds were provided for '
                                   f'parameter "{param}". Upper bound '
                                   f'{param_dict["bounds"][1]} is lower than '
                                   f'lower bound ({param_dict["bounds"][0]}).')
            else:
                self._model_parameters_bounds[param] = param_dict['bounds']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set normalization bounds
            if self.is_normalized_parameters:
                # Set lower and upper bounds
                self._model_parameters_norm_bounds[param] = (-1.0, 1.0)
                # Normalize initial value
                init_val = self.transform_parameter(param, init_val,
                                                    mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model parameter
            self._model_parameters[param] = torch.nn.Parameter(
                torch.tensor(init_val, requires_grad=True))
    # -------------------------------------------------------------------------
    def sync_material_model_parameters(self):
        """Synchronize material model parameters with learnable parameters."""
        # Get material constitutive model
        material_model_name = self._material_model_name
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over learnable parameters
        for param, value in self._model_parameters.items():
            # Get updated material constitutive model parameter
            if self.is_normalized_parameters:
                sync_val = self.transform_parameter(param, value,
                                                    mode='denormalize')
            else:
                sync_val = 1.0*value
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Synchronize material constitutive model parameter
            if self._material_model_name in ('von_mises', 'von_mises_vmap'):
                # Set valid learnable parameters
                valid_learnable_parameters = ('E', 'v', 's0', 'a', 'b')
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                material_parameters = self.get_material_model_parameters()
                # Get hardening parameters
                hardening_parameters = \
                    material_parameters['hardening_parameters']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Synchronize parameter
                if param in ('s0', 'a', 'b'):
                    hardening_parameters[param] = sync_val
                else:
                    material_parameters[param] = sync_val
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._material_model_name in ('von_mises_mixed',
                                               'von_mises_mixed_vmap'):
                # Set valid learnable parameters
                valid_learnable_parameters = ('E', 'v', 's0', 'a', 'b',
                                              'kin_s0', 'kin_a', 'kin_b')
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                material_parameters = self.get_material_model_parameters()
                # Get isotropic hardening parameters
                hardening_parameters = \
                    material_parameters['hardening_parameters']
                # Get kinematic hardening parameters
                kinematic_hardening_parameters = \
                    material_parameters['kinematic_hardening_parameters']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Synchronize parameter
                if param in ('s0', 'a', 'b'):
                    hardening_parameters[param] = sync_val
                elif param in ('kin_s0', 'kin_a', 'kin_b'):
                    kinematic_hardening_parameters[param[4:]] = sync_val
                else:
                    material_parameters[param] = sync_val
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._material_model_name in \
                    ('drucker_prager', 'drucker_prager_vmap'):
                # Set valid learnable parameters
                valid_learnable_parameters = ('E', 'v', 's0', 'a', 'b',
                                              'yield_cohesion_parameter',
                                              'yield_pressure_parameter',
                                              'flow_pressure_parameter',
                                              'friction_angle')
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                material_parameters = self.get_material_model_parameters()
                # Get hardening parameters
                hardening_parameters = \
                    material_parameters['hardening_parameters']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Synchronize parameter
                if param in ('s0', 'a', 'b'):
                    hardening_parameters[param] = sync_val
                elif param == 'friction_angle':
                    material_parameters[param] = sync_val
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set frictional angle
                    friction_angle = sync_val
                    # Set dilatancy angle
                    dilatancy_angle = friction_angle
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute angle-related material parameters (matching with
                    # Mohr-Coulomb under uniaxial tension and compression)
                    # Set yield surface cohesion parameter
                    yield_cohesion_parameter = \
                        (2.0/math.sqrt(3.0))*torch.cos(friction_angle)
                    # Set yield pressure parameter
                    yield_pressure_parameter = \
                        (3.0/math.sqrt(3.0))*torch.sin(friction_angle)
                    # Set plastic flow pressure parameter
                    flow_pressure_parameter = \
                        (3.0/math.sqrt(3.0))*torch.sin(dilatancy_angle)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Synchronize angle-related material parameters
                    material_parameters['yield_cohesion_parameter'] = \
                        yield_cohesion_parameter
                    material_parameters['yield_pressure_parameter'] = \
                        yield_pressure_parameter
                    material_parameters['flow_pressure_parameter'] = \
                        flow_pressure_parameter
                else:
                    material_parameters[param] = sync_val
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._material_model_name in \
                    ('lou_zhang_yoon', 'lou_zhang_yoon_vmap'):
                # Set valid learnable parameters
                valid_learnable_parameters = (
                    'E', 'v', 's0', 'a', 'b',
                    'yield_a_s0', 'yield_a_a', 'yield_a_b',
                    'yield_b_s0', 'yield_b_a', 'yield_b_b',
                    'yield_c_s0', 'yield_c_a', 'yield_c_b',
                    'yield_d_s0', 'yield_d_a', 'yield_d_b')
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                material_parameters = self.get_material_model_parameters()
                # Get hardening parameters
                hardening_parameters = \
                    material_parameters['hardening_parameters']
                # Get yield surface parameters
                a_hardening_parameters = \
                    material_parameters['a_hardening_parameters']
                b_hardening_parameters = \
                    material_parameters['b_hardening_parameters']
                c_hardening_parameters = \
                    material_parameters['c_hardening_parameters']
                d_hardening_parameters = \
                    material_parameters['d_hardening_parameters']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Synchronize parameter
                if param in ('s0', 'a', 'b'):
                    hardening_parameters[param] = sync_val
                elif param in ('yield_a_s0', 'yield_a_a', 'yield_a_b'):
                    a_hardening_parameters[param[8:]] = sync_val
                elif param in ('yield_b_s0', 'yield_b_a', 'yield_b_b'):
                    b_hardening_parameters[param[8:]] = sync_val
                elif param in ('yield_c_s0', 'yield_c_a', 'yield_c_b'):
                    c_hardening_parameters[param[8:]] = sync_val
                elif param in ('yield_d_s0', 'yield_d_a', 'yield_d_b'):
                    d_hardening_parameters[param[8:]] = sync_val
                else:
                    material_parameters[param] = sync_val
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._material_model_name == 'elastic':
                # Set valid learnable parameters
                valid_learnable_parameters = ('E', 'v')
                # Check learnable parameter
                if param not in valid_learnable_parameters:
                    raise RuntimeError(f'Parameter "{param}" is not a valid '
                                       f'learnable parameter for model '
                                       f'"{material_model_name}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                material_parameters = self.get_material_model_parameters()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Synchronize parameter
                material_parameters[param] = sync_val
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown material constitutive model.')
    # -------------------------------------------------------------------------
    def get_material_model_name(self):
        """Get material constitutive model name.
        
        Returns
        -------
        material_model_name : str
            Material constitutive model name.
        """
        return self._material_model_name
    # -------------------------------------------------------------------------
    def transform_parameter(self, name, value, mode='normalize'):
        """Transform model parameter by means of min-max scaling.
        
        Parameters
        ----------
        name : str
            Parameter name.
        value : float
            Parameter value.
        mode : {'normalize', 'denormalize'}, default='normalize'
            Transformation type.

        Returns
        -------
        transformed_value : float
            Transformed parameter value.
        """
        # Set parameter original and transformed bounds according with the
        # transformation type
        if mode == 'normalize':
            omin, omax = self._model_parameters_bounds[name]
            tmin, tmax = self._model_parameters_norm_bounds[name]
        elif mode == 'denormalize':
            omin, omax = self._model_parameters_norm_bounds[name]
            tmin, tmax = self._model_parameters_bounds[name]
        else:
            raise RuntimeError('Invalid transformation type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transform parameter
        transformed_value = tmin + ((tmax - tmin)/(omax - omin))*(value - omin)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_value
    # -------------------------------------------------------------------------
    def freeze_parameter(self, name):
        """Freeze model parameter (disable gradient computation).
        
        Parameters
        ----------
        name : str
            Parameter name.
        """
        # Check model parameter
        if name in self._model_parameters.keys():
            # Freeze model parameter
            self._model_parameters[name].requires_grad = False
        else:
            # Unknown parameter
            raise RuntimeError(f'Unknown model parameter \'{name}\'.')
    # -------------------------------------------------------------------------
    def get_model_parameters(self):
        """Get model parameters.
        
        Returns
        -------
        model_parameters : torch.nn.ParameterDict
            Model parameters.
        """
        return self._model_parameters
    # -------------------------------------------------------------------------
    def get_model_parameters_bounds(self):
        """Get model parameters bounds.
        
        Returns
        -------
        model_parameters_bounds : dict
            Model learnable parameters bounds. For each parameter (key, str),
            the corresponding bounds are stored as a
            tuple(lower_bound, upper_bound) (item, tuple).
        """
        return copy.deepcopy(self._model_parameters_bounds)
    # -------------------------------------------------------------------------
    def get_model_parameters_norm_bounds(self):
        """Get model parameters normalized bounds.
        
        Returns
        -------
        model_parameters_norm_bounds : dict
            Model learnable parameters normalized bounds. For each parameter
            (key, str), the corresponding bounds are stored as a
            tuple(lower_bound, upper_bound) (item, tuple).
        """
        return copy.deepcopy(self._model_parameters_norm_bounds)
    # -------------------------------------------------------------------------
    def enforce_parameters_bounds(self):
        """Enforce bounds in model parameters.
        
        Bounds are enforced by means of in-place parameters updates.

        """
        # Get model parameters
        model_parameters = self.get_model_parameters()
        # Enforce bounds on model parameters
        for param, value in model_parameters.items():
            # Get parameter bounds
            if self.is_normalized_parameters:
                lower_bound, upper_bound = \
                    self.get_model_parameters_norm_bounds()[param]
            else:
                lower_bound, upper_bound = \
                    self.get_model_parameters_bounds()[param]
            # Enforce bounds
            value.data.clamp_(lower_bound, upper_bound)
    # -------------------------------------------------------------------------
    def enforce_parameters_constraints(self):
        """Enforce material model-dependent parameters constraints.
        
        Constraints are enforced by means of in-place parameters updates.

        """
        # Get material model name
        material_model_name = self.get_material_model_name()
        # Get model parameters
        model_parameters = self.get_model_parameters()
        # Get model material parameters
        material_model_parameters = self.get_material_model_parameters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce model-dependent parameters constraints
        if 'lou_zhang_yoon' in material_model_name:
            # Get Lou-Zhang-Yoon model convexity-related yielding parameters
            yield_c_s0 = \
                material_model_parameters['c_hardening_parameters']['s0']
            yield_d_s0 = \
                material_model_parameters['d_hardening_parameters']['s0']
            # Initialize learning parameters flags
            is_learnable_yield_c = False
            is_learnable_yield_d = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get Lou-Zhang-Yoon model convexity-related yielding parameters
            if 'yield_c_s0' in model_parameters.keys():
                # Set learning parameter flag
                is_learnable_yield_c = True
                # Get learning parameter
                yield_c_s0 = model_parameters['yield_c_s0'].detach().clone()
            if 'yield_d_s0' in model_parameters.keys():
                # Set learning parameter flag
                is_learnable_yield_d = True
                # Get learning parameter
                yield_d_s0 = model_parameters['yield_d_s0'].detach().clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Enforce constraints if learnable parameters
            if is_learnable_yield_c or is_learnable_yield_d:
                # Denormalize yielding parameters
                if self.is_normalized_parameters:
                    if is_learnable_yield_c:
                        yield_c_s0 = self.transform_parameter(
                            'yield_c_s0', yield_c_s0, mode='denormalize')
                    if is_learnable_yield_d:
                        yield_d_s0 = self.transform_parameter(
                            'yield_d_s0', yield_d_s0, mode='denormalize')
                # Perform convexity return-mapping
                is_convex, yield_c_s0, yield_d_s0 = \
                    LouZhangYoon.convexity_return_mapping(yield_c_s0,
                                                          yield_d_s0)
                # Update yielding parameters
                if not is_convex:
                    # Normalize yielding parameters
                    if self.is_normalized_parameters:
                        if is_learnable_yield_c:
                            yield_c_s0 = self.transform_parameter(
                                'yield_c_s0', yield_c_s0, mode='normalize')
                        if is_learnable_yield_d:
                            yield_d_s0 = self.transform_parameter(
                                'yield_d_s0', yield_d_s0, mode='normalize')
                    # Update yielding parameters (enforcing convexity)
                    with torch.no_grad():
                        if is_learnable_yield_c:
                            self.get_model_parameters()['yield_c_s0'].copy_(
                                yield_c_s0)
                        if is_learnable_yield_d:
                            self.get_model_parameters()['yield_d_s0'].copy_(
                                yield_d_s0)
    # -------------------------------------------------------------------------
    def get_detached_model_parameters(self, is_normalized_out=False):
        """Get model parameters detached of gradients.
        
        Parameters
        ----------
        is_normalized_out : bool, default=False
            If True, then model parameters are normalized.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        # Get detached model parameters
        base_model_parameters = {param: float(value.data) for param, value
                                 in self._model_parameters.items()}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model parameters
        model_parameters = copy.deepcopy(base_model_parameters)
        # Normalize/Denormalize model parameters
        if self.is_normalized_parameters:
            # Denormalize model parameters
            if not is_normalized_out:
                for param, value in base_model_parameters.items():
                    model_parameters[param] = self.transform_parameter(
                        param, value, mode='denormalize')
        else:
            # Normalize model parameters
            if is_normalized_out:
                for param, value in base_model_parameters.items():
                    model_parameters[param] = self.transform_parameter(
                        param, value, mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters
    # -------------------------------------------------------------------------
    def get_material_model_parameters(self):
        """Get current material constitutive model parameters.
        
        Returns
        -------
        model_parameters : dict
            Material constitutive model parameters.
        """
        return self._constitutive_model._model_parameters
    # -------------------------------------------------------------------------
    def forward(self, features_in):
        """Forward propagation.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
            
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Synchronize material model parameters with learnable parameters
        if self._is_auto_sync_parameters:
            self.sync_material_model_parameters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Forward propagation: Material constitutive model
        if bool(re.search(r'_vmap$', self._material_model_name)):
            features_out = self._vrecurrent_constitutive_model(features_in)
        else:
            features_out = self._recurrent_constitutive_model(features_in)
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
            # Check path stress history
            if self._is_check_su_fail:
                if torch.isnan(stress_path).any():
                    raise RuntimeError(f'NaNs were detected in the stress '
                                       f'path history. This may have resulted '
                                       f'from a state update convergence '
                                       f'failure.')
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
        # Initialize last converged material constitutive state variables
        state_variables_old = state_variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(1, n_time):
            # Get previous and current strain tensors
            strain_tensor_old = self.vbuild_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx - 1, :],
                device=self._device)
            strain_tensor = self.vbuild_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx, :],
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
            if self._is_check_su_fail:
                if state_variables['is_su_fail']:
                    raise RuntimeError('Material state update convergence '
                                       'failure.')
            # Update last converged material constitutive state variables
            state_variables_old = state_variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get stress tensor
            if self._strain_formulation == 'infinitesimal':
                # Get Cauchy stress tensor
                stress = vget_tensor_from_mf(state_variables['stress_mf'],
                                             self._n_dim, self._comp_order_sym,
                                             device=self._device)
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store stress tensor
            stress_path[time_idx, :] = \
                self.vstore_tensor_comps(stress_comps_order, stress,
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
                        strain = vget_tensor_from_mf(
                            state_variables[state_var], self._n_dim,
                            self._comp_order_sym, device=self._device)
                    else:
                        raise RuntimeError('Not implemented.')
                    # Store strain tensor
                    state_path[state_var][time_idx, :] = \
                        self.vstore_tensor_comps(strain_comps_order, strain,
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
        # Get device from input tensor
        if device is None:
            device = comps_array.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Get device from input tensor
        if device is None:
            device = tensor.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    def _vrecurrent_constitutive_model_outdated(self, strain_paths):
        """Compute material response.
        
        Compatible with vectorized mapping.
        
        This method can be removed once new (vmap-based) version is validated.
        
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
        # Initialize material response paths data
        response_paths_data = []
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
            stress_path, state_path = self._vcompute_stress_path(strain_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate path stress and state features
            response_path_data = torch.cat((stress_path, state_path), dim=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check number of output features
            if response_path_data.shape[1] != self._n_features_out:
                raise RuntimeError(f'Material response number of dimensions '
                                   f'({response_path_data.shape[1]}) does not '
                                   f'match the model number of output '
                                   f'features ({self._n_features_out}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store material response path data
            response_paths_data.append(response_path_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build material response paths
        if is_batched:
            response_paths = torch.stack(response_paths_data, dim=1)
        else:
            response_paths = response_paths_data[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return response_paths
    # -------------------------------------------------------------------------
    def _vrecurrent_constitutive_model(self, strain_paths):
        """Compute material response.
        
        Compatible with vectorized mapping.
        
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
        # Set batching dimension
        batch_dim = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if batched input
        is_batched = len(strain_paths.shape) == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batched dimension
        if not is_batched:
            strain_paths = strain_paths.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized material response computation (batch along paths)
        vmap_compute_stress_path = torch.vmap(
            self._vcompute_stress_path, in_dims=(1,), out_dims=(1, 0))
        # Compute paths material response
        stress_paths, state_paths = vmap_compute_stress_path(strain_paths)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check paths stress history
        if self._is_check_su_fail:
            if torch.isnan(stress_paths).any():
                raise RuntimeError(f'NaNs were detected in the stress paths '
                                   f'history. This may have resulted from a '
                                   f'state update convergence failure.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fix batch dimension (required to handle case when there are no state
        # features)
        state_paths = state_paths.permute(1, 0, 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate path stress and state features
        response_paths = torch.cat((stress_paths, state_paths), dim=2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of output features
        if response_paths.shape[2] != self._n_features_out:
            raise RuntimeError(f'Material response number of dimensions '
                               f'({response_paths.shape[2]}) does not '
                               f'match the model number of output '
                               f'features ({self._n_features_out}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            response_paths = response_paths.squeeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return response_paths
    # -------------------------------------------------------------------------
    def _vcompute_stress_path(self, strain_path):
        """Compute material response for given strain path.

        Compatible with vectorized mapping.
        
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
        state_path : torch.Tensor(2d)
            State path history stored as torch.Tensor(2d) of shape
            (sequence_length, n_features).
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
        # Initialize stress path history steps
        stress_path_steps = []
        # Set initial stress components
        stress_comps = torch.zeros(len(stress_comps_order),
                                   device=self._device)
        # Store initial stress tensor
        stress_path_steps.append(stress_comps)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check state features
        is_state_features_out = False
        if len(self._state_features_out) > 0:
            is_state_features_out = True
        # Store initial state features tensor
        if is_state_features_out:
            # Initialize state features path history steps
            state_path_steps = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize state features
            state_var_features = []
            for state_var, n_features in self._state_features_out.items():
                state_var_features.append(torch.zeros((1, n_features),
                                                      device=self._device))
            # Set initial state features tensor
            state_comps = torch.cat(state_var_features, dim=1) 
            # Store initial state features tensor
            state_path_steps.append(state_comps)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize constitutive model state variables
        state_variables = self._constitutive_model.state_init()
        # Initialize last converged material constitutive state variables
        state_variables_old = state_variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(1, n_time):
            # Get previous and current strain tensors
            strain_tensor_old = self.vbuild_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx - 1, :],
                device=self._device)
            strain_tensor = self.vbuild_tensor_from_comps(
                self._n_dim, strain_comps_order, strain_path[time_idx, :],
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
            if self._is_check_su_fail:
                if state_variables['is_su_fail']:
                    raise RuntimeError('Material state update convergence '
                                       'failure.')
            # Update last converged material constitutive state variables
            state_variables_old = state_variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get stress tensor
            if self._strain_formulation == 'infinitesimal':
                # Get Cauchy stress tensor
                stress = vget_tensor_from_mf(state_variables['stress_mf'],
                                             self._n_dim, self._comp_order_sym,
                                             device=self._device)
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get stress components
            stress_comps = self.vstore_tensor_comps(stress_comps_order, stress,
                                                    device=self._device)
            # Store stress tensor
            stress_path_steps.append(stress_comps) 
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store state features
            if is_state_features_out:
                # Initialize state features
                state_var_features = []
                for state_var, n_features in self._state_features_out.items():
                    # Skip if state variable is not available
                    if state_var not in state_variables.keys():
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store state features
                    if 'strain_mf' in state_var:
                        # Build strain tensor
                        if self._strain_formulation == 'infinitesimal':
                            strain = vget_tensor_from_mf(
                                state_variables[state_var], self._n_dim,
                                self._comp_order_sym, device=self._device)
                        else:
                            raise RuntimeError('Not implemented.')
                        # Get strain components
                        state_comps = self.vstore_tensor_comps(
                            strain_comps_order, strain, device=self._device)
                    else:
                        # Get generic state variable components
                        state_comps = \
                            state_variables[state_var].reshape(1, n_features)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store state features
                    state_var_features.append(state_comps)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set state features tensor
                state_comps = torch.cat(state_var_features, dim=1)
                # Store state features tensor
                state_path_steps.append(state_comps)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build stress path
        stress_path = torch.stack(stress_path_steps, dim=0)
        # Build state features path
        state_path = torch.zeros((n_time, 0), device=self._device)
        if is_state_features_out:
            state_path = torch.cat(state_path_steps, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_path, state_path
    # -------------------------------------------------------------------------
    @classmethod
    def vbuild_tensor_from_comps(cls, n_dim, comps, comps_array, device=None):
        """Build strain/stress tensor from given components.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comps : tuple[str]
            Strain/Stress components order.
        comps_array : torch.Tensor(1d)
            Strain/Stress components array.
        device : torch.device, default=None
            Device on which torch.Tensor is allocated.
        
        Returns
        -------
        tensor : torch.Tensor(2d)
            Strain/Stress tensor.
        """
        # Get device from input tensor
        if device is None:
            device = comps_array.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set row major components order       
        row_major_order = tuple(f'{i + 1}{j + 1}' for i, j
                                in itertools.product(range(n_dim), repeat=2))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build indexing mapping
        index_map = [comps.index(x) if x in comps
                     else comps.index(x[::-1]) for x in row_major_order]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build tensor
        tensor = comps_array[index_map].view(n_dim, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tensor
    # -------------------------------------------------------------------------
    @classmethod
    def vstore_tensor_comps(cls, comps, tensor, device=None):
        """Store strain/stress tensor components in array.
        
        Compatible with vectorized mapping.
        
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
        # Get device from input tensor
        if device is None:
            device = tensor.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build indexing mapping
        index_map = tuple([int(x[i]) - 1 for x in comps] for i in range(2))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build tensor components array
        comps_array = tensor[index_map]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return comps_array
    # -------------------------------------------------------------------------
    def move_state_tensors_to_device(self, state_variables):
        """Move state variables tensors to model device.
        
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
        model_init_args['is_normalized_parameters'] = \
            self.is_normalized_parameters
        model_init_args['is_model_in_normalized'] = self.is_model_in_normalized
        model_init_args['is_model_out_normalized'] = \
            self.is_model_out_normalized
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