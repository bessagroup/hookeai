"""DARPA METALS PROJECT: Finding material model by inverse engineering.

Classes
-------
MaterialModelFinder(torch.nn.Module)
    Find material model by inverse engineering experimental results.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
import re
import itertools
# Third-party
import torch
# Local
from simulators.fetorch.element.integrations.internal_forces import \
    compute_element_internal_forces, compute_infinitesimal_inc_strain, \
    compute_infinitesimal_strain
from simulators.fetorch.element.derivatives.gradients import \
    eval_shapefun_deriv, vbuild_discrete_sym_gradient
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf
from simulators.fetorch.math.voigt_notation import vget_stress_vmf, \
    vget_strain_from_vmf
from utilities.data_scalers import TorchMinMaxScaler
from rnn_base_model.data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
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
class MaterialModelFinder(torch.nn.Module):
    """Find material model by inverse engineering experimental results.
    
    This class implements the devised data-driven machine learning process
    to find a unknown material constitutive model from a single experiment
    on a given specimen. After discretizing the specimen in a suitable finite
    element mesh, the experimental results are translated to the nodes
    displacement history (input data) and reaction forces history
    (output data).
    
    The data-driven machine learning process is designed to learn a surrogate
    model that characterizes the material behavior and predicts the observed
    experimental data accurately.
    
    Attributes
    ----------
    _specimen_data : SpecimenNumericalData
        Specimen numerical data translated from experimental results.
    _specimen_material_state : StructureMaterialState
        FETorch specimen material state.
    _is_force_normalization : bool, default=False
        If True, then normalize forces prior to the computation of the force
        equilibrium loss.
    model_directory : str
        Directory where model is stored.
    model_name : str
        Name of model.
    _material_models_dir : str
        Model subdirectory where material models are stored.
    _internal_data_normalization_dir : str
        Model subdirectory where the internal data normalization parameters
        are stored.
    _temp_dir : str
        Model subdirectory where temporary data is stored.
    _device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    _set_model_subdirs(self)
        Set model subdirectories.
    _set_material_models_dirs(self)
        Set material models directories.
    set_specimen_data(self, specimen_data, specimen_material_state,
                      force_minimum=None, force_maximum=None)
        Set specimen data and material state.
    _set_model_parameters(self)
        Set model parameters (collect material models parameters).
    get_model_parameters(self)
        Get model parameters (material models parameters).
    get_detached_model_parameters(self)
        Get model parameters (material models) detached of gradients.
    get_model_parameters_bounds(self)
        Get model parameters (material models) bounds.
    enforce_parameters_bounds(self)
        Enforce bounds in model parameters (material models).
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, sequential_mode='sequential_element')
        Forward propagation.
    forward_sequential_time(self)
        Forward propagation (sequential time).
    forward_sequential_element(self, is_store_local_paths=False)
        Forward propagation (sequential element).
    compute_element_internal_forces_hist(self, strain_formulation, \
                                         problem_type, element_type, \
                                         element_material, element_state_old, \
                                         nodes_coords_hist, nodes_disps_hist, \
                                         nodes_inc_disps_hist, time_hist, \
                                         is_recurrent_model)
        Compute history of finite element internal forces.
    recurrent_material_state_update(self, strain_formulation, problem_type, \
                                    constitutive_model, strain_hist, time_hist)
        Material state update for any given recurrent constitutive model.
    force_equilibrium_loss(self, internal_forces_mesh, external_forces_mesh, \
                           reaction_forces_mesh, dirichlet_bool_mesh)
        Compute force equilibrium loss for given discrete time.
    build_tensor_from_comps(cls, n_dim, comps, comps_array, is_symmetric=False,
                            device=None)
        Build strain/stress tensor from given components.
    store_tensor_comps(cls, comps, tensor, device=None)
        Store strain/stress tensor components in array.
    _init_data_scalers(self)
        Initialize model data scalers.
    set_fitted_force_data_scalers(self, force_minimum, force_maximum)
        Set fitted forces data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    data_scaler_transform(self, tensor, features_type, mode='normalize')
        Perform data scaling operation on features PyTorch tensor.
    set_material_models_fitted_data_scalers(self, models_scaling_type, \
                                            models_scaling_parameters)
    build_element_local_samples(self, strain_formulation, problem_type,
                                element_type, time_hist, element_state_hist)
        Build element Gauss integration points local strain-stress paths.
    save_model_state(self, epoch=None, is_best_state=False, \
                     is_remove_posterior=True)
        Save model state to file.
    load_model_state(self, load_model_state=None, is_remove_posterior=True)
        Load model state from file.
    _check_state_file(self, filename)
        Check if file is model training epoch state file.
    _check_best_state_file(self, filename)
        Check if file is model training epoch best state file.
    _remove_posterior_state_files(self, epoch)
        Delete model training epoch state files posterior to given epoch.
    _remove_best_state_files(self)
        Delete existent model best state files.
    """
    def __init__(self, model_directory, model_name='material_model_finder',
                 is_force_normalization=False, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        model_directory : str
            Directory where model is stored.
        model_name : str, default='material_model_finder'
            Name of model.
        is_force_normalization : bool, default=False
            If True, then normalize forces prior to the computation of the
            force equilibrium loss.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(MaterialModelFinder, self).__init__()
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
        # Set model subdirectories
        self._set_model_subdirs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set force normalization
        self._is_force_normalization = is_force_normalization
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize specimen numerical data
        self._specimen_data = None
        # Initialize specimen material state
        self._specimen_material_state = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize parameters
        self._model_parameters = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        self._data_scalers = None
        if self._is_force_normalization:
            self._init_data_scalers()
    # -------------------------------------------------------------------------
    def _set_model_subdirs(self):
        """Set model subdirectories."""
        # Set material models subdirectory
        self._material_models_dir = os.path.join(
            os.path.normpath(self.model_directory), 'material_models')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect model subdirectories
        subdirs = (self._material_models_dir,)
        # Create model subdirectories
        for subdir in subdirs:
            make_directory(subdir, is_overwrite=True)
    # -------------------------------------------------------------------------
    def _set_material_models_dirs(self):
        """Set material models directories."""
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Loop over material models
        for model_key, model in material_models.items():
            # Set material model directory
            model_dir = \
                os.path.join(os.path.normpath(self._material_models_dir),
                             f'model_{model_key}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create material model directory
            make_directory(model_dir, is_overwrite=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update material model directory
            model.model_directory = model_dir
    # -------------------------------------------------------------------------
    def set_specimen_data(self, specimen_data, specimen_material_state,
                          force_minimum=None, force_maximum=None):
        """Set specimen data and material state.
        
        Parameters
        ----------
        specimen_data : SpecimenNumericalData
            Specimen numerical data translated from experimental results.
        specimen_material_state : StructureMaterialState
            FETorch specimen material state.
        force_minimum : torch.Tensor(1d), default=None
            Forces normalization minimum tensor stored as a torch.Tensor with
            shape (n_dim,). Only required if force normalization is set to
            True, otherwise ignored.
        force_maximum : torch.Tensor(1d)
            Forces normalization maximum tensor stored as a torch.Tensor with
            shape (n_dim,). Only required if force normalization is set to
            True, otherwise ignored.
        """
        # Set specimen numerical data
        self._specimen_data = specimen_data
        # Set specimen material state
        self._specimen_material_state = specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material models directories
        self._set_material_models_dirs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect specimen underlying material models parameters
        self._set_model_parameters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit force data scalers
        if self._is_force_normalization:
            self.set_fitted_force_data_scalers(force_minimum, force_maximum)
    # -------------------------------------------------------------------------
    def get_material_models(self):
        """Get material models.
        
        Returns
        -------
        material_models : dict
            FETorch material constitutive models (key, str[int], item,
            ConstitutiveModel). Models are labeled from 1 to n_mat_model.
        """
        return self._specimen_material_state.get_material_models()
    # -------------------------------------------------------------------------
    def _set_model_parameters(self):
        """Set model parameters (collect material models parameters)."""
        # Initialize parameters
        self._model_parameters = torch.nn.ParameterDict({})
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Loop over material models
        for model_key, model in material_models.items():
            # Assemble material model parameters
            if hasattr(model, 'get_model_parameters'):
                self._model_parameters[model_key] = \
                    model.get_model_parameters()
    # -------------------------------------------------------------------------
    def get_model_parameters(self):
        """Get model parameters (material models parameters).
        
        Returns
        -------
        model_parameters : torch.nn.ParameterDict
            Model parameters.
        """
        return self._model_parameters
    # -------------------------------------------------------------------------
    def get_detached_model_parameters(self):
        """Get model parameters (material models) detached of gradients.
        
        Only collects parameters from material models with explicit learnable
        parameters.
        
        Parameters names are prefixed by corresponding model label.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        # Initialize model parameters
        model_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Get model selection procedure
        is_collect_model_parameters = \
            self._specimen_material_state.get_material_model_param_nature
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for model_key, model in material_models.items():
            # Check if model parameters are collected
            is_collect_params = is_collect_model_parameters(int(model_key))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip model parameters
            if not is_collect_params:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get detached model parameters
            detached_parameters = \
                model.get_detached_model_parameters(is_normalized=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Collect parameters (prefix with model label)
            for param, value in detached_parameters.items():
                # Store parameter
                model_parameters[f'model_{model_key}_{param}'] = value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters
    # -------------------------------------------------------------------------
    def get_model_parameters_bounds(self):
        """Get model parameters (material models) bounds.
        
        Only collects parameters bounds from material models with explicit
        learnable parameters.
        
        Parameters names are prefixed by corresponding model label.

        Returns
        -------
        model_parameters_bounds : dict
            Model learnable parameters bounds. For each parameter (key, str),
            the corresponding bounds are stored as a
            tuple(lower_bound, upper_bound) (item, tuple).
        """
        # Initialize model parameters
        model_parameters_bounds = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Get model selection procedure
        is_collect_model_parameters = \
            self._specimen_material_state.get_material_model_param_nature
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for model_key, model in material_models.items():
            # Check if model parameters are collected
            is_collect_params = is_collect_model_parameters(int(model_key))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip model parameters
            if not is_collect_params:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model parameters bounds
            parameters_bounds = model.get_model_parameters_bounds()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Collect parameters bounds (prefix with model label)
            for param, bounds in parameters_bounds.items():
                # Store parameter
                model_parameters_bounds[f'model_{model_key}_{param}'] = bounds
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters_bounds
    # -------------------------------------------------------------------------
    def enforce_parameters_bounds(self):
        """Enforce bounds in model parameters (material models).
        
        Only enforces bounds in parameters from material models with explicit
        learnable parameters.
        
        """
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Get model selection procedure
        is_collect_model_parameters = \
            self._specimen_material_state.get_material_model_param_nature
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over parameters
        for model_key, param_dict in self.get_model_parameters().items():
            # Check if model parameters are collected
            is_collect_params = is_collect_model_parameters(int(model_key))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip model parameters
            if not is_collect_params:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material model
            model = material_models[model_key]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material model parameters
            for param in model.get_model_parameters().keys():
                # Get parameter bounds
                if model.is_normalized_parameters:
                    lower_bound, upper_bound = \
                        model.get_model_parameters_norm_bounds()[param]
                else:
                    lower_bound, upper_bound = \
                        model.get_model_parameters_bounds()[param]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get learnable parameter
                value = param_dict[param]
                # Enforce bounds
                value.data.clamp_(lower_bound, upper_bound)
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
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def forward(self, sequential_mode='sequential_element'):
        """Forward propagation.
        
        Parameters
        ----------
        specimen_data : SpecimenNumericalData
            Specimen numerical data translated from experimental results.
        specimen_material_state : StructureMaterialState
            FETorch structure material state.
        sequential_mode : {'sequential_time', 'sequential_element', \
                           'sequential_element_vmap}, \
                          default='sequential_element'
                          
            'sequential_time' : Internal forces are computed in the standard
            way, processing each time step sequentially. Currently only
            available for inference.
            
            'sequential_element' : Internal forces are computed such that each
            element is processed sequentially (taking into account the
            corresponding deformation history). Available for both training
            and inference. Significantly limited with respect to memory costs.
            
            'sequential_element_vmap' : Similar to 'sequential_element' but
            leveraging vectorizing maps (significant improvement of processing
            time and memory efficiency). Available for both training and
            inference.

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
        if sequential_mode == 'sequential_time':
            force_equilibrium_hist_loss = self.forward_sequential_time()
        elif sequential_mode == 'sequential_element':
            #force_equilibrium_hist_loss = self.forward_sequential_element()
            force_equilibrium_hist_loss = self.vforward_sequential_element()
        elif sequential_mode == 'sequential_element_vmap':
            force_equilibrium_hist_loss = self.vforward_sequential_element()
        else:
            raise RuntimeError('Unknown sequential mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_time(self):
        """Forward propagation (sequential time).

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
        # Get specimen numerical data
        specimen_data = self._specimen_data
        # Get specimen material state
        specimen_material_state = self._specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get specimen finite element mesh
        specimen_mesh = specimen_data.specimen_mesh
        # Get number of elements of finite element mesh
        n_elem = specimen_mesh.get_n_elem()
        # Get elements type
        elements_type = specimen_mesh.get_elements_type()
        # Get degrees of freedom subject to Dirichlet boundary conditions
        dirichlet_bool_mesh = specimen_mesh.get_dirichlet_bool_mesh()
        # Get time history length
        n_time = specimen_data.time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain formulation and problem type
        strain_formulation = specimen_material_state.get_strain_formulation()
        problem_type = specimen_material_state.get_problem_type()
        # Get problem type parameters
        n_dim, _, _ = get_problem_type_parameters(problem_type)
        # Get elements material
        elements_material = specimen_material_state.get_elements_material()
        # Set finite element mesh nodes coordinates update flag
        if strain_formulation == 'infinitesimal':
            is_update_coords = False
        else:
            is_update_coords = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize force equilibrium history loss
        force_equilibrium_hist_loss = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(n_time):
            # Update mesh configuration with known displacement history
            specimen_data.update_specimen_mesh_configuration(
                time_idx, is_update_coords=is_update_coords)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize elements internal forces
            elements_internal_forces = {}
            # Loop over elements
            for i in range(n_elem):
                # Get element label
                elem_id = i + 1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get element type
                element_type = elements_type[str(elem_id)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get element nodes coordinates and displacements
                nodes_coords, nodes_disps = \
                    specimen_mesh.get_element_configuration(elem_id,
                                                            time='current')
                # Get element nodes last converged displacements
                _, nodes_disps_old = \
                    specimen_mesh.get_element_configuration(elem_id,
                                                            time='last')
                # Compute element nodes incremental displacements
                nodes_inc_disps = nodes_disps - nodes_disps_old
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get element material model
                element_material = elements_material[str(elem_id)]
                # Get element last converged material constitutive state
                # variables
                element_state_old = \
                    specimen_material_state.get_element_state(elem_id,
                                                              time='last')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute element internal forces
                internal_forces, element_state = \
                    compute_element_internal_forces(
                        strain_formulation, problem_type, element_type,
                        element_material, element_state_old, nodes_coords,
                        nodes_disps, nodes_inc_disps)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store element internal forces
                elements_internal_forces[str(elem_id)] = internal_forces
                # Update element material constitutive state variables
                specimen_material_state.update_element_state(
                    elem_id, element_state, time='current')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble element internal forces of finite element mesh nodes
            internal_forces_mesh = \
                specimen_mesh.element_assembler(elements_internal_forces)
            internal_forces_mesh = internal_forces_mesh.reshape(-1, n_dim)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update elements last converged material constitutive state
            # variables
            specimen_material_state.update_converged_elements_state()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set null external forces of finite element mesh nodes
            external_forces_mesh = torch.zeros_like(internal_forces_mesh)
            # Get reaction forces (Dirichlet boundary conditions) of finite
            # element mesh nodes
            reaction_forces_mesh = \
                specimen_data.reaction_forces_mesh_hist[:, :, time_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add contribution to force equilibrium history loss
            force_equilibrium_hist_loss += self.force_equilibrium_loss(
                internal_forces_mesh, external_forces_mesh,
                reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_element(self, is_store_local_paths=False):
        """Forward propagation (sequential element).
        
        Parameters
        ----------
        is_store_local_paths : bool, default=False
            If True, then store data set of specimen local (Gauss integration
            points) strain-stress paths in dedicated model subdirectory.
            Overwrites existing data set.

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
        # Get specimen numerical data
        specimen_data = self._specimen_data
        # Get specimen material state
        specimen_material_state = self._specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get specimen finite element mesh
        specimen_mesh = specimen_data.specimen_mesh
        # Get number of nodes of finite element mesh
        n_node_mesh = specimen_mesh.get_n_node_mesh()
        # Get number of elements of finite element mesh
        n_elem = specimen_mesh.get_n_elem()
        # Get elements type
        elements_type = specimen_mesh.get_elements_type()
        # Get degrees of freedom subject to Dirichlet boundary conditions
        dirichlet_bool_mesh = specimen_mesh.get_dirichlet_bool_mesh()
        # Get time history
        time_hist = specimen_data.time_hist
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain formulation and problem type
        strain_formulation = specimen_material_state.get_strain_formulation()
        problem_type = specimen_material_state.get_problem_type()
        # Get problem type parameters
        n_dim, _, _ = get_problem_type_parameters(problem_type)
        # Get elements material
        elements_material = specimen_material_state.get_elements_material()
        # Set finite element mesh nodes coordinates update flag
        if strain_formulation == 'infinitesimal':
            is_update_coords = False
        else:
            is_update_coords = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize specimen local strain-stress paths data set samples
        if is_store_local_paths:
            specimen_local_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element internal forces of finite element mesh nodes
        internal_forces_mesh_hist = torch.zeros(((n_node_mesh, n_dim, n_time)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for i in range(n_elem):
            # Get element label
            elem_id = i + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element type
            element_type = elements_type[str(elem_id)]
            # Get element type number of nodes
            n_node = element_type.get_n_node()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element material model
            element_material = elements_material[str(elem_id)]
            # Get element material model recurrent structure
            is_recurrent_model = \
                specimen_material_state.get_element_model_recurrency(elem_id)
            # Get element initial material constitutive state variables
            element_state_old = \
                specimen_material_state.get_element_state(elem_id, time='last',
                                                          is_copy=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize element nodes coordinates, displacements and
            # incremental displacements histories
            nodes_coords_hist = \
                torch.zeros((n_node, n_dim, n_time), dtype=torch.float)
            nodes_disps_hist = torch.zeros_like(nodes_coords_hist)
            nodes_inc_disps_hist = torch.zeros_like(nodes_coords_hist)
            # Loop over discrete time
            for time_idx in range(n_time):
                # Update mesh configuration with known displacement history
                specimen_data.update_specimen_mesh_configuration(
                    time_idx, is_update_coords=is_update_coords)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get element nodes coordinates and displacements
                nodes_coords, nodes_disps = \
                    specimen_mesh.get_element_configuration(elem_id,
                                                            time='current')
                # Get element nodes last converged displacements
                _, nodes_disps_old = \
                    specimen_mesh.get_element_configuration(elem_id,
                                                            time='last')
                # Compute element nodes incremental displacements
                nodes_inc_disps = nodes_disps - nodes_disps_old
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble element nodes history
                nodes_coords_hist[:, :, time_idx] = nodes_coords
                nodes_disps_hist[:, :, time_idx] = nodes_disps
                nodes_inc_disps_hist[:, :, time_idx] = nodes_inc_disps
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute history of finite element internal forces
            element_internal_forces_hist, element_state_hist = \
                self.compute_element_internal_forces_hist(
                    strain_formulation, problem_type, element_type,
                    element_material, element_state_old, nodes_coords_hist,
                    nodes_disps_hist, nodes_inc_disps_hist, time_hist,
                    is_recurrent_model)
            # Update element material constitutive state variables
            specimen_material_state.update_element_state(
                elem_id, element_state_hist[-1], time='current', is_copy=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete time
            for time_idx in range(n_time):
                # Reshape element internal forces into mesh format
                internal_forces_mesh = specimen_mesh.element_assembler(
                    {str(elem_id): element_internal_forces_hist[:, time_idx]})
                # Assemble element internal forces of finite element mesh nodes
                internal_forces_mesh_hist[:, :, time_idx] += \
                    internal_forces_mesh.reshape(-1, n_dim)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble element local strain-stress paths
            if is_store_local_paths:
                # Build element local strain-stress paths
                element_local_samples = self.build_element_local_samples(
                    strain_formulation, problem_type, element_type, time_hist,
                    element_state_hist)
                # Assemble element local strain-stress paths
                specimen_local_samples += element_local_samples
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elements last converged material constitutive state variables
        specimen_material_state.update_converged_elements_state(is_copy=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize force equilibrium history loss
        force_equilibrium_hist_loss = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(n_time):
            # Get internal forces of finite element mesh nodes
            internal_forces_mesh = internal_forces_mesh_hist[:, :, time_idx]
            # Set null external forces of finite element mesh nodes
            external_forces_mesh = torch.zeros_like(internal_forces_mesh)
            # Get reaction forces (Dirichlet boundary conditions) of finite
            # element mesh nodes
            reaction_forces_mesh = \
                specimen_data.reaction_forces_mesh_hist[:, :, time_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add contribution to force equilibrium history loss
            force_equilibrium_hist_loss += self.force_equilibrium_loss(
                internal_forces_mesh, external_forces_mesh,
                reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store specimen local strain-stress paths data set
        if is_store_local_paths:
            # Create strain-stress material response path data set
            dataset = TimeSeriesDatasetInMemory(specimen_local_samples)
            # Set data set file basename
            dataset_basename = 'ss_paths_dataset'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set directory
            dataset_directory = \
                os.path.join(os.path.normpath(self.model_directory),
                             'local_response_dataset')
            # Create model directory
            make_directory(dataset_directory, is_overwrite=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save data set
            save_dataset(dataset, dataset_basename, dataset_directory,
                         is_append_n_sample=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def compute_element_internal_forces_hist(
        self, strain_formulation, problem_type, element_type, element_material,
        element_state_old, nodes_coords_hist, nodes_disps_hist,
        nodes_inc_disps_hist, time_hist, is_recurrent_model):
        """Compute history of finite element internal forces.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        element_type : Element
            FETorch finite element.
        element_material : ConstitutiveModel
            FETorch material constitutive model.
        element_state_old : dict
            Last converged material constitutive model state variables
            (item, dict) for each Gauss integration point (key, str[int]).
        nodes_coords_hist : torch.Tensor(3d)
            Coordinates history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        nodes_disps_hist : torch.Tensor(3d)
            Displacements history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        nodes_inc_disps_hist : torch.Tensor(3d)
            Incremental displacements history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        time_hist : torch.Tensor(1d)
            Discrete time history.
        is_recurrent_model : bool
            True if the material constitutive model has a recurrent structure
            (processes full deformation path when called), False otherwise.

        Returns
        -------
        element_internal_forces_hist : torch.Tensor(2d)
            Element internal forces history stored as torch.Tensor(2d) of shape
            (n_node*n_dof_node, n_time).
        element_state_hist : list[dict]
            Material constitutive model state variables history (item, dict)
            for each Gauss integration point (key, str[int]).
        """
        # Get problem type parameters
        n_dim, comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element number of nodes
        n_node = element_type.get_n_node()
        # Get element number of degrees of freedom per node
        n_dof_node = element_type.get_n_dof_node()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element number of Gauss quadrature integration points
        n_gauss = element_type.get_n_gauss()
        # Get element Gauss quadrature integration points local coordinates
        # and weights
        gp_coords, gp_weights = element_type.get_gauss_integration_points()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element internal forces history
        element_internal_forces_hist = torch.zeros((n_node*n_dof_node, n_time))
        # Initialize element material constitutive model state variables
        # history
        element_state_hist = \
            [{key: None for key in gp_coords.keys()} for _ in range(n_time)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Gauss integration points
        for i in range(n_gauss):
            # Get Gauss integration point local coordinates and weight
            local_coords = gp_coords[str(i + 1)]
            weight = gp_weights[str(i + 1)]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Gauss integration point last converged material
            # constitutive model state variables
            if not is_recurrent_model:
                state_variables_old = \
                    copy.deepcopy(element_state_old[str(i + 1)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Gauss integration point strain tensor history
            if strain_formulation == 'infinitesimal':
                if is_recurrent_model:
                    strain_hist = \
                        torch.zeros((n_dim, n_dim, n_time), dtype=torch.float)
                else:
                    inc_strain_hist = \
                        torch.zeros((n_dim, n_dim, n_time), dtype=torch.float)
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete time
            for time_idx in range(n_time):
                # Evaluate shape functions derivates and Jacobian
                shape_fun_deriv, _, jacobian_det = eval_shapefun_deriv(
                    element_type, nodes_coords_hist[:, :, time_idx],
                    local_coords)
                # Build discrete symmetric gradient operator
                grad_operator_sym = vbuild_discrete_sym_gradient(
                    shape_fun_deriv, comp_order_sym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute strain tensor
                if strain_formulation == 'infinitesimal':
                    if is_recurrent_model:
                        # Compute infinitesimal strain tensor (Voigt matricial
                        # form)
                        strain_vmf = compute_infinitesimal_strain(
                            grad_operator_sym,
                            nodes_disps_hist[:, :, time_idx])
                        # Get strain tensor
                        strain_hist[:, :, time_idx] = vget_strain_from_vmf(
                            strain_vmf, n_dim, comp_order_sym)
                    else:
                        # Compute incremental infinitesimal strain tensor
                        # (Voigt matricial form)
                        inc_strain_vmf = compute_infinitesimal_inc_strain(
                            grad_operator_sym,
                            nodes_inc_disps_hist[:, :, time_idx])
                        # Get incremental strain tensor
                        inc_strain_hist[:, :, time_idx] = vget_strain_from_vmf(
                            inc_strain_vmf, n_dim, comp_order_sym)
                else:
                    raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Gauss integration point stress history
            if is_recurrent_model:
                # Material state update
                state_variables_hist = self.recurrent_material_state_update(
                    strain_formulation, problem_type, element_material,
                    strain_hist, time_hist)
                # Loop over discrete time
                for time_idx in range(n_time):
                    # Store Gaussian integration point material constitutive
                    # model state variables
                    element_state_hist[time_idx][str(i + 1)] = \
                        state_variables_hist[time_idx]
            else:
                # Loop over discrete time
                for time_idx in range(n_time):
                    # Material state update
                    state_variables, _ = material_state_update(
                        strain_formulation, problem_type, element_material,
                        inc_strain_hist[:, :, time_idx], state_variables_old,
                        def_gradient_old=None)
                    # Store Gauss integration point material constitutive model
                    # state variables
                    element_state_hist[time_idx][str(i + 1)] = state_variables 
                    # Update Gauss integration point last converged material
                    # constitutive model state variables
                    state_variables_old = copy.deepcopy(state_variables)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete time
            for time_idx in range(n_time):
                # Get stress tensor
                if strain_formulation == 'infinitesimal':
                    # Get Cauchy stress tensor
                    stress = vget_tensor_from_mf(
                        element_state_hist[time_idx][str(i + 1)]['stress_mf'],
                        n_dim, comp_order_sym)
                    # Get Cauchy stress tensor (Voigt matricial form)
                    stress_vmf = vget_stress_vmf(stress, n_dim, comp_order_sym)
                else:
                    raise RuntimeError('Not implemented.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Evaluate shape functions derivates and Jacobian
                shape_fun_deriv, _, jacobian_det = eval_shapefun_deriv(
                    element_type, nodes_coords_hist[:, :, time_idx],
                    local_coords)
                # Build discrete symmetric gradient operator
                grad_operator_sym = vbuild_discrete_sym_gradient(
                    shape_fun_deriv, comp_order_sym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add Gauss integration point contribution to element internal
                # forces
                element_internal_forces_hist[:, time_idx] += (
                    weight*torch.matmul(grad_operator_sym.T, stress_vmf)*
                    jacobian_det)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_internal_forces_hist, element_state_hist
    # -------------------------------------------------------------------------
    def recurrent_material_state_update(self, strain_formulation, problem_type,
                                        constitutive_model, strain_hist,
                                        time_hist):
        """Material state update for any given recurrent constitutive model.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        constitutive_model : ConstitutiveModel
            Recurrent material constitutive model.
        strain_hist : torch.Tensor(3d)
            Strain tensor history stored as torch.Tensor(3d) of shape
            (n_dim, n_dim, n_time).
        time_hist : torch.Tensor(1d)
            Discrete time history.

        Returns
        -------
        state_variables_hist : list[dict]
            Material constitutive model state variables history.
        """
        # Get problem type parameters
        n_dim, comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material constitutive model state variables history
        state_variables_hist = [{} for _ in range(n_time)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor of input features
        if strain_formulation == 'infinitesimal':
            features_in = torch.zeros((n_time, len(comp_order_sym)))
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build tensor of input features
        for time_idx in range(n_time):
            # Store strain tensor
            if strain_formulation == 'infinitesimal':
                features_in[time_idx, :] = self.store_tensor_comps(
                    comp_order_sym, strain_hist[:, :, time_idx],
                    device=self._device)
            else:
                raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute output features
        features_out = constitutive_model(features_in, is_normalized=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(n_time):
            # Build and store strain and stress tensors
            if strain_formulation == 'infinitesimal':
                # Build strain tensor
                strain = self.vbuild_tensor_from_comps(
                    n_dim, comp_order_sym,
                    features_in[time_idx, :len(comp_order_sym)],
                    device=self._device)
                # Store strain tensor
                state_variables_hist[time_idx]['strain_mf'] = \
                    vget_tensor_mf(strain, n_dim, comp_order_sym,
                                   device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build stress tensor
                stress = self.vbuild_tensor_from_comps(
                    n_dim, comp_order_sym,
                    features_out[time_idx, :len(comp_order_sym)],
                    device=self._device)
                # Store stress tensor
                state_variables_hist[time_idx]['stress_mf'] = \
                    vget_tensor_mf(stress, n_dim, comp_order_sym,
                                   device=self._device)
            else:
                raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables_hist
    # -------------------------------------------------------------------------
    def force_equilibrium_loss(self, internal_forces_mesh,
                               external_forces_mesh, reaction_forces_mesh,
                               dirichlet_bool_mesh):
        """Compute force equilibrium loss.
        
        Parameters
        ----------
        internal_forces_mesh : torch.Tensor(2d)
            Internal forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        external_forces_mesh : torch.Tensor(2d)
            External forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        reaction_forces_mesh : torch.Tensor(2d)
            Reaction forces (Dirichlet boundary conditions) of finite element
            mesh nodes stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim).
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
            
        Returns
        -------
        force_equilibrium_loss : float
            Force equilibrium loss.
        """
        # Initialize force equilibrium loss
        force_equilibrium_loss = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize forces
        if self._is_force_normalization:
            # Normalize internal forces
            internal_forces_mesh = self.data_scaler_transform(
                internal_forces_mesh, features_type='forces', mode='normalize')
            # Normalize external forces
            external_forces_mesh = self.data_scaler_transform(
                external_forces_mesh, features_type='forces', mode='normalize')
            # Normalize reaction forces
            reaction_forces_mesh = self.data_scaler_transform(
                reaction_forces_mesh, features_type='forces', mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of nodes of finite element mesh
        n_node_mesh = internal_forces_mesh.shape[0]
        # Get number of spatial dimensions
        n_dim = internal_forces_mesh.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over nodes
        for i in range(n_node_mesh):
            # Loop over dimensions
            for j in range(n_dim):
                # Add contribution of degree of freedom
                if dirichlet_bool_mesh[i, j] == 1:
                    # Constrained degree of freedom (Dirichlet boundary
                    # condition)
                    force_equilibrium_loss += (internal_forces_mesh[i, j]
                                               - external_forces_mesh[i, j]
                                               - reaction_forces_mesh[i, j])**2
                else:
                    # Free degree of freedom
                    force_equilibrium_loss += (internal_forces_mesh[i, j]
                                               - external_forces_mesh[i, j])**2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_loss
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
        tensor = torch.zeros((n_dim, n_dim), dtype=torch.float, device=device)
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
    def store_tensor_comps(cls, comps, tensor, device=None):
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
        comps_array = torch.zeros(len(comps), dtype=torch.float, device=device)
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
    def _init_data_scalers(self):
        """Initialize model data scalers."""
        self._data_scalers = {}
        self._data_scalers['forces'] = None
    # -------------------------------------------------------------------------
    def set_fitted_force_data_scalers(self, force_minimum, force_maximum):
        """Set fitted forces data scalers.
        
        Parameters
        ----------
        force_minimum : torch.Tensor(1d)
            Forces normalization minimum tensor stored as a torch.Tensor with
            shape (n_dim,).
        force_maximum : torch.Tensor(1d)
            Forces normalization maximum tensor stored as a torch.Tensor with
            shape (n_dim,).
        """
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of spatial dimensions
        n_dim = self._specimen_data.get_n_dim()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate forces data scaler
        scaler_forces = \
            TorchMinMaxScaler(n_features=n_dim, device_type=self._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data scaler normalization factors
        scaler_forces.set_minimum_and_maximum(force_minimum, force_maximum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['forces'] = scaler_forces
    # -------------------------------------------------------------------------
    def get_fitted_data_scaler(self, features_type):
        """Get fitted model data scalers.
        
        Parameters
        ----------
        features_type : str
            Features for which data scaler is required:
            
            'forces'  : Forces

        Returns
        -------
        data_scaler : {TorchMinMaxScaler, TorchStandardScaler}
            Fitted data scaler.
        """
        # Get fitted data scaler
        if features_type not in self._data_scalers.keys():
            raise RuntimeError(f'Unknown data scaler for {features_type}.')
        elif self._data_scalers[features_type] is None:
            raise RuntimeError(f'Data scaler for {features_type} has not '
                               f'been fitted.')
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
            
            'forces'  : Forces

        mode : {'normalize', 'denormalize'}, default=normalize
            Data scaling transformation type.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Transformed features PyTorch tensor.
        """
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
    def set_material_models_fitted_data_scalers(self, models_scaling_type,
                                                models_scaling_parameters):
        """Set material constitutive models fitted data scalers.
        
        Data scalers are only fitted for material models that support data
        normalization and for which the corresponding data scaling type and
        parameters are provided.
        
        Parameters
        ----------
        models_scaling_type : dict
            Type of data scaling (str, {'min-max', 'mean-std'}) for each
            material model (key, str[int]). Models are labeled from 1 to
            n_mat_model. Min-Max scaling ('min-max') or standardization
            ('mean-std').
        models_scaling_type : dict
            Features data scaling parameters (item, dict) for each material
            model (key, str[int]), stored as data scaling parameters
            (item, tuple[2]) for each features type (key, str). Models are
            labeled from 1 to n_mat_model. Each data scaling parameter is set
            as a torch.Tensor(1d) according to the corresponding number of
            features. For 'min-max' data scaling, the parameters are the
            'minimum'[0] and 'maximum'[1] tensors, while for 'mean-std' data
            scaling the parameters are the 'mean'[0] and 'std'[1] tensors.
        """
        # Check specimen material state
        if self._specimen_data is None:
            raise RuntimeError('The specimen material data and material state '
                               'must be set prior to set the material '
                               'constitutive models fitted data scalers '
                               '(check method set_specimen_data()).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for model_key, model in self.get_material_models().items():
            # Check if material model supports data normalization
            if not hasattr(model, 'is_data_normalization'):
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check material model data normalization
            if not (model.is_data_normalization
                    and model_key in models_scaling_type.keys()):
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material model data scaling type
            scaling_type = models_scaling_type[model_key]
            # Get material model data scaling parameters
            scaling_parameters = models_scaling_parameters[model_key]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material model fitted data scalers
            model.set_fitted_data_scalers(scaling_type, scaling_parameters)
    # -------------------------------------------------------------------------
    def build_element_local_samples(self, strain_formulation, problem_type,
                                    element_type, time_hist,
                                    element_state_hist):
        """Build element Gauss integration points local strain-stress paths.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        element_type : Element
            FETorch finite element.
        time_hist : torch.Tensor(1d)
            Discrete time history.
        element_state_hist : list[dict]
            Material constitutive model state variables history (item, dict)
            for each Gauss integration point (key, str[int]).
            
        Returns
        -------
        element_local_samples : list[dict]
            Element local strain-stress paths, each corresponding to a given
            element Gauss integration point. Each path is stored as a
            dictionary where each feature (key, str) data is a torch.Tensor(2d)
            of shape (sequence_length, n_features).
        """
        # Get problem type parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain and stress components
        if strain_formulation == 'infinitesimal':
            strain_comps_order = comp_order_sym
            stress_comps_order = comp_order_sym
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element number of Gauss quadrature integration points
        n_gauss = element_type.get_n_gauss()
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element local strain-stress paths
        element_local_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Gauss integration points
        for i in range(n_gauss):
            # Initialize strain path
            strain_path = torch.zeros((n_time, len(strain_comps_order)),
                                      dtype=torch.float)
            # Initialize stress path
            stress_path = torch.zeros((n_time, len(stress_comps_order)),
                                      dtype=torch.float)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete time
            for time_idx in range(n_time):
                # Get strain tensor (matricial form)
                strain_mf = \
                    element_state_hist[time_idx][str(i + 1)]['strain_mf']
                # Get strain tensor
                strain = vget_tensor_from_mf(strain_mf, n_dim,
                                             strain_comps_order)
                # Store strain components
                strain_path[time_idx, :] = \
                    self.vstore_tensor_comps(comp_order_sym, strain)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get stress tensor (matricial form)
                stress_mf = \
                    element_state_hist[time_idx][str(i + 1)]['stress_mf']
                # Get stress tensor
                stress = vget_tensor_from_mf(stress_mf, n_dim,
                                             stress_comps_order)
                # Store stress components
                stress_path[time_idx, :] = \
                    self.vstore_tensor_comps(comp_order_sym, stress)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize material response path data
            response_path = {}
            # Assemble strain-stress material response path
            response_path['strain_comps_order'] = strain_comps_order
            response_path['strain_path'] = strain_path.detach()
            response_path['stress_comps_order'] = stress_comps_order
            response_path['stress_path'] = stress_path.detach()
            # Assemble time path
            response_path['time_hist'] = time_hist.detach().reshape(-1, 1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material response path
            element_local_samples.append(response_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_local_samples
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
    def vforward_sequential_element(self):
        """Forward propagation (sequential element).
        
        Compatible with vectorized mapping.
        
        Returns
        -------
        force_equilibrium_hist_loss : torch.Tensor(0d)
            Force equilibrium history loss.
        """
         # Get specimen numerical data
        specimen_data = self._specimen_data
         # Get specimen material state
        specimen_material_state = self._specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain formulation and problem type
        strain_formulation = specimen_material_state.get_strain_formulation()
        problem_type = specimen_material_state.get_problem_type()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get specimen finite element mesh
        specimen_mesh = specimen_data.specimen_mesh
        # Get elements type
        elements_type = specimen_mesh.get_elements_type()
        # Get element type
        if specimen_mesh.get_n_element_type() > 1:
            raise RuntimeError('Vectorized forward propagation requires that '
                               'all the elements share the same element type.')
        else:
            # Get unique element type
            element_type = elements_type['1']
            # Get number of degrees of freedom per node
            n_dof_node = element_type.get_n_dof_node()
        # Get number of spatial dimensions
        n_dim = specimen_mesh.get_n_dim()
        # Get number of nodes of finite element mesh
        n_node_mesh = specimen_mesh.get_n_node_mesh()
        # Build elements nodes degrees of freedom mesh indexes
        elements_mesh_indexes = \
            specimen_mesh.build_elements_mesh_indexing(n_dof_node)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get elements material
        elements_material = specimen_material_state.get_elements_material()
        # Get element constitutive material model
        if specimen_material_state.get_n_element_material_type() > 1:
            raise RuntimeError('Vectorized forward propagation requires that '
                               'all the elements share the same material.')  
        else:
            # Get unique constitutive material model
            element_material = elements_material['1']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get batched finite element mesh configuration history
        elements_coords_hist, elements_disps_hist = \
            specimen_data.get_batched_mesh_configuration_hist(
                is_update_coords=strain_formulation != 'infinitesimal',
                device=self._device)
        # Get time history
        time_hist = specimen_data.time_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute history of finite element mesh internal forces
        elements_internal_forces_hist, _ = \
            self.vcompute_elements_internal_forces_hist(
                strain_formulation, problem_type, element_type,
                element_material, elements_coords_hist, elements_disps_hist,
                time_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build internal forces history of finite element mesh nodes
        internal_forces_mesh_hist = self.vbuild_internal_forces_mesh_hist(
            elements_internal_forces_hist, elements_mesh_indexes, n_node_mesh,
            n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set null external forces of finite element mesh nodes
        external_forces_mesh_hist = torch.zeros_like(internal_forces_mesh_hist)
        # Get reaction forces (Dirichlet boundary conditions) history of finite
        # element mesh nodes
        reaction_forces_mesh_hist = \
            specimen_data.reaction_forces_mesh_hist.to(self._device)
        # Get degrees of freedom subject to Dirichlet boundary conditions
        dirichlet_bool_mesh = \
            specimen_mesh.get_dirichlet_bool_mesh().to(self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium history loss
        force_equilibrium_hist_loss = \
            self.vforce_equilibrium_hist_loss(internal_forces_mesh_hist,
                                              external_forces_mesh_hist,
                                              reaction_forces_mesh_hist,
                                              dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def vcompute_elements_internal_forces_hist(self, strain_formulation,
        problem_type, element_type, element_material, elements_coords_hist,
        elements_disps_hist, time_hist):
        """Compute history of finite elements internal forces.
        
        Compatible with vectorized mapping.
        
        Vectorization constraints require that all the elements share the same
        element type (FETorch finite element) and share the same material
        constitutive model (FETorch material constitutive model).
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        element_type : Element
            FETorch finite element.
        element_material : ConstitutiveModel
            FETorch material constitutive model.
        elements_coords_hist : torch.Tensor(4d)
            Coordinates history of finite elements nodes stored as
            torch.Tensor(4d) of shape (n_elem, n_node, n_dim, n_time).
        elements_disps_hist : torch.Tensor(4d)
            Displacements history of finite elements nodes stored as
            torch.Tensor(4d) of shape (n_elem, n_node, n_dim, n_time).
        time_hist : torch.Tensor(1d)
            Discrete time history.
        
        Returns
        -------
        elements_internal_forces_hist : torch.Tensor(3d)
            Internal forces history of finite elements nodes stored as
            torch.Tensor(3d) of shape (n_elem, n_node*n_dim, n_time).
        elements_state_hist : torch.Tensor(4d)
            Gauss integration points strain and stress path history of finite
            elements stored as torch.Tensor(4d) of shape
            (n_elem, n_gauss, n_time, n_strain_comps + n_stress_comps).
        """
        # Set vectorized element internal forces history computation (batch
        # along element)
        vmap_compute_element_internal_forces_hist = torch.vmap(
            self.vcompute_element_internal_forces_hist,
            in_dims=(0, 0, None, None, None, None, None),
            out_dims=(0, 0))
        # Compute elements internal forces history
        elements_internal_forces_hist, elements_state_hist = \
            vmap_compute_element_internal_forces_hist(
                elements_coords_hist, elements_disps_hist, strain_formulation,
                problem_type, element_type, element_material, time_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check elements internal forces history
        if torch.isnan(elements_internal_forces_hist).any():
            raise RuntimeError('NaNs were detected in the tensor storing the '
                               'elements internal forces history. This may '
                               'have resulted from a state update convergence '
                               'failure.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elements_internal_forces_hist, elements_state_hist
    # -------------------------------------------------------------------------
    def vcompute_element_internal_forces_hist(
        self, nodes_coords_hist, nodes_disps_hist, strain_formulation,
        problem_type, element_type, element_material, time_hist):
        """Compute history of finite element internal forces.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        nodes_coords_hist : torch.Tensor(3d)
            Coordinates history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        nodes_disps_hist : torch.Tensor(3d)
            Displacements history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        element_type : Element
            FETorch finite element.
        element_material : ConstitutiveModel
            FETorch material constitutive model.
        time_hist : torch.Tensor(1d)
            Discrete time history.

        Returns
        -------
        element_internal_forces_hist : torch.Tensor(2d)
            Element internal forces history stored as torch.Tensor(2d) of shape
            (n_node*n_dim, n_time).
        element_state_hist : torch.Tensor(3d)
            Element Gauss integration points strain and stress path history
            stored as torch.Tensor(3d) of shape
            (n_gauss, n_time, n_strain_comps + n_stress_comps).
        """
        # Get element Gauss quadrature integration points local coordinates
        # and weights
        gp_coords_tensor, gp_weights_tensor = \
            element_type.get_batched_gauss_integration_points(
                device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized strain computation (batch along Gauss integration
        # points)
        vmap_compute_local_internal_forces_hist = torch.vmap(
            self.vcompute_local_internal_forces_hist,
            in_dims=(0, 0, None, None, None, None, None, None, None),
            out_dims=(0, 0))
        # Compute Gauss integration points contribution history to element
        # internal forces
        gps_local_internal_forces_hist, element_state_hist = \
            vmap_compute_local_internal_forces_hist(
                gp_coords_tensor, gp_weights_tensor, strain_formulation,
                problem_type, element_type, nodes_coords_hist,
                nodes_disps_hist, time_hist, element_material)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute element internal forces history
        element_internal_forces_hist = \
            torch.sum(gps_local_internal_forces_hist, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_internal_forces_hist, element_state_hist
    # -------------------------------------------------------------------------
    def vcompute_local_internal_forces_hist(
        self, local_coords, weight, strain_formulation, problem_type,
        element_type, nodes_coords_hist, nodes_disps_hist, time_hist,
        element_material):
        """Compute local integration point internal force contribution history.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        local_coords : torch.Tensor(1d)
            Local integration point coordinates.
        weight : torch.Tensor(0d)
            Local integration point weight.
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        element_type : Element
            FETorch finite element.
        nodes_coords_hist : torch.Tensor(3d)
            Coordinates history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        nodes_disps_hist : torch.Tensor(3d)
            Displacements history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        time_hist : torch.Tensor(1d)
            Discrete time history.
        element_material : ConstitutiveModel
            FETorch material constitutive model.
            
        Returns
        -------
        local_internal_forces_hist : torch.Tensor(2d)
            Local integration point contribution history to finite element
            internal forces stored as torch.Tensor(2d) of
            shape (n_node*n_dim, n_time).
        local_state_variables_hist : torch.Tensor(2d)
            Local integration point strain and stress path history stored as
            torch.Tensor(2d) of
            shape (n_time, n_strain_comps + n_stress_comps).
        """
        # Get problem type parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized strain computation (batch along time)
        vmap_compute_local_strain = \
            torch.vmap(self.vcompute_local_strain,
                       in_dims=(2, 2, None, None, None, None, None),
                       out_dims=(0, 0, 0))
        # Compute Gauss integration point strain history
        if strain_formulation == 'infinitesimal':
            # Compute infinitesimal strain tensor history
            strain_hist, jacobian_det_hist, grad_operator_sym_hist = \
                vmap_compute_local_strain(nodes_coords_hist, nodes_disps_hist,
                                          local_coords, strain_formulation,
                                          n_dim, comp_order_sym, element_type)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Material state update
        state_variables_hist = self.vrecurrent_material_state_update(
            strain_formulation, problem_type, element_material,
            strain_hist, time_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set stress components index range
        if strain_formulation == 'infinitesimal':
            stress_indexes = \
                torch.arange(len(comp_order_sym), 2*len(comp_order_sym))
        else:
            raise RuntimeError('Not implemented.')
        # Extract stress tensor history
        stress_vmf_hist = state_variables_hist[:, stress_indexes]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized internal force computation (batch along time)
        vmap_compute_local_internal_forces = \
            torch.vmap(self.vcompute_local_internal_forces,
                       in_dims=(0, 0, 0, None), out_dims=(1,))
        # Compute Gauss integration point contribution history to element
        # internal forces
        local_internal_forces_hist = vmap_compute_local_internal_forces(
            stress_vmf_hist, grad_operator_sym_hist, jacobian_det_hist, weight)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return local_internal_forces_hist, state_variables_hist
    # -------------------------------------------------------------------------
    def vcompute_local_strain(self, nodes_coords, nodes_disps, local_coords,
                              strain_formulation, n_dim, comp_order,
                              element_type):
        """Compute strain tensor at given local point of element.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        nodes_coords : torch.Tensor(2d)
            Nodes coordinates stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        nodes_disps : torch.Tensor(2d)
            Nodes displacements stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        local_coords : torch.Tensor(1d)
            Local coordinates of point where strain is computed.
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        n_dim : int
            Number of spatial dimensions.
        comp_order : tuple
            Strain/Stress components order associated to matricial form.
        element_type : Element
            FETorch finite element.
            
        Returns
        -------
        strain : torch.Tensor(2d)
            Strain tensor at given local coordinates.
        jacobian_det : torch.Tensor(0d)
            Determinant of element Jacobian at given local coordinates.
        grad_operator_sym : torch.Tensor(2d)
            Discrete symmetric gradient operator evaluated at given local
            coordinates.
        """
        # Evaluate shape functions derivates and Jacobian
        shape_fun_deriv, _, jacobian_det = \
            eval_shapefun_deriv(element_type, nodes_coords, local_coords)
        # Build discrete symmetric gradient operator
        grad_operator_sym = vbuild_discrete_sym_gradient(shape_fun_deriv,
                                                         comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute strain tensor
        if strain_formulation == 'infinitesimal':
            # Compute infinitesimal strain tensor (Voigt matricial form)
            strain_vmf = \
                compute_infinitesimal_strain(grad_operator_sym, nodes_disps)
            # Get strain tensor
            strain = vget_strain_from_vmf(strain_vmf, n_dim, comp_order)
        else:
            raise RuntimeError('Not implemented.')      
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain, jacobian_det, grad_operator_sym
    # -------------------------------------------------------------------------
    def vrecurrent_material_state_update(self, strain_formulation,
                                         problem_type, constitutive_model,
                                         strain_hist, time_hist):
        """Material state update for recurrent constitutive model.

        Compatible with vectorized mapping.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        constitutive_model : ConstitutiveModel
            Recurrent material constitutive model.
        strain_hist : torch.Tensor(3d)
            Strain tensor history stored as torch.Tensor(3d) of shape
            (n_time, n_dim, n_dim).
        time_hist : torch.Tensor(1d)
            Discrete time history.

        Returns
        -------
        state_variables_hist : torch.Tensor(2d)
            Strain and stress path history stored as torch.Tensor(2d) of shape
            (n_time, n_strain_comps + n_stress_comps).
        """
        # Get problem type parameters
        _, comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input features data
        features_in_data = \
            [self.vstore_tensor_comps(comp_order_sym, strain_hist[t, :, :],
                                      device=self._device)
             for t in range(n_time)]
        # Build input features tensor
        features_in = torch.stack(features_in_data, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute output features
        features_out = constitutive_model(features_in, is_normalized=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build state path history
        if strain_formulation == 'infinitesimal':
            state_variables_hist = torch.cat(
                (features_in[:, 0:len(comp_order_sym)],
                 features_out[:, 0:len(comp_order_sym)]), dim=1)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables_hist
    # -------------------------------------------------------------------------
    def vcompute_local_internal_forces(self, stress_vmf, grad_operator_sym,
                                       jacobian_det, weight):
        """Compute local integration point internal forces contribution.
        
        Compatible with vectorized mapping.
        
        Internal forces are computed in the spatial configuration, i.e., based
        on the discrete symmetric gradient operator and the Cauchy stress
        tensor.
        
        Parameters
        ----------
        stress_vmf : torch.Tensor(1d)
            Cauchy stress tensor stored in Voigt matricial form.
        grad_operator_sym : torch.Tensor(2d)
            Discrete symmetric gradient operator evaluated at given local
            coordinates.
        jacobian_det : torch.Tensor(0d)
            Determinant of element jacobian evaluated at given local
            coordinates.
        weight : torch.Tensor(0d)
            Local integration point weight.

        Returns
        -------
        internal_forces : torch.Tensor(1d)
            Integration point contribution to element internal forces.
        """
        # Compute local integration point contribution to element internal
        # forces
        internal_forces = weight*torch.matmul(grad_operator_sym.t(),
                                              stress_vmf)*jacobian_det
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return internal_forces
    # -------------------------------------------------------------------------
    def vbuild_internal_forces_mesh_hist(self, elements_internal_forces_hist,
                                         elements_mesh_indexes, n_node_mesh,
                                         n_dim):
        """Build internal forces history of finite element mesh.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        elements_internal_forces_hist : torch.Tensor(3d)
            Internal forces history of finite elements nodes stored as
            torch.Tensor(3d) of shape (n_elem, n_node*n_dim, n_time).
        elements_mesh_indexes : torch.Tensor(2d)
            Elements nodes degrees of freedom mesh indexes stored as
            torch.Tensor(2d) of shape (n_elem, n_node*n_dof_node).
        n_node_mesh : int
            Number of nodes of finite element mesh.
        n_dim : int
            Number of spatial dimensions.
        
        Returns
        -------
        internal_forces_mesh_hist : torch.Tensor(3d)
            Internal forces history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        """
        # Set vectorized internal forces assembly (batch along time)
        vmap_assemble_internal_forces = \
            torch.vmap(self.vassemble_internal_forces,
                       in_dims=(2, None, None, None), out_dims=(2,))
        # Compute internal forces history
        internal_forces_mesh_hist = vmap_assemble_internal_forces(
            elements_internal_forces_hist, elements_mesh_indexes, n_node_mesh,
            n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return internal_forces_mesh_hist
    # -------------------------------------------------------------------------
    def vassemble_internal_forces(self, elements_internal_forces,
                                  elements_mesh_indexes, n_node_mesh, n_dim):
        """Assemble element internal forces into mesh counterpart.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        elements_internal_forces : torch.Tensor(2d)
            Internal forces of finite elements nodes stored as
            torch.Tensor(2d) of shape (n_elem, n_node*n_dim).
        elements_mesh_indexes : torch.Tensor(2d)
            Elements nodes degrees of freedom mesh indexes stored as
            torch.Tensor(2d) of shape (n_elem, n_node*n_dof_node).
        n_node_mesh : int
            Number of nodes of finite element mesh.
        n_dim : int
            Number of spatial dimensions.

        Returns
        -------
        internal_forces_mesh : torch.Tensor(2d)
            Internal forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        """
        # Assemble internal forces of finite element mesh
        internal_forces_mesh = torch.stack(
            [elements_internal_forces[elements_mesh_indexes == index].sum()
             for index in range(n_node_mesh*n_dim)]).view(n_node_mesh, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return internal_forces_mesh
    # -------------------------------------------------------------------------
    def vforce_equilibrium_hist_loss(self, internal_forces_mesh_hist,
                                     external_forces_mesh_hist,
                                     reaction_forces_mesh_hist,
                                     dirichlet_bool_mesh):
        """Compute force equilibrium history loss.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        internal_forces_mesh_hist : torch.Tensor(3d)
            Internal forces history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        external_forces_mesh_hist : torch.Tensor(3d)
            External forces history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        reaction_forces_mesh_hist : torch.Tensor(3d)
            Reaction forces (Dirichlet boundary conditions) history of finite
            element mesh nodes stored as torch.Tensor(3d) of shape
            (n_node_mesh, n_dim, n_time).
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
            
        Returns
        -------
        force_equilibrium_hist_loss : torch.Tensor(0d)
            Force equilibrium history loss.
        """
        # Set vectorized force equilibrium history loss computation (batch
        # along time)
        vmap_force_equilibrium_loss = \
            torch.vmap(self.vforce_equilibrium_loss,
                       in_dims=(2, 2, 2, None), out_dims=(0,))
        # Compute force equilibrium history loss
        force_equilibrium_hist_loss = torch.sum(
            vmap_force_equilibrium_loss(internal_forces_mesh_hist,
                                        external_forces_mesh_hist,
                                        reaction_forces_mesh_hist,
                                        dirichlet_bool_mesh))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def vforce_equilibrium_loss(self, internal_forces_mesh,
                                external_forces_mesh, reaction_forces_mesh,
                                dirichlet_bool_mesh):
        """Compute force equilibrium loss.
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        internal_forces_mesh : torch.Tensor(2d)
            Internal forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        external_forces_mesh : torch.Tensor(2d)
            External forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        reaction_forces_mesh : torch.Tensor(2d)
            Reaction forces (Dirichlet boundary conditions) of finite element
            mesh nodes stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim).
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
            
        Returns
        -------
        force_equilibrium_loss : torch.Tensor(0d)
            Force equilibrium loss.
        """
        # Normalize forces
        if self._is_force_normalization:
            # Normalize internal forces
            internal_forces_mesh = self.data_scaler_transform(
                internal_forces_mesh, features_type='forces', mode='normalize')
            # Normalize external forces
            external_forces_mesh = self.data_scaler_transform(
                external_forces_mesh, features_type='forces', mode='normalize')
            # Normalize reaction forces
            reaction_forces_mesh = self.data_scaler_transform(
                reaction_forces_mesh, features_type='forces', mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium loss
        force_equilibrium_loss = \
            torch.sum((internal_forces_mesh - external_forces_mesh
                       - torch.where(dirichlet_bool_mesh == 1,
                                     reaction_forces_mesh, 0.0))**2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_loss
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