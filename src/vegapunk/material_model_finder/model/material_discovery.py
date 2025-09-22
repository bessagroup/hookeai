"""Material model finder forward model.

Classes
-------
MaterialModelFinder(torch.nn.Module)
    Material model finder forward model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
import itertools
import pickle
import warnings
# Third-party
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt
# Local
from model_architectures.hybrid_base_model.model.hybrid_model import \
    HybridModel
from simulators.fetorch.element.integrations.internal_forces import \
    compute_element_internal_forces, compute_infinitesimal_inc_strain, \
    compute_infinitesimal_strain
from simulators.fetorch.element.derivatives.gradients import \
    eval_shapefun_deriv, vbuild_discrete_sym_gradient, \
    vbuild_discrete_gradient, vexpand_grad_operator_sym_2d_to_3d, \
    vreduce_grad_operator_sym_3d_to_2d
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf
from simulators.fetorch.math.voigt_notation import vget_stress_vmf, \
    vget_strain_from_vmf, get_projection_tensors_vmf
from utilities.data_scalers import TorchMinMaxScaler
from model_architectures.procedures.model_data_scaling import \
    set_fitted_data_scalers, data_scaler_transform
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset
from ioput.iostandard import make_directory
from ioput.plots import plot_xy_data, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class MaterialModelFinder(torch.nn.Module):
    """Material model finder forward model.
    
    Attributes
    ----------
    _specimen_data : SpecimenNumericalData
        Specimen numerical data translated from experimental results.
    _specimen_material_state : StructureMaterialState
        FETorch specimen material state.
    _force_equilibrium_loss_type : str
        Type of force equilibrium loss:
        
        'pointwise'      : Force equilibrium strictly based on pointwise
                           internal, external and reaction forces.

        'dirichlet_sets' : Force equilibrium (1) based on pointwise
                           internal and external forces (non-Dirichlet
                           degrees of freedom) and (2) based on pointwise
                           internal, external and set-based reaction forces
                           (Dirichlet constrained degrees of freedom).
    _is_force_normalization : bool, default=False
        If True, then normalize forces prior to the computation of the force
        equilibrium loss.
    _data_scalers : dict
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str).
    _loss_scaling_factor : torch.Tensor(0d)
        Loss scaling factor. If provided, then loss is pre-multiplied by
        loss scaling factor.
    _loss_time_weights : torch.Tensor(1d), default=None
        Loss time weights stored as torch.Tensor(1d) of shape (n_time).
        If provided, then each discrete time loss contribution is
        pre-multiplied by corresponding weight. If None, time weights are
        set to 1.0.
    _is_store_force_equilibrium_loss_hist : bool
        If True, then store force equilibrium loss components history.
    _is_store_local_paths : bool
        If True, then store data set of specimen local (Gauss integration
        points) strain-stress paths in dedicated model subdirectory.
        Overwrites existing data set.
    _local_paths_elements : list[int]
        Elements for which local (Gauss integration points) strain-stress
        paths are stored as part of the specimen local data set. Elements
        are labeled from 1 to n_elem. If None, then all elements are
        stored. Only effective if is_store_local_paths=True.
    _is_compute_sets_reaction_hist : bool
        If True, then compute reaction forces history of Dirichlet boundary
        sets. Only available for 'dirichlet_sets' force equilibrium loss type.
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
    check_force_equilibrium_loss_type(cls, force_equilibrium_loss_type)
        Check if force equilibrium loss type is available.
    set_specimen_data(self, specimen_data, specimen_material_state,
                      force_minimum=None, force_maximum=None)
        Set specimen data and material state.
    get_detached_model_parameters(self)
        Get model parameters (material models) detached of gradients.
    get_model_parameters_bounds(self)
        Get model parameters (material models) bounds.
    enforce_parameters_bounds(self)
        Enforce bounds in model parameters (material models).
    enforce_parameters_constraints(self)
        Enforce material model-dependent parameters constraints.
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
    build_element_local_samples(self, strain_formulation, problem_type, \
                                element_type, time_hist, element_state_hist)
        Build element Gauss integration points local strain-stress paths.
    compute_dirichlet_sets_reaction_hist(self, dirichlet_bc_mesh_hist, \
                                          dirichlet_bool_mesh_hist)
        Compute Dirichlet boundary sets reaction forces history.
    compute_dirichlet_sets_reaction(self, internal_forces_mesh, \
                                    external_forces_mesh, dirichlet_bc_mesh)
        Compute reaction forces of Dirichlet boundary sets.
    store_dirichlet_sets_reaction_hist(self, dirichlet_sets_reaction_hist, \
                                       is_plot=True)
        Store reaction forces history of Dirichlet boundary sets.
    build_tensor_from_comps(cls, n_dim, comps, comps_array, \
                            is_symmetric=False, device=None)
        Build strain/stress tensor from given components.
    store_tensor_comps(cls, comps, tensor, device=None)
        Store strain/stress tensor components in array.
    vforward_sequential_element(self)
        Forward propagation (sequential element).
    vcompute_elements_internal_forces_hist(self, strain_formulation, \
                                           problem_type, element_type, \
                                           element_material, \
                                           elements_coords_hist, \
                                           elements_disps_hist, time_hist)
        Compute history of finite elements internal forces.
    vcompute_element_internal_forces_hist(self, nodes_coords_hist, \
                                          nodes_disps_hist, \
                                          strain_formulation, \
                                          problem_type, element_type, \
                                          element_material, time_hist)
        Compute history of finite element internal forces.
    vcompute_element_vol_grad_hist(self, nodes_coords_hist, nodes_disps_hist, \
                                   strain_formulation, problem_type, \
                                   element_type, time_hist)
        Compute history of finite element volumetric gradient operator.
    vcompute_local_vol_grad_operator_hist(self, local_coords, weight, \
                                          strain_formulation, problem_type, \
                                          element_type, nodes_coords_hist, \
                                          nodes_disps_hist, time_hist)
        Compute local integration point gradient contribution history.
    vcompute_local_gradient(self, nodes_coords, local_coords, comp_order, \
                            element_type, is_symmetric=True)
        Compute discrete gradient operator at given local point of element.
    vcompute_local_vol_sym_gradient(self, grad_operator_sym, n_dim)
        Compute discrete volumetric symmetric gradient operator.
    vcompute_local_internal_forces_hist(self, local_coords, weight, \
                                        strain_formulation, problem_type, \
                                        element_type, nodes_coords_hist, \
                                        nodes_disps_hist, time_hist, \
                                        element_material, \
                                        is_volumetric_bar=False, \
                                        avg_vol_grad_operator_hist=None)
        Compute local integration point internal force contribution history.
    vcompute_local_strain(self, nodes_coords, nodes_disps, local_coords, \
                          strain_formulation, n_dim, comp_order, element_type)
        Compute strain tensor at given local point of element.
    vcompute_local_strain_vbar(self, nodes_coords, nodes_disps, \
                               avg_vol_grad_operator, local_coords, \
                               strain_formulation, n_dim, comp_order, \
                               element_type)
        Compute strain tensor at given local point of element.
    vcompute_local_dev_sym_gradient(self, grad_operator_sym, n_dim)
        Compute discrete deviatoric symmetric gradient operator.
    vrecurrent_material_state_update(self, strain_formulation, problem_type, \
                                     constitutive_model, strain_hist, \
                                     time_hist)
        Material state update for recurrent constitutive model.
    vcompute_local_internal_forces(self, stress_vmf, grad_operator_sym, \
                                   jacobian_det, weight)
        Compute local integration point internal forces contribution.
    vbuild_internal_forces_mesh_hist(self, elements_internal_forces_hist, \
                                     elements_mesh_indexes, n_node_mesh, n_dim)
        Build internal forces history of finite element mesh.
    vassemble_internal_forces(self, elements_internal_forces, \
                              elements_mesh_indexes, n_node_mesh, n_dim)
        Assemble element internal forces into mesh counterpart.
    vforce_equilibrium_hist_loss(self, internal_forces_mesh_hist, \
                                 external_forces_mesh_hist, \
                                 reaction_forces_mesh_hist, \
                                 dirichlet_bc_mesh_hist)
        Compute force equilibrium history loss.
    vforce_equilibrium_loss(self, internal_forces_mesh, external_forces_mesh, \
                            reaction_forces_mesh, dirichlet_bc_mesh)
        Compute force equilibrium loss.
    force_equilibrium_loss_components_hist(self, internal_forces_mesh_hist, \
                                           external_forces_mesh_hist, \
                                           reaction_forces_mesh_hist, \
                                           dirichlet_bc_mesh_hist)
        Compute force equilibrium loss components history (output purposes).
    store_force_equilibrium_loss_components_hist( \
        self, force_equilibrium_loss_components_hist, is_plot=True)
        Store force equilibrium loss components history.
    build_elements_local_samples(self, strain_formulation, problem_type, \
                                 time_hist, elements_state_hist)
        Build elements local strain-stress paths.
    compute_dirichlet_sets_reaction_hist(self, internal_forces_mesh_hist, \
                                         external_forces_mesh_hist, \
                                         dirichlet_bc_mesh_hist)
        Compute reaction forces history of Dirichlet boundary sets.
    compute_dirichlet_sets_reaction(self, internal_forces_mesh, \
                                    external_forces_mesh, \
                                    dirichlet_bc_mesh)
        Compute reaction forces of Dirichlet boundary sets.
    store_dirichlet_sets_reaction_hist(self, dirichlet_sets_reaction_hist, \
                                       is_export_csv=True, is_plot=True)
        Store reaction forces history of Dirichlet boundary sets.
    vbuild_tensor_from_comps(cls, n_dim, comps, comps_array, device=None)
        Build strain/stress tensor from given components.
    vstore_tensor_comps(cls, comps, tensor, device=None)
        Store strain/stress tensor components in array.
    features_out_extractor(cls, model_output)
        Extract output features from generic model output.
    _init_data_scalers(self)
        Initialize model data scalers.
    set_fitted_force_data_scalers(self, force_minimum, force_maximum)
        Set fitted forces data scalers.
    set_material_models_fitted_data_scalers(self, models_scaling_type, \
                                            models_scaling_parameters)
    check_model_in_normalized(cls, model)
        Check if generic model expects normalized input features.
    check_model_out_normalized(cls, model)
        Check if generic model expects normalized output features.
    """
    def __init__(self, model_directory, model_name='material_model_finder',
                 force_equilibrium_loss_type='pointwise',
                 is_force_normalization=False,
                 is_store_force_equilibrium_loss_hist=False,
                 is_store_local_paths=False, local_paths_elements=None,
                 is_compute_sets_reaction_hist=False,
                 is_detect_autograd_anomaly=False,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        model_directory : str
            Directory where model is stored.
        model_name : str, default='material_model_finder'
            Name of model.
        force_equilibrium_loss_type : str, default='pointwise'
            Type of force equilibrium loss:
            
            'pointwise'      : Force equilibrium strictly based on pointwise
                               internal, external and reaction forces.

            'dirichlet_sets' : Force equilibrium (1) based on pointwise
                               internal and external forces (non-Dirichlet
                               degrees of freedom) and (2) based on pointwise
                               internal, external and set-based reaction forces
                               (Dirichlet constrained degrees of freedom).
            
        is_force_normalization : bool, default=False
            If True, then normalize forces prior to the computation of the
            force equilibrium loss.
        is_store_force_equilibrium_loss_hist : bool, default=False
            If True, then store force equilibrium loss components history.
        is_store_local_paths : bool, default=False
            If True, then store data set of specimen local (Gauss integration
            points) strain-stress paths in dedicated model subdirectory.
            Overwrites existing data set.
        local_paths_elements : list[int], default=None
            Elements for which local (Gauss integration points) strain-stress
            paths are stored as part of the specimen local data set. Elements
            are labeled from 1 to n_elem. If None, then all elements are
            stored. Only effective if is_store_local_paths=True.
        is_compute_sets_reaction_hist : bool, default=False
            If True, then compute reaction forces history of Dirichlet
            boundary sets. Only available for 'dirichlet_sets' force
            equilibrium loss type.
        is_detect_autograd_anomaly : bool, default=False
            If True, then set context-manager that enables anomaly detection
            for the autograd engine. Should only be enabled for debugging
            purposes as it degrades performance.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(MaterialModelFinder, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set autograd engine anomaly detection
        if is_detect_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)
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
        # Set force equilibrium loss type
        self._force_equilibrium_loss_type = \
            self.check_force_equilibrium_loss_type(force_equilibrium_loss_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set storage of force equilibrium loss components history
        self._is_store_force_equilibrium_loss_hist = \
            is_store_force_equilibrium_loss_hist
        # Set storage of specimen local strain-stress paths
        self._is_store_local_paths = is_store_local_paths
        # Set elements of specimen local strain-stress data set
        self._local_paths_elements = local_paths_elements
        # Set computation of Dirichlet boundary sets reaction forces
        if (is_compute_sets_reaction_hist
                and self._force_equilibrium_loss_type != 'dirichlet_sets'):
            warnings.warn('The computation of Dirichlet boundary sets '
                          'reaction forces is only available for the '
                          '\'dirichlet_sets\' force equilibrium loss type.',
                          category=UserWarning)
            self._is_compute_sets_reaction_hist = False
        else:
            self._is_compute_sets_reaction_hist = is_compute_sets_reaction_hist
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
    @classmethod
    def check_force_equilibrium_loss_type(cls, force_equilibrium_loss_type):
        """Check if force equilibrium loss type is available.
        
        Parameters
        ----------
        force_equilibrium_loss_type : str
            Type of force equilibrium loss.
        
        Returns
        -------
        force_equilibrium_loss_type : str
            Type of force equilibrium loss.
        """
        # Set available force equilibrium loss types
        available_force_equilibrium_loss_types = \
            ('pointwise', 'dirichlet_sets')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check force equilibrium loss type
        if (force_equilibrium_loss_type not in
                available_force_equilibrium_loss_types):
            raise RuntimeError(f'Invalid force equilibrium loss type: '
                               f'{force_equilibrium_loss_type}. \n\n'
                               f'Available types are: '
                               f'{available_force_equilibrium_loss_types}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_loss_type
    # -------------------------------------------------------------------------
    def set_specimen_data(self, specimen_data, specimen_material_state,
                          force_minimum=None, force_maximum=None,
                          loss_scaling_factor=None, loss_time_weights=None):
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
        force_maximum : torch.Tensor(1d), default=None
            Forces normalization maximum tensor stored as a torch.Tensor with
            shape (n_dim,). Only required if force normalization is set to
            True, otherwise ignored.
        loss_scaling_factor : torch.Tensor(0d), default=None
            Loss scaling factor. If provided, then loss is pre-multiplied by
            loss scaling factor.
        loss_time_weights : torch.Tensor(1d), default=None
            Loss time weights stored as torch.Tensor(1d) of shape (n_time).
            If provided, then each discrete time loss contribution is
            pre-multiplied by corresponding weight. If None, time weights are
            set to 1.0.
        """
        # Check Dirichlet boundary constraints admissibility
        specimen_data.check_dirichlet_bc_mesh_hist(
            self._force_equilibrium_loss_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen numerical data
        self._specimen_data = specimen_data
        # Set specimen material state (material models parameters linkage)
        self._specimen_material_state = specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material models directories
        self._set_material_models_dirs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit force data scalers
        if self._is_force_normalization:
            # Check forces normalization minimum and maximum tensors
            if force_minimum is None or force_maximum is None:
                raise RuntimeError('Forces normalization minimum and maximum '
                                   'tensors must be provided to perform '
                                   'force normalization.')
            # Set fitted force data scalers
            self.set_fitted_force_data_scalers(force_minimum, force_maximum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set loss scaling factor
        self._loss_scaling_factor = loss_scaling_factor
        # Check loss time weights
        if isinstance(loss_time_weights, torch.Tensor):
            # Get number of time steps
            n_time = len(self._specimen_data.time_hist)
            # Check loss time weights
            if len(loss_time_weights.shape) != 1:
                raise RuntimeError('Loss time weights must be provided as '
                                   'torch.Tensor(1d) of shape (n_time).')
            elif (loss_time_weights.shape[0]
                  != len(self._specimen_data.time_hist)):
                raise RuntimeError(f'Loss time weights must be provided as '
                                   f'torch.Tensor(1d) of shape (n_time), '
                                   f'where n_time = {n_time} (got '
                                   f'{loss_time_weights.shape[0]}).')
            # Set loss time weights
            self._loss_time_weights = loss_time_weights
        else:
            # Get number of time steps
            n_time = len(self._specimen_data.time_hist)
            # Set loss time weights
            self._loss_time_weights = torch.ones(n_time, device=self._device)
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for model_key, model in material_models.items():
            # Collect material models explicit learnable parameters
            if isinstance(model, HybridModel):
                # Get hybridized material models names
                submodels_names = model.get_hybridized_models_names()
                # Get hybridized material models
                submodels = model.get_hybridized_models()
                # Loop over material submodels
                for submodel_name, submodel in \
                        list(zip(submodels_names, submodels)):
                    # Check if submodel parameters are collected
                    is_collect_params = \
                        (hasattr(submodel, 'is_explicit_parameters')
                         and submodel.is_explicit_parameters)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip submodel parameters
                    if not is_collect_params:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get detached submodel parameters
                    detached_parameters = \
                        submodel.get_detached_model_parameters(
                            is_normalized_out=False)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Collect parameters (prefix with model and submodel label)
                    for param, value in detached_parameters.items():
                        # Set parameter label
                        param_label = \
                            f'model_{model_key}_{submodel_name}_{param}'
                        # Store parameter
                        model_parameters[param_label] = value  
            else:
                # Check if model parameters are collected
                is_collect_params = (hasattr(model, 'is_explicit_parameters')
                                     and model.is_explicit_parameters)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Skip model parameters
                if not is_collect_params:
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get detached model parameters
                detached_parameters = model.get_detached_model_parameters(
                    is_normalized_out=False)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Collect parameters (prefix with model label)
                for param, value in detached_parameters.items():
                    # Set parameter label
                    param_label = f'model_{model_key}_{param}'
                    # Store parameter
                    model_parameters[param_label] = value
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for model_key, model in material_models.items():
            # Collect material models explicit learnable parameters
            if isinstance(model, HybridModel):
                # Get hybridized material models names
                submodels_names = model.get_hybridized_models_names()
                # Get hybridized material models
                submodels = model.get_hybridized_models()
                # Loop over material submodels
                for submodel_name, submodel in \
                        list(zip(submodels_names, submodels)):
                    # Check if submodel parameters are collected
                    is_collect_params = \
                        (hasattr(submodel, 'is_explicit_parameters')
                         and submodel.is_explicit_parameters)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip submodel parameters
                    if not is_collect_params:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get model parameters bounds
                    parameters_bounds = submodel.get_model_parameters_bounds()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Collect parameters bounds (prefix with model and submodel
                    # label)
                    for param, bounds in parameters_bounds.items():
                        # Set parameter label
                        param_label = \
                            f'model_{model_key}_{submodel_name}_{param}'
                        # Store parameter
                        model_parameters_bounds[param_label] = bounds
            else:
                # Check if model parameters are collected
                is_collect_params = (hasattr(model, 'is_explicit_parameters')
                                     and model.is_explicit_parameters)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Skip model parameters
                if not is_collect_params:
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get model parameters bounds
                parameters_bounds = model.get_model_parameters_bounds()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Collect parameters bounds (prefix with model label)
                for param, bounds in parameters_bounds.items():
                    # Set parameter label
                    param_label = f'model_{model_key}_{param}'
                    # Store parameter
                    model_parameters_bounds[param_label] = bounds
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters_bounds
    # -------------------------------------------------------------------------
    def enforce_parameters_bounds(self):
        """Enforce bounds in model parameters (material models).
        
        Only enforces bounds in parameters from material models with explicit
        learnable parameters.
        
        Bounds are enforced by means of in-place parameters updates.
        
        """
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for _, model in material_models.items():
            # Enforce bounds in models explicit learnable parameters
            if isinstance(model, HybridModel):
                # Get hybridized material models
                submodels = model.get_hybridized_models()
                # Loop over material submodels
                for submodel in submodels:
                    # Check if submodel parameters are collected
                    is_collect_params = \
                        (hasattr(submodel, 'is_explicit_parameters')
                         and submodel.is_explicit_parameters)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip submodel parameters
                    if not is_collect_params:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get submodel parameters
                    param_dict = submodel.get_model_parameters()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over submodel parameters
                    for param in param_dict.keys():
                        # Get parameter bounds
                        if submodel.is_normalized_parameters:
                            lower_bound, upper_bound = \
                                submodel.get_model_parameters_norm_bounds(
                                    )[param]
                        else:
                            lower_bound, upper_bound = \
                                submodel.get_model_parameters_bounds()[param]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Get learnable parameter
                        value = param_dict[param]
                        # Enforce bounds
                        value.data.clamp_(lower_bound, upper_bound)
            else:
                # Check if model parameters are collected
                is_collect_params = (hasattr(model, 'is_explicit_parameters')
                                     and model.is_explicit_parameters)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Skip model parameters
                if not is_collect_params:
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material model parameters
                param_dict = model.get_model_parameters()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over material model parameters
                for param in param_dict.keys():
                    # Get parameter bounds
                    if model.is_normalized_parameters:
                        lower_bound, upper_bound = \
                            model.get_model_parameters_norm_bounds()[param]
                    else:
                        lower_bound, upper_bound = \
                            model.get_model_parameters_bounds()[param]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get learnable parameter
                    value = param_dict[param]
                    # Enforce bounds
                    value.data.clamp_(lower_bound, upper_bound)
    # -------------------------------------------------------------------------
    def enforce_parameters_constraints(self):
        """Enforce material model-dependent parameters constraints.
        
        Only enforces constraints in parameters from material models with
        explicit learnable parameters.
        
        Constraints are enforced by means of in-place parameters updates.
        
        """
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for _, model in material_models.items():
            # Enforce bounds in models explicit learnable parameters
            if isinstance(model, HybridModel):
                # Get hybridized material models
                submodels = model.get_hybridized_models()
                # Loop over material submodels
                for submodel in submodels:
                    # Check if submodel parameters are collected
                    is_collect_params = \
                        (hasattr(submodel, 'is_explicit_parameters')
                         and submodel.is_explicit_parameters)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip submodel parameters
                    if not is_collect_params:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Enforce material model-dependent parameters constraints
                    submodel.enforce_parameters_constraints()
            else:
                # Check if model parameters are collected
                is_collect_params = (hasattr(model, 'is_explicit_parameters')
                                     and model.is_explicit_parameters)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Skip model parameters
                if not is_collect_params:
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Enforce material model-dependent parameters constraints
                model.enforce_parameters_constraints()
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
        force_equilibrium_hist_loss : torch.Tensor(0d)
            Force equilibrium history loss.
        """
        # Compute force equilibrium history loss
        if sequential_mode == 'sequential_time':
            # Check device support
            if self._device_type == 'cuda':
                raise RuntimeError(
                    'FETorch standard sequential time computation does not '
                    'currently support CUDA. Please set torch device as CPU '
                    'to use \'sequential_time\' mode.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            force_equilibrium_hist_loss = self.forward_sequential_time()
        elif sequential_mode == 'sequential_element':
            force_equilibrium_hist_loss = self.forward_sequential_element()
        elif sequential_mode == 'sequential_element_vmap':
            force_equilibrium_hist_loss = self.vforward_sequential_element()
        else:
            raise RuntimeError('Unknown sequential mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply loss scaling factor
        if self._loss_scaling_factor is not None:
            force_equilibrium_hist_loss = \
                self._loss_scaling_factor*force_equilibrium_hist_loss
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_time(self):
        """Forward propagation (sequential time).

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
            # Get discrete time loss weight
            time_weight = self._loss_time_weights[time_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add contribution to force equilibrium history loss
            force_equilibrium_hist_loss += \
                time_weight*self.force_equilibrium_loss(
                    internal_forces_mesh, external_forces_mesh,
                    reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_element(self):
        """Forward propagation (sequential element).

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
        if self._is_store_local_paths:
            # Initialize data set samples
            specimen_local_samples = []
            # Set elements of specimen local strain-stress data set
            if isinstance(self._local_paths_elements, list):
                local_paths_elements = self._local_paths_elements
            else:
                local_paths_elements = [x + 1 for x in range(n_elem)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element internal forces of finite element mesh nodes
        internal_forces_mesh_hist = torch.zeros((n_node_mesh, n_dim, n_time),
                                                device=self._device)
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
                torch.zeros((n_node, n_dim, n_time), device=self._device)
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
            if self._is_store_local_paths and elem_id in local_paths_elements:
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
                specimen_data.reaction_forces_mesh_hist[:, :, time_idx].to(
                    self._device)    
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get discrete time loss weight
            time_weight = self._loss_time_weights[time_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add contribution to force equilibrium history loss
            force_equilibrium_hist_loss += \
                time_weight*self.force_equilibrium_loss(
                    internal_forces_mesh, external_forces_mesh,
                    reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store specimen local strain-stress paths data set
        if self._is_store_local_paths:
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
        element_internal_forces_hist = torch.zeros((n_node*n_dof_node, n_time),
                                                   device=self._device)
        # Initialize element material constitutive model state variables
        # history
        element_state_hist = \
            [{key: None for key in gp_coords.keys()} for _ in range(n_time)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Gauss integration points
        for i in range(n_gauss):
            # Get Gauss integration point local coordinates and weight
            local_coords = gp_coords[str(i + 1)].to(self._device)
            weight = gp_weights[str(i + 1)].to(self._device)
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
                    strain_hist = torch.zeros((n_dim, n_dim, n_time),
                                              device=self._device)
                else:
                    inc_strain_hist = torch.zeros((n_dim, n_dim, n_time),
                                                  device=self._device)
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
            features_in = torch.zeros((n_time, len(comp_order_sym)),
                                      device=self._device)
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
        features_out = constitutive_model(features_in)
        # Extract output features
        features_out = self.features_out_extractor(features_out)
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
            internal_forces_mesh = data_scaler_transform(self,
                internal_forces_mesh, features_type='forces', mode='normalize')
            # Normalize external forces
            external_forces_mesh = data_scaler_transform(self,
                external_forces_mesh, features_type='forces', mode='normalize')
            # Normalize reaction forces
            reaction_forces_mesh = data_scaler_transform(self,
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
                                      device=self._device)
            # Initialize stress path
            stress_path = torch.zeros((n_time, len(stress_comps_order)),
                                      device=self._device)
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
            response_path['strain_path'] = strain_path.detach().cpu()
            response_path['stress_comps_order'] = stress_comps_order
            response_path['stress_path'] = stress_path.detach().cpu()
            # Assemble time path
            response_path['time_hist'] = time_hist.detach().reshape(-1, 1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material response path
            element_local_samples.append(response_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_local_samples
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
        # Synchronize material model parameters with learnable parameters
        if hasattr(element_material, 'sync_material_model_parameters') \
                and callable(element_material.sync_material_model_parameters):
            element_material.sync_material_model_parameters()
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
        elements_internal_forces_hist, elements_state_hist = \
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
        # Get Dirichlet boundary constraints history
        dirichlet_bc_mesh_hist = \
            specimen_data.dirichlet_bc_mesh_hist.to(self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium history loss
        force_equilibrium_hist_loss = \
            self.vforce_equilibrium_hist_loss(internal_forces_mesh_hist,
                                              external_forces_mesh_hist,
                                              reaction_forces_mesh_hist,
                                              dirichlet_bc_mesh_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium loss components history
        if self._is_store_force_equilibrium_loss_hist:
            # Compute force equilibrium loss components history
            force_equilibrium_loss_hist = \
                self.force_equilibrium_loss_components_hist(
                    internal_forces_mesh_hist, external_forces_mesh_hist,
                    reaction_forces_mesh_hist, dirichlet_bc_mesh_hist)  
            # Store force equilibrium loss components history
            self.store_force_equilibrium_loss_components_hist(
                force_equilibrium_loss_hist, is_plot=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store specimen local strain-stress paths data set
        if self._is_store_local_paths:
            # Build elements local strain-stress paths
            specimen_local_samples = self.build_elements_local_samples(
                strain_formulation, problem_type, time_hist,
                elements_state_hist)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Compute Dirichlet boundary constrained sets reaction forces
        if (self._force_equilibrium_loss_type == 'dirichlet_sets'
                and self._is_compute_sets_reaction_hist):
            # Compute Dirichlet boundary constrained sets reaction forces
            dirichlet_sets_reaction_hist = \
                self.compute_dirichlet_sets_reaction_hist(
                    internal_forces_mesh_hist, external_forces_mesh_hist,
                    dirichlet_bc_mesh_hist)
            # Store Dirichlet boundary constrained sets reaction forces
            self.store_dirichlet_sets_reaction_hist(
                dirichlet_sets_reaction_hist, is_plot=True)
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
        # Set average volumetric strain formulation flag
        # (must be refactored as StructureMesh attribute)
        is_volumetric_bar = False
        # Compute element average volumetric gradient operator history
        if strain_formulation == 'infinitesimal' and is_volumetric_bar:
            avg_vol_grad_operator_hist = self.vcompute_element_vol_grad_hist(
                nodes_coords_hist, nodes_disps_hist, strain_formulation,
                problem_type, element_type, time_hist)
        else:
            avg_vol_grad_operator_hist = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized strain computation (batch along Gauss integration
        # points)
        vmap_compute_local_internal_forces_hist = torch.vmap(
            self.vcompute_local_internal_forces_hist,
            in_dims=(0, 0, None, None, None, None, None, None, None, None,
                     None),
            out_dims=(0, 0))
        # Compute Gauss integration points contribution history to element
        # internal forces
        gps_local_internal_forces_hist, element_state_hist = \
            vmap_compute_local_internal_forces_hist(
                gp_coords_tensor, gp_weights_tensor, strain_formulation,
                problem_type, element_type, nodes_coords_hist,
                nodes_disps_hist, time_hist, element_material,
                is_volumetric_bar, avg_vol_grad_operator_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute element internal forces history
        element_internal_forces_hist = \
            torch.sum(gps_local_internal_forces_hist, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_internal_forces_hist, element_state_hist
    # -------------------------------------------------------------------------
    def vcompute_element_vol_grad_hist(
        self, nodes_coords_hist, nodes_disps_hist, strain_formulation,
        problem_type, element_type, time_hist):
        """Compute history of finite element volumetric gradient operator.
        
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
        time_hist : torch.Tensor(1d)
            Discrete time history.
            
        Returns
        -------
        avg_vol_grad_operator_hist : torch.Tensor(3d)
            Element average volumetric gradient operator history stored as
            torch.Tensor(3d) of shape (n_time, n_strain_comp, n_node*n_dim).
        """
        # Get element Gauss quadrature integration points local coordinates
        # and weights
        gp_coords_tensor, gp_weights_tensor = \
            element_type.get_batched_gauss_integration_points(
                device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized gradient computation (batch along Gauss integration
        # points)
        vmap_compute_local_vol_grad_operator_hist = torch.vmap(
            self.vcompute_local_vol_grad_operator_hist,
            in_dims=(0, 0, None, None, None, None, None, None),
            out_dims=(0, 0))
        # Compute Gauss integration points contribution history to element
        # average volumetric gradient operator and volume history
        gps_local_vol_grad_operator_hist, gps_local_vol_hist = \
            vmap_compute_local_vol_grad_operator_hist(
                gp_coords_tensor, gp_weights_tensor, strain_formulation,
                problem_type, element_type, nodes_coords_hist,
                nodes_disps_hist, time_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute element volumetric gradient operator history
        vol_grad_operator_hist = \
            torch.sum(gps_local_vol_grad_operator_hist, dim=0)
        # Compute element volume history
        vol_hist = torch.sum(gps_local_vol_hist, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute element average volumetric gradient operator history
        avg_vol_grad_operator_hist = \
            torch.mul(1/vol_hist[:, None, None], vol_grad_operator_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return avg_vol_grad_operator_hist
    # -------------------------------------------------------------------------
    def vcompute_local_vol_grad_operator_hist(
        self, local_coords, weight, strain_formulation, problem_type,
        element_type, nodes_coords_hist, nodes_disps_hist, time_hist):
        """Compute local integration point gradient contribution history.
        
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
            
        Returns
        -------
        vol_grad_operator_hist : torch.Tensor(3d)
            Local integration point contribution history to finite element
            average volumetric gradient operator history stored as
            torch.Tensor(3d) of shape (n_time, n_strain_comp, n_node*n_dim).
        vol_hist : torch.Tensor(1d)
            Local integration point contribution history to finite element
            volume history stored as torch.Tensor(1d) of shape (n_time,).
        """
        # Get problem type parameters
        n_dim, comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized discrete gradient computation (batch along time)
        vmap_compute_local_gradient = \
            torch.vmap(self.vcompute_local_gradient,
                       in_dims=(2, None, None, None, None),
                       out_dims=(0, 0))
        # Compute Gauss integration point discrete gradient history
        if strain_formulation == 'infinitesimal':
            # Set symmetric gradient flag
            is_symmetric = True
            # Compute discrete symmetric gradient operator history
            jacobian_det_hist, grad_operator_hist = \
                vmap_compute_local_gradient(nodes_coords_hist, local_coords,
                                            comp_order_sym, element_type,
                                            is_symmetric)
            # Set vectorized volumetric gradient operator computation (batch
            # along time)
            vmap_compute_local_vol_sym_gradient = torch.vmap(
                self.vcompute_local_vol_sym_gradient,
                in_dims=(0, None), out_dims=(0,))
            # Compute volumetric gradient operator history
            vol_grad_operator_hist = vmap_compute_local_vol_sym_gradient(
                grad_operator_hist, n_dim)
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute local integration point contribution to element average
        # volumetric gradient operator history
        vol_grad_operator_hist = weight*vol_grad_operator_hist
        # Compute local integration point contribution to element volume
        vol_hist = weight*jacobian_det_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return vol_grad_operator_hist, vol_hist   
    # -------------------------------------------------------------------------
    def vcompute_local_gradient(self, nodes_coords, local_coords, comp_order,
                                element_type, is_symmetric=True):
        """Compute discrete gradient operator at given local point of element.

        Compatible with vectorized mapping.
        
        Parameters
        ----------
        nodes_coords : torch.Tensor(2d)
            Nodes coordinates stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        local_coords : torch.Tensor(1d)
            Local coordinates of point where strain is computed.
        comp_order : tuple
            Strain/Stress components order associated to matricial form.
        element_type : Element
            FETorch finite element.
        is_symmetric : bool, default=True
            If True, then compute discrete symmetric gradient operator.
            Otherwise, compute non-symmetric discrete gradient operator.
            
        Returns
        -------
        jacobian_det : torch.Tensor(0d)
            Determinant of element Jacobian at given local coordinates.
        grad_operator : torch.Tensor(2d)
            Discrete gradient operator evaluated at given local coordinates.
        """
        # Evaluate shape functions derivates and Jacobian
        shape_fun_deriv, _, jacobian_det = \
            eval_shapefun_deriv(element_type, nodes_coords, local_coords)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build discrete gradient operator
        if is_symmetric:
            # Build discrete symmetric gradient operator
            grad_operator = \
                vbuild_discrete_sym_gradient(shape_fun_deriv, comp_order)
        else:
            # Build discrete non-symmetric gradient operator
            grad_operator = \
                vbuild_discrete_gradient(shape_fun_deriv, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return jacobian_det, grad_operator
    # -------------------------------------------------------------------------
    def vcompute_local_vol_sym_gradient(self, grad_operator_sym, n_dim):
        """Compute discrete volumetric symmetric gradient operator.
        
        Parameters
        ----------
        grad_operator_sym : torch.Tensor(2d)
            Discrete symmetric gradient operator.
        n_dim : int
            Number of spatial dimensions.
            
        Returns
        -------
        vol_grad_operator_sym : torch.Tensor(2d)
            Discrete volumetric symmetric gradient operator.
        """
        # Get device
        device = grad_operator_sym.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2D > 3D conversion
        if n_dim == 2:
            # Expand 2D symmetric gradient operator to 3D
            grad_operator_sym = \
                vexpand_grad_operator_sym_2d_to_3d(grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem type parameters
        _, comp_order_sym, _ = get_problem_type_parameters(4)
        # Get volumetric and deviatoric projection tensors
        vol_proj_vmf, _ = get_projection_tensors_vmf(
            n_dim=3, comp_order_sym=comp_order_sym, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute discrete volumetric symmetric gradient operator
        vol_grad_operator_sym = torch.matmul(vol_proj_vmf, grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3D > 2D conversion
        if n_dim == 2:
            # Reduce 3D volumetric symmetric gradient operator to 2D
            vol_grad_operator_sym = \
                vreduce_grad_operator_sym_3d_to_2d(vol_grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return vol_grad_operator_sym
    # -------------------------------------------------------------------------
    def vcompute_local_internal_forces_hist(
        self, local_coords, weight, strain_formulation, problem_type,
        element_type, nodes_coords_hist, nodes_disps_hist, time_hist,
        element_material, is_volumetric_bar=False,
        avg_vol_grad_operator_hist=None):
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
        is_volumetric_bar : bool, default=False
            If True, then use volumetric strain averaging formulation (e.g.,
            B-bar formulation under infinitesimal strains).
        avg_vol_grad_operator_hist : torch.Tensor(3d), default=None
            Element average volumetric gradient operator history stored as
            torch.Tensor(3d) of shape (n_time, n_strain_comp, n_node*n_dim).
            Required only if volumetric strain averaging formulation is used.
            
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Apply volumetric strain averaging formulation
            if is_volumetric_bar:
                # Check average volumetric gradient operator history
                if avg_vol_grad_operator_hist is None:
                    raise RuntimeError('The average volumetric gradient '
                                       'operator history must be provided '
                                       'when using a volumetric strain '
                                       'averaging formulation.')
                # Set vectorized strain computation with volumetric strain
                # averaging formulation (batch along time)
                vmap_compute_local_strain_vbar = \
                    torch.vmap(self.vcompute_local_strain_vbar,
                               in_dims=(2, 2, 0, None, None, None, None, None),
                               out_dims=(0, 0))
                # Compute infinitesimal strain tensor history with volumetric
                # strain averaging formulation
                strain_hist, grad_operator_sym_hist = \
                    vmap_compute_local_strain_vbar(
                        nodes_coords_hist, nodes_disps_hist,
                        avg_vol_grad_operator_hist, local_coords,
                        strain_formulation, n_dim, comp_order_sym,
                        element_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Detach computation graph (minimize memory costs)
        state_variables_hist = state_variables_hist.detach()
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
    def vcompute_local_strain_vbar(self, nodes_coords, nodes_disps,
                                   avg_vol_grad_operator, local_coords,
                                   strain_formulation, n_dim, comp_order,
                                   element_type):
        """Compute strain tensor at given local point of element.
        
        The strain tensor is computed using a volumetric strain averaging
        formulation (e.g., B-bar formulation under infinitesimal strains).
        
        Compatible with vectorized mapping.
        
        Parameters
        ----------
        nodes_coords : torch.Tensor(2d)
            Nodes coordinates stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        nodes_disps : torch.Tensor(2d)
            Nodes displacements stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        avg_vol_grad_operator : torch.Tensor(2d)
            Element average volumetric gradient operator.
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
        vbar_grad_operator_sym : torch.Tensor(2d)
            Modified discrete symmetric gradient operator evaluated at given
            local coordinates.
        """
        # Evaluate shape functions derivates and Jacobian
        shape_fun_deriv, _, _ = \
            eval_shapefun_deriv(element_type, nodes_coords, local_coords)
        # Build discrete symmetric gradient operator
        grad_operator_sym = vbuild_discrete_sym_gradient(shape_fun_deriv,
                                                         comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       # Compute discrete deviatoric symmetric gradient operator
        dev_grad_operator_sym = self.vcompute_local_dev_sym_gradient(
            grad_operator_sym, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute modified discrete symmetric gradient operator
        vbar_grad_operator_sym = avg_vol_grad_operator + dev_grad_operator_sym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute strain tensor
        if strain_formulation == 'infinitesimal':
            # Compute infinitesimal strain tensor (Voigt matricial form)
            strain_vmf = compute_infinitesimal_strain(
                vbar_grad_operator_sym, nodes_disps)
            # Get strain tensor
            strain = vget_strain_from_vmf(strain_vmf, n_dim, comp_order)
        else:
            raise RuntimeError('Not implemented.')      
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain, vbar_grad_operator_sym
    # -------------------------------------------------------------------------
    def vcompute_local_dev_sym_gradient(self, grad_operator_sym, n_dim):
        """Compute discrete deviatoric symmetric gradient operator.
        
        Parameters
        ----------
        grad_operator_sym : torch.Tensor(2d)
            Discrete symmetric gradient operator.
        n_dim : int
            Number of spatial dimensions.
            
        Returns
        -------
        dev_grad_operator_sym : torch.Tensor(2d)
            Discrete deviatoric symmetric gradient operator.
        """
        # Get device
        device = grad_operator_sym.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2D > 3D conversion
        if n_dim == 2:
            # Expand 2D symmetric gradient operator to 3D
            grad_operator_sym = \
                vexpand_grad_operator_sym_2d_to_3d(grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem type parameters
        _, comp_order_sym, _ = get_problem_type_parameters(4)
        # Get volumetric and deviatoric projection tensors
        _, dev_proj_vmf = \
            get_projection_tensors_vmf(n_dim, comp_order_sym, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute discrete deviatoric symmetric gradient operator
        dev_grad_operator_sym = torch.matmul(dev_proj_vmf, grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3D > 2D conversion
        if n_dim == 2:
            # Reduce 3D volumetric symmetric gradient operator to 2D
            dev_grad_operator_sym = \
                vreduce_grad_operator_sym_3d_to_2d(dev_grad_operator_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dev_grad_operator_sym
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
        # Normalize input features tensor
        if self.check_model_in_normalized(constitutive_model):
            # Normalize input features
            features_in = data_scaler_transform(constitutive_model,
                features_in, 'features_in', mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute output features
        features_out = constitutive_model(features_in)
        # Extract output features
        features_out = self.features_out_extractor(features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features tensor
        if self.check_model_out_normalized(constitutive_model):
            # Denormalize output features
            features_out = data_scaler_transform(constitutive_model,
                features_out, 'features_out', mode='denormalize')
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
                                     dirichlet_bc_mesh_hist):
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
        dirichlet_bc_mesh_hist : torch.Tensor(3d)
            Dirichlet boundary constraints history of finite element mesh nodes
            stored as torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
            Encodes if each degree of freedom is free (assigned 0) or
            constrained (greater than 0) under Dirichlet boundary conditions.
            The encoding depends on the selected force equilibrium loss type.
            
        Returns
        -------
        force_equilibrium_hist_loss : torch.Tensor(0d)
            Force equilibrium history loss.
        """
        # Get initial Dirichlet boundary constraints
        # (vectorized mapping does not support dynamic shaping due to potential
        # dynamic Dirichlet boundary constrains throughout time)
        dirichlet_bc_mesh_init = dirichlet_bc_mesh_hist[:, :, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized force equilibrium history loss computation (batch
        # along time)
        vmap_force_equilibrium_loss = \
            torch.vmap(self.vforce_equilibrium_loss,
                       in_dims=(2, 2, 2, None), out_dims=(0,))
        # Compute force equilibrium history loss
        force_equilibrium_hist_loss = torch.sum(
            self._loss_time_weights
            *vmap_force_equilibrium_loss(internal_forces_mesh_hist,
                                         external_forces_mesh_hist,
                                         reaction_forces_mesh_hist,
                                         dirichlet_bc_mesh_init))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def vforce_equilibrium_loss(self, internal_forces_mesh,
                                external_forces_mesh, reaction_forces_mesh,
                                dirichlet_bc_mesh):
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
        dirichlet_bc_mesh : torch.Tensor(2d)
            Dirichlet boundary constraints of finite element mesh nodes
            stored as torch.Tensor(2d) of shape (n_node_mesh, n_dim).
            Encodes if each degree of freedom is free (assigned 0) or
            constrained (greater than 0) under Dirichlet boundary conditions.
            The encoding depends on the selected force equilibrium loss type.
            
        Returns
        -------
        force_equilibrium_loss : torch.Tensor(0d)
            Force equilibrium loss.
        """
        # Normalize forces
        if self._is_force_normalization:
            # Normalize internal forces
            internal_forces_mesh = data_scaler_transform(self,
                internal_forces_mesh, features_type='forces', mode='normalize')
            # Normalize external forces
            external_forces_mesh = data_scaler_transform(self,
                external_forces_mesh, features_type='forces', mode='normalize')
            # Normalize reaction forces
            reaction_forces_mesh = data_scaler_transform(self,
                reaction_forces_mesh, features_type='forces', mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium loss
        if self._force_equilibrium_loss_type == 'pointwise':
            # Force equilibrium strictly based on pointwise internal, external
            # and reaction forces
            force_equilibrium_loss = \
                torch.sum((internal_forces_mesh - external_forces_mesh
                           - torch.where(dirichlet_bc_mesh == 1,
                                         reaction_forces_mesh, 0.0))**2)
        elif self._force_equilibrium_loss_type == 'dirichlet_sets':
            # Flatten mesh data
            internal_forces_mesh_flat = internal_forces_mesh.view(-1)
            external_forces_mesh_flat = external_forces_mesh.view(-1)
            reaction_forces_mesh_flat = reaction_forces_mesh.view(-1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Flatten mesh sets
            dirichlet_bc_mesh_flat = dirichlet_bc_mesh.view(-1).long()
            # Get unique set labels
            unique_labels = torch.unique(dirichlet_bc_mesh_flat)
            # Set contiguous set labels
            dense_indices = \
                torch.bucketize(dirichlet_bc_mesh_flat, unique_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get total number of degrees of freedom
            n_total = dirichlet_bc_mesh_flat.numel()
            # Get number of sets
            n_sets = unique_labels.numel()
            # Initialize one-hot encoding of mesh degrees of freedom
            one_hot = torch.zeros(n_total, n_sets,
                                  device=dirichlet_bc_mesh.device)
            # Scatter one-hot encoding based on contiguous set labels
            one_hot.scatter_(dim=1, index=dense_indices.unsqueeze(1),
                             value=1.0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get one-hot enconding for non-Dirichlet and Dirichlet constrained
            # sets
            one_hot_ndbc = one_hot[:, 0]
            one_hot_dbc = one_hot[:, 1:]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set mask for reaction forces (0 for non-Dirichlet degrees of
            # freedom, 1 for Dirichlet constrained degrees of freedom)
            reaction_forces_mask = 1.0 - one_hot_ndbc.unsqueeze(1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute force equilibrium (all degrees of freedom)
            force_equilibrium = (
                internal_forces_mesh_flat - external_forces_mesh_flat
                - reaction_forces_mesh_flat*reaction_forces_mask.squeeze(1))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Output total reaction forces per Dirichlet constrained set
            #torch.set_printoptions(linewidth=1000)
            #reaction_dbc = \
            #    reaction_forces_mesh_flat*reaction_forces_mask.squeeze(1)
            #print((reaction_dbc.unsqueeze(1)*one_hot_dbc).sum(dim=0))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set force equilibrium loss multi-objective weights flag
            is_multi_objective_weights = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute force equilibrium loss terms
            if is_multi_objective_weights:
                # Compute sets number of degrees of freedom
                sets_n_dof = one_hot.sum(dim=0)
                # Compute sets weights
                sets_weights = sets_n_dof/n_total
                # Set weights for non-Dirichlet and Dirichlet constrained sets
                weight_ndbc = sets_weights[0]
                weight_dbc = sets_weights[1:]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute force equilibrium loss term
                # (non-Dirichlet constrained set)
                force_equilibrium_loss_ndbc = weight_ndbc*torch.sum(
                    (force_equilibrium*one_hot_ndbc)**2)
                # Compute force equilibrium loss term
                # (Dirichlet constrained sets)
                force_equilibrium_loss_dbc = torch.sum(weight_dbc*((
                    force_equilibrium.unsqueeze(1)*one_hot_dbc).sum(dim=0)**2))
            else:
                # Compute force equilibrium loss term
                # (non-Dirichlet constrained set)
                force_equilibrium_loss_ndbc = torch.sum(
                    (force_equilibrium*one_hot_ndbc)**2)
                # Compute force equilibrium loss term
                # (Dirichlet constrained sets)
                force_equilibrium_loss_dbc = torch.sum(((
                    force_equilibrium.unsqueeze(1)*one_hot_dbc).sum(dim=0)**2))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute force equilibrium loss
            force_equilibrium_loss = \
                force_equilibrium_loss_ndbc + force_equilibrium_loss_dbc
        else:
            raise RuntimeError(f'Unknown force equilibrium loss type: '
                               f'\'{self._force_equilibrium_loss_type}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_loss
    # -------------------------------------------------------------------------
    def force_equilibrium_loss_components_hist(
        self, internal_forces_mesh_hist, external_forces_mesh_hist,
        reaction_forces_mesh_hist, dirichlet_bc_mesh_hist):
        """Compute force equilibrium loss components history (output purposes).
        
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
        dirichlet_bc_mesh_hist : torch.Tensor(3d)
            Dirichlet boundary constraints history of finite element mesh nodes
            stored as torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
            Encodes if each degree of freedom is free (assigned 0) or
            constrained (greater than 0) under Dirichlet boundary conditions.
            The encoding depends on the selected force equilibrium loss type.

        Returns
        -------
        force_equilibrium_loss_components_hist : torch.Tensor(2d)
            Force equilibrium loss components history stored as
            torch.Tensor(2d) of shape (1 + n_loss_comp, n_time).
        """
        # Set number of force equilibrium loss components
        if self._force_equilibrium_loss_type == 'pointwise':
            n_loss_comp = 0
        elif self._force_equilibrium_loss_type == 'dirichlet_sets':
            n_loss_comp = 2
        else:
            raise RuntimeError(f'Unknown force equilibrium loss type: '
                               f'\'{self._force_equilibrium_loss_type}\'.')
        # Get history length
        n_time = internal_forces_mesh_hist.shape[2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize forces
        if self._is_force_normalization:
            # Normalize internal forces
            internal_forces_mesh = data_scaler_transform(
                self, internal_forces_mesh_hist, features_type='forces',
                mode='normalize')
            # Normalize external forces
            external_forces_mesh = data_scaler_transform(
                self, external_forces_mesh_hist, features_type='forces',
                mode='normalize')
            # Normalize reaction forces
            reaction_forces_mesh = data_scaler_transform(
                self, reaction_forces_mesh_hist, features_type='forces',
                mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize force equilibrium loss components history
        force_equilibrium_loss_components_hist = \
            torch.zeros(1 + n_loss_comp, n_time,
                        device=internal_forces_mesh_hist.device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over time
        for i in range(n_time):
            # Get internal forces at time step
            internal_forces_mesh = internal_forces_mesh_hist[:, :, i]
            # Get external forces at time step
            external_forces_mesh = external_forces_mesh_hist[:, :, i]
            # Get reaction forces at time step
            reaction_forces_mesh = reaction_forces_mesh_hist[:, :, i]
            # Get Dirichlet boundary constraints at time step
            dirichlet_bc_mesh = dirichlet_bc_mesh_hist[:, :, i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute force equilibrium loss components
            if self._force_equilibrium_loss_type == 'pointwise':
                # Force equilibrium strictly based on pointwise internal,
                # external and reaction forces
                force_equilibrium_loss = \
                    torch.sum((internal_forces_mesh - external_forces_mesh
                               - torch.where(dirichlet_bc_mesh == 1,
                                             reaction_forces_mesh, 0.0))**2)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store force equilibrium loss and components
                force_equilibrium_loss_components_hist[0, i] = \
                    force_equilibrium_loss
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._force_equilibrium_loss_type == 'dirichlet_sets':
                # Flatten mesh data
                internal_forces_mesh_flat = internal_forces_mesh.view(-1)
                external_forces_mesh_flat = external_forces_mesh.view(-1)
                reaction_forces_mesh_flat = reaction_forces_mesh.view(-1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Flatten mesh sets
                dirichlet_bc_mesh_flat = dirichlet_bc_mesh.view(-1).long()
                # Get unique set labels
                unique_labels = torch.unique(dirichlet_bc_mesh_flat)
                # Set contiguous set labels
                dense_indices = \
                    torch.bucketize(dirichlet_bc_mesh_flat, unique_labels)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get total number of degrees of freedom
                n_total = dirichlet_bc_mesh_flat.numel()
                # Get number of sets
                n_sets = unique_labels.numel()
                # Initialize one-hot encoding of mesh degrees of freedom
                one_hot = torch.zeros(n_total, n_sets,
                                      device=dirichlet_bc_mesh.device)
                # Scatter one-hot encoding based on contiguous set labels
                one_hot.scatter_(dim=1, index=dense_indices.unsqueeze(1),
                                 value=1.0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get one-hot enconding for non-Dirichlet and Dirichlet
                # constrained sets
                one_hot_ndbc = one_hot[:, 0]
                one_hot_dbc = one_hot[:, 1:]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set mask for reaction forces (0 for non-Dirichlet degrees of
                # freedom, 1 for Dirichlet constrained degrees of freedom)
                reaction_forces_mask = 1.0 - one_hot_ndbc.unsqueeze(1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute force equilibrium (all degrees of freedom)
                force_equilibrium = (
                    internal_forces_mesh_flat - external_forces_mesh_flat
                    - reaction_forces_mesh_flat
                    *reaction_forces_mask.squeeze(1))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set force equilibrium loss multi-objective weights flag
                is_multi_objective_weights = False
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute force equilibrium loss terms
                if is_multi_objective_weights:
                    # Compute sets number of degrees of freedom
                    sets_n_dof = one_hot.sum(dim=0)
                    # Compute sets weights
                    sets_weights = sets_n_dof/n_total
                    # Set weights for non-Dirichlet and Dirichlet constrained
                    # sets
                    weight_ndbc = sets_weights[0]
                    weight_dbc = sets_weights[1:]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute force equilibrium loss term
                    # (non-Dirichlet constrained set)
                    force_equilibrium_loss_ndbc = weight_ndbc*torch.sum(
                        (force_equilibrium*one_hot_ndbc)**2)
                    # Compute force equilibrium loss term
                    # (Dirichlet constrained sets)
                    force_equilibrium_loss_dbc = torch.sum(weight_dbc*(
                        (force_equilibrium.unsqueeze(1)*one_hot_dbc).sum(
                            dim=0)**2))
                else:
                    # Compute force equilibrium loss term
                    # (non-Dirichlet constrained set)
                    force_equilibrium_loss_ndbc = torch.sum(
                        (force_equilibrium*one_hot_ndbc)**2)
                    # Compute force equilibrium loss term
                    # (Dirichlet constrained sets)
                    force_equilibrium_loss_dbc = torch.sum((
                        (force_equilibrium.unsqueeze(1)*one_hot_dbc).sum(
                            dim=0)**2))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute force equilibrium loss
                force_equilibrium_loss = \
                    force_equilibrium_loss_ndbc + force_equilibrium_loss_dbc
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store force equilibrium loss and components
                force_equilibrium_loss_components_hist[0, i] = \
                    force_equilibrium_loss
                force_equilibrium_loss_components_hist[1, i] = \
                    force_equilibrium_loss_ndbc
                force_equilibrium_loss_components_hist[2, i] = \
                    force_equilibrium_loss_dbc
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply loss time weights
        if self._loss_time_weights is not None:
            force_equilibrium_loss_components_hist = \
                (self._loss_time_weights[None, :] \
                 *force_equilibrium_loss_components_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply loss scaling factor
        if self._loss_scaling_factor is not None:
            force_equilibrium_loss_components_hist = \
                (self._loss_scaling_factor
                 *force_equilibrium_loss_components_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_loss_components_hist
    # -------------------------------------------------------------------------
    def store_force_equilibrium_loss_components_hist(
        self, force_equilibrium_loss_components_hist, is_plot=True):
        """Store force equilibrium loss components history.
        
        Parameters
        ----------
        force_equilibrium_loss_components_hist : torch.Tensor(2d)
            Force equilibrium loss components history stored as
            torch.Tensor(2d) of shape (1 + n_loss_comp, n_time).
        is_plot : bool, default=True
            If True, then plot force equilibrium loss components history.
        """
        # Set storage directory
        save_dir = os.path.join(os.path.normpath(self.model_directory),
                                'force_equilibrium_loss_components')
        # Create storage directory
        make_directory(save_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build force equilibrium loss data
        force_equilibrium_loss_data = {}
        force_equilibrium_loss_components_hist = \
            force_equilibrium_loss_components_hist.detach().cpu().numpy()
        force_equilibrium_loss_data['force_equilibrium_loss_components_hist'] \
            = force_equilibrium_loss_components_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data file path
        data_file_path = \
            os.path.join(save_dir, 'force_equilibrium_loss_data.pkl')
        # Save data
        with open(data_file_path, 'wb') as data_file:
            pickle.dump(force_equilibrium_loss_data, data_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot force equilibrium loss components history
        if is_plot:
            # Get number of loss components
            n_loss_comp = force_equilibrium_loss_components_hist.shape[0] - 1
            # Get number of time steps
            n_time = force_equilibrium_loss_components_hist.shape[1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize data labels
            data_labels = []
            # Build data array and set data labels
            data_array = np.zeros((n_time, 2*(1 + n_loss_comp)))
            for i in range(n_loss_comp + 1):
                data_array[:, 2*i] = np.arange(n_time)
                data_array[:, 2*i + 1] = \
                    force_equilibrium_loss_components_hist[i, :]
                if i == 0:
                    data_labels.append('Total')
                else:
                    data_labels.append(f'Component {i}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot force equilibrium loss components history
            figure, _ = plot_xy_data(data_array,
                                     data_labels=data_labels,
                                     x_lims=(0, None),
                                     y_lims=(None, None),
                                     x_label=f'Time step',
                                     y_label=f'Force equilibrium loss',
                                     y_scale='log',
                                     is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set file name
            filename = f'force_equilibrium_loss_components_hist'
            # Save figure
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Close figure
            plt.close('all')
    # -------------------------------------------------------------------------
    def build_elements_local_samples(self, strain_formulation, problem_type,
                                     time_hist, elements_state_hist):
        """Build elements local strain-stress paths.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        time_hist : torch.Tensor(1d)
            Discrete time history.
        elements_state_hist : torch.Tensor(4d)
            Gauss integration points strain and stress path history of finite
            elements stored as torch.Tensor(4d) of shape
            (n_elem, n_gauss, n_time, n_strain_comps + n_stress_comps).
 
        Returns
        -------
        elements_local_samples : list[dict]
            Elements local strain-stress paths, each corresponding to a given
            element Gauss integration point. Each path is stored as a
            dictionary where each feature (key, str) data is a torch.Tensor(2d)
            of shape (sequence_length, n_features).
        """
        # Get problem type parameters
        _, comp_order_sym, _ = get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain and stress components
        if strain_formulation == 'infinitesimal':
            strain_comps_order = comp_order_sym
            stress_comps_order = comp_order_sym
        else:
            raise RuntimeError('Not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain indexes
        strain_slice = slice(0, len(comp_order_sym))
        # Set stress indexes
        stress_slice = slice(len(comp_order_sym), 2*len(comp_order_sym))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elements of specimen local strain-stress data set
        if isinstance(self._local_paths_elements, list):
            elements_idxs = [int(x) - 1 for x in self._local_paths_elements]
        else:
            elements_idxs = [x for x in range(elements_state_hist.shape[0])]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements local strain-stress paths
        elements_local_samples = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for i in elements_idxs:
            # Loop over Gauss integration points
            for j in range(elements_state_hist.shape[1]):
                # Get strain path
                strain_path = elements_state_hist[i, j, :, strain_slice]
                # Get stress path
                stress_path = elements_state_hist[i, j, :, stress_slice]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize material response path data
                response_path = {}
                # Assemble strain-stress material response path
                response_path['strain_comps_order'] = strain_comps_order
                response_path['strain_path'] = strain_path.detach().cpu()
                response_path['stress_comps_order'] = stress_comps_order
                response_path['stress_path'] = stress_path.detach().cpu()
                # Assemble time path
                response_path['time_hist'] = time_hist.detach().reshape(-1, 1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble material response path
                elements_local_samples.append(response_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elements_local_samples
    # -------------------------------------------------------------------------
    def compute_dirichlet_sets_reaction_hist(self, internal_forces_mesh_hist,
                                             external_forces_mesh_hist,
                                             dirichlet_bc_mesh_hist):
        """Compute reaction forces history of Dirichlet boundary sets.
        
        
        At a given time step, the reaction force of each Dirichlet boundary
        set is computed to strictly satisfy the total force equilibrium,
        assuming that the internal and external forces are known.

        Parameters
        ----------
        internal_forces_mesh_hist : torch.Tensor(3d)
            Internal forces history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        external_forces_mesh_hist : torch.Tensor(3d)
            External forces history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        dirichlet_bc_mesh_hist : torch.Tensor(3d)
            Dirichlet boundary constraints history of finite element mesh nodes
            stored as torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
            Encodes if each degree of freedom is free (assigned 0) or
            constrained (greater than 0) under Dirichlet boundary conditions.
            The encoding depends on the selected force equilibrium loss type.
            
        Returns
        -------
        dirichlet_sets_reaction_hist : torch.Tensor(3d)
            Reaction forces history of Dirichlet boundary sets stored as
            torch.Tensor(3d) of shape (n_sets, 1, n_time). Sets are sorted
            according with their encoding labels and are associated with a
            single spatial dimension.
        """
        # Get initial Dirichlet boundary constraints
        # (vectorized mapping does not support dynamic shaping due to potential
        # dynamic Dirichlet boundary constrains throughout time)
        dirichlet_bc_mesh_init = dirichlet_bc_mesh_hist[:, :, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized Dirichlet boundary constrained sets reaction forces
        # computation (batch along time)
        vmap_compute_dirichlet_sets_reaction = \
            torch.vmap(self.compute_dirichlet_sets_reaction,
                       in_dims=(2, 2, None), out_dims=(2,))
        # Compute force equilibrium history loss
        dirichlet_sets_reaction_hist = \
            vmap_compute_dirichlet_sets_reaction(internal_forces_mesh_hist,
                                                 external_forces_mesh_hist,
                                                 dirichlet_bc_mesh_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dirichlet_sets_reaction_hist
    # -------------------------------------------------------------------------
    def compute_dirichlet_sets_reaction(self, internal_forces_mesh,
                                        external_forces_mesh,
                                        dirichlet_bc_mesh):
        """Compute reaction forces of Dirichlet boundary sets.
        
        Parameters
        ----------
        internal_forces_mesh : torch.Tensor(2d)
            Internal forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        external_forces_mesh : torch.Tensor(2d)
            External forces of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
            (n_node_mesh, n_dim).
        dirichlet_bc_mesh : torch.Tensor(2d)
            Dirichlet boundary constraints of finite element mesh nodes
            stored as torch.Tensor(2d) of shape (n_node_mesh, n_dim).
            Encodes if each degree of freedom is free (assigned 0) or
            constrained (greater than 0) under Dirichlet boundary conditions.
            The encoding depends on the selected force equilibrium loss type.
            
        Returns
        -------
        dirichlet_sets_reaction : torch.Tensor(2d)
            Reaction forces of Dirichlet boundary sets stored as
            torch.Tensor(2d) of shape (n_sets, 1). Sets are sorted according
            with their encoding labels and are associated with a single spatial
            dimension.
        """
        # Flatten mesh data
        internal_forces_mesh_flat = internal_forces_mesh.view(-1)
        external_forces_mesh_flat = external_forces_mesh.view(-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Flatten mesh sets
        dirichlet_bc_mesh_flat = dirichlet_bc_mesh.view(-1).long()
        # Get unique set labels
        unique_labels = torch.unique(dirichlet_bc_mesh_flat)
        # Set contiguous set labels
        dense_indices = \
            torch.bucketize(dirichlet_bc_mesh_flat, unique_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get total number of degrees of freedom
        n_total = dirichlet_bc_mesh_flat.numel()
        # Get number of sets
        n_sets = unique_labels.numel()
        # Initialize one-hot encoding of mesh degrees of freedom
        one_hot = torch.zeros(n_total, n_sets, device=dirichlet_bc_mesh.device)
        # Scatter one-hot encoding based on contiguous set labels
        one_hot.scatter_(dim=1, index=dense_indices.unsqueeze(1), value=1.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute force equilibrium (all degrees of freedom)
        force_equilibrium = (internal_forces_mesh_flat
                             - external_forces_mesh_flat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute reaction forces of Dirichlet constrained sets
        dirichlet_sets_reaction = \
            (force_equilibrium.unsqueeze(1)*one_hot).sum(dim=0).view(n_sets, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dirichlet_sets_reaction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def store_dirichlet_sets_reaction_hist(self, dirichlet_sets_reaction_hist,
                                           is_export_csv=True, is_plot=True):
        """Store reaction forces history of Dirichlet boundary sets.
        
        Parameters
        ----------
        dirichlet_sets_reaction_hist : torch.Tensor(3d)
            Reaction forces history of Dirichlet boundary sets stored as
            torch.Tensor(3d) of shape (n_sets, 1, n_time).
        is_export_csv : bool, default=True
            If True, then export the reaction force history of Dirichlet
            boundary sets to a '.csv' file.
        is_plot : bool, default=True
            If True, then plot the reaction force history of each Dirichlet
            boundary set.
        """
        # Set storage directory
        save_dir = os.path.join(os.path.normpath(self.model_directory),
                                'dirichlet_sets_reaction_forces')
        # Create storage directory
        make_directory(save_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build Dirichlet boundary sets data
        dirichlet_sets_data = {}
        dirichlet_sets_data['dirichlet_sets_reaction_hist'] = \
            dirichlet_sets_reaction_hist.detach().cpu().numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data file path
        data_file_path = \
            os.path.join(save_dir, 'dirichlet_sets_data.pkl')
        # Save data
        with open(data_file_path, 'wb') as data_file:
            pickle.dump(dirichlet_sets_data, data_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Export Dirichlet boundary sets reaction force history to '.csv' file
        if is_export_csv:
            # Get number of Dirichlet boundary sets
            n_sets = dirichlet_sets_reaction_hist.shape[0]
            # Set data to export CSV file
            csv_data = dirichlet_sets_reaction_hist.squeeze(1).T
            # Set data frame headers
            headers = [f"SET {i}" for i in range(n_sets)]
            # Build data frame
            df = pandas.DataFrame(csv_data.detach().cpu().numpy(),
                                  columns=headers)
            # Set '.csv' data file path
            csv_file_path = \
                os.path.join(save_dir, 'dirichlet_reaction_hist_sets.csv')
            # Export '.csv' data file
            df.to_csv(csv_file_path, index=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot Dirichlet boundary sets reaction force history
        if is_plot:
            # Get number of Dirichlet boundary sets
            n_sets = dirichlet_sets_reaction_hist.shape[0]
            # Get number of time steps
            n_time = dirichlet_sets_reaction_hist.shape[2]
            # Loop over Dirichlet boundary sets
            for i in range(n_sets):
                # Get Dirichlet boundary set reaction force history
                dirichlet_set_reaction_hist = \
                    dirichlet_sets_reaction_hist[
                        i, 0, :].detach().cpu().numpy()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build data array
                data_array = np.zeros((n_time, 2))
                data_array[:, 0] = np.arange(n_time)
                data_array[:, 1] = dirichlet_set_reaction_hist
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot Dirichlet boundary set reaction force history
                figure, _ = plot_xy_data(data_array,
                                         x_lims=(0, None),
                                         x_label=f'Time step',
                                         y_label=f'Reaction force',
                                         is_latex=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set file name
                filename = f'dirichlet_reaction_hist_set_{i}'
                # Save figure
                save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    @classmethod
    def features_out_extractor(cls, model_output):
        """Extract output features from generic model output.
        
        Parameters
        ----------
        model_output : {torch.Tensor, tuple}
            Model output.
        
        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Extract output features
        if isinstance(model_output, tuple):
            # Assume output features are stored in the first output index
            # of model output
            features_out = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            # Output features correspond directly to model output
            features_out = model_output
        else:
            raise RuntimeError(f'Unexpected model output of type '
                               f'({type(model_output)}). Output features '
                               f'extraction is not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
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
            # Check if material model supports features normalization
            if ((not hasattr(model, 'is_model_in_normalized'))
                 and (not hasattr(model, 'is_model_out_normalized'))):
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check material model features normalization
            if ((not model.is_model_in_normalized)
                    and (not model.is_model_out_normalized)):
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material model data scaling type
            scaling_type = models_scaling_type[model_key]
            # Get material model data scaling parameters
            scaling_parameters = models_scaling_parameters[model_key]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material model fitted data scalers
            set_fitted_data_scalers(model, scaling_type, scaling_parameters)
    # -------------------------------------------------------------------------
    @classmethod
    def check_model_in_normalized(cls, model):
        """Check if generic model expects normalized input features.
        
        A model expects normalized input features if it has an attribute
        'is_model_in_normalized' set to True.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model.
        
        Returns
        -------
        is_model_in_normalized : bool
            If True, then model expects normalized input features (normalized
            input data has been seen during model training).
        """
        # Get model input features normalization
        if hasattr(model, 'is_model_in_normalized'):
            is_model_in_normalized = model.is_model_in_normalized
        else:
            is_model_in_normalized = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_model_in_normalized
    # -------------------------------------------------------------------------
    @classmethod
    def check_model_out_normalized(cls, model):
        """Check if generic model expects normalized output features.
        
        A model expects normalized output features if it has an attribute
        'is_model_out_normalized' set to True.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model.
        
        Returns
        -------
        is_model_out_normalized : bool
            If True, then model expects normalized output features (normalized
            output data has been seen during model training).
        """
        # Get model output features normalization
        if hasattr(model, 'is_model_out_normalized'):
            is_model_out_normalized = model.is_model_out_normalized
        else:
            is_model_out_normalized = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_model_out_normalized