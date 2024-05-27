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
import copy
# Third-party
import torch
# Local
from simulators.fetorch.element.integrations.internal_forces import \
    compute_element_internal_forces, compute_infinitesimal_inc_strain
from simulators.fetorch.element.derivatives.gradients import \
    eval_shapefun_deriv, build_discrete_sym_gradient
from simulators.fetorch.material.material_su import material_state_update
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_from_mf
from simulators.fetorch.math.voigt_notation import get_strain_from_vfm, \
    get_stress_vfm
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
    _device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, sequential_mode, specimen_data, specimen_material_state)
        Forward propagation.
    forward_sequential_time(self, specimen_data, specimen_material_state)
        Forward propagation (sequential time).
    forward_sequential_element(self, specimen_data, specimen_material_state)
        Forward propagation (sequential element).
    force_equilibrium_loss(internal_forces_mesh, external_forces_mesh, \
                           reaction_forces_mesh, dirichlet_bool_mesh)
        Compute force equilibrium loss for given discrete time.
    """
    def __init__(self, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(MaterialModelFinder, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        self.set_device(device_type)
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
    def forward(self, sequential_mode, specimen_data, specimen_material_state):
        """Forward propagation.
        
        Parameters
        ----------
        sequential_mode : {'sequential_time', 'sequential_element'}
            If 'sequential_time', then internal forces are computed in the
            standard way, processing each time step sequentially.
            If 'sequential_element', then internal forces are computed such
            that each element is processed sequentially, taking into account
            the corresponding deformation history.
        specimen_data : SpecimenNumericalData
            Specimen numerical data translated from experimental results.
        specimen_material_state : StructureMaterialState
            FETorch structure material state.

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
        if sequential_mode == 'sequential_time':
            force_equilibrium_hist_loss = self.forward_sequential_time(
                specimen_data, specimen_material_state)
        elif sequential_mode == 'sequential_element':
            force_equilibrium_hist_loss = self.forward_sequential_element(
                specimen_data, specimen_material_state)
        else:
            raise RuntimeError('Unknown sequential mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_time(self, specimen_data, specimen_material_state):
        """Forward propagation (sequential time).
        
        Parameters
        ----------
        specimen_data : SpecimenNumericalData
            Specimen numerical data translated from experimental results.
        specimen_material_state : StructureMaterialState
            FETorch structure material state.

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
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
            force_equilibrium_hist_loss += type(self).force_equilibrium_loss(
                internal_forces_mesh, external_forces_mesh,
                reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def forward_sequential_element(self, specimen_data,
                                   specimen_material_state):
        """Forward propagation (sequential element).
        
        Parameters
        ----------
        specimen_data : SpecimenNumericalData
            Specimen numerical data translated from experimental results.
        specimen_material_state : StructureMaterialState
            FETorch structure material state.

        Returns
        -------
        force_equilibrium_hist_loss : float
            Force equilibrium history loss.
        """
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
            # Get element last converged material constitutive state
            # variables
            element_state_old = \
                specimen_material_state.get_element_state(elem_id, time='last')
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
                    nodes_disps_hist, nodes_inc_disps_hist, time_hist)
            # Update element material constitutive state variables
            specimen_material_state.update_element_state(
                elem_id, element_state_hist[-1], time='current')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete time
            for time_idx in range(n_time):
                # Reshape element internal forces into mesh format
                internal_forces_mesh = specimen_mesh.element_assembler(
                    {str(elem_id): element_internal_forces_hist[:, time_idx]})
                # Assemble element internal forces of finite element mesh nodes
                internal_forces_mesh_hist[:, :, time_idx] += \
                    internal_forces_mesh.reshape(-1, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elements last converged material constitutive state variables
        specimen_material_state.update_converged_elements_state()
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
            force_equilibrium_hist_loss += type(self).force_equilibrium_loss(
                internal_forces_mesh, external_forces_mesh,
                reaction_forces_mesh, dirichlet_bool_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return force_equilibrium_hist_loss
    # -------------------------------------------------------------------------
    def compute_element_internal_forces_hist(
        self, strain_formulation, problem_type, element_type, element_material,
        element_state_old, nodes_coords_hist, nodes_disps_hist,
        nodes_inc_disps_hist, time_hist):
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
            state_variables_old = copy.deepcopy(element_state_old[str(i + 1)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Gauss integration point incremental strain tensor
            # history
            if strain_formulation == 'infinitesimal':
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
                grad_operator_sym = build_discrete_sym_gradient(
                    shape_fun_deriv, comp_order_sym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute incremental strain tensor
                if strain_formulation == 'infinitesimal':
                    # Compute incremental infinitesimal strain tensor (Voigt
                    # matricial form)
                    inc_strain_vmf = compute_infinitesimal_inc_strain(
                        grad_operator_sym,
                        nodes_inc_disps_hist[:, :, time_idx])
                    # Get incremental strain tensor
                    inc_strain_hist[:, :, time_idx] = get_strain_from_vfm(
                        inc_strain_vmf, n_dim, comp_order_sym)
                else:
                    raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if False:
                raise RuntimeError('Missing implementation of recurrent '
                                   'data-driven constitutive model.')
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
                    stress = get_tensor_from_mf(
                        element_state_hist[time_idx][str(i + 1)]['stress_mf'],
                        n_dim, comp_order_sym)
                    # Get Cauchy stress tensor (Voigt matricial form)
                    stress_vmf = get_stress_vfm(stress, n_dim, comp_order_sym)
                else:
                    raise RuntimeError('Not implemented.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Evaluate shape functions derivates and Jacobian
                shape_fun_deriv, _, jacobian_det = eval_shapefun_deriv(
                    element_type, nodes_coords_hist[:, :, time_idx],
                    local_coords)
                # Build discrete symmetric gradient operator
                grad_operator_sym = build_discrete_sym_gradient(
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
    @staticmethod
    def force_equilibrium_loss(internal_forces_mesh, external_forces_mesh,
                               reaction_forces_mesh, dirichlet_bool_mesh):
        """Compute force equilibrium loss for given discrete time.
        
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