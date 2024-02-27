"""DARPA METALS PROJECT: Finding material model by inverse engineering.

Classes
-------
MaterialModelFinder(torch.nn.Module)
    Find material model by inverse engineering experimental results.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.integrations.internal_forces import \
    compute_element_internal_forces
from simulators.fetorch.math.matrixops import get_problem_type_parameters
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
    forward(self, specimen_data, specimen_material_state)
        Forward propagation.
    force_equilibrium_loss(internal_forces_mesh, external_forces_mesh, \
                           reaction_forces_mesh, dirichlet_bool_mesh):
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
    def forward(self, specimen_data, specimen_material_state):
        """Forward propagation.
        
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
            mesh nodes stored as torch.Tensor(3d) of shape
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