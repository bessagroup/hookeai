# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import psutil
import time
# Third-party
import numpy as np
import torch
# Local
from simulators.fetorch.structure.structure_mesh import StructureMesh
from simulators.fetorch.element.type.quad4 import FEQuad4
# =============================================================================
# Summary:
# =============================================================================
def generate_quad4_uniform_mesh(n_elem_dim):
    """Generate uniform mesh of 4-node quadrilateral elements.
    
    Parameters
    ----------
    n_elem_dim : tuple
        Number of elements per dimension.

    Returns
    -------
    nodes_coords_mesh_init : torch.Tensor(2d)
        Initial coordinates of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    elements_type : dict
        FETorch element type (item, ElementType) of each finite element
        mesh element (str[int]). Elements labels must be within the range
        of 1 to n_elem (included).
    connectivities : dict
        Nodes (item, tuple[int]) of each finite element mesh element
        (key, str[int]). Elements labels must be within the range of
        1 to n_elem (included).
    dirichlet_bool_mesh : torch.Tensor(2d)
        Degrees of freedom of finite element mesh subject to Dirichlet
        boundary conditions. Stored as torch.Tensor(2d) of shape
        (n_node_mesh, n_dim) where constrained degrees of freedom are
        labeled 1, otherwise 0.
    """
    # Set size per dimension
    size_dim = (1.0, 1.0)
    # Compute incremental size per dimension
    inc_size_dim = tuple([size_dim[i]/n_elem_dim[i] for i in range(2)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of nodes per dimension
    n_nodes_dim = [int(x + 1) for x in n_elem_dim]
    # Compute number of nodes
    n_node_mesh = np.prod(n_nodes_dim)
    # Compute number of elements
    n_elem = np.prod(n_elem_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize nodes coordinates
    nodes_coords_mesh_init = torch.zeros((n_node_mesh, 2))
    # Initialize node label
    node_label = 0
    # Loop over nodes
    for j_node in range(n_nodes_dim[1]):
        # Loop over elements
        for i_node in range(n_nodes_dim[0]):
            # Increment node label
            node_label += 1
            # Get node coordinates
            node_coords = torch.tensor((i_node*inc_size_dim[0],
                                        j_node*inc_size_dim[1]))
            # Assemble node coordinates
            nodes_coords_mesh_init[node_label - 1, :] = node_coords
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize elements type
    elements_type = {}
    # Loop over elements
    for i_elem in range(n_elem):
        # Set element type
        elements_type[str(i_elem + 1)] = FEQuad4()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize element connectivities
    connectivities = {}
    # Initialize element label
    element_label = 0
    # Loop over elements
    for j_elem in range(n_elem_dim[1]):
        # Loop over elements
        for i_elem in range(n_elem_dim[0]):
            # Increment element label
            element_label += 1
            # Get element nodes
            node_1 = j_elem*n_nodes_dim[0] + i_elem + 1
            node_2 = node_1 + 1
            node_3 = (j_elem + 1)*n_nodes_dim[0] + i_elem + 2
            node_4 = node_3 - 1
            # Assemble element connectivites
            connectivities[str(element_label)] = \
                tuple([node_1, node_2, node_3, node_4])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Dirichlet boundary conditions (boolean)
    dirichlet_bool_mesh = torch.zeros((n_node_mesh, 2))   
    # Loop over left facet boundary nodes
    for node_label in range(1, n_node_mesh, n_nodes_dim[0]):
        dirichlet_bool_mesh[node_label - 1, :] = torch.ones(2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return nodes_coords_mesh_init, elements_type, connectivities, \
        dirichlet_bool_mesh
# =============================================================================
def get_dirichlet_nodes(dirichlet_bool_mesh):
    """Get nodes with Dirichlet boundary conditions.
    
    Returns
    -------
    dirichlet_nodes : tuple[int]
        Finite element mesh nodes with Dirichlet boundary conditions.
    """
    # Initialize nodes with Dirichlet boundary conditions
    dirichlet_nodes = []
    # Loop over nodes
    for i_node in range(dirichlet_bool_mesh.shape[0]):
        # Check Dirichlet boundary conditions
        if any(dirichlet_bool_mesh[i_node, :] == 1):
            # Set node label
            node_label = i_node + 1
            # Store node
            dirichlet_nodes.append(node_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert to tuple
    dirichlet_nodes = tuple(dirichlet_nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    return dirichlet_nodes
# =============================================================================
def compute_element_internal_forces_hist(n_dim, n_node, n_time, param_1):
    """Compute history of finite element internal forces.
    
    Returns
    -------
    element_internal_forces_hist : torch.Tensor(2d)
        Element internal forces history stored as torch.Tensor(2d) of shape
        (n_node*n_dof_node, n_time).
    """
    # Initialize history of finite element internal forces
    element_internal_forces_hist = torch.zeros((n_node*n_dim, n_time))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for i_node in range(n_node):
        # Loop over dimensions
        for j_dof in range(n_dim):
            # Loop over discrete time
            for k_time in range(n_time):
                # Compute internal force
                internal_force = (i_node + 1)*(j_dof + 1)*(k_time + 1)*param_1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble internal force
                element_internal_forces_hist[i_node*n_dim + j_dof, k_time] = \
                    internal_force
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return element_internal_forces_hist
# =============================================================================
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of nodes of finite element mesh
    n_node_mesh = internal_forces_mesh.shape[0]
    # Get number of spatial dimensions
    n_dim = internal_forces_mesh.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return force_equilibrium_loss
# =============================================================================
# Get the process ID (PID) of the current process
process = psutil.Process(os.getpid())
# Start timer
start_time_sec = time.time()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set loss differentiation strategy
loss_diff_strategy = ('naive', 'memory_efficient')[1]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set number of spatial dimensions
n_dim = 2
# Set number of elements per dimension
n_elem_dim = tuple([5, 3])
# Set time history length
n_time = 3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate uniform mesh of 4-node quadrilateral elements
nodes_coords_mesh_init, elements_type, connectivities, dirichlet_bool_mesh = \
    generate_quad4_uniform_mesh(n_elem_dim)
# Get nodes with Dirichlet boundary conditions
dirichlet_nodes = get_dirichlet_nodes(dirichlet_bool_mesh)
# Initialize structure mesh
specimen_mesh = StructureMesh(nodes_coords_mesh_init, elements_type,
                              connectivities, dirichlet_bool_mesh)
# Get number of mesh nodes
n_node_mesh = specimen_mesh.get_n_node_mesh()
# Set number of elements
n_elem = specimen_mesh.get_n_elem()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize reaction forces history
reaction_forces_mesh_hist = torch.zeros(((n_node_mesh, n_dim, n_time)))
# Set reaction forces history
for k_time in range(n_time):
    # Loop over nodes
    for i_node in range(n_node_mesh):
        # Loop over dimensions
        for j_dof in range(n_dim):
            # Set reaction force
            if dirichlet_bool_mesh[i_node, j_dof] == 1:
                reaction_forces_mesh_hist[i_node, j_dof, k_time] = 1.0 + k_time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize external forces history
external_forces_mesh_hist = torch.zeros_like(reaction_forces_mesh_hist)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set material learnable parameter
param_1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize internal forces history
internal_forces_mesh_hist = torch.zeros(((n_node_mesh, n_dim, n_time)))



# CODE ENHANCEMENT
if loss_diff_strategy == 'memory_efficient':
    # Initialize internal forces gradients history
    grad_mesh_hist = torch.zeros((n_node_mesh, n_dim, n_time))


# Loop over elements
for i in range(n_elem):
    # Get element label
    elem_id = i + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get element type
    element_type = elements_type[str(elem_id)]
    # Get element type number of nodes
    n_node = element_type.get_n_node()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute history of finite element internal forces
    element_internal_forces_hist = \
        compute_element_internal_forces_hist(n_dim, n_node, n_time, param_1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    # CODE ENHANCEMENT
    if loss_diff_strategy == 'memory_efficient':
        # Initialize history of finite element internal forces gradients
        element_grad_hist = torch.zeros((n_node*n_dim, n_time))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over element nodes
        for j in range(n_node):
            # Loop over dimensions
            for k in range(n_dim):
                # Loop over discrete time
                for time_idx in range(n_time):
                    # Get internal force
                    internal_force = \
                        element_internal_forces_hist[j*n_dim + k, time_idx]
                    # Compute gradient
                    internal_force.backward(retain_graph=True)
                    # Store gradient
                    element_grad_hist[j*n_dim + k, time_idx] = param_1.grad
                    # Reset parameter gradient
                    param_1.grad = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Release element internal forces history computation graph
        element_internal_forces_hist.detach_()

    
        
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for time_idx in range(n_time):
        # Reshape element internal forces into mesh format
        internal_forces_mesh = specimen_mesh.element_assembler(
            {str(elem_id): element_internal_forces_hist[:, time_idx]})
        # Assemble element internal forces of finite element mesh nodes
        internal_forces_mesh_hist[:, :, time_idx] += \
            internal_forces_mesh.reshape(-1, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if loss_diff_strategy == 'memory_efficient':
            # Reshape element internal forces gradients into mesh format
            grad_mesh = specimen_mesh.element_assembler(
                {str(elem_id): element_grad_hist[:, time_idx]})
            # Assemble element internal forces gradients of finite element mesh
            # nodes
            grad_mesh_hist[:, :, time_idx] += \
                grad_mesh.reshape(-1, n_dim)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# CODE ENHANCEMENT
# Compute loss gradient with respect to material parameter
if loss_diff_strategy == 'memory_efficient':
    # Initialize loss gradient
    loss_gradient = torch.tensor(0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes
    for i in range(n_node_mesh):
        # Loop over dimensions
        for j in range(n_dim):
            # Loop over discrete time
            for k in range(n_time):
                # Assemble contribution to loss gradient
                loss_gradient += 2*grad_mesh_hist[i, j, k]*(              
                    internal_forces_mesh_hist[i, j, k]
                    - external_forces_mesh_hist[i, j, k]
                    - reaction_forces_mesh_hist[i, j, k])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store loss gradient
    param_1.grad = loss_gradient




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize force equilibrium history loss
force_equilibrium_hist_loss = 0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over discrete time
for time_idx in range(n_time):
    # Get internal forces of finite element mesh nodes
    internal_forces_mesh = internal_forces_mesh_hist[:, :, time_idx]
    # Set null external forces of finite element mesh nodes
    external_forces_mesh = external_forces_mesh_hist[:, :, time_idx]
    # Get reaction forces (Dirichlet boundary conditions) of finite
    # element mesh nodes
    reaction_forces_mesh = reaction_forces_mesh_hist[:, :, time_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add contribution to force equilibrium history loss
    force_equilibrium_hist_loss += force_equilibrium_loss(
        internal_forces_mesh, external_forces_mesh,
        reaction_forces_mesh, dirichlet_bool_mesh)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute gradient
if loss_diff_strategy == 'naive':
    force_equilibrium_hist_loss.backward()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display force equilibrium loss differentiation strategy
print(f'\nGradient strategy: {loss_diff_strategy}')
# Display force equilibrium loss
print(f'\nForce equilibrium loss         : ', force_equilibrium_hist_loss)
# Display force equilibrium loss gradient with respect to material parameter
print(f'Force equilibrium loss gradient: ', param_1.grad)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute total execution time
total_time_sec = time.time() - start_time_sec
# Display total execution time
print(f'\nExecution time: {total_time_sec:.4f} s')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get process non-swapped physical memory usage
memory_usage = process.memory_info().rss
memory_usage_mb = memory_usage/(1024**2)
# Display memory usage
print(f'Memory usage  : {memory_usage_mb:.2f} MB')


"""
# Loop over nodes
for i_node in range(n_node_mesh):
    # Loop over dimensions
    for j_dof in range(n_dim):
        # Loop over discrete time
        for k_time in range(n_time):
            internal_force = i_node*j_dof*k_time*param_1
            
            if loss_diff_strategy == 'efficient':
                internal_force_squared = internal_force**2
                internal_force_squared.backward()
                internal_forces_mesh_hist[i_node, j_dof, k_time] = internal_force.detach()
            else:
                internal_forces_mesh_hist[i_node, j_dof, k_time] = internal_force
"""

"""
# Create an input tensor with requires_grad=True
x = torch.tensor(1.0, requires_grad=True)

# Perform some operation
y = x ** 2  # y = [1.0, 4.0, 9.0]

y = torch.zeros(2)
y[0] = 2*x
y[1] = 3*x

# Initialize a list to hold gradients for each element
grads = []

# Loop through each element in y
for i in range(y.size(0)):
    # Zero the gradients before each backward call
    x.grad = None
    
    # Compute the gradient for the i-th element
    y[i].backward(retain_graph=True)
    
    # Store the gradient for this particular output
    grads.append(x.grad.clone())

# Stack gradients to get a tensor where each row corresponds to the gradient
# of a specific output with respect to the inputs
grads = torch.stack(grads)

print(grads)

"""