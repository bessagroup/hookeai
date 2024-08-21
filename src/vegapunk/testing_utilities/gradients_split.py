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
import matplotlib.pyplot as plt
# Local
from simulators.fetorch.structure.structure_mesh import StructureMesh
from simulators.fetorch.element.type.quad4 import FEQuad4
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Testing explicit gradient handling to minimize memory costs
# =============================================================================
def get_memory_usage():
    # Get the process ID (PID) of the current process
    process = psutil.Process(os.getpid())
    # Get process non-swapped physical memory usage (bytes)
    memory_usage = process.memory_info().rss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return memory_usage
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
def compute_element_internal_forces_hist(n_dim, n_node, n_time, params):
    """Compute history of finite element internal forces.
    
    Parameters
    ----------
    ...
    
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
                # Compute parameter dependent term
                param_term = sum(params)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute internal force
                internal_force = \
                    (i_node + 1)*(j_dof + 1)*(k_time + 1)*param_term
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
def forward_sequential_element(strategy, n_elem_dim, n_time,
                               n_params):
    """Forward propagation (sequential element).
    
    Parameters
    ----------
    ...

    """
    # Set number of spatial dimensions
    n_dim = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate uniform mesh of 4-node quadrilateral elements
    nodes_coords_mesh_init, elements_type, connectivities, \
        dirichlet_bool_mesh = generate_quad4_uniform_mesh(n_elem_dim)
    # Initialize structure mesh
    specimen_mesh = StructureMesh(nodes_coords_mesh_init, elements_type,
                                  connectivities, dirichlet_bool_mesh)
    # Get number of mesh nodes
    n_node_mesh = specimen_mesh.get_n_node_mesh()
    # Set number of elements
    n_elem = specimen_mesh.get_n_elem()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                    reaction_forces_mesh_hist[i_node, j_dof, k_time] = \
                        1.0 + k_time
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize external forces history
    external_forces_mesh_hist = torch.zeros_like(reaction_forces_mesh_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of parameters
    n_params = 2
    # Set material learnable parameters
    params = [torch.nn.Parameter(torch.rand(1)[0], requires_grad=True)
              for _ in range(n_params)]
    params = [torch.nn.Parameter(torch.tensor(p + 1.0), requires_grad=True)
              for p in range(n_params)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize internal forces history
    internal_forces_mesh_hist = torch.zeros(((n_node_mesh, n_dim, n_time)))


    # CODE ENHANCEMENT
    if strategy == 'memory_efficient':
        # Initialize internal forces gradients history
        grad_mesh_hist = torch.zeros((n_node_mesh, n_dim, n_time, n_params))


    # Loop over elements
    for i in range(n_elem):
        # Get element label
        elem_id = i + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element type
        element_type = elements_type[str(elem_id)]
        # Get element type number of nodes
        n_node = element_type.get_n_node()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute history of finite element internal forces
        element_internal_forces_hist = \
            compute_element_internal_forces_hist(n_dim, n_node, n_time, params)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


        # CODE ENHANCEMENT
        if strategy == 'memory_efficient':
            # Initialize history of finite element internal forces gradients
            element_grad_hist = torch.zeros((n_node*n_dim, n_time, n_params))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over parameters
                        for p, param in enumerate(params):
                            # Store gradient
                            element_grad_hist[j*n_dim + k, time_idx, p] = \
                                param.grad
                            # Reset parameter gradient
                            param.grad = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Release element internal forces history computation graph
            element_internal_forces_hist.detach_()

  
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete time
        for time_idx in range(n_time):
            # Reshape element internal forces into mesh format
            internal_forces_mesh = specimen_mesh.element_assembler(
                {str(elem_id): element_internal_forces_hist[:, time_idx]})
            # Assemble element internal forces of finite element mesh nodes
            internal_forces_mesh_hist[:, :, time_idx] += \
                internal_forces_mesh.reshape(-1, n_dim)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if strategy == 'memory_efficient':
                # Loop over parameters
                for p, param in enumerate(params):
                    # Reshape element internal forces gradients into mesh
                    # format
                    grad_mesh = specimen_mesh.element_assembler(
                        {str(elem_id): element_grad_hist[:, time_idx, p]})
                    # Assemble element internal forces gradients of finite
                    # element mesh nodes
                    grad_mesh_hist[:, :, time_idx, p] += \
                        grad_mesh.reshape(-1, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # CODE ENHANCEMENT
    if strategy == 'memory_efficient':
        # Loop over parameters
        for p, param in enumerate(params):
            # Initialize loss gradient
            loss_gradient = torch.tensor(0.0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over nodes
            for i in range(n_node_mesh):
                # Loop over dimensions
                for j in range(n_dim):
                    # Loop over discrete time
                    for k in range(n_time):
                        # Assemble contribution to loss gradient
                        loss_gradient += 2*grad_mesh_hist[i, j, k, p]*(              
                            internal_forces_mesh_hist[i, j, k]
                            - external_forces_mesh_hist[i, j, k]
                            - reaction_forces_mesh_hist[i, j, k])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store loss gradient
            param.grad = loss_gradient

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize force equilibrium history loss
    force_equilibrium_hist_loss = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete time
    for time_idx in range(n_time):
        # Get internal forces of finite element mesh nodes
        internal_forces_mesh = internal_forces_mesh_hist[:, :, time_idx]
        # Set null external forces of finite element mesh nodes
        external_forces_mesh = external_forces_mesh_hist[:, :, time_idx]
        # Get reaction forces (Dirichlet boundary conditions) of finite
        # element mesh nodes
        reaction_forces_mesh = reaction_forces_mesh_hist[:, :, time_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add contribution to force equilibrium history loss
        force_equilibrium_hist_loss += force_equilibrium_loss(
            internal_forces_mesh, external_forces_mesh,
            reaction_forces_mesh, dirichlet_bool_mesh)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute gradient
    if strategy == 'full_graph':
        force_equilibrium_hist_loss.backward()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display force equilibrium loss
    print(f'\n  > Force equilibrium loss: ', force_equilibrium_hist_loss)
    # Display force equilibrium loss gradient with respect to material parameter
    print(f'\n  > Force equilibrium loss gradient: ')
    for p, param in enumerate(params):
        print(f'    > Parameter {p} gradient: ', param.grad)        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    # Set loss differentiation strategy
    #strategy_range = ('full_graph', )
    #strategy_range = ('memory_efficient', )
    strategy_range = ('full_graph', 'memory_efficient')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of parameters
    n_params_range = (1,)
    # Set number of elements per dimension
    n_elem_range = (10000,)
    n_elem_range = (1, 3, 10, 30, 100)
    # Set time history length
    n_time_range = (100,)
    #n_time_range = (1, 10, 100, 1000)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize performance data
    peak_memory_data = torch.zeros((len(n_elem_range), len(n_time_range),
                                    len(n_params_range), 2))
    exec_time_data = torch.zeros_like(peak_memory_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display
    print('\nStarting performance parametric analysis'
          '\n----------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over numbers of parameters
    for p, n_params in enumerate(n_params_range):
        # Loop over numbers of elements
        for i, n_elem in enumerate(n_elem_range):
            # Set number of elements per dimension
            n_elem_dim = 2*(n_elem,)
            # Loop over time history lengths
            for j, n_time in enumerate(n_time_range):
                # Loop over differentiation strategies
                for s, strategy in enumerate(strategy_range):
                    # Display
                    print(f'\n\nIteration: n_params = {n_params} '
                            f'| n_elem_dim = {n_elem} | n_time = {n_time} '
                            f'| strategy = {strategy}')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Initialize timer
                    start_time_sec = time.time()
                    # Initialize memory usage
                    start_memory_usage = get_memory_usage()                    
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute force equilibrium loss for given discrete time
                    forward_sequential_element(strategy, n_elem_dim,
                                               n_time, n_params)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute peak memory usage
                    peak_memory = get_memory_usage() - start_memory_usage
                    # Compute execution time
                    exec_time_sec = time.time() - start_time_sec
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Display peak memory usage
                    peak_memory_mb = peak_memory/(1024**2)
                    print(f'\n  > Peak memory usage: {peak_memory_mb:.2f} MB')
                    # Display execution time
                    print(f'\n  > Execution time: {exec_time_sec:.4f} s')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store performance data
                    peak_memory_data[i, j, p, s] = peak_memory_mb
                    exec_time_data[i, j, p, s] = exec_time_sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots
    is_plot = True
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/5_global_specimens/memory_bottleneck')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot: Computational time vs Number of elements
    if is_plot and len(n_elem_range) > 1:
        # Set data array
        data_array = np.zeros((len(n_elem_range), 4))
        data_array[:, 0] = [x**2 for x in n_elem_range]
        data_array[:, 1] = exec_time_data[:, 0, 0, 0]
        data_array[:, 2] = data_array[:, 0]
        data_array[:, 3] = exec_time_data[:, 0, 0, 1]
        # Set data labels
        data_labels = strategy_range
        # Set title
        title = (f'$n_e = \Delta$ / $n_t = {n_time}$ / $n_p = {n_params}$')
        # Set axes labels
        x_label = 'Number of elements'
        y_label = 'Computational time (s)'
        # Set axes limits
        x_lims = (None, None)
        y_lims = (None, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'log'
        # Plot data
        figure, axes = plot_xy_data(data_array, data_labels=data_labels,
                                    x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    x_scale=x_scale, y_scale=y_scale,
                                    title=title, marker='o', is_latex=True)
        # Display figure
        plt.show()
        # Set file name
        filename = f'time_vs_ne_for_nt_{n_time}_np_{n_params}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=plots_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(figure)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot: Computational time vs Number of time steps
    if is_plot and len(n_time_range) > 1:
        # Set data array
        data_array = np.zeros((len(n_time_range), 4))
        data_array[:, 0] = [x**2 for x in n_time_range]
        data_array[:, 1] = exec_time_data[0, :, 0, 0]
        data_array[:, 2] = data_array[:, 0]
        data_array[:, 3] = exec_time_data[0, :, 0, 1]
        # Set data labels
        data_labels = strategy_range
        # Set title
        title = (f'$n_e = {n_elem}$ / $n_t = \Delta$ / $n_p = {n_params}$')
        # Set axes labels
        x_label = 'Number of time steps'
        y_label = 'Computational time (s)'
        # Set axes limits
        x_lims = (None, None)
        y_lims = (None, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'log'
        # Plot data
        figure, axes = plot_xy_data(data_array, data_labels=data_labels,
                                    x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    x_scale=x_scale, y_scale=y_scale,
                                    title=title, marker='o', is_latex=True)
        # Display figure
        plt.show()
        # Set file name
        filename = f'time_vs_nt_for_ne_{n_elem}_np_{n_params}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=plots_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(figure)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot: Computational time vs Number of parameters
    if is_plot and len(n_params_range) > 1:
        # Set data array
        data_array = np.zeros((len(n_params_range), 4))
        data_array[:, 0] = [x**2 for x in n_params_range]
        data_array[:, 1] = exec_time_data[0, 0, :, 0]
        data_array[:, 2] = data_array[:, 0]
        data_array[:, 3] = exec_time_data[0, 0, :, 1]
        # Set data labels
        data_labels = strategy_range
        # Set title
        title = (f'$n_e = {n_elem}$ / $n_t = {n_time}$ / $n_p = \Delta$')
        # Set axes labels
        x_label = 'Number of parameters'
        y_label = 'Computational time (s)'
        # Set axes limits
        x_lims = (None, None)
        y_lims = (None, None)
        # Set axes scale
        x_scale = 'log'
        y_scale = 'log'
        # Plot data
        figure, axes = plot_xy_data(data_array, data_labels=data_labels,
                                    x_lims=x_lims, y_lims=y_lims,
                                    x_label=x_label, y_label=y_label,
                                    x_scale=x_scale, y_scale=y_scale,
                                    title=title, marker='o', is_latex=True)
        # Display figure
        plt.show()
        # Set file name
        filename = f'time_vs_np_for_ne_{n_elem}_nt_{n_time}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=plots_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(figure)
# =============================================================================
"""Annotations

np = 1
------

nt = 1:

              Full Graph            Memory Efficiency
         --------------------     --------------------
  ne      time(s)  memory(MB)      time(s)  memory(MB)
    1      0.01       12.3         0.0028     12.2
   10     0.007       13.2         0.0157     12.3
  100      0.06       21.7         0.1598     13.0
 1000      0.55       93.8         1.4198     18.9
10000         7      883.3        16.3368     70.9


nt = 5:

              Full Graph            Memory Efficiency
         --------------------     --------------------
  ne      time(s)  memory(MB)      time(s)  memory(MB)
    1      0.018      12.8          0.030     12.2
   10      0.028      16.4          0.258     12.5
  100      0.288      55.8          2.800     13.0
 1000      2.835     390.3          25.23     19.3
10000      43.72      4184          290.6     72.6


nt = 10:

              Full Graph            Memory Efficiency
         --------------------     --------------------
  ne      time(s)  memory(MB)      time(s)  memory(MB)
    1     0.023        13             0.1      13
   10      0.05        20             0.9      13
  100       0.6        98              10      14
 1000         5       762              90      20
10000        96      8311            1035      73

"""