"""Post-process specimen history data files. 

Functions
---------
plot_nodes_displacement_history(specimen_name, specimen_history_dir, \
                                nodes_labels=None, n_dim=3, save_dir=None)
    Plot specimen nodes displacement history from specimen history data.
plot_reaction_forces_history(specimen_name, specimen_history_dir, n_dim=3, \
                             reference_displacement_node=None, save_dir=None)
    Plot specimen reaction forces history from specimen history data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import pandas
import torch
# Local
from ioput.plots import plot_xy_data, save_figure
from user_scripts.global_model_update.material_finder.gen_specimen_data \
    import get_specimen_history_paths, get_specimen_numerical_data
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def plot_nodes_displacement_history(specimen_name, specimen_history_dir,
                                    nodes_labels=None, n_dim=3, save_dir=None):
    """Plot specimen nodes displacement history from specimen history data.
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    specimen_history_dir : str
        Specimen history data directory.
    nodes_labels : {int, list, 'all'}, default=None
        If provided, then displacement history of the nodes with the given
        labels are plotted. If None, then displacement history of all nodes is
        plotted.
    n_dim : int, default=3
        Number of spatial dimensions.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    """
    # Get specimen history data file paths
    specimen_history_paths = get_specimen_history_paths(specimen_history_dir,
                                                        specimen_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe first specimen history data file to get mesh parameters
    df = pandas.read_csv(specimen_history_paths[0])
    # Get number of mesh nodes
    n_node_mesh = df.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get specimen numerical data
    nodes_disps_mesh_hist, _, _, time_hist = get_specimen_numerical_data(
        specimen_history_paths, n_dim, n_node_mesh)
    # Get history length
    n_time = len(time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process nodes labels
    if isinstance(nodes_labels, int):
        nodes_labels = [nodes_labels,]
    elif isinstance(nodes_labels, list):
        pass
    elif nodes_labels == 'all':
        nodes_labels = list(range(1, n_node_mesh+1))
    else:
        raise ValueError('The node labels (\'nodes_labels\') must be provided '
                         'as an int, a list of ints or \'all\'.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over nodes labels
    for node_label in nodes_labels:
        # Get node index
        node_index = node_label - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data array
        data_array = np.zeros((n_time, 2*n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set x-axis label
        x_label = 'Time step'
        # Set x-axis limits
        x_lims = (0, None)
        # Set y-axis label
        y_label = 'Displacement'
        # Set y-axis limits
        y_lims = (None, None)
        # Set data labels
        data_labels = [f'Dim {i + 1}' for i in range(n_dim)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over spatial dimensions
        for i in range(n_dim):
            # Set x-axis data
            data_array[:, 2*i] = np.arange(n_time)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get node displacement history along spatial dimension
            node_disp_hist = \
                nodes_disps_mesh_hist[node_index, i, :].detach().cpu().numpy()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set y-axis data
            data_array[:, 2*i + 1] = node_disp_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot node displacement history
        figure, _ = plot_xy_data(data_array, data_labels=data_labels,
                                 x_lims=x_lims, y_lims=y_lims,
                                 x_label=x_label, y_label=y_label,
                                 is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set file name
        filename = f'displacement_hist_node_{node_label}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def plot_reaction_forces_history(specimen_name, specimen_history_dir, n_dim=3,
                                 reference_displacement_node=None,
                                 save_dir=None):
    """Plot specimen reaction forces history from specimen history data.
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    specimen_history_dir : str
        Specimen history data directory.
    n_dim : int, default=3
        Number of spatial dimensions.
    reference_displacement_node : tuple, default=None
        If provided, then displacement history of the reference node along a
        given spatial dimension replaces the discrete time steps in all
        reaction forces history plots x-axis. Tuple contains [0] the reference
        node label (int, between 1 and n_node_mesh) and [1] the spatial
        dimension index (int, between 0 and n_dim-1).
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    """
    # Get specimen history data file paths
    specimen_history_paths = get_specimen_history_paths(specimen_history_dir,
                                                        specimen_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe first specimen history data file to get mesh parameters
    df = pandas.read_csv(specimen_history_paths[0])
    # Get number of mesh nodes
    n_node_mesh = df.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get specimen numerical data
    nodes_disps_mesh_hist, reaction_forces_mesh_hist, \
        dirichlet_bc_mesh_hist, time_hist = get_specimen_numerical_data(
            specimen_history_paths, n_dim, n_node_mesh)
    # Get history length
    n_time = len(time_hist)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get initial Dirichlet boundary constraints
    dirichlet_bc_mesh_init = dirichlet_bc_mesh_hist[:, :, 0]
    # Get number of unique Dirichlet boundary constraints labels per dimension
    n_labels_per_dim = torch.tensor(
        [torch.unique(dirichlet_bc_mesh_init[:, i]).numel()
         for i in range(dirichlet_bc_mesh_init.shape[1])])
    # Get total number of unique Dirichlet boundary constraints labels
    n_labels_total = sum(n_labels_per_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize reaction forces history per Dirichlet boundary constraint
    # label
    dirichlet_sets_reaction_hist = torch.zeros(n_labels_total, 1, n_time)
    # Initialize assembly labels
    assembly_labels = []
    # Initialize assembly label index
    assembly_index = 0
    # Loop over dimensions
    for i in range(n_dim):
        # Get unique Dirichlet boundary constraint labels
        unique_labels_dim = torch.unique(dirichlet_bc_mesh_init[:, i])
        # Loop over unique Dirichlet boundary constraint labels
        for label in unique_labels_dim:
            # Get label nodes mask
            label_mask = torch.where(dirichlet_bc_mesh_init[:, i] == label)[0]
            # Loop over time steps
            for t in range(n_time):
                # Assemble reaction forces history per Dirichlet boundary
                # constraint label
                dirichlet_sets_reaction_hist[assembly_index, 0, t] = \
                    reaction_forces_mesh_hist[label_mask, i, t].sum()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store assembly label
            assembly_labels.append(f'dim_{i}_label_{label.item()}')
            # Increment assembly label index
            assembly_index += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over Dirichlet boundary constraint labels
    for i in range(n_labels_total):
        # Get Dirichlet boundary constraint label reaction force history
        dirichlet_set_reaction_hist = \
            dirichlet_sets_reaction_hist[i, 0, :].detach().cpu().numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data array
        data_array = np.zeros((n_time, 2))
        # Get x-axis data and label
        if isinstance(reference_displacement_node, tuple):
            # Get reference node label and spatial dimension index
            ref_node_label, ref_dim_index = reference_displacement_node
            # Get reference node index
            ref_node_index = ref_node_label - 1
            # Get reference node displacement history
            ref_node_disp_hist = \
                nodes_disps_mesh_hist[ref_node_index, ref_dim_index, :]
            # Set x-axis data as reference node displacement history
            data_array[:, 0] = ref_node_disp_hist.detach().cpu().numpy()
            # Set x-axis label
            x_label = 'Displacement'
            # Set x-axis limits
            x_lims = (None, None)
        else:
            # Set x-axis data as discrete time steps
            data_array[:, 0] = np.arange(n_time)
            # Set x-axis label
            x_label = 'Time step'
            # Set x-axis limits
            x_lims = (0, None)
        # Get y-axis data
        data_array[:, 1] = dirichlet_set_reaction_hist
        # Set y-axis label
        y_label = 'Reaction force'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot Dirichlet boundary constrain label reaction force history
        figure, _ = plot_xy_data(data_array, x_lims=x_lims,
                                 x_label=x_label,y_label=y_label,
                                 is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set file name
        filename = f'dirichlet_reaction_hist_{assembly_labels[i]}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
if __name__ == '__main__':
    # Set specimen name
    specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
    # Set specimen history data directory
    specimen_history_dir = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
         '1_DTP1U_data/loss_dirichlet_sets/'
         '0_abaqus_simulation_hexa8_8GP_gt_parameters/0_simulation/'
         'debug_plots/specimen_history_data')
    # Set number of spatial dimensions
    n_dim = 3
    # Set plots directory
    save_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
                '1_DTP1U_data/loss_dirichlet_sets/'
                '0_abaqus_simulation_hexa8_8GP_gt_parameters/0_simulation/'
                'debug_plots/plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set nodes labels
    nodes_labels = [131, 1301]
    # Plot nodes displacement history
    plot_nodes_displacement_history(
        specimen_name, specimen_history_dir, nodes_labels=nodes_labels,
        n_dim=n_dim, save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reference node displacement history
    reference_displacement_node = (131, 1)
    # Plot reaction forces history
    plot_reaction_forces_history(
        specimen_name, specimen_history_dir, n_dim=n_dim,
        reference_displacement_node=reference_displacement_node,
        save_dir=save_dir)