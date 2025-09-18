"""Post-process specimen history data files. 

Functions
---------
plot_nodes_displacement_history(specimen_name, specimen_history_dir, \
                                nodes_labels=None, n_dim=3, save_dir=None)
    Plot specimen nodes displacement history from specimen history data.
plot_reaction_forces_history(specimen_name, specimen_history_dir, n_dim=3, \
                             reference_displacement_node=None, save_dir=None)
    Plot specimen reaction forces history from specimen history data.
build_dirichlet_sets_reaction_history(specimen_name, specimen_history_dir, \
                                      n_dim=3)
    Build Dirichlet sets reaction forces history from specimen data.
compare_reaction_forces_history(dirichlet_sets_reaction_hist, \
                                dirichlet_sets_data_labels, save_dir=None)
    Compare reaction forces history between different Dirichlet sets.
copy_dbc_labels(specimen_name, src_specimen_history_dir, \
                target_specimen_history_dir)
    Copy Dirichlet boundary constraints labels from specimen history data.
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
import pickle
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
    # Build Dirichlet sets reaction forces history
    dirichlet_sets_reaction_hist, dirichlet_sets_data_labels, \
        nodes_disps_mesh_hist = build_dirichlet_sets_reaction_history(
            specimen_name, specimen_history_dir, n_dim)    
    # Get total number of Dirichlet boundary constraint labels
    n_labels_total = dirichlet_sets_reaction_hist.shape[0]
    # Get history length
    n_time = dirichlet_sets_reaction_hist.shape[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over Dirichlet boundary constraint labels
    for i in range(n_labels_total):
        # Get Dirichlet boundary constraint label reaction force history
        dirichlet_set_reaction_hist = dirichlet_sets_reaction_hist[i, 0, :]
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
            data_array[:, 0] = ref_node_disp_hist
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
        filename = f'dirichlet_reaction_hist_{dirichlet_sets_data_labels[i]}'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def build_dirichlet_sets_reaction_history(specimen_name, specimen_history_dir,
                                          n_dim=3):
    """Build Dirichlet sets reaction forces history from specimen data.
    
    Parameters
    ----------
    specimen_name : str
        Specimen name.
    specimen_history_dir : str
        Specimen history data directory.
    n_dim : int, default=3
        Number of spatial dimensions.
        
    Returns
    -------
    dirichlet_sets_reaction_hist : numpy.ndarray(3d)
        Reaction forces history of Dirichlet boundary sets stored as
        numpy.ndarray(3d) of shape (n_sets, 1, n_time).
    dirichlet_sets_data_labels : list[str]
        Dirichlet boundary sets data labels.
    nodes_disps_mesh_hist : numpy.ndarray(3d)
        Nodes displacements history stored as numpy.ndarray(3d) of shape
        (n_node_mesh, n_dim, n_time).
    """
    # Get specimen history data file paths
    specimen_history_paths = \
        get_specimen_history_paths(specimen_history_dir, specimen_name)
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
    # Initialize Dirichlet sets data labels
    dirichlet_sets_data_labels = []
    # Initialize assembly index
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
            # Store Dirichlet sets data labels
            dirichlet_sets_data_labels.append(
                f'dim_{i + 1}_label_{label.item()}')
            # Increment assembly index
            assembly_index += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert to numpy arrays
    dirichlet_sets_reaction_hist = \
        dirichlet_sets_reaction_hist.detach().cpu().numpy()
    nodes_disps_mesh_hist = \
        nodes_disps_mesh_hist.detach().cpu().numpy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dirichlet_sets_reaction_hist, dirichlet_sets_data_labels, \
        nodes_disps_mesh_hist
# =============================================================================
def compare_reaction_forces_history(dirichlet_sets_reaction_hist,
                                    dirichlet_sets_data_labels,
                                    save_dir=None):
    """Compare reaction forces history between different Dirichlet sets.
    
    Instead of plotting reaction forces history for each Dirichlet boundary
    set separately, this function plots them all together for comparison. All
    Dirichlet boundary sets reaction forces history are expected to have the
    same history length.
    
    Parameters
    ----------
    dirichlet_sets_reaction_hist : numpy.ndarray(3d)
        Reaction forces history of Dirichlet boundary sets stored as
        numpy.ndarray(3d) of shape (n_sets, 1, n_time).
    dirichlet_sets_data_labels : list[str]
        List of strings with the Dirichlet boundary sets data labels.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    """
    # Check Dirichlet boundary sets data labels
    if (len(dirichlet_sets_data_labels)
        != dirichlet_sets_reaction_hist.shape[0]):
        raise RuntimeError('The number of Dirichlet boundary sets data labels '
                           f'({len(dirichlet_sets_data_labels)}) must be '
                           'equal to the number of Dirichlet boundary '
                           'sets for which the reaction forces history is '
                           'provided.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of Dirichlet boundary sets
    n_set = dirichlet_sets_reaction_hist.shape[0]
    # Get history length
    n_time = dirichlet_sets_reaction_hist.shape[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array
    data_array = np.zeros((n_time, 2*n_set))
    # Build data array
    for i in range(n_set):
        data_array[:, 2*i] = np.arange(n_time)
        data_array[:, 2*i + 1] = dirichlet_sets_reaction_hist[i, 0, :]
    # Set data labels
    data_labels = dirichlet_sets_data_labels
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set x-axis label
    x_label = 'Time step'
    # Set x-axis limits
    x_lims = (0, None)
    # Set y-axis label
    y_label = 'Reaction force'
    # Set y-axis limits
    y_lims = (None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Dirichlet boundary sets reaction forces history
    figure, _ = plot_xy_data(data_array, data_labels=data_labels,
                             x_lims=x_lims, y_lims=y_lims, x_label=x_label,
                             y_label=y_label, is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set file name
    filename = 'dirichlet_reaction_hist_comparison'
    # Save figure
    save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
def copy_dbc_labels(specimen_name, src_specimen_history_dir,
                    target_specimen_history_dir):
    """Copy Dirichlet boundary constraints labels from specimen history data.
    
    Given a source and target specimen history data directories for a given
    specimen, copies the Dirichlet boundary constraints labels from the source
    to the target directory files. Source and target specimen history data
    files must be consistent (same specimen, mesh and history length).

    Parameters
    ----------
    specimen_name : str
        Specimen name.
    src_specimen_history_dir : str
        Source specimen history data directory.
    target_specimen_history_dir : str
        Target specimen history data directory.
    """
    # Get source specimen history data file paths
    src_specimen_history_paths = get_specimen_history_paths(
        src_specimen_history_dir, specimen_name)
    # Get target specimen history data file paths
    target_specimen_history_paths = get_specimen_history_paths(
        target_specimen_history_dir, specimen_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get time history length
    if len(src_specimen_history_paths) != len(target_specimen_history_paths):
        raise RuntimeError('Source and target specimen history data files '
                           'must have the same history length. Source has '
                           f'{len(src_specimen_history_paths)} time steps, '
                           f'target has {len(target_specimen_history_paths)} '
                           'time steps.')
    else:
        n_time = len(src_specimen_history_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for i in range(n_time):
        # Load source specimen history data file
        df_src = pandas.read_csv(src_specimen_history_paths[i])
        # Load target specimen history data file
        df_target = pandas.read_csv(target_specimen_history_paths[i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if source specimen history data file has Dirichlet boundary
        # constraints labels
        if not all(col in df_src.columns for col in ['DBC1', 'DBC2', 'DBC3']):
            raise RuntimeError('Source specimen history data file does not '
                               'have Dirichlet boundary constraints labels '
                               '(\'DBC1\', \'DBC2\', \'DBC3\').')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Copy Dirichlet boundary constraints labels from source to target
        df_target[['DBC1', 'DBC2', 'DBC3']] = df_src[['DBC1', 'DBC2', 'DBC3']]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save modified target specimen history data file
        df_target.to_csv(target_specimen_history_paths[i], encoding='utf-8',
                         index=False)   
# =============================================================================
if __name__ == '__main__':
    # Set computation process
    process = ('plot_nodes_displacement_history',
               'plot_reaction_forces_history',
               'compare_reaction_forces_history',
               'copy_dbc_labels')[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot nodes displacement history or reaction forces history
    if process in ('plot_nodes_displacement_history',
                   'plot_reaction_forces_history'):
        # Set specimen name
        specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
        # Set specimen history data directory
        specimen_history_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '1_DTP1U_data/loss_dirichlet_sets/'
             '0_abaqus_simulation_hexa8_8GP_gt_parameters/0_simulation/'
             'specimen_history_data')
        # Set number of spatial dimensions
        n_dim = 3
        # Set plots directory
        save_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '1_DTP1U_data/loss_dirichlet_sets/'
             '0_abaqus_simulation_hexa8_8GP_gt_parameters/0_simulation/'
             'debug_plots_2')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes labels
        nodes_labels = [131, 1301]
        # Plot nodes displacement history
        plot_nodes_displacement_history(
            specimen_name, specimen_history_dir, nodes_labels=nodes_labels,
            n_dim=n_dim, save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference node displacement history
        reference_displacement_node = (131, 1)
        # Plot reaction forces history
        plot_reaction_forces_history(
            specimen_name, specimen_history_dir, n_dim=n_dim,
            reference_displacement_node=reference_displacement_node,
            save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif process == 'compare_reaction_forces_history':
        # Set plots directory
        save_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '2_DTP1U_V2_data/loss_dirichlet_sets/'
             '3_comparison_with_synthetic_dtp1v4')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist_data = []
        dirichlet_sets_data_labels = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen name
        specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
        # Set specimen history data directory
        specimen_history_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '0_DTP1_V4_data/loss_dirichlet_sets/'
             '0_abaqus_simulation_hexa8_8GP_gt_parameters/0_simulation/'
             'specimen_history_data')
        # Set number of spatial dimensions
        n_dim = 3
        # Build Dirichlet sets reaction forces history
        dirichlet_sets_reaction_hist, _, _ = \
            build_dirichlet_sets_reaction_history(
                specimen_name, specimen_history_dir, n_dim=n_dim)
        # Store Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist_data.append(
            dirichlet_sets_reaction_hist[4:5, :, :])
        # Set Dirichlet sets data label
        dirichlet_sets_data_labels.append('DTP1_V4 (Synth)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Dirichlet sets reaction forces history file
        dirichlet_sets_reaction_hist_file_path = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '0_DTP1_V4_data/loss_dirichlet_sets/'
             '2_adimu_forward_hexa8_1GP_opt_parameters/'
             '2_discover_rc_von_mises_adimu_force_displacement/'
             'material_model_finder/3_model/dirichlet_sets_reaction_forces/'
             'dirichlet_sets_data.pkl')
        # Load Dirichlet sets reaction forces history data
        with open(dirichlet_sets_reaction_hist_file_path, 'rb') as file:
            dirichlet_sets_reaction_hist = \
                pickle.load(file)['dirichlet_sets_reaction_hist']
        # Store Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist_data.append(
            dirichlet_sets_reaction_hist[2:3, :, :])
        # Set Dirichlet sets data label
        dirichlet_sets_data_labels.append('DTP1_V4 (Optim)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set specimen name
        specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
        # Set specimen history data directory
        specimen_history_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '2_DTP1U_V2_data/loss_dirichlet_sets/'
             '2_adimu_forward_hexa8_1GP_opt_parameters/'
             '2_discover_rc_von_mises_adimu_force_displacement/'
             'material_model_finder/0_simulation/specimen_history_data')
        # Set number of spatial dimensions
        n_dim = 3
        # Build Dirichlet sets reaction forces history
        dirichlet_sets_reaction_hist, _, _ = \
            build_dirichlet_sets_reaction_history(
                specimen_name, specimen_history_dir, n_dim=n_dim)
        # Store Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist_data.append(
            dirichlet_sets_reaction_hist[4:5, :, :])
        # Set Dirichlet sets data label
        dirichlet_sets_data_labels.append('DTP1U_V2 (Exp)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Dirichlet sets reaction forces history file
        dirichlet_sets_reaction_hist_file_path = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             '2_DTP1U_V2_data/loss_dirichlet_sets/'
             '2_adimu_forward_hexa8_1GP_opt_parameters/'
             '2_discover_rc_von_mises_adimu_force_displacement/'
             'material_model_finder/3_model/dirichlet_sets_reaction_forces/'
             'dirichlet_sets_data.pkl')
        # Load Dirichlet sets reaction forces history data
        with open(dirichlet_sets_reaction_hist_file_path, 'rb') as file:
            dirichlet_sets_reaction_hist = \
                pickle.load(file)['dirichlet_sets_reaction_hist']
        # Store Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist_data.append(
            dirichlet_sets_reaction_hist[2:3, :, :])
        # Set Dirichlet sets data label
        dirichlet_sets_data_labels.append('DTP1U_V2 (Optim)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate Dirichlet sets reaction forces history data
        dirichlet_sets_reaction_hist = \
            np.concatenate(dirichlet_sets_reaction_hist_data, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compare reaction forces history between different Dirichlet sets
        compare_reaction_forces_history(
            dirichlet_sets_reaction_hist, dirichlet_sets_data_labels,
            save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif process == 'copy_dbc_labels':
        # Set specimen name
        specimen_name = 'Ti6242_HIP2_UT_Specimen2_J2'
        # Set source specimen history data directory
        src_specimen_history_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/'
             '3_dtp1_j2_rowan_data/testing/dtp1u/specimen_history_data')
        # Set target specimen history data directory
        target_specimen_history_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_antonios/dtp_validation/3_dtp1_j2_rowan_data/'
             'testing/dtp1v4/specimen_history_data')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Copy Dirichlet boundary constraints labels from source to target
        copy_dbc_labels(specimen_name, src_specimen_history_dir,
                        target_specimen_history_dir)