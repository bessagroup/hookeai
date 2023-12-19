"""Data set graph samples files.

Functions
---------
generate_dataset_samples_files
    Generate data set graph samples files.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import time
import datetime
# Third-party
import numpy as np
import torch
import tqdm
import pandas
# Local
from gnn_base_model.data.graph_data import GraphData
from gnn_base_model.data.graph_dataset import \
    write_graph_dataset_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
#
# =============================================================================
def generate_dataset_samples_files(dataset_directory, dataset_csv_file_path,
                                   sample_file_basename='material_patch_graph',
                                   is_save_sample_plot=False,
                                   is_verbose=False):
    """Generate data set graph samples files.

    Parameters
    ----------
    dataset_directory : str
        Directory where the data set is stored (all ata set samples files).
        All existent files are overridden when saving sample data files.
    dataset_csv_file_path : str
        Data set csv file path.
    sample_file_basename : str, default='sample_graph'
        Basename of data set sample file. The basename is appended with sample
        index.
    is_save_sample_plot : bool, default=False
        Save plot of each sample graph in the same directory where the data set
        is stored.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset_directory : str
        Directory where the data set is stored (all data set samples files).
    dataset_sample_files : list[str]
        Data set samples files paths. Each sample file contains a
        torch_geometric.data.Data object describing a homogeneous graph.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGenerate shell knock down data set'
              '\n----------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data set directory
    if not os.path.isdir(dataset_directory):
        raise RuntimeError('The data set directory has not been found:'
                           '\n\n' + dataset_directory)
    else:
        dataset_directory = os.path.normpath(dataset_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data set file
    if not os.path.isfile(dataset_csv_file_path):
        raise RuntimeError('The data set file has not been found:'
                           '\n\n' + dataset_csv_file_path)
    else:
        dataset_csv_file_path = os.path.normpath(dataset_csv_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    df = pandas.read_csv(dataset_csv_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get shells ids
    shells_ids = set(df.loc[:, 'shell_id'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize shells data
    shells_data = []
    # Loop over shells
    for shell_id in shells_ids:
        # Get shell data set
        df_shell_id = df.loc[df['shell_id'] == shell_id]
        # Initialize shell data
        shell_data = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize shell defects data
        shell_defects = []
        # Loop over shell defects
        for i, row in df_shell_id.iterrows():
            # Set defects attributes
            defect_attr = ('defect_id', 'theta', 'phi', 'delta', 'lambda')
            # Initialize defect data
            defect = {}
            # Build defect data
            for key in defect_attr:
                if key in ('defect_id',):
                    defect[key] = int(row[key])
                else:
                    defect[key] = row[key]
            # Assemble defect data
            shell_defects.append(defect)
        # Assemble shell defects data
        shell_data['defects'] = shell_defects
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set shell global attributes
        global_attr = ('shell_id', 'knock_down', 'radius', 'eta', 'nu')
        # Build shell global data
        for key in global_attr:
            value = df_shell_id.iloc[0, df_shell_id.columns.get_loc(key)] 
            if key in ('shell_id',):
                shell_data[key] = int(value)
            else:
                shell_data[key] = value
        # Assemble shell global data
        shells_data.append(shell_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of samples
    n_sample = len(shells_data)
    # Initialize data set samples files
    dataset_sample_files = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Starting graphs generation process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in tqdm.tqdm(range(n_sample), desc='> Generating graphs: ',
                       disable=not is_verbose):
        # Get sample
        shell_data = shells_data[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get shell radius
        shell_radius = shell_data['radius']
        # Get shell thickness
        shell_thickness = shell_data['radius']/shell_data['eta']
        # Get shell number of defects
        n_defects = len(shell_data['defects'])
        # Get shell Poisson's ratio
        poisson = shell_data['nu']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize defects (nodes) spherical and cartesian coordinates
        defects_coords_spherical = np.zeros((n_defects, 3))
        defects_coords = np.zeros((n_defects, 3))
        # Loop over defects
        for j, defect in enumerate(shell_data['defects']):
            # Assemble defect spherical coordinates
            defects_coords_spherical[j, :] = \
                np.array([shell_radius, defect['theta'], defect['phi']])
            # Assemble defect cartesian coordinates
            defects_coords[j, :] = np.array(
                [shell_radius*np.sin(defect['phi'])*np.cos(defect['theta']),
                 shell_radius*np.sin(defect['phi'])*np.sin(defect['theta']),
                 shell_radius*np.cos(defect['phi'])])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize defects (nodes) geometry
        defects_geometry = np.zeros((n_defects, 2))
        # Loop over defects
        for j, defect in enumerate(shell_data['defects']):
            # Assemble defect geometry
            defects_geometry[j, :] = \
                np.array([defect['delta'], defect['lambda']])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate graph data
        graph_data = GraphData(n_dim=3, nodes_coords=defects_coords)
        # Set connectivity radius (maximum distance between two nodes leading
        # to an edge)
        connect_radius = get_critical_buckling_wavelength(
            poisson, shell_radius, shell_thickness)
        # Set graph edges
        graph_data.set_graph_edges_indexes(connect_radius=connect_radius)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node features
        node_features = ('radius', 'theta', 'phi', 'delta', 'lambda')
        # Set node features matrix
        node_features_matrix = np.hstack((defects_coords, defects_geometry))
        # Set graph node features
        graph_data.set_node_features_matrix(node_features_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set global targets matrix
        global_targets_matrix = \
            np.array([shell_data['knock_down'],]).reshape(1, -1)
        # Set graph global targets
        graph_data.set_global_targets_matrix(global_targets_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get PyG homogeneous graph data object
        pyg_graph = graph_data.get_torch_data_object()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample file name
        sample_file_name = sample_file_basename + '_' + str(i) + '.pt'
        # Set sample file path
        sample_file_path = os.path.join(os.path.normpath(dataset_directory),
                                        sample_file_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save graph sample file
        torch.save(pyg_graph, sample_file_path)
        # Save graph sample file path
        dataset_sample_files.append(sample_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save graph sample plot
        if is_save_sample_plot:
            # Set sample plot name
            sample_file_name = sample_file_basename + '_' + str(i) + '_plot'
            # Save sample plot
            graph_data.plot_material_patch_graph(
                is_save_plot=is_save_sample_plot,
                save_directory=dataset_directory,
                plot_name=sample_file_name,
                is_overwrite_file=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished graphs generation process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Data set size (number of graphs): {n_sample}')
        if node_features_matrix is not None:
            print(f'\n> Node features ({node_features_matrix.shape[1]}): '
                f'{" || ".join([x for x in node_features])}')
        else:
            print(f'\n> Node features (0): None')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total generation time and average generation time per patch
    total_time_sec = time.time() - start_time_sec
    avg_time_sec = total_time_sec/n_sample
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Data set directory: ', dataset_directory)
        print(f'\n> Total generation time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. generation time per graph: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sec)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for graph data set generation
    write_graph_dataset_summary_file(dataset_directory, n_sample,
                                     total_time_sec, avg_time_sec,
                                     node_features=node_features)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    return dataset_directory, dataset_sample_files
# =============================================================================
def get_critical_buckling_wavelength(poisson, shell_radius, shell_thickness):
    """Compute shell critical buckling wavelength.
    
    Parameters
    ----------
    poisson : float
        Material Poisson ratio.
    shell_radius : float
        Shell radius.
    shell_thickness : float
        Shell thickness.
        
    Returns
    -------
    critical_bw : float
        Critical buckling wavelength.
    """
    # Compute critical buckling wavelength
    critical_bw = 2*np.pi*((12*(1 - poisson**2))**(-1/4))*(
        (shell_radius*shell_thickness)**(1/2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return critical_bw