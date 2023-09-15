"""GNN-based material patch graph data.

Classes
-------
GNNPatchGraphData
    GNN-based material patch graph data.
GNNPatchFeaturesGenerator:
    GNN-based material patch input and output features generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
# Third-party
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import torch
import torch_geometric.data
# Local
from ioput.iostandard import new_file_path_with_int
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class GNNPatchGraphData:
    """GNN-based material patch graph data.
    
    Attributes
    ----------
    _n_dim : int
        Number of spatial dimensions.
    _nodes_coords : numpy.ndarray(2d), default=None
        Coordinates of nodes stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    _node_features_matrix : numpy.ndarray(2d), default=None
        Nodes input features matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_features).
    _edge_features_matrix : numpy.ndarray(2d), default=None
        Edges input features matrix stored as a numpy.ndarray(2d) of shape
        (n_edges, n_features).
    _global_features_matrix : numpy.ndarray(2d), default=None
        Global input features matrix stored as a numpy.ndarray(2d) of shape
        (1, n_features).
    _node_targets_matrix : numpy.ndarray(2d), default=None
        Nodes targets matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_targets).
    _edge_targets_matrix : numpy.ndarray(2d), default=None
        Edges targets matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_targets).
    _global_targets_matrix : numpy.ndarray(2d), default=None
        Global targets matrix stored as a numpy.ndarray(2d) of shape
        (1, n_targets).
    _edges_indexes : numpy.ndarray(2d)
        Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
        (num_edges, 2), where the i-th edge is stored in
        edges_indexes[i, :] as (start_node_index, end_node_index).
    
    Methods
    -------
    get_torch_data_object(self)
        Get PyG homogeneous graph data object.
    set_graph_edges_indexes(self, connect_radius=None, \
                            edges_indexes_mesh=None)
        Set material patch graph edges indexes.
    get_graph_edges_indexes(self)
        Get material patch graph edges indexes.
    set_node_features_matrix(self, node_features_matrix)
        Set nodes input features matrix.
    get_node_features_matrix(self)
        Set nodes input features matrix.
    set_edge_features_matrix(self, edge_features_matrix)
        Set edges input features matrix.
    get_edge_features_matrix(self)
        Get edges input features matrix.
    set_global_features_matrix(self, global_features_matrix)
        Set global input features matrix.
    get_global_features_matrix(self)
        Get global input features matrix.  
    set_node_targets_matrix(self, node_targets_matrix)
        Set node targets matrix.
    get_node_targets_matrix(self)
        Get node targets matrix.  
    set_edge_targets_matrix(self, edge_targets_matrix)
        Set edge targets matrix.   
    get_edge_targets_matrix(self)
        Get edge targets matrix.
    set_global_targets_matrix(self, global_targets_matrix)
        Set global targets matrix.
    get_global_targets_matrix(self)
        Get global targets matrix.
    plot_material_patch_graph(self, is_show_plot=False, is_save_plot=False, \
                              save_directory=None, plot_name=None, \
                              is_overwrite_file=False)
        Generate plot of material patch graph.
    _get_edges_from_local_radius(nodes_coords, connect_radius)
        Get edges between nodes that are within a given connectivity radius.
    get_undirected_unique_edges(edges_indexes)
        Get set of undirected unique edges indexes.
    _check_edges_indexes_matrix(edges_indexes)
        Check if given edges indexes matrix is valid.
    """
    def __init__(self, n_dim, nodes_coords):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        nodes_coords : numpy.ndarray(2d)
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        """
        self._n_dim = n_dim
        self._nodes_coords = nodes_coords[:, :n_dim]
        # Initialize graph edges
        self._edges_indexes=None
        # Initialize features matrices
        self._node_features_matrix = None
        self._edge_features_matrix = None
        self._global_features_matrix = None
        self._node_targets_matrix = None
        self._edge_targets_matrix = None
        self._global_targets_matrix = None
    # -------------------------------------------------------------------------  
    def get_torch_data_object(self):
        """Get PyG homogeneous graph data object.
        
        Returns
        -------
        pyg_graph : torch_geometric.data.Data
            PyG data object describing a homogeneous graph.
        """        
        # Set PyG node feature matrix
        x = None
        if self._node_features_matrix is not None:
            x = torch.tensor(copy.deepcopy(self._node_features_matrix),
                             dtype=torch.float)
        # Set PyG graph connectivity
        edge_index = None
        if self._edges_indexes is not None:
            edge_index = torch.tensor(
                np.transpose(copy.deepcopy(self._edges_indexes)),
                dtype=torch.long)
        # Set PyG edge feature matrix
        edge_attr = None
        # Set PyG ground-truth labels
        y = None
        if self._node_targets_matrix is not None:
            y = torch.tensor(copy.deepcopy(self._node_targets_matrix),
                             dtype=torch.float)
        # Set PyG node position matrix
        pos = None
        if self._nodes_coords is not None:
            pos = torch.tensor(copy.deepcopy(self._nodes_coords),
                               dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate PyG homogeneous graph data object
        pyg_graph = torch_geometric.data.Data(x=x, edge_index=edge_index,
                                              edge_attr=edge_attr, y=y,
                                              pos=pos)
        # Validate graph data object
        pyg_graph.validate(raise_on_error=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return pyg_graph
    # -------------------------------------------------------------------------
    def set_graph_edges_indexes(self, connect_radius=None,
                                edges_indexes_mesh=None):
        """Set material patch graph edges indexes.
        
        Parameters
        ----------
        connect_radius : float, default=None
            Connectivity radius that sets the maximum distance between two
            nodes that leads to an edge. If None, then no edges are generated
            from distance-based search.
        edges_indexes_mesh : numpy.ndarray(2d), default=None
            Edges stemming from any relevant mesh representation (e.g., finite
            element mesh) and that should be accounted for. Edges indexes
            matrix stored as numpy.ndarray[int](2d) with shape (n_edges, 2),
            where the i-th edge is stored in edges_indexes[i, :] as
            (start_node_index, end_node_index).
        """
        # Initialize edges indexes
        edges_indexes = np.empty((0, 2), dtype=int)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get edges from distance-based search within given connectivity radius
        if connect_radius is not None:
            edge_indexes_radius = type(self)._get_edges_from_local_radius(
                nodes_coords=self._nodes_coords, connect_radius=connect_radius)
            # Append distance-based edges indexes
            edges_indexes = \
                np.concatenate((edges_indexes, edge_indexes_radius), axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append given edges stemming from mesh representation
        if edges_indexes_mesh is not None:
            # Check mesh-based edges indexes
            GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes_mesh)
            # Append mesh-based edges indexes
            edges_indexes = np.concatenate((edges_indexes, edges_indexes_mesh),
                                           axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove any existent duplicated edges
        edges_indexes = np.unique(edges_indexes, axis=0) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edges indexes
        self._edges_indexes = edges_indexes
    # -------------------------------------------------------------------------
    def get_nodes_coords(self):
        """Get material patch graph node coordinates.
        
        Returns
        -------
        nodes_coords : numpy.ndarray(2d), default=None
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        """
        return copy.deepcopy(self._nodes_coords)
    # -------------------------------------------------------------------------
    def get_graph_edges_indexes(self):
        """Get material patch graph edges indexes.
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        return copy.deepcopy(self._edges_indexes)
    # -------------------------------------------------------------------------
    def set_node_features_matrix(self, node_features_matrix):
        """Set nodes input features matrix.
        
        Parameters
        ----------
        node_features_matrix : numpy.ndarray(2d)
            Nodes input features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        if not isinstance(node_features_matrix, np.ndarray):
            raise RuntimeError('Nodes input features matrix must be provided '
                               'as a numpy 2d array of shape '
                               '(n_nodes, n_features).')
        elif node_features_matrix.shape[0] != self._nodes_coords.shape[0]:
            raise RuntimeError('Nodes input features matrix shape is not '
                               'compatible with number of nodes of material '
                               'patch graph.')
        self._node_features_matrix = copy.deepcopy(node_features_matrix)
    # -------------------------------------------------------------------------
    def get_node_features_matrix(self):
        """Set nodes input features matrix.
        
        Returns
        -------
        node_features_matrix : numpy.ndarray(2d)
            Nodes input features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        return copy.deepcopy(self._node_features_matrix)
    # -------------------------------------------------------------------------
    def set_edge_features_matrix(self, edge_features_matrix):
        """Set edges input features matrix.
        
        Parameters
        ----------
        edge_features_matrix : numpy.ndarray(2d)
            Edges input features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features).
        """
        if self._edges_indexes is None:
            raise RuntimeError('Edges indexes must be set in order to set '
                               'the corresponding input features matrix.')
        elif not isinstance(edge_features_matrix, np.ndarray):
            raise RuntimeError('Edges input features matrix must be provided '
                               'as a numpy 2d array of shape '
                               '(n_edges, n_features).')
        elif edge_features_matrix.shape[0] != self._edges_indexes.shape[0]:
            raise RuntimeError('Edges input features matrix shape is not '
                               'compatible with number of edges of material '
                               'patch graph.')
        self._edge_features_matrix = copy.deepcopy(edge_features_matrix)
    # -------------------------------------------------------------------------
    def get_edge_features_matrix(self):
        """Get edges input features matrix.
        
        Returns
        -------
        edge_features_matrix : numpy.ndarray(2d)
            Edges input features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features).
        """
        return copy.deepcopy(self._edge_features_matrix)
    # -------------------------------------------------------------------------   
    def set_global_features_matrix(self, global_features_matrix):
        """Set global input features matrix.
        
        Parameters
        ----------
        global_features_matrix : numpy.ndarray(2d)
            Global input features matrix stored as a numpy.ndarray(2d) of
            shape (1, n_features).
        """
        if not isinstance(global_features_matrix, np.ndarray):
            raise RuntimeError('Global input features matrix must be provided '
                               'as a numpy 2d array of shape (1, n_features).')
        elif global_features_matrix.shape[0] != 1:
            raise RuntimeError('Global input features matrix must be provided '
                               'as a numpy 2d array of shape (1, n_features).')
        self._global_features_matrix = copy.deepcopy(global_features_matrix)
    # -------------------------------------------------------------------------    
    def get_global_features_matrix(self):
        """Get global input features matrix.
        
        Returns
        -------
        global_features_matrix : numpy.ndarray(2d)
            Global input features matrix stored as a numpy.ndarray(2d) of
            shape (1, n_features).
        """
        return copy.deepcopy(self._global_features_matrix)
    # -------------------------------------------------------------------------   
    def set_node_targets_matrix(self, node_targets_matrix):
        """Set node targets matrix.
        
        Parameters
        ----------
        node_targets_matrix : numpy.ndarray(2d)
            Nodes targets matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_targets).
        """
        if not isinstance(node_targets_matrix, np.ndarray):
            raise RuntimeError('Nodes targets matrix must be provided '
                               'as a numpy 2d array of shape '
                               '(n_nodes, n_targets).')
        elif node_targets_matrix.shape[0] != self._nodes_coords.shape[0]:
            raise RuntimeError('Nodes targets matrix shape is not '
                               'compatible with number of nodes of material '
                               'patch graph.')
        self._node_targets_matrix = copy.deepcopy(node_targets_matrix)
    # -------------------------------------------------------------------------    
    def get_node_targets_matrix(self):
        """Get node targets matrix.
        
        Returns
        -------
        node_targets_matrix : numpy.ndarray(2d)
            Nodes targets matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_targets).
        """
        return copy.deepcopy(self._node_targets_matrix)
    # -------------------------------------------------------------------------   
    def set_edge_targets_matrix(self, edge_targets_matrix):
        """Set edge targets matrix.
        
        Parameters
        ----------
        edge_targets_matrix : numpy.ndarray(2d)
            Edges targets matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_targets).
        """
        if self._edges_indexes is None:
            raise RuntimeError('Edges indexes must be set in order to set '
                               'the corresponding targets matrix.')
        elif not isinstance(edge_targets_matrix, np.ndarray):
            raise RuntimeError('Edges targets matrix must be provided '
                               'as a numpy 2d array of shape '
                               '(n_edges, n_targets).')
        elif edge_targets_matrix.shape[0] != self._edges_indexes.shape[0]:
            raise RuntimeError('Edges targets matrix shape is not '
                               'compatible with number of edges of material '
                               'patch graph.')
        self._edge_targets_matrix = copy.deepcopy(edge_targets_matrix)
    # -------------------------------------------------------------------------    
    def get_edge_targets_matrix(self):
        """Get edge targets matrix.
        
        Returns
        -------
        edge_targets_matrix : numpy.ndarray(2d)
            Edges targets matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_targets).
        """
        return copy.deepcopy(self._edge_targets_matrix)
    # -------------------------------------------------------------------------   
    def set_global_targets_matrix(self, global_targets_matrix):
        """Set global targets matrix.
        
        Parameters
        ----------
        global_targets_matrix : numpy.ndarray(2d)
            Global targets matrix stored as a numpy.ndarray(2d) of shape
            (1, n_targets).
        """
        if not isinstance(global_targets_matrix, np.ndarray):
            raise RuntimeError('Global targets matrix must be provided '
                               'as a numpy 2d array of shape (1, n_targets).')
        elif global_targets_matrix.shape[0] != 1:
            raise RuntimeError('Global targets matrix must be provided '
                               'as a numpy 2d array of shape (1, n_targets).')
        self._global_targets_matrix = copy.deepcopy(global_targets_matrix)
    # -------------------------------------------------------------------------    
    def get_global_targets_matrix(self):
        """Get global targets matrix.
        
        Returns
        -------
        global_targets_matrix : numpy.ndarray(2d)
            Global targets matrix stored as a numpy.ndarray(2d) of shape
            (1, n_targets).
        """
        return copy.deepcopy(self._global_targets_matrix)
    # -------------------------------------------------------------------------
    def plot_material_patch_graph(self, is_show_plot=False, is_save_plot=False,
                                  save_directory=None, plot_name=None,
                                  is_overwrite_file=False):
        """Generate plot of material patch graph.
        
        Parameters
        ----------
        is_show_plot : bool, default=False
            Display plot of material patch graph if True.
        is_save_plot : bool, default=False
            Save plot of material patch graph. Plot is only saved if
            `save_directory` is provided and exists.
        save_directory : str, default=None
            Directory where plot of material patch graph is stored.
        plot_name : str, default=None
            Filename of material patch graph plot.
        is_overwrite_file : bool, default=False
            Overwrite plot of material patch graph if True, otherwise generate
            generate non-existent file path by extending the original file path
            with an integer.
        """
        # Get nodes coordinates  
        nodes_coords = self._nodes_coords
        # Get edges
        edges_indexes = self._edges_indexes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate plot
        fig, ax = plt.subplots()
        # Plot nodes
        if nodes_coords is not None:
            ax.plot(nodes_coords[:, 0], nodes_coords[:, 1],
                    'o', color='#d62728', label='Nodes', zorder=10)
        # Plot edges
        if nodes_coords is not None and edges_indexes is not None:
            for (i, j) in edges_indexes:
                ax.plot([nodes_coords[i, 0], nodes_coords[j, 0]],
                        [nodes_coords[i, 1], nodes_coords[j, 1]],
                        '-', color='#1d77b4', zorder=5)          
        ax.plot([], [], '-', color='#1d77b4', label='Edges', zorder=5)   
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plot legend
        ax.legend(loc='center', ncol=3, numpoints=1, frameon=True,
                    fancybox=True, facecolor='inherit', edgecolor='inherit',
                    fontsize=10, framealpha=1.0,
                    bbox_to_anchor=(0, 1.05, 1.0, 0.1),
                    borderaxespad=0.1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_show_plot:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure (pdf format)
        if is_save_plot and os.path.exists(str(save_directory)):
            # Set figure path
            if plot_name is None:
                plot_name = 'material_patch_graph'
            fig_path = \
                os.path.join(os.path.normpath(save_directory), plot_name) \
                    + '.pdf'
            if os.path.isfile(fig_path) and not is_overwrite_file:
                fig_path = new_file_path_with_int(fig_path)
            # Set figure size (inches)
            fig.set_figheight(3.6, forward=False)
            fig.set_figwidth(3.6, forward=False)
            # Save figure file
            fig.savefig(fig_path, transparent=False, dpi=300,
                        bbox_inches='tight')
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_edges_from_local_radius(nodes_coords, connect_radius):
        """Get edges between nodes that are within a given connectivity radius.
        
        Parameters
        ----------
        nodes_coords : numpy.ndarray(2d)
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        connect_radius : float
            Connectivity radius that sets the maximum distance between two
            nodes that leads to an edge.
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        # Initialize k-d tree
        kd_tree = scipy.spatial.KDTree(nodes_coords)
        # Find all edges between nodes that are at most within a given distance
        # between them
        edges_indexes = kd_tree.query_pairs(r=connect_radius, p=2.0,
                                            output_type='ndarray')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get undirected edges indexes
        edges_indexes = \
            GNNPatchGraphData.get_undirected_unique_edges(edges_indexes) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def get_edges_indexes_mesh(connected_nodes):
        """Convert set of mesh connected nodes to edges indexes matrix.
        
        It is assumed that nodes are labeled from 1 to n_nodes, such that
        node 1 and node n_nodes are associated with indexes 0 and n_nodes-1,
        respectively.
        
        Parameters
        ----------
        connected_nodes : tuple[tuple(2)]
            A set containing all pairs of nodes that are connected by any
            relevant mesh representation (e.g., finite element mesh). Each
            connection is stored a single time as a tuple(node[int], node[int])
            and is independent of the corresponding nodes storage order.
            
        Returns
        -------
        edges_indexes_mesh : numpy.ndarray(2d), default=None
            Edges stemming from any relevant mesh representation (e.g., finite
            element mesh) and that should be accounted for. Edges indexes
            matrix stored as numpy.ndarray[int](2d) with shape (n_edges, 2),
            where the i-th edge is stored in edges_indexes[i, :] as
            (start_node_index, end_node_index).
        """
        # Initialize mesh edges indexes matrix
        edges_indexes_mesh = np.zeros((len(connected_nodes), 2), dtype=int)        
        # Loop over mesh edges
        for i, edge in enumerate(connected_nodes):
            # Assemble edge indexes
            edges_indexes_mesh[i, :] = (edge[0] - 1, edge[1] - 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get unidirected unique edges indexes
        edges_indexes_mesh = \
            GNNPatchGraphData.get_undirected_unique_edges(edges_indexes_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes_mesh
    # -------------------------------------------------------------------------
    @staticmethod
    def get_undirected_unique_edges(edges_indexes):
        """Get set of undirected unique edges indexes.
        
        This function processes the given matrix of edges indexes and
        transforms all edges into undirected edges. In addition, it also
        removes any duplicated edges.
        
        Parameters
        ----------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        # Check provided edges indexes matrix
        GNNPatchGraphData._check_edges_indexes_matrix(edges_indexes)
        # Transforms all edges into undirected edges
        edges_indexes = \
            np.concatenate((edges_indexes, edges_indexes[:, ::-1]), axis=0)
        # Remove duplicated edges
        edges_indexes = np.unique(edges_indexes, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def _check_edges_indexes_matrix(edges_indexes):
        """Check if given edges indexes matrix is valid.
        
        Parameters
        ----------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in edges_indexes[i, :]
            as (start_node_index, end_node_index).
        """
        if not isinstance(edges_indexes, np.ndarray):
            raise RuntimeError('Edges indexes matrix is not a numpy.array.')
        elif edges_indexes.dtype != int:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of dtype int.')
        elif len(edges_indexes.shape) != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (num_edges, 2).')
        elif edges_indexes.shape[1] != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (num_edges, 2).')
# =============================================================================
class GNNPatchFeaturesGenerator:
    """GNN-based material patch input and output features generator.
    
    Attributes
    ----------
    _nodes_coords : numpy.ndarray(2d)
        Coordinates of nodes stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    _nodes_coords_hist : numpy.ndarray(3d)
        Nodes coordinates history stored as a numpy.ndarray(3d) with shape
        (n_nodes, n_dim, n_time_steps). Coordinates of i-th node at k-th
        time step are stored in nodes_coords[i, :, k].   
    _edges_indexes : numpy.ndarray(2d)
        Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
        (num_edges, 2), where the i-th edge is stored in
        edges_indexes[i, :] as (start_node_index, end_node_index).
    _nodes_disps_hist : numpy.ndarray(3d)
        Nodes displacements history stored as a numpy.ndarray(3d) with
        shape (n_nodes, n_dim, n_time_steps). Displacements of i-th node at
        k-th time step are stored in nodes_disps_hist[i, :, k].
    _nodes_int_forces_hist : numpy.ndarray(2d)
        Nodes internal forces history stored as a numpy.ndarray(3d) with
        shape (n_nodes, n_dim, n_time_steps). Internal forces of i-th node
        at k-th time step are stored in nodes_int_forces_hist[i, :, k]. 
            
    Methods
    -------
    build_nodes features_matrix(self, features=(), n_time_steps=1)
        Build nodes features matrix.
    get_available_nodes_features()
        Get available nodes features.
    build_edges_features_matrix(self, features=(), n_time_steps=1)
        Build edges features matrix.
    get_available_edges_input_features()
        Get available edges features.   
    """
    def __init__(self, nodes_coords_hist, edges_indexes=None,
                 nodes_disps_hist=None, nodes_int_forces_hist=None):
        """Constructor.
        
        Parameters
        ----------
        nodes_coords_hist : numpy.ndarray(3d)
            Nodes coordinates history stored as a numpy.ndarray(3d) with shape
            (n_nodes, n_dim, n_time_steps). Coordinates of i-th node at k-th
            time step are stored in nodes_coords[i, :, k].   
        edges_indexes : numpy.ndarray(2d), default=None
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        nodes_disps_hist : numpy.ndarray(3d), default=None
            Nodes displacements history stored as a numpy.ndarray(3d) with
            shape (n_nodes, n_dim, n_time_steps). Displacements of i-th node at
            k-th time step are stored in nodes_disps_hist[i, :, k].
        nodes_int_forces_hist : numpy.ndarray(2d), default=None
            Nodes internal forces history stored as a numpy.ndarray(3d) with
            shape (n_nodes, n_dim, n_time_steps). Internal forces of i-th node
            at k-th time step are stored in nodes_int_forces_hist[i, :, k]. 
        """
        self._nodes_coords_hist = copy.deepcopy(nodes_coords_hist)
        self._edges_indexes = copy.deepcopy(edges_indexes)
        self._nodes_disps_hist = copy.deepcopy(nodes_disps_hist)
        self._nodes_int_forces_hist = copy.deepcopy(nodes_int_forces_hist)
        # Set current nodes coordinates
        self._nodes_coords = nodes_coords_hist[:, :, -1]
    # -------------------------------------------------------------------------
    def build_nodes_features_matrix(self, features=(), n_time_steps=1):
        """Build nodes features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Nodes features.
        n_time_steps : int, default=1
            Number of history time steps to account for in history-based
            features. Defaults to the last time step only.
        
        Returns
        -------
        node_features_matrix : numpy.ndarray(2d), default=None
            Nodes features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        # Get number of nodes
        n_nodes = self._nodes_coords.shape[0]
        # Get number of spatial dimensions
        n_dim = self._nodes_coords.shape[1]
        # Initialize nodes features matrix
        node_features_matrix = np.empty((n_nodes, 0))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get available nodes
        available_features = type(self).get_available_nodes_features()
        # Check features
        if not isinstance(features, tuple):
            raise RuntimeError('Nodes features must be specified as a tuple '
                               'of strings.')
        for feature in features:
            if feature not in available_features:
                raise RuntimeError('Unavailable node feature: ' + str(feature))
        # Check number of time steps
        if not isinstance(n_time_steps, int) or n_time_steps < 1:
            raise RuntimeError('Invalid number of history time steps.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for feature in features:
            if feature == 'coord_hist':
                # Check required data
                if n_time_steps > self._nodes_coords_hist.shape[1]:
                    raise RuntimeError('Number of time steps exceeds length '
                                       'of feature history data: '
                                       + str(feature))
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_time_steps*n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    # Loop over last time steps
                    for j in range(n_time_steps):
                        # Assemble node feature
                        feature_matrix[i, j*n_dim:(j + 1)*n_dim] = \
                            self._nodes_coords_hist[i, :n_dim,
                                                    -n_time_steps + j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'disp_hist':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                elif n_time_steps > self._nodes_disps_hist.shape[1]:
                    raise RuntimeError('Number of time steps exceeds length '
                                       'of feature history data: '
                                       + str(feature))
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_time_steps*n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    # Loop over last time steps
                    for j in range(n_time_steps):
                        # Assemble node feature
                        feature_matrix[i, j*n_dim:(j + 1)*n_dim] = \
                            self._nodes_disps_hist[i, :n_dim,
                                                   -n_time_steps + j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'int_force':
                # Check required data
                if self._nodes_int_forces_hist is None:
                    raise RuntimeError('Nodes internal forces must be set in '
                                       'order to build feature '
                                       + str(feature))
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    # Assemble node feature
                    feature_matrix[i, :] = \
                        self._nodes_int_forces_hist[i, :n_dim, -1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
            # Assemble node feature
            node_features_matrix = \
                np.concatenate((node_features_matrix, feature_matrix), axis=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_nodes_features():
        """Get available nodes features.
        
        Returns
        -------
        available_features : tuple[str]
            Available nodes features.
            
            'coord_hist' : numpy.ndarray(1d) of shape (n_time_steps*n_dim,) \
                           with the node coordinates history.
                          
            'disp_hist'  : numpy.ndarray(1d) of shape (n_time_steps*n_dim,) \
                           with the node displacements history.
            
            'int_force'  : numpy.ndarray(1d) of shape (n_dim,) with the node \
                           internal forces.
        """
        available_features = ('coord_hist', 'disp_hist', 'int_force')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features
    # -------------------------------------------------------------------------
    def build_edges_features_matrix(self, features=(), n_time_steps=1):
        """Build edges features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Edges features.
        n_time_steps : int, default=1
            Number of history time steps to account for in history-based
            features. Defaults to the last time step only.
        
        Returns
        -------
        edge_features_matrix : numpy.ndarray(2d), default=None
            Edges features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features).
        """
        # Check required data
        if self._edges_indexes is None:
            raise RuntimeError('Edges indexes must be set in order to build '
                               'edges features matrix.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of edges
        n_edges = self._edges_indexes.shape[0]
        # Get number of spatial dimensions
        n_dim = self._nodes_coords.shape[1]
        # Initialize edges features matrix
        edge_features_matrix = np.empty((n_edges, 0))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get available edges
        available_features = type(self).get_available_edges_features()
        # Check features
        if not isinstance(features, tuple):
            raise RuntimeError('Edges features must be specified as a tuple '
                               'of strings.')
        for feature in features:
            if feature not in available_features:
                raise RuntimeError('Unavailable edge feature: ' + str(feature))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for feature in features:
            if feature == 'edge_vector':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edges, n_dim))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = self._nodes_coords[i, :] \
                        - self._nodes_coords[j, :]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            elif feature == 'edge_vector_norm':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edges, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_coords[i, :]
                        - self._nodes_coords[j, :])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            elif feature == 'relative_disp':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edges, n_dim))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = \
                        self._nodes_disps_hist[i, :n_dim, -1] \
                        - self._nodes_disps_hist[j, :n_dim, -1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            elif feature == 'relative_disp_norm':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edges, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_disps_hist[i, :n_dim, -1]
                        - self._nodes_disps_hist[j, :n_dim, -1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
            # Assemble edge feature
            edge_features_matrix = \
                np.concatenate((edge_features_matrix, feature_matrix), axis=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features_matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_edges_features():
        """Get available edges features.
        
        Returns
        -------
        available_features : tuple[str]
            Available edges features:
            
            'edge_vector' : numpy.ndarray(1d) of shape (n_dim,) resulting \
                            from the difference of current coordinates \
                            between the starting node and the end node.
            
            'edge_vector_norm' : float corresponding to the norm of the \
                                 difference of current coordinates between \
                                 the starting node and the end node.
                                 
            'relative_disp' : numpy.ndarray(1d) of shape (n_dim,) resulting \
                              from the difference of current displacements \
                              between the starting node and the end node.
                              
            'relative_disp_norm' : float corresponding to the norm of the \
                                   difference of current displacements \
                                   between the starting node and the end node.
        """
        available_features = ('edge_vector', 'edge_vector_norm',
                              'relative_disp', 'relative_disp_norm')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features