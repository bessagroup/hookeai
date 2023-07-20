"""Generate GNN-based material patch graph data.

Classes
-------
GNNPatchGraphData
    GNN-based material patch graph data.

Functions
---------

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
    n_dim : int
        Number of spatial dimensions.
    nodes_coords : numpy.ndarray(2d), default=None
        Coordinates of nodes stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    node_features_matrix : numpy.ndarray(2d), default=None
        Nodes input features matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_features).
    global_features_matrix : numpy.ndarray(2d), default=None
        Global input features matrix stored as a numpy.ndarray(2d) of shape
        (1, n_features).
    node_targets_matrix : numpy.ndarray(2d), default=None
        Nodes targets matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_targets).
    global_targets_matrix : numpy.ndarray(2d), default=None
        Global targets matrix stored as a numpy.ndarray(2d) of shape
        (1, n_targets).
    edge_indexes : numpy.ndarray(2d)
        Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
        (num_edges, 2), where the i-th edge is stored in
        edges_indexes[i, :] as (start_node_index, end_node_index).
    
    Methods
    -------
    set_graph_edges_indexes(self, connect_radius=None, \
                            edges_indexes_mesh=None)
        Set material patch graph edges indexes.
    plot_material_patch_graph(self, is_show_plot=False, is_save_plot=False, \
                              save_directory=None, plot_name=None, \
                              is_overwrite_file=False)
        Generate plot of material patch graph.
    _get_edges_from_local_radius(nodes_coords, connect_radius)
        Get edges between nodes that are within a given connectivity radius.
    get_undirected_unique_edges(edge_indexes)
        Get set of undirected unique edges indexes.
    _check_edges_indexes_matrix(edge_indexes)
        Check if given edges indexes matrix is valid.
    """
    def __init__(self, n_dim, nodes_coords=None, node_features_matrix=None,
                 global_features_matrix=None, node_targets_matrix=None,
                 global_targets_matrix=None):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        nodes_coords : numpy.ndarray(2d), default=None
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        node_features_matrix : numpy.ndarray(2d), default=None
            Nodes input features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        global_features_matrix : numpy.ndarray(2d), default=None
            Global input features matrix stored as a numpy.ndarray(2d) of shape
            (1, n_features).
        node_targets_matrix : numpy.ndarray(2d), default=None
            Nodes targets matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_targets).
        global_targets_matrix : numpy.ndarray(2d), default=None
            Global targets matrix stored as a numpy.ndarray(2d) of shape
            (1, n_targets).
        """
        self._n_dim = n_dim
        self._node_features_matrix = node_features_matrix
        self._global_features_matrix = global_features_matrix
        self._node_targets_matrix = node_targets_matrix
        self._global_targets_matrix = global_targets_matrix
        self._nodes_coords = nodes_coords
        if nodes_coords is None:
            self._nodes_coords = nodes_coords
        else:
            self._nodes_coords = nodes_coords[:, :n_dim]
        # Initialize graph edges
        self._edges_indexes=None
    # -------------------------------------------------------------------------  
    def get_torch_data_object(self):
        """Get PyG homogeneous graph data object.
        
        Returns
        -------
        data : torch_geometric.data.Data
            PyG data object describing a homogeneous graph.
        """        
        # Set PyG node feature matrix
        x = None
        if self._node_features_matrix is not None:
            x = torch.tensor(copy.deepcopy(self._node_features_matrix),
                             dtype=torch.float)
        # Set PyG graph connectivity
        if self._edges_indexes is not None:
            edge_index = torch.tensor(
                np.transpose(copy.deepcopy(self._edges_indexes)),
                dtype=torch.long)
        # Set PyG edge feature matrix
        edge_attr = None
        # Set PyG ground-truth labels
        y = None
        if self._node_targets_matrix is not None:
            y = torch.tensor(copy.deepcopy(self.node_targets_matrix),
                             dtype=torch.float)
        # Set PyG node position matrix
        pos = None
        if self._nodes_coords is not None:
            pos = torch.tensor(copy.deepcopy(self._nodes_coords),
                               dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate PyG homogeneous graph data object
        data = torch_geometric.data.Data(x=x, edge_index=edge_index,
                                         edge_attr=edge_attr, y=y, pos=pos)
        # Validate graph data object
        data.validate(raise_on_error=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data
    # -------------------------------------------------------------------------
    def get_graph_edges_indexes(self):
        """Get material patch graph edges indexes.
        
        Returns
        -------
        edge_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        return copy.deepcopy(self._edges_indexes)
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
            element mesh) and that should be accounted for. Must be given as an
            edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).       
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
        if edges_indexes is not None:
            ax.plot(nodes_coords[:, 0], nodes_coords[:, 1],
                    'o', color='#d62728', label='Nodes', zorder=10)
        # Plot edges
        if edges_indexes is not None:
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
        edge_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        
        # Initialize k-d tree
        kd_tree = scipy.spatial.KDTree(nodes_coords)
        # Find all edges between nodes that are at most within a given distance
        # between them
        edge_indexes = kd_tree.query_pairs(r=connect_radius, p=2.0,
                                           output_type='ndarray')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get undirected edges indexes
        edge_indexes = \
            GNNPatchGraphData.get_undirected_unique_edges(edge_indexes) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def get_undirected_unique_edges(edge_indexes):
        """Get set of undirected unique edges indexes.
        
        This function processes the given matrix of edges indexes and
        transforms all edges into undirected edges. In addition, it also
        removes any duplicated edges.
        
        Parameters
        ----------
        edge_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        
        Returns
        -------
        edge_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        # Check provided edges indexes matrix
        GNNPatchGraphData._check_edges_indexes_matrix(edge_indexes)
        # Transforms all edges into undirected edges
        edge_indexes = \
            np.concatenate((edge_indexes, edge_indexes[:, ::-1]), axis=0)
        # Remove duplicated edges
        edge_indexes = np.unique(edge_indexes, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def _check_edges_indexes_matrix(edge_indexes):
        """Check if given edges indexes matrix is valid.
        
        Parameters
        ----------
        edge_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in edge_indexes[i, :]
            as (start_node_index, end_node_index).
        """
        if not isinstance(edge_indexes, np.ndarray):
            raise RuntimeError('Edges indexes matrix is not a numpy.array.')
        elif edge_indexes.dtype != int:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of dtype int.')
        elif len(edge_indexes.shape) != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (num_edges, 2).')
        elif edge_indexes.shape[1] != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (num_edges, 2).')
