"""Finite element mesh graph features.

Classes
-------
FEMMeshFeaturesGenerator
    Finite element mesh input and output features generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class FEMMeshFeaturesGenerator:
    """Finite element mesh input and output features generator.
    
    Attributes
    ----------
    _n_dim : int
        Number of spatial dimensions.
    _nodes_coords : numpy.ndarray(2d)
        Nodes current coordinates stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    _nodes_coords_hist : numpy.ndarray(3d)
        Nodes coordinates history stored as a numpy.ndarray(3d) with shape
        (n_nodes, n_dim, n_time_steps). Coordinates of i-th node at k-th
        time step are stored in nodes_coords[i, :, k].
    _n_edge : int
        Number of edges.
    _edges_indexes : numpy.ndarray(2d)
        Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
        (num_edges, 2), where the i-th edge is stored in
        edges_indexes[i, :] as (start_node_index, end_node_index).
    _nodes_disps_hist : numpy.ndarray(3d)
        Nodes displacements history stored as a numpy.ndarray(3d) with
        shape (n_nodes, n_dim, n_time_steps). Displacements of i-th node at
        k-th time step are stored in nodes_disps_hist[i, :, k].
    _time_hist : tuple
        Discrete time history.
            
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
    def __init__(self, n_dim, nodes_coords_hist, n_edge=None,
                 edges_indexes=None, nodes_disps_hist=None, time_hist=None):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        nodes_coords_hist : numpy.ndarray(3d)
            Nodes coordinates history stored as a numpy.ndarray(3d) with shape
            (n_nodes, n_dim, n_time_steps). Coordinates of i-th node at k-th
            time step are stored in nodes_coords[i, :, k].
        n_edge : int, default=None
            Number of edges. If None, then number of edges is inferred from
            edges indexes matrix if the later is known.
        edges_indexes : numpy.ndarray(2d), default=None
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index). Required
            to compute edge features based on corresponding node features.
        nodes_disps_hist : numpy.ndarray(3d), default=None
            Nodes displacements history stored as a numpy.ndarray(3d) with
            shape (n_nodes, n_dim, n_time_steps). Displacements of i-th node at
            k-th time step are stored in nodes_disps_hist[i, :, k].
        time_hist : tuple, default=None
            Discrete time history.
        """
        self._n_dim = n_dim
        self._nodes_coords_hist = \
            copy.deepcopy(nodes_coords_hist)[:, :n_dim, :]
        self._n_edge = n_edge
        self._edges_indexes = copy.deepcopy(edges_indexes)
        self._nodes_disps_hist = copy.deepcopy(nodes_disps_hist)[:, :n_dim, :]
        self._time_hist = time_hist
        # Set current nodes coordinates
        self._nodes_coords = nodes_coords_hist[:, :n_dim, -1]
        # Check number of edges
        if self._n_edge is not None and self._edges_indexes is not None:
            if self._n_edge != self._edges_indexes.shape[0]:
                raise RuntimeError(f'Mismatch between number of edges '
                                   f'n_edges ({n_edge}) and edges_indexes '
                                   f'({self._edges_indexes.shape[0]}).')
        elif self._n_edge is None and self._edges_indexes is not None:
            # Set number of edges from edges indexes
            self._n_edge = self._edges_indexes.shape[0]
    # -------------------------------------------------------------------------
    def build_nodes_features_matrix(self, features=(), features_kwargs={}):
        """Build nodes features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Nodes features.
        features_kwargs : dict, default={}
            Parameters (item, dict) associated with the computation of node
            features (key, str).
        
        Returns
        -------
        node_features_matrix : {numpy.ndarray(2d), None}
            Nodes features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        # Get number of nodes
        n_nodes = self._nodes_coords.shape[0]
        # Get number of spatial dimensions
        n_dim = self._n_dim
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for feature in features:
            if feature in ('coord_init', 'coord_old', 'coord'):
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_dim))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set time step index
                if feature == 'coord_init':
                    time_id = 0
                elif feature == 'coord_old':
                    time_id = -2
                else:
                    time_id = -1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over nodes
                for i in range(n_nodes):
                    feature_matrix[i, :] = \
                        self._nodes_coords_hist[i, :n_dim, time_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'disp':
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_dim))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over nodes
                for i in range(n_nodes):
                    feature_matrix[i, :] = \
                        self._nodes_disps_hist[i, :n_dim, -1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'time':
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, 1))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over nodes
                for i in range(n_nodes):
                    feature_matrix[i, :] = self._time_hist[-1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
            # Assemble node feature
            node_features_matrix = \
                np.concatenate((node_features_matrix, feature_matrix), axis=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes features matrix as None in the absence of node features
        if node_features_matrix.shape[1] == 0:
            node_features_matrix = None
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

            'coord_init'     : numpy.ndarray(1d) of shape (n_dim,) with the \
                               node initial coordinates.

            'coord_old'      : numpy.ndarray(1d) of shape (n_dim,) with the \
                               node previous coordinates.

            'coord'          : numpy.ndarray(1d) of shape (n_dim,) with the \
                               node current coordinates.

            'disp'           : numpy.ndarray(1d) of shape (n_dim,) with the \
                               node current displacements.
            
            'time'           : current time (float)
        """
        available_features = ('coord_init', 'coord_old', 'coord', 'disp',
                              'time')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features
    # -------------------------------------------------------------------------
    def build_edges_features_matrix(self, features=(), features_kwargs={}):
        """Build edges features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Edges features.
        features_kwargs : dict, default={}
            Parameters (item, dict) associated with the computation of edge
            features (key, str).
        
        Returns
        -------
        edge_features_matrix : {numpy.ndarray(2d), None}
            Edges features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features). Set as None in the absence of edges
            features.
        """
        # Check number of edges
        if self._n_edge is None and self._edges_indexes is None:
            raise RuntimeError('The number of edges must be known in order '
                               'to build the edges features matrix. Provide '
                               'either n_edges or edges_indexes when '
                               'initializing the features generator.')
        # Check edge features requiring edge indexes
        edge_indexes_features = ('edge_vector_old', 'edge_vector_old_norm')
        if (self._edges_indexes is None
                and any([x in features for x in edge_indexes_features])):
            raise RuntimeError('Edges indexes must be set in order to build '
                               'edges features matrix based on the '
                               'corresponding node features.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of edges
        n_edge = self._n_edge
        # Get number of spatial dimensions
        n_dim = self._n_dim
        # Initialize edges features matrix
        edge_features_matrix = np.empty((n_edge, 0))
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
            if feature == 'edge_vector_init':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, n_dim))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = \
                        self._nodes_coords_hist[i, :n_dim, 0] \
                        - self._nodes_coords_hist[j, :n_dim, 0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'edge_vector_init_norm':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_coords_hist[i, :n_dim, 0]
                        - self._nodes_coords_hist[j, :n_dim, 0])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'edge_vector_old':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, n_dim))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = \
                        self._nodes_coords_hist[i, :n_dim, -2] \
                        - self._nodes_coords_hist[j, :n_dim, -2]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'edge_vector_old_norm':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_coords_hist[i, :n_dim, -2]
                        - self._nodes_coords_hist[j, :n_dim, -2])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
            # Assemble edge feature
            edge_features_matrix = \
                np.concatenate((edge_features_matrix, feature_matrix), axis=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edges features matrix as None in the absence of edge features
        if edge_features_matrix.shape[1] == 0:
            edge_features_matrix = None
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
            
            'edge_vector_init' : numpy.ndarray(1d) of shape (n_dim,) \
                                 resulting from the difference of the initial \
                                 coordinates between the starting node and \
                                 the end node.
            
            'edge_vector_init_norm' : float corresponding to the norm of the \
                                      difference of the initial coordinates \
                                      between the starting node and the end \
                                      node.

            'edge_vector_old' : numpy.ndarray(1d) of shape (n_dim,) resulting \
                                from the difference of the previous \
                                coordinates between the starting node and the \
                                end node.
            
            'edge_vector_old_norm' : float corresponding to the norm of the \
                                     difference of the previous coordinates \
                                     between the starting node and the end \
                                     node.
        """
        available_features = ('edge_vector_init', 'edge_vector_init_norm',
                              'edge_vector_old', 'edge_vector_old_norm')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features