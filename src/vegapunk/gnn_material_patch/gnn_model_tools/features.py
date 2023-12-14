"""GNN-based material patch graph features.

Classes
-------
GNNPatchFeaturesGenerator
    GNN-based material patch input and output features generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import itertools
# Third-party
import numpy as np
import sklearn.preprocessing
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class GNNPatchFeaturesGenerator:
    """GNN-based material patch input and output features generator.
    
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
    def __init__(self, n_dim, nodes_coords_hist, n_edge=None,
                 edges_indexes=None, nodes_disps_hist=None,
                 nodes_int_forces_hist=None):
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
        nodes_int_forces_hist : numpy.ndarray(2d), default=None
            Nodes internal forces history stored as a numpy.ndarray(3d) with
            shape (n_nodes, n_dim, n_time_steps). Internal forces of i-th node
            at k-th time step are stored in nodes_int_forces_hist[i, :, k]. 
        """
        self._n_dim = n_dim
        self._nodes_coords_hist = \
            copy.deepcopy(nodes_coords_hist)[:, :n_dim, :]
        self._n_edge = n_edge
        self._edges_indexes = copy.deepcopy(edges_indexes)
        self._nodes_disps_hist = copy.deepcopy(nodes_disps_hist)[:, :n_dim, :]
        self._nodes_int_forces_hist = \
            copy.deepcopy(nodes_int_forces_hist)[:, :n_dim, :]
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
    def build_nodes_features_matrix(self, features=(), n_time_steps=1,
                                    features_kwargs={}):
        """Build nodes features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Nodes features.
        n_time_steps : {int, dict}, default=1
            Number of history time steps to account for in history-based
            features, starting with the last time step available. If int, then
            the same number of time steps is considered for all history-based
            features. If dict, then number of time steps (item, int) is
            specified for each history-based feature (key, str), defaulting
            to 1. Defaults to the last time step only.
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
        # Process number of time steps
        if isinstance(n_time_steps, int):
            if n_time_steps < 1:
                raise RuntimeError(f'Invalid number of history time steps '
                                   f'({n_time_steps}).')
            else:
                n_time_steps = {feature: n_time_steps for feature in features}
        elif isinstance(n_time_steps, dict):
            for feature in features:
                if feature not in n_time_steps.keys():
                    n_time_steps[feature] = 1
        else:
            raise RuntimeError('Invalid specification of number of steps.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for feature in features:
            if feature == 'coord_init':
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    feature_matrix[i, :] = \
                        self._nodes_coords_hist[i, :n_dim, 0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'coord_hist':
                # Get number of time steps
                n_steps = n_time_steps[feature]
                # Check required data
                if n_steps > self._nodes_coords_hist.shape[1]:
                    raise RuntimeError('Number of time steps exceeds length '
                                       'of feature history data: '
                                       + str(feature))
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_steps*n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    # Loop over last time steps
                    for j in range(n_steps):
                        # Assemble node feature
                        feature_matrix[i, j*n_dim:(j + 1)*n_dim] = \
                            self._nodes_coords_hist[i, :n_dim, -n_steps + j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'disp_hist':
                # Get number of time steps
                n_steps = n_time_steps[feature]
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                elif n_steps > self._nodes_disps_hist.shape[1]:
                    raise RuntimeError('Number of time steps exceeds length '
                                       'of feature history data: '
                                       + str(feature))
                # Initialize node feature matrix
                feature_matrix = np.zeros((n_nodes, n_steps*n_dim))
                # Loop over nodes
                for i in range(n_nodes):
                    # Loop over last time steps
                    for j in range(n_steps):
                        # Assemble node feature
                        feature_matrix[i, j*n_dim:(j + 1)*n_dim] = \
                            self._nodes_disps_hist[i, :n_dim, -n_steps + j]
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
            elif feature == 'polycoord_disp':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set default feature parameters
                feature_kwargs = {'coord_order': 1, 'is_bias': False}
                # Set feature parameters (override defaults)
                if feature in features_kwargs.keys():
                    # Override default parameters
                    for parameter, value in features_kwargs[feature].items():
                        feature_kwargs[parameter] = value
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize polynomial basis generator
                poly_basis = sklearn.preprocessing.PolynomialFeatures(
                    degree=feature_kwargs['coord_order'],
                    include_bias=feature_kwargs['is_bias'])
                # Get node coordinates (last time step)
                nodes_coords_last = self._nodes_coords_hist[:, :n_dim, -1]
                # Compute node coordinates polynomial basis
                poly_coord = poly_basis.fit_transform(nodes_coords_last)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get node displacements (last time step)
                nodes_disps_last = self._nodes_disps_hist[:, :n_dim, -1]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build node coordinates-displacements products
                poly_coord_disp = \
                    [poly_coord*nodes_disps_last[:, i][:, np.newaxis]
                     for i in range(n_dim)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build node feature matrix
                feature_matrix = np.concatenate(poly_coord_disp, axis=1)
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

            'coord_hist'     : numpy.ndarray(1d) of shape \
                               (n_time_steps*n_dim,) with the node \
                               coordinates history.
                          
            'disp_hist'      : numpy.ndarray(1d) of shape \
                               (n_time_steps*n_dim,) with the node \
                               displacements history.
            
            'int_force'      : numpy.ndarray(1d) of shape (n_dim,) with the \
                               current node internal forces.
                           
            'polycoord_disp' : numpy.ndarray(1d) of shape (n_dim*0.5n*(n+1),) \
                               with a n-order polynomial basis of current \
                               node coordinates (bias term excluded) \
                               multiplied by each node displacement component.
        """
        available_features = ('coord_init', 'coord_hist', 'disp_hist',
                              'int_force', 'polycoord_disp')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features
    # -------------------------------------------------------------------------
    def build_edges_features_matrix(self, features=(), n_time_steps=1,
                                    features_kwargs={}):
        """Build edges features matrix.
        
        Parameters
        ----------
        features : tuple[str]
            Edges features.
        n_time_steps : {int, dict}, default=1
            Number of history time steps to account for in history-based
            features, starting with the last time step available. If int, then
            the same number of time steps is considered for all history-based
            features. If dict, then number of time steps (item, int) is
            specified for each history-based feature (key, str), defaulting
            to 1. Defaults to the last time step only.
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
        edge_indexes_features = ('edge_vector', 'edge_vector_norm',
                                 'relative_disp', 'relative_disp_norm')
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
        # Process number of time steps
        if isinstance(n_time_steps, int):
            if n_time_steps < 1:
                raise RuntimeError(f'Invalid number of history time steps '
                                   f'({n_time_steps}).')
            else:
                n_time_steps = {feature: n_time_steps for feature in features}
        elif isinstance(n_time_steps, dict):
            for feature in features:
                if feature not in n_time_steps.keys():
                    n_time_steps[feature] = 1
        else:
            raise RuntimeError('Invalid specification of number of steps.')
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
            elif feature == 'edge_vector':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, n_dim))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = self._nodes_coords[i, :n_dim] \
                        - self._nodes_coords[j, :n_dim]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            elif feature == 'edge_vector_norm':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_coords[i, :n_dim]
                        - self._nodes_coords[j, :n_dim])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            elif feature == 'relative_disp':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, n_dim))
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
                feature_matrix = np.zeros((n_edge, 1))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Assemble edge feature
                    feature_matrix[k, :] = np.linalg.norm(
                        self._nodes_disps_hist[i, :n_dim, -1]
                        - self._nodes_disps_hist[j, :n_dim, -1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'disp_gradient_init':
                # Check required data
                if self._nodes_disps_hist is None:
                    raise RuntimeError('Nodes displacements must be set in '
                                       'order to build feature '
                                       + str(feature))
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, n_dim**2))
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Loop over dimensional product
                    for m, (p, q) in enumerate(itertools.product(
                            range(n_dim), range(n_dim))):
                        # Compute displacement difference
                        disp_diff = (self._nodes_disps_hist[i, p, -1]
                                     - self._nodes_disps_hist[j, p, -1])
                        # Compute initial coordinates difference
                        coord_init_diff = (self._nodes_coords_hist[i, q, 0]
                                           - self._nodes_coords_hist[j, q, 0])
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Set flags to avoid numerical issues
                        is_disp_close = \
                            np.isclose(self._nodes_disps_hist[i, p, -1],
                                       self._nodes_disps_hist[j, p, -1])
                        is_coord_close = \
                            np.isclose(self._nodes_coords_hist[i, q, 0],
                                       self._nodes_coords_hist[j, q, 0])
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute gradient
                        if is_disp_close:
                            # Null gradient
                            gradient = 0
                        elif is_coord_close or coord_init_diff == 0:
                            # Set null gradient if coordinates difference is
                            # very small compared to displacements difference
                            # (avoid exploding or unknown gradient)
                            gradient = 0
                        else:
                            gradient = disp_diff/coord_init_diff
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Assemble edge feature
                        feature_matrix[k, m] = gradient
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif feature == 'edge_vector_norm_ratio':
                # Initialize edge feature matrix
                feature_matrix = np.zeros((n_edge, 1))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over edges
                for k, (i, j) in enumerate(self._edges_indexes):
                    # Compute current coordinates difference norm
                    coord_diff_norm = np.linalg.norm(
                        self._nodes_coords_hist[i, :n_dim, -1]
                        - self._nodes_coords_hist[j, :n_dim, -1])
                    # Compute initial coordinates difference norm
                    coord_init_diff_norm = np.linalg.norm(
                        self._nodes_coords_hist[i, :n_dim, 0]
                        - self._nodes_coords_hist[j, :n_dim, 0])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set flags to avoid numerical issues
                    is_coord_close = \
                        np.allclose(self._nodes_coords_hist[i, :n_dim, -1],
                                    self._nodes_coords_hist[j, :n_dim, -1])
                    is_coord_init_close = \
                        np.allclose(self._nodes_coords_hist[i, :n_dim, 0],
                                    self._nodes_coords_hist[j, :n_dim, 0])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute norm ratio
                    if is_coord_close:
                        # Null norm ratio
                        norm_ratio = 0
                    elif is_coord_init_close or coord_init_diff_norm == 0:
                        # Set null norm ratio if initial coordinates difference
                        # is very small compared to current coordinates
                        # difference (avoid exploding or unknown gradient)
                        norm_ratio = 0
                    else:
                        norm_ratio = coord_diff_norm/coord_init_diff_norm
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble edge feature
                    feature_matrix[k, :] = norm_ratio
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
                                 resulting from the difference of initial \
                                 coordinates between the starting node and \
                                 the end node.
                                 
            'edge_vector_init_norm' : float corresponding to the norm of the \
                                      difference of initial coordinates \
                                      between the starting node and the end \
                                      node.
            
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
            
            'disp_gradient_init' : numpy.ndarray(1d) of shape (n_dim**2,) \
                                   that corresponds to the gradient of \
                                   displacements with respect to the initial \
                                   coordinates.
                                   
            'edge_vector_norm_ratio' : float corresponding to the ratio of \
                                       the norms of the difference of \
                                       coordinates between the starting node \
                                       and the end node for current and
                                       initial times.
        """
        available_features = ('edge_vector_init', 'edge_vector_init_norm',
                              'edge_vector', 'edge_vector_norm',
                              'relative_disp', 'relative_disp_norm',
                              'disp_gradient_init', 'edge_vector_norm_ratio')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_features