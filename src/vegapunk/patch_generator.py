"""Finite element material patch generator.

Classes
-------
class FiniteElementPatchGenerator
    Finite element material patch generator.
    
Functions
---------
rotation_tensor_from_euler_angles
    Set rotation tensor from Euler angles (Bunge convention).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
# Local
from patch import FiniteElementPatch
from finite_element import FiniteElement
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class FiniteElementPatchGenerator:
    """Finite element material patch generator.
    
    Attributes
    ----------
    _n_corners : int
        Number of corners of the patch.
    _n_edges : int
        Number of edges of the patch.
    _corners_mapping : dict
        For each corner label (key, str[int]), store the corner internal index.
        Corners are labeled from 1 to number of corners.
    _corners_coords_ref : numpy.ndarray(2d)
        Patch corners coordinates in the reference configuration
        (numpy.ndarray(n_corners, n_dim)).
    _edges_per_dim : dict
        For each dimension (key, str[int]), store the edges
        (item, tuple[tuple]) oriented along that dimension in the reference
        configuration. Each edge is stored as tuple[int](2) containing the
        indexes of the two corresponding corners (sorted in asceding order of
        coordinate).
    _edges_euler_angles_per_dim : dict
        For each dimension (key, str[int]), store the corresponding edges
        deformation plans orientations (item, tuple[numpy.ndarray(2d)]). For
        each edge, the deformation plans orientations are stored as a
        numpy.ndarray[float](n_def_plans, 3), where each orientation is defined
        by the the Euler angles (degrees) sorted according to Bunge convention
        (Z1-X2-Z3). Axis colinear with the edge always points towards the
        positive coordinate, while the orthogonal axis is always positive
        towards the outside of the patch.
    _edges_mapping : dict
        For each edge label (key, str[int]), store the edge orientation
        dimension and internal index consistent with `_edges_per_dim`
        (tuple(2)). Edges are labeled from 1 to number of edges.

    Methods
    -------
    generate_deformed_patch(self, elem_type, n_elems_per_dim, \
                            corners_lab_bc=None, corners_lab_disp_range=None, \
                            edges_lab_def_order=None, \
                            edges_lab_disp_range=None, max_iter=10, \
                            is_verbose=False)
        Generate finite element deformed patch.
    _build_corners_bc(self, corners_lab_bc=None)
        Build boundary conditions on patch corners.
    _build_corners_disp_range(self, corners_lab_disp_range=None, \
                              corners_bc=None)
        Build patch corners displacement range.
    _build_edges_poly_orders(self, edges_lab_def_order=None)
        Build patch edges deformation polynomials orders.
    _build_edges_disp_range(self, edges_lab_disp_range=None)
        Build patch edges displacements range.
    _set_corners_attributes(self)
        Set patch corners attributes.
    _set_corners_coords_ref(self)
        Set patch corners coordinates (reference configuration).
    _set_edges_attributes(self)
        Set patch edges attributes.
    _get_n_edge_nodes_per_dim(self, elem_type, n_elems_per_dim)
        Get number of patch edge nodes along each dimension.
    _get_elem_type_attributes(self, elem_type)
        Get finite element type attributes.
    _get_corners_random_displacements(self, corners_disp_range, \
                                      edges_poly_orders=None)
        Compute patch corners random displacements.
    _get_deformed_boundary_edge(self, nodes_coords_ref, left_node_def, \
                                right_node_def, poly_order,
                                poly_bounds_range=None, is_plot=False)
    _polynomial_sampler(order, left_point, right_point, lower_bound=None, \
                        upper_bound=None, is_plot=False)
        Generate random polynomial by sampling points within given bounds.
    _is_admissible_geometry(self, edges_coords)
        Check whether patch is geometrically admissible.
    _get_orthogonal_dims(self, dim)
        Get orthogonal dimensions to given dimension.
    _rotate_coords_array(coords_array, r)
        Rotate coordinates array.
    _generate_finite_element_mesh(self, elem_type, n_elems_per_dim)
        Generate patch finite element mesh (reference configuration).
    _get_elem_node_index(self, elem_type, n_elems_per_dim, global_index)
        Get element node local index from global mesh index.
    _get_mesh_boundary_nodes_disps(self, edges_coords_ref, edges_coords_def, \
                                   mesh_nodes_coords_ref)
        Compute finite element patch boundary displacements.
    _get_node_label_from_coords(self, mesh_nodes_coords_ref, node_coords)
        Get finite element mesh node label from coordinates.
    """
    def __init__(self, n_dim, patch_dims):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        patch_dims : tuple[float]
            Patch size in each dimension.
        """
        self._n_dim = n_dim
        self._patch_dims = copy.deepcopy(patch_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set patch number of corners and edges
        if self._n_dim == 2:
            self._n_corners = 4
            self._n_edges = 4
        else:
            self._n_corners = 8
            self._n_edges = 12
        # Set corners attributes
        self._set_corners_attributes()  
        # Set patch corners coordinates (reference configuration) 
        self._set_corners_coords_ref()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edges attributes
        self._set_edges_attributes()
    # -------------------------------------------------------------------------  
    def generate_deformed_patch(self, elem_type, n_elems_per_dim,
                                corners_lab_bc=None,
                                corners_lab_disp_range=None,
                                edges_lab_def_order=None,
                                edges_lab_disp_range=None,
                                max_iter=10, is_verbose=False):
        """Generate finite element deformed patch.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension.
        corners_lab_bc : dict, default=None
            Boundary conditions (item, tuple[int](n_dim)) for each patch
            corner label (key, str[int]). Corners are labeled from 1 to number
            of corners. The tuple[int](n_dim) prescribes 0 (free) or 1 (fixed)
            for each degree of freedom. Corners are labeled from 1 to number of
            corners.Unspecified corners are assumed free by default.
        corners_lab_disp_range : dict, default=None
            Displacement range along each dimension (item, tuple[tuple(2)]) for
            each corner label (key, str[int]). Corners are labeled from 1 to
            number of corners. Range is specified as tuple(min, max) for each
            dimension.  If None, a null displacement range is set by default.
        edges_lab_def_order : {int, dict}, default=None
            Deformation polynomial order (item, int) for each edge label
            (key, str[int]). Edges are labeled from 1 to number of edges.
            If a single order is provided, then it is assumed for all the
            edges. Zero order is assumed for unspecified edges. If None, zero
            order is assumed for all the edges.
        edges_lab_disp_range : dict, default=None
            Displacement range (item, tuple[float](2)) for each edge
            label (key, str[int]). Edges are labeled from 1 to number of edges.
            Range is specified as a tuple(min, max) where: (1) displacement
            range is orthogonal to the edge (reference configuration),
            (2) displacement range is relative to midplane defined by both
            limiting corners (deformed configuration), and
            (3) positive/negative displacement corresponds to outward/inward
            direction with respect to the patch. Null displacement range is
            assumed for unspecified edges. If None, null displacement range is
            set by default.
        max_iter : int, default=10
            Maximum number of iterations to get a geometrically admissible
            deformed patch configuration.
        is_verbose : bool, default=False
            If True, enable verbose output.

        Returns
        -------
        is_admissible : bool
            If True, the patch is geometrically admissible.
        patch : FiniteElementPatch
            Finite element patch. If `is_admissible` is False, then returns
            None.
        """
        if is_verbose:
            print('\nGenerating finite element deformed material patch'
                  '\n-------------------------------------------------')
            print('\n> Setting patch reference configuration...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set corners boundary conditions
        corners_bc = self._build_corners_bc(corners_lab_bc=corners_lab_bc)
        # Build corners displacement range
        corners_disp_range = self._build_corners_disp_range(
            corners_lab_disp_range=corners_lab_disp_range,
            corners_bc=corners_bc)
        # Build edges deformation polynomials orders and displacement range
        edges_poly_orders = self._build_edges_poly_orders(
            edges_lab_def_order=edges_lab_def_order)
        edges_disp_range = self._build_edges_disp_range(
            edges_lab_disp_range=edges_lab_disp_range)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get corners coordinates (reference configuration)
        corners_coords_ref = self._corners_coords_ref
        # Get edges attributes
        edges_per_dim = self._edges_per_dim
        edges_euler_angles_per_dim = self._edges_euler_angles_per_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = self._get_n_edge_nodes_per_dim(elem_type,
                                                              n_elems_per_dim)
        # Set edges nodes coordinates (reference configuration)
        edges_coords_ref = {}
        for i in range(self._n_dim):
            # Get number of patch edge nodes
            n_edge_nodes = n_edge_nodes_per_dim[i]
            # Initialize edges nodes coordinates along dimension
            edges_coords_ref[str(i)] = []
            # Loop over edges
            for (cid_l, cid_r) in edges_per_dim[str(i)]:
                # Build edge nodes coordinates assuming a regular
                # discretization (evenly spaced nodes)
                coords = np.zeros((n_edge_nodes, self._n_dim))
                # Loop over dimensions
                for j in range(self._n_dim):
                    coords[:, j] = np.linspace(corners_coords_ref[cid_l, j],
                                               corners_coords_ref[cid_r, j],
                                               num=n_edge_nodes)
                # Store edge nodes coordinates
                edges_coords_ref[str(i)].append(coords)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Random generation iterative loop:')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize outputs
        is_admissible = False
        patch = None
        # Loop over randomly generated deformed configurations
        for iter in range(max_iter):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print('\n  > Iteration ' + str(iter) + ':')
                print('    > Generating corners displacements...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute corners random displacements
            corners_disp, corners_disp_range = \
                self._get_corners_random_displacements(
                    corners_disp_range, edges_poly_orders=edges_poly_orders)
            # Compute corners deformed coordinates
            corners_coords_def = self._corners_coords_ref + corners_disp
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print('    > Generating edges random displacements...')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute edges nodes coordinates (deformed configuration)   
            edges_coords_def = {}
            for i in range(self._n_dim):
                # Initialize edges nodes coordinates along dimension
                edges_coords_def[str(i)] = []
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over edges
                for j, (cid_l, cid_r) in enumerate(edges_per_dim[str(i)]):
                    # Get edge nodes coordinates (reference configuration)
                    nodes_coords_ref = edges_coords_ref[str(i)][j]
                    # Get edge corners coordinates (deformed configuration)
                    left_node_def = corners_coords_def[cid_l, :]
                    right_node_def = corners_coords_def[cid_r, :]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get edge deformation planes orientation (Euler angles,
                    # Bunge convention)
                    euler_degs = edges_euler_angles_per_dim[str(i)][j]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over edge deformation planes
                    for k in range(euler_degs.shape[0]):
                        # Get edge deformation plane orientation (Euler angles,
                        # Bunge convention)
                        euler_deg = tuple(euler_degs[k, :])
                        # Get edge rotation matrix to the deformation plane
                        rotation = rotation_tensor_from_euler_angles(
                            euler_deg)[:self._n_dim, :self._n_dim]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Rotate edge and corners coordinates to deformation
                        # plane
                        rot_nodes_coords_ref = \
                            type(self)._rotate_coords_array(nodes_coords_ref,
                                                            rotation)
                        rot_left_node_def = np.matmul(rotation, left_node_def)
                        rot_right_node_def = np.matmul(rotation,
                                                       right_node_def)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Get edge deformation polynomial order
                        poly_order = edges_poly_orders[str(i)][j]
                        # Get edge displacement range
                        disp_amp = edges_disp_range[str(i)][j]        
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Generate randomly deformed boundary edge node
                        # coordinates (deformed plane)
                        rot_nodes_coords_def, rot_nodes_disp = \
                            self._get_deformed_boundary_edge(
                                rot_nodes_coords_ref, rot_left_node_def,
                                rot_right_node_def, poly_order,
                                poly_bounds_range=disp_amp)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Get edge and corners coordinates (deformed
                        # configuration) in the original space
                        nodes_coords_def = type(self)._rotate_coords_array(
                            rot_nodes_coords_def, np.transpose(rotation))
                        left_node_def = np.matmul(np.transpose(rotation),
                                                left_node_def)
                        right_node_def = np.matmul(np.transpose(rotation),
                                                right_node_def)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Store edge nodes coordinates (deformed configuration)
                        edges_coords_def[str(i)].append(nodes_coords_def)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
            # Check whether patch deformation is geometrically admissible
            is_admissible = self._is_admissible_geometry(edges_coords_def)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            if is_verbose:
                print('    > Is admissible deformation? ', is_admissible)  
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
            # If admissible deformation is achieved, then generate finite
            # element patch and leave iterative loop
            if is_admissible:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print('    > Generating finite element mesh...')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate finite element mesh (reference configuration)
                mesh_nodes_matrix, mesh_nodes_coords_ref = \
                    self._generate_finite_element_mesh(elem_type,
                                                       n_elems_per_dim)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print('    > Computing boundary nodes displacements...')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute finite element mesh boundary nodes displacements
                mesh_boundary_nodes_disps = \
                    self._get_mesh_boundary_nodes_disps(
                        edges_coords_ref, edges_coords_def,
                        mesh_nodes_coords_ref)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print('    > Generating finite element patch...')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate finite element patch
                patch = FiniteElementPatch(
                    self._n_dim, copy.deepcopy(self._patch_dims), elem_type,
                    copy.deepcopy(n_elems_per_dim),
                    copy.deepcopy(mesh_nodes_matrix),
                    copy.deepcopy(mesh_nodes_coords_ref),
                    copy.deepcopy(mesh_boundary_nodes_disps))
                # Leave iterative loop
                break 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            if is_admissible:
                print('\n> Generation status: Success\n')
            else:
                print('\n> Generation status: Failure\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_admissible, patch
    # ------------------------------------------------------------------------- 
    def _build_corners_bc(self, corners_lab_bc=None, is_random_min=False):
        """Build boundary conditions on patch corners.
        
        If boundary conditions are not provided, then a random set of minimal
        constraints is prescribed.
        
        Parameters
        ----------
        corners_lab_bc : dict, default=None
            Boundary conditions (item, tuple[int](n_dim)) for each patch
            corner label (key, str[int]). The tuple[int](n_dim)] prescribes 0
            (free) or 1 (fixed) for each degree of freedom. Unspecified corners
            are assumed free by default.
        is_random_min : bool, False
            Prescribe a random set of minimal constraints. This option
            overrides any prescribed boundary conditions.
        
        Returns
        -------
        corners_bc : tuple[tuple]
            Boundary conditions applied to the corners of the patch. For each
            node, a tuple(n_dim) prescribes 0 (free) or 1 (fixed) for each
            degree of freedom.
        """
        # Build corners boundary conditions
        if is_random_min:
            # Set number of minimal constraints
            if self._n_dim == 2:
                n_min_bc = 3
            else:
                n_min_bc = 6
            # Generate random set of minimal constraints
            random_bc = np.zeros(self._n_corners*self._n_dim, dtype=int)
            random_bc[:n_min_bc] = 1
            np.random.shuffle(random_bc)
            # Build corners boundary conditions 
            corners_bc = []
            for i in range(self._n_corners):
                corners_bc.append(
                    tuple([random_bc[k]
                        for k in range(i*self._n_dim, (i + 1)*self._n_dim)]))
        else:
            # Initialize corners boundary conditions
            corners_bc = [self._n_dim*(0,) for i in range(self._n_corners)]
            # Build corners boundary conditions
            if isinstance(corners_lab_bc, dict):
                # Loop over prescribed corners
                for label in corners_lab_bc.keys():
                    # Get corner internal index
                    index = self._corners_mapping[label]
                    # Check corner prescribed boundary condition
                    if not isinstance(corners_lab_bc[label], tuple):
                        raise RuntimeError('Corner boundary conditions must '
                                           'be prescribed as a '
                                           'tuple[int](n_dim).')
                    elif len(corners_lab_bc[label]) != self._n_dim:
                        raise RuntimeError('Corner boundary conditions must '
                                           'be prescribed as a '
                                           'tuple[int](n_dim).')                
                    # Set corner boundary conditions
                    corners_bc[index] = corners_lab_bc[label]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(corners_bc) 
    # -------------------------------------------------------------------------
    def _build_corners_disp_range(self, corners_lab_disp_range=None, 
                                  corners_bc=None):
        """Build patch corners displacement range.
         
        Parameters
        ----------
        corners_lab_disp_range : dict, default=None
            Displacement range along each dimension (item, tuple[tuple(2)]) for
            each corner label (key, str[int]). Corners are labeled from 1 to
            number of corners. Range is specified as tuple(min, max) for each
            dimension.  If None, a null displacement range is set by default.
        corners_bc : tuple[tuple], default=None
            Boundary conditions applied to the corners of the patch. For each
            node, a tuple(n_dim) prescribes 0 (free) or 1 (fixed) for each
            degree of freedom.

        Returns
        -------
        corners_disp_range : numpy.ndarray(3d)
            Patch corners displacements (numpy.ndarray(n_corners, n_dim, k),
            where k=0 (min) and k=1 (max)).
        """
        # Initialize corners displacement range
        corners_disp_range = np.zeros((self._n_corners, self._n_dim, 2))
        # Set corners displacement range
        if corners_lab_disp_range is None:
            # Loop over corners
            for i in range(self._n_corners):
                # Loop over dimensions
                    for j in range(self._n_dim):
                        # Set null displacement range
                        corners_disp_range[i, j, :] = [0, 0]
        elif isinstance(corners_lab_disp_range, dict):
            # Get mapping between corners labels and internal indexing
            corners_mapping = self._corners_mapping
            # Loop over corners
            for i in range(self._n_corners):
                # Get corner label, dimension and internal index
                label = str(i + 1)
                index = corners_mapping[label]
                if label in corners_lab_disp_range.keys():
                    # Check displacement ranges
                    if not isinstance(corners_lab_disp_range[label], tuple):
                        raise RuntimeError('Corner displacement ranges '
                                           'must be a tuple of tuples.')
                    elif len(corners_lab_disp_range[label]) != self._n_dim:
                        raise RuntimeError('Corner displacement range '
                                           'must be specified for all '
                                           'dimensions.')
                    else:
                        range_dims = corners_lab_disp_range[label]
                    # Loop over dimensions
                    for j in range(self._n_dim):
                        # Check displacement range
                        if not isinstance(range_dims[j], tuple):
                            raise RuntimeError('Corner displacement range '
                                               'must be a tuple(min, max) '
                                               'along each dimension.')
                        elif len(range_dims) != 2:
                            raise RuntimeError('Corner displacement range '
                                               'must be a tuple(min, max) '
                                               'along each dimension.')
                        # Set corner displacement range
                        corners_disp_range[index, j, :] = range_dims[j]       
        else:
            raise RuntimeError('Invalid specification of '
                               'corners_lab_disp_range.')         
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce boundary conditions
        if corners_bc is not None:
            for i in range(self._n_corners):
                for j in range(self._n_dim):
                    if corners_bc[i][j] == 1:
                        corners_disp_range[i, j, :] = [0, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        return corners_disp_range             
    # -------------------------------------------------------------------------
    def _build_edges_poly_orders(self, edges_lab_def_order=None):
        """Build patch edges deformation polynomials orders.
         
        Parameters
        ----------
        edges_lab_def_order : {int, dict}, default=None
            Deformation polynomial order (item, int) for each edge label
            (key, str[int]). Edges are labeled from 1 to number of edges.
            If a single order is provided, then it is assumed for all the
            edges. Zero order is assumed for unspecified edges. If None, zero
            order is assumed for all the edges.
            
        Returns
        -------
        edge_poly_orders : dict
            For each dimension (key, str[int]), store the corresponding edges
            deformation polynomials orders (item, tuple[int]).
        """
        # Set number of edges oriented along each dimension
        if self._n_dim == 2:
            n_edges_per_dim = 2
        else:
            n_edges_per_dim = 4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize edges deformation polynomials orders
        edge_poly_orders = {str(i): n_edges_per_dim*[0,]
                            for i in range(self._n_dim)}
        # Build edges deformation polynomials orders
        if edges_lab_def_order is None:
            # Set zero order deformation polynomial by default
            for i in range(self._n_dim):
                edge_poly_orders[str(i)] = self.n_edges_per_dim*(0,)
        elif isinstance(edges_lab_def_order, int):
            # Enforce minimum of zero order polynomial
            edges_lab_def_order = np.max((0, edges_lab_def_order))
            # Set edges deformation polynomials orders
            for i in range(self._n_dim):
                edge_poly_orders[str(i)] = \
                    n_edges_per_dim*(edges_lab_def_order,)
        elif isinstance(edges_lab_def_order, dict):
            # Get mapping between edges labels and internal indexing
            edges_mapping = self._edges_mapping
            # Set edges deformation polynomials orders
            for i in range(self._n_edges):
                # Get edge label, dimension and internal index
                label = str(i + 1)
                dim = str(edges_mapping[label][0])
                index = edges_mapping[label][1]
                # Set edge deformation polynomials order
                if label in edges_lab_def_order.keys():                    
                    # Get deformation polynomial order
                    order = edges_lab_def_order[label]
                    if not isinstance(order, int):
                        raise RuntimeError('Edge polynomial order must be a '
                                           'non-negative integer.')
                    # Enforce minimum polynomial order
                    order = np.max((0, order))
                else:
                    # Assume zero order polynomial
                    order = 0
                # Set edge deformation polynomial order
                edge_poly_orders[dim][index] = order
        else:
            raise RuntimeError('Invalid specification of edges_lab_def_order.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        return edge_poly_orders
    # -------------------------------------------------------------------------
    def _build_edges_disp_range(self, edges_lab_disp_range=None):
        """Build patch edges displacements range.
         
        Parameters
        ----------
        edges_lab_disp_range : dict, default=None
            Displacement range (item, tuple[float](2)) for each edge
            label (key, str[int]). Edges are labeled from 1 to number of edges.
            Range is specified as a tuple(min, max) where: (1) displacement
            range is orthogonal to the edge (reference configuration),
            (2) displacement range is relative to midplane defined by both
            limiting corners (deformed configuration), and
            (3) positive/negative displacement corresponds to outward/inward
            direction with respect to the patch. Null displacement range is
            assumed for unspecified edges. If None, null displacement range is
            set by default.
            
        Returns
        -------
        edge_disp_range : dict
            For each dimension (key, str[int]), store the corresponding edges
            displacement range (item, tuple[tuple(min, max)]).
        """
        # Set number of edges oriented along each dimension
        if self._n_dim == 2:
            n_edges_per_dim = 2
        else:
            n_edges_per_dim = 4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize edges displacements range
        edge_disp_range = {str(i): n_edges_per_dim*[(),]
                           for i in range(self._n_dim)}
        # Build edges displacements range
        if edges_lab_disp_range is None:
            # Set null displacement range by default
            for i in range(self._n_dim):
                edge_disp_range[str(i)] = n_edges_per_dim*((0, 0),)
        elif isinstance(edges_lab_disp_range, dict):
            # Get mapping between edges labels and internal indexing
            edges_mapping = self._edges_mapping
            # Set edges displacements range
            for i in range(self._n_edges):
                # Get edge label, dimension and internal index
                label = str(i + 1)
                dim = str(edges_mapping[label][0])
                index = edges_mapping[label][1]
                # Set edge displacement range
                if label in edges_lab_disp_range.keys():                    
                    # Get displacement range
                    disp_range = edges_lab_disp_range[label]
                    if not isinstance(disp_range, tuple):
                        raise RuntimeError('Edge displacement range must '
                                           'be a tuple(min, max).')
                else:
                    # Assume null displacement range
                    disp_range = (0, 0)
                # Set edge displacement range
                edge_disp_range[dim][index] = disp_range
            # Loop over dimensions
            for i in range(self._n_dim):
                edge_disp_range[str(i)] = tuple(edge_disp_range[str(i)])
        else:
            raise RuntimeError('Invalid specification of '
                               'edges_lab_disp_range.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        return edge_disp_range
    # -------------------------------------------------------------------------
    def _set_corners_attributes(self):
        """Set patch corners attributes.""" 
        # Set mapping between corners labels and internal indexing
        corners_mapping = {str(i + 1): i for i in range(self._n_corners)}     
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._corners_mapping = corners_mapping
    # -------------------------------------------------------------------------
    def _set_corners_coords_ref(self):
        """Set patch corners coordinates (reference configuration)."""
        # Get patch dimensions
        dims = self._patch_dims
        # Set corners coordinates
        corners_coords = np.zeros((self._n_corners, self._n_dim))
        if self._n_dim == 2:
            corners_coords[0, :] = np.array([0.0, 0.0])
            corners_coords[1, :] = np.array([dims[0], 0.0])
            corners_coords[2, :] = np.array([dims[0], dims[1]])
            corners_coords[3, :] = np.array([0.0, dims[1]])
        else:
            corners_coords[0, :] = np.array([0.0, 0.0, 0.0])
            corners_coords[1, :] = np.array([dims[0], 0.0, 0.0])
            corners_coords[2, :] = np.array([dims[0], dims[1], 0.0])
            corners_coords[3, :] = np.array([0.0, dims[1], 0.0])
            corners_coords[4, :] = np.array([0.0, 0.0, dims[2]])
            corners_coords[5, :] = np.array([dims[0], 0.0, dims[2]])
            corners_coords[6, :] = np.array([dims[0], dims[1], dims[2]])
            corners_coords[7, :] = np.array([0.0, dims[1], dims[2]])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._corners_coords_ref = corners_coords
    # -------------------------------------------------------------------------
    def _set_edges_attributes(self):
        """Set patch edges attributes.""" 
        if self._n_dim == 2:
            # Set edges connectities with respect to corners
            edges_per_dim = {'0': ((0, 1), (3, 2)),
                             '1': ((0, 3), (1, 2))}
            # Set edges deformation plans orientations (Euler angles (degrees)
            # sorted according to Bunge convention (Z1-X2-Z3))
            edges_euler_angles_per_dim = {'0': (np.array([[0, -180, 0],]),
                                                np.array([[0, 0, 0],])),
                                          '1': (np.array([[-90, 0, 0],]),
                                                np.array([[90, -180, 0],]))}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set mapping between edges labels and internal indexing
            edges_mapping = {'1': ('0', 0), '2': ('0', 1),
                             '3': ('1', 0), '4': ('1', 1)}
        else:
            raise RuntimeError('Missing 3D implementation.')     
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._edges_per_dim = edges_per_dim
        self._edges_euler_angles_per_dim = edges_euler_angles_per_dim
        self._edges_mapping = edges_mapping
    # -------------------------------------------------------------------------
    def _get_n_edge_nodes_per_dim(self, elem_type, n_elems_per_dim): 
        """Get number of patch edge nodes along each dimension.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension.
            
        Returns
        -------
        n_edge_nodes_per_dim : tuple[int]
            Number of patch edge nodes along each dimension.
        """
        # Get number of edge nodes per element
        n_edge_nodes_elem = FiniteElement(elem_type).get_n_edge_nodes()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = []
        # Loop over each dimension
        for i in range(self._n_dim):
            # Compute number of edge nodes
            n_edge_nodes_per_dim.append(n_elems_per_dim[i]*n_edge_nodes_elem
                                        - (n_elems_per_dim[i] - 1))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(n_edge_nodes_per_dim)
    # -------------------------------------------------------------------------
    def _get_corners_random_displacements(self, corners_disp_range,
                                          edges_poly_orders=None):
        """Compute patch corners random displacements.
        
        Parameters
        ----------
        corners_disp_range : numpy.ndarray(3d)
            Patch corners displacements (numpy.ndarray(n_corners, n_dim, k),
            where k=0 (min) and k=1 (max)).
        edges_poly_orders : dict, default=None
            For each dimension (key, str[int]), store the corresponding edges
            deformation polynomials orders (item, tuple[int]).
              
        Returns
        -------
        corners_disp : numpy.ndarray(2d)
            Patch corners displacements (numpy.ndarray(n_corners, n_dim)).
        corners_disp_range : numpy.ndarray(3d)
            Patch corners displacements (numpy.ndarray(n_corners, n_dim, k),
            where k=0 (min) and k=1 (max)).
        """
        # Initialize corners displacements
        corners_disp = np.zeros((self._n_corners, self._n_dim))
        # Loop over corners
        for i in range(self._n_corners):
            # Loop over dimensions
            for j in range(self._n_dim):
                # Get displacement bounds
                bounds = (corners_disp_range[i, j, 0],
                          corners_disp_range[i, j, 1])
                # Sample random displacement along dimension: uniform
                # distribution
                corners_disp[i, j] = np.random.uniform(low=bounds[0],
                                                       high=bounds[1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce edge zero order deformation polynomial
        if edges_poly_orders is not None:
            # Loop over dimensions
            for i in range(self._n_dim):
                # Get orthogonal dimensions
                orth_dims = self._get_orthogonal_dims(i)
                # Loop over edges
                for j, (cid_l, cid_r) in enumerate(
                        self._edges_per_dim[str(i)]):
                    # If edge zero order deformation polynomial
                    if edges_poly_orders[str(i)][j] == 0:
                        # Loop over orthogonal dimensions
                        for k in orth_dims:
                            # Get displacement ranges
                            range_l = (corners_disp_range[cid_l, k, 0],
                                       corners_disp_range[cid_l, k, 1])
                            range_r = (corners_disp_range[cid_r, k, 0],
                                       corners_disp_range[cid_r, k, 1])
                            # Get displacement range intersection
                            min_max = np.min((range_l[1], range_r[1]))
                            max_min = np.max((range_l[0], range_r[0]))
                            if min_max > max_min:
                                bounds = (max_min, min_max)
                            else:
                                bounds = (0, 0)
                            # Sample random displacement along dimension:
                            # uniform distribution
                            disp = np.random.uniform(low=bounds[0],
                                                     high=bounds[1])
                            # Enforce the same displacement on both corners
                            corners_disp[cid_l, k] = disp
                            corners_disp[cid_r, k] = disp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        return corners_disp, corners_disp_range
    # -------------------------------------------------------------------------
    def _get_deformed_boundary_edge(self, nodes_coords_ref, left_node_def,
                                    right_node_def, poly_order,
                                    poly_bounds_range=None, is_plot=False):
        """Get randomly deformed boundary edge node coordinates in 2D plane.
        
        The boundary edge nodes (reference configuration) must be sorted in
        ascending order along the first dimension, taken as the edge
        'reference' direction.
        
        The deformed configuration of the boundary edge is computed by sampling
        a random polynomial along the second dimension.
        
        The boundary edge nodes are then projected to the deformed
        configuration and the corresponding displacements are computed based on
        the reference coordinates.
        
        Parameters
        ----------
        nodes_coords_ref : numpy.ndarray(2d)
            Boundary edge nodes coordinates in the reference configuration
            (numpy.ndarray(n_edge_nodes, 2)). 
        left_node_def : numpy.ndarray(1d)
            Boundary edge leftmost node coordinates in the deformed
            configuration (numpy.ndarray(2)).
        right_node_def : numpy.ndarray(1d)
            Boundary edge rightmost node coordinates in the deformed
            configuration (numpy.ndarray(2)).
        poly_order : int
            Order of random polynomial sampled to generate the deformed
            configuration of the boundary edge.
        poly_bounds_range : tuple[float], default=None
            Polynomial range along second dimension. Range is relative to
            midplane defined by both boundary edge nodes along the second
            dimension.
        is_plot : bool, default=False
            If True, plot boundary edge reference and deformed configurations.
        
        Returns
        -------
        nodes_coords_def : numpy.ndarray(2d)
            Boundary edge nodes coordinates in the deformed configuration
            (numpy.ndarray(n_edge_nodes, 2)).
        nodes_disp : numpy.ndarray(2d)
            Boundary edge nodes displacements (numpy.ndarray(n_edge_nodes, 2)).
        """
        # Check limit nodes
        if left_node_def[0] >= right_node_def[0]:
            raise RuntimeError('Invalid boundary edge limit nodes coordinates '
                               'along first dimension.')
        # Check if boundary edge nodes are sorted along first dimension
        is_sorted = lambda arr: np.all(arr[:-1] <= arr[1:])
        if not is_sorted(nodes_coords_ref[:, 0]):
            raise RuntimeError('Boundary edge nodes must be sorted in '
                               'ascending order along the first dimension.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get reference coordinate
        ref_coord = np.mean((left_node_def[1], right_node_def[1]))
        # Compute polynomial bounds
        if poly_bounds_range is None:
            # Assume null polynomial bounds range
            poly_lower_bound = ref_coord
            poly_upper_bound = ref_coord
        else:
            if not isinstance(poly_bounds_range, tuple):
                raise RuntimeError('Polynomial bounds range must be '
                                   'specified as a tuple(min, max).')
            elif len(poly_bounds_range) != 2:
                raise RuntimeError('Polynomial bounds range must be '
                                   'specified as a tuple(min, max).')
            # Compute polynomial bounds from range
            poly_lower_bound = ref_coord + poly_bounds_range[0]
            poly_upper_bound = ref_coord + poly_bounds_range[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate random polynomial
        coefficients = type(self)._polynomial_sampler(
            poly_order, left_node_def, right_node_def,
            lower_bound=poly_lower_bound, upper_bound=poly_upper_bound)
        # Polynomial evaluation
        def polynomial(x, coefficients):
            return np.sum([coefficients[i]*x**i
                           for i in range(len(coefficients))])
        # 1D coordinates linear mapping
        def linear_coord_map(x1, x1_lower, x1_upper, x2_lower, x2_upper):
            ratio = (x2_upper - x2_lower)/(x1_upper - x1_lower)
            x2 = x2_lower + ratio*(x1 - x1_lower)
            return x2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of points for polynomial first-order approximation
        n_poly_points = 1000
        # Compute point index step length
        point_step = 1.0/(n_poly_points - 1)
        # Compute first-order approximation of polynomial cord lengths
        cord_lengths = np.empty(0)
        for i in range(n_poly_points - 1):
            # Get cord points coordinates along first dimension
            x1_l = linear_coord_map(i*point_step, 0, 1,
                                    left_node_def[0], right_node_def[0])
            x1_r = linear_coord_map((i + 1)*point_step, 0, 1,
                                    left_node_def[0], right_node_def[0])
            # Compute cord points coordinates from polynomial
            point_l = np.array([x1_l, polynomial(x1_l, coefficients)])
            point_r = np.array([x1_r, polynomial(x1_r, coefficients)])
            # Compute linear distance between cord points
            length = np.linalg.norm(point_r - point_l)
            # Store cord length
            cord_lengths = np.append(cord_lengths, length)            
        # Compute points fractional positions along polynomial
        total_length = np.sum(cord_lengths)
        frac_point_positions = \
            (1.0/total_length)*np.array([np.sum(cord_lengths[:i])
                                         for i in range(n_poly_points)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of boundary edge nodes
        n_nodes = nodes_coords_ref.shape[0]
        # Initialize boundary edge nodes deformed coordinates
        nodes_coords_def = np.zeros((n_nodes, 2))
        # Set boundary edge limit nodes deformed coordinates
        nodes_coords_def[0, :] = left_node_def
        nodes_coords_def[-1, :] = right_node_def
        # Compute first-order approximation of boundary edge reference
        # configuration cord lengths
        cord_lengths = np.empty(0)
        for i in range(n_nodes - 1):
            # Get boundary cord points coordinates
            point_l = nodes_coords_ref[i, :]
            point_r = nodes_coords_ref[i + 1, :]
            # Compute linear distance between cord points
            length = np.linalg.norm(point_r - point_l)
            # Store cord length
            cord_lengths = np.append(cord_lengths, length) 
        # Compute nodes fractional positions along boundary edge
        total_length = np.sum(cord_lengths)
        frac_node_positions = \
            (1.0/total_length)*np.array([np.sum(cord_lengths[:i])
                                         for i in range(n_nodes)])
        # Loop over boundary edge interior nodes
        for i in range(1, n_nodes - 1):
            # Get boundary edge node fractional position
            frac_node_position = frac_node_positions[i]
            # Find node position index with respect to polynomial points
            index = np.searchsorted(frac_point_positions, frac_node_position,
                                    side='right')
            # Get adjacent polynomial points fractional position
            index_l = index - 1
            fp_l = frac_point_positions[index_l]
            index_r = index
            fp_r = frac_point_positions[index_r]
            # Get node index fractional position
            t = index_l*point_step \
                + ((frac_node_position - fp_l)/(fp_r - fp_l))*point_step
            # Compute node coordinate along first dimension
            x1 = linear_coord_map(t, 0, 1, left_node_def[0], right_node_def[0])
            # Store node deformed coordinates
            nodes_coords_def[i, :] = \
                np.array([x1, polynomial(x1, coefficients)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute boundary edge nodes displacements
        nodes_disp = nodes_coords_def - nodes_coords_ref
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot boundary edge reference and deformed configurations
        if is_plot:
            # Generate plot
            _, ax = plt.subplots()
            # Plot reference boundary edge
            ax.plot([nodes_coords_ref[i][0] for i in range(n_nodes)],
                    [nodes_coords_ref[i][1] for i in range(n_nodes)],
                    '-o', color='k') 
            # Plot deformed boundary edge polynomial
            x1 = np.linspace(left_node_def[0], right_node_def[0], 100)
            x2 = [polynomial(val, coefficients) for val in x1]
            ax.plot(x1, x2, color='#1f77b4')
            # Plot deformed boundary edge nodes
            ax.plot([nodes_coords_def[i][0] for i in range(n_nodes)],
                    [nodes_coords_def[i][1] for i in range(n_nodes)],
                    'o', color='#d62728')
            # Set axes properties
            ax.set(xlabel='x', ylabel='y')
            ax.set_aspect('equal', adjustable='box')       
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return nodes_coords_def, nodes_disp
    # -------------------------------------------------------------------------
    @staticmethod
    def _polynomial_sampler(order, left_point, right_point, lower_bound=None,
                            upper_bound=None, is_plot=False):
        """Generate random polynomial by sampling points within given bounds.
        
        Arguments
        ---------
        order : int
            Order of polynomial.
        left_point : tuple[float]
            Leftmost control point of polynomial. Sets sampling lower bound
            along first dimension.
        right_point : tuple[float]
            Rightmost control point of polynomial. Sets sampling upper bound
            along first dimension.
        lower_bound : float, default=None
            Sampling lower bound along second dimension. If None, lower bound
            is set from limit control points.
        upper_bound : float, default=None
            Sampling upper bound along second dimension. If None, upper bound
            is set from limit control points.
        is_plot : bool, default=False
            If True, plot randomly generated polynomial.
        """
        # Set number of sampled internal points
        n_int = np.max((0, order - 1))
        # Set total number of points
        n_point = 2 + n_int
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check sampling boundaries
        if lower_bound is None:
            lower_bound = np.min((left_point[1], right_point[1]))
        if upper_bound is None:
            upper_bound = np.max((left_point[1], right_point[1]))
        # Set sampling boundaries
        sampling_bounds = ((left_point[0], right_point[0]),
                           (lower_bound, upper_bound))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Get sampling points coordinates along first dimension: evenly spaced
        x1 = np.linspace(start=sampling_bounds[0][0],
                         stop=sampling_bounds[0][1],
                         num=n_point)[1:-1]
        # Get sampling points coordinates along second dimension: uniform
        # distribution
        x2 = np.random.uniform(low=sampling_bounds[1][0],
                               high=sampling_bounds[1][1],
                               size=n_int)
        # Set sampling points
        sampling_points = [(x1[i], x2[i]) for i in range(n_int)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble polynomial points
        control_points = [left_point, right_point] + sampling_points
        # Build matrix and right-hand side
        matrix = np.zeros((n_point, n_point))
        rhs = np.zeros(n_point)
        for i in range(n_point):
            matrix[i, :] = [control_points[i][0]**k for k in range(n_point)]
            rhs[i] = control_points[i][1]
        # Solve system of equations for polynomial coefficients
        coefficients = np.linalg.solve(matrix, rhs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot polynomial
        if is_plot:
            # Evaluate polynomial
            def polynomial(x, coefficients):
                return np.sum([coefficients[i]*x**i
                               for i in range(len(coefficients))])
            # Generate plot
            _, ax = plt.subplots()
            # Plot reference line (second dimension)
            ax.plot([left_point[0], right_point[0]],
                    [0, 0], 'k-')
            # Plot sampling bounds (second dimension)
            ax.plot([left_point[0], right_point[0]],
                    [lower_bound, lower_bound], 'k--')
            ax.plot([left_point[0], right_point[0]],
                    [upper_bound, upper_bound], 'k--')
            # Plot polynomial
            x1 = np.linspace(left_point[0], right_point[0], 100)
            x2 = [polynomial(val, coefficients) for val in x1]
            ax.plot(x1, x2, color='#1f77b4')
            # Plot polynomial randomly sampled control points
            ax.scatter([control_points[i][0] for i in range(n_point)],
                       [control_points[i][1] for i in range(n_point)],
                       color='#d62728', zorder=10)
            # Set axes properties
            ax.set(xlabel='x1', ylabel='x2')
            ax.set_aspect('equal', adjustable='box')
            ax.grid()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return coefficients
    # -------------------------------------------------------------------------
    def _is_admissible_geometry(self, edges_coords):
        """Check whether patch is geometrically admissible.
        
        Parameters
        ----------
        edges_coords : dict[list[numpy.ndarray(2d)]]
            For each dimension (key, str[int]), store the corresponding edges
            coordinates (item, list[numpy.ndarray(2d)]). Each edge coordinates
            are stored as a numpy.ndarray(n_edge_nodes, n_dim). Corner nodes
            are assumed part of the edge.
            
        Returns
        -------
        is_admissible : bool
            If True, the patch is geometrically admissible.
        """
        if self._n_dim == 2:
            # Initialize polygon coordinates
            coords_array = np.empty((0, 2))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set edges clockwise order based on corners indexes
            edges_clockwise = ((0, 1), (1, 2), (2, 3), (3, 0))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over edges in clockwise order
            for target in edges_clockwise:
                # Loop over edges
                for label in self._edges_mapping.keys():
                    # Get edge dimension and internal index
                    dim = self._edges_mapping[label][0]
                    index = self._edges_mapping[label][1]
                    # Get edge corners
                    corners = self._edges_per_dim[dim][index]
                    # Check if target edge
                    is_target_edge = \
                        set(target) == set(self._edges_per_dim[dim][index])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Append edge 
                    if is_target_edge:
                        # Get edge nodes coordinates
                        edge_coords = edges_coords[dim][index]
                        # Set nodes sorting
                        is_ascending = target == corners
                        # Sort edge nodes according to clockwise order
                        if is_ascending:
                            edge_coords = edge_coords[
                                np.argsort(edge_coords[:, int(dim)])]
                        else:
                            edge_coords = edge_coords[
                                np.argsort(edge_coords[:, int(dim)])][::-1, :]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Append edge nodes to polygon coordinates
                        coords_array = np.append(coords_array,
                                                 edge_coords[:-1, :], axis=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Close polygon
            coords_array = np.append(coords_array, coords_array[0:1, :],
                                     axis=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate polygon
            polygon = shapely.geometry.Polygon(coords_array)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if material patch is geometricaly admissible
            is_admissible = polygon.is_valid
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Missing 3D implementation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_admissible
    # -------------------------------------------------------------------------
    def _get_orthogonal_dims(self, dim):
        """Get orthogonal dimensions to given dimension.
        
        Parameters
        ----------
        dim : int
            Dimension.
            
        Returns
        -------
        orthogonal_dims : tuple[int]
            Orthogonal dimensions to given dimension.
        """
        orthogonal_dims = \
            tuple({i for i in range(self._n_dim)}.difference({dim,}))
        return orthogonal_dims
    # -------------------------------------------------------------------------
    @staticmethod
    def _rotate_coords_array(coords_array, r):
        """Rotate coordinates array.
        
        Parameters
        ----------
        coords_array : numpy.ndarray(2d)
            Coordinates array (numpy.ndarray(n_points, n_dim)).
        r : numpy.ndarray (2d)
            Rotation tensor (for given rotation angle theta, active
            transformation (+ theta) and passive transformation (- theta)).
        
        Returns
        -------
        rot_coords_array : numpy.ndarray(2d)
            Coordinates array (numpy.ndarray(n_points, n_dim)).
        """
        # Get number of points
        n_points = coords_array.shape[0]
        # Compute rotated coordinates array
        rot_coords_array = np.zeros_like(coords_array)
        for i in range(n_points):
            rot_coords_array[i, :] = np.matmul(r, coords_array[i, :])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return rot_coords_array
    # -------------------------------------------------------------------------
    def _generate_finite_element_mesh(self, elem_type, n_elems_per_dim):
        """Generate patch finite element mesh (reference configuration).
        
        The finite element mesh is regular and uniform, assuming a spatial
        discretization in quadrilateral (2D) / hexahedral (3D) elements.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension. 
        
        Returns
        -------
        mesh_nodes_matrix : numpy.ndarray(2d or 3d)
            Finite element mesh nodes matrix
            (numpy.ndarray[int](n_edge_nodes_per_dim) where each element
            corresponds to a given node position and whose value is set either
            as the global node label or zero (if the node does not exist).
            Nodes are labeled from 1 to n_nodes.
        mesh_nodes_coords : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]). Nodes are labeled from 1 to n_nodes.   
        """
        # Get finite element
        elem = FiniteElement(elem_type)
        # Get number of edge nodes per element
        n_edge_nodes_elem = elem.get_n_edge_nodes()
        # Get element nodes matrix
        nodes_matrix = elem.get_nodes_matrix()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = []
        # Loop over each dimension
        for i in range(self._n_dim):
            # Compute number of edge nodes
            n_edge_nodes_per_dim.append(n_elems_per_dim[i]*n_edge_nodes_elem
                                        - (n_elems_per_dim[i] - 1))
        # Compute node coordinates step along each dimension
        coord_step_per_dim = tuple(
            [self._patch_dims[i]/(n_elems_per_dim[i]*(n_edge_nodes_elem - 1))
            for i in range(self._n_dim)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize mesh nodes matrix
        mesh_nodes_matrix = np.zeros(n_edge_nodes_per_dim, dtype=int)
        # Initialize mesh nodes coordinates
        mesh_nodes_coords = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize node label
        node_label = 1
        # Build mesh nodes matrix and coordinates
        if self._n_dim == 2:
            for j in range(n_edge_nodes_per_dim[1]):
                for i in range(n_edge_nodes_per_dim[0]):
                    # Get element node local index
                    elem_node_index = self._get_elem_node_index(
                        elem_type, n_elems_per_dim, (i, j))
                    # Set mesh node
                    if nodes_matrix[elem_node_index] != 0:
                        # Set mesh node label
                        mesh_nodes_matrix[i, j] = node_label
                        # Set mesh node coordinates
                        mesh_nodes_coords[str(node_label)] = \
                            np.array([i*coord_step_per_dim[0],
                                      j*coord_step_per_dim[1]])
                        # Increment nodel label
                        node_label += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mesh_nodes_matrix, mesh_nodes_coords
    # -------------------------------------------------------------------------
    def _get_elem_node_index(self, elem_type, n_elems_per_dim, global_index):
        """Get element node local index from global mesh index.
        
        Shared nodes between adjacent elements are assumed to belong to the
        element from the lower coordinate side.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension.
        global_index : tuple[int]
            Global mesh node index.
            
        Returns
        -------
        local_index : tuple[int]
            Finite element node index.
        """
        # Get finite element
        elem = FiniteElement(elem_type)
        # Get number of edge nodes per element
        n_edge_nodes_elem = elem.get_n_edge_nodes()
        # Get number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = self._get_n_edge_nodes_per_dim(elem_type,
                                                              n_elems_per_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element node index
        if self._n_dim == 2:
            # Get global mesh finite element index
            elem_index = tuple([np.max((1, int(np.ceil(
                (global_index[k]/(n_edge_nodes_per_dim[k] - 1))
                *n_elems_per_dim[k])))) for k in range(2)])
            # Get finite element node local index
            local_index = tuple([global_index[k] - (
                (elem_index[k] - 1)*n_edge_nodes_elem - (elem_index[k] - 1))
                for k in range(2)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return local_index
    # -------------------------------------------------------------------------
    def _get_mesh_boundary_nodes_disps(self, edges_coords_ref,
                                       edges_coords_def,
                                       mesh_nodes_coords_ref):
        """Compute finite element patch boundary displacements.
        
        Parameters
        ----------
        edges_coords_ref : dict[list[numpy.ndarray(2d)]]
            For each dimension (key, str[int]), store the corresponding edges
            coordinates in the reference configuration
            (item, list[numpy.ndarray(2d)]). Each edge coordinates are stored
            as a numpy.ndarray(n_edge_nodes, n_dim). Corner nodes are assumed
            part of the edge.
        edges_coords_def : dict[list[numpy.ndarray(2d)]]
            For each dimension (key, str[int]), store the corresponding edges
            coordinates in the deformed configuration
            (item, list[numpy.ndarray(2d)]). Each edge coordinates are stored
            as a numpy.ndarray(n_edge_nodes, n_dim). Corner nodes are assumed
            part of the edge.
        mesh_nodes_coords_ref : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]) in the reference configuration. Nodes are
            labeled from 1 to n_nodes.
        
        Returns
        -------
        mesh_boundary_nodes_disps : dict
            Displacements (item, numpy.ndarray(n_dim)) prescribed on each
            finite element mesh boundary node (key, str[int]). Free degrees of
            freedom must be set as None.
        """
        # Initialize finite element mesh boundary nodes displacements
        mesh_boundary_nodes_disps = {}
        # Loop over dimensions
        for i in range(self._n_dim):
            # Loop over edges
            for j in range(len(self._edges_per_dim[str(i)])):
                # Get edge nodes coordinates
                edge_nodes_coords_ref = edges_coords_ref[str(i)][j]
                edge_nodes_coords_def = edges_coords_def[str(i)][j]
                # Loop over edge nodes
                for k in range(edge_nodes_coords_ref.shape[0]):
                    # Get node coordinates
                    coord_ref = edge_nodes_coords_ref[k, :]
                    coord_def = edge_nodes_coords_def[k, :]
                    # Compute node displacement
                    disp = coord_def - coord_ref
                    # Get node label
                    label = self._get_node_label_from_coords(
                        mesh_nodes_coords_ref, coord_ref)
                    # Store node displacement
                    if str(label) not in mesh_boundary_nodes_disps.keys():
                        mesh_boundary_nodes_disps[label] = disp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
        return mesh_boundary_nodes_disps
    # -------------------------------------------------------------------------
    def _get_node_label_from_coords(self, mesh_nodes_coords_ref, node_coords):
        """Get finite element mesh node label from coordinates.
        
        Parameters
        ----------
        mesh_nodes_coords_ref : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]) in the reference configuration. Nodes are
            labeled from 1 to n_nodes.
        node_coords : numpy.ndarray(1d)
            Target node coordinates (numpy.ndarray(n_dim)). 
        
        Returns
        -------
        nodel_label : int
            Target node label.
        """
        # Initialize node label
        node_label = None
        # Loop over mesh nodes
        for label, coords in mesh_nodes_coords_ref.items():
            # Check for coordinates match
            if np.allclose(node_coords, coords):
                node_label = label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        if node_label is None:
            raise RuntimeError('Node label has not been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        return node_label
# =============================================================================
def rotation_tensor_from_euler_angles(euler_deg):
    """Set rotation tensor from Euler angles (Bunge convention).

    The rotation tensor is defined as

    .. math::

       \\mathbf{R} =
           \\begin{bmatrix}
               c_1 c_3 - c_2 s_1 s_3 & -c_1 s_3 - c_2 c_3 s_1 & s_1 s_2 \\\\
               c_3 s_1 + c_1 c_2 s_3 & c_1 c_2 c_3 - s_1 s_3 & - c_1 s_2 \\\\
               s_2 s_3 & c_3 s_2 & c_2
           \\end{bmatrix}

    where

    .. math::

       \\begin{align}
           c_1 = \\cos(\\alpha) \\qquad s_1 = \\sin(\\alpha) \\\\
           c_2 = \\cos(\\beta) \\qquad s_2 = \\sin(\\beta) \\\\
           c_3 = \\cos(\\gamma) \\qquad s_3 = \\sin(\\gamma)
        \\end{align}

    and :math:`(\\alpha, \\beta, \\gamma)` are the Euler angles corresponding
    to the Bunge convention (Z1-X2-Z3).

    ----

    Parameters
    ----------
    euler_deg : tuple
        Euler angles (degrees) sorted according to Bunge convention (Z1-X2-Z3).

    Returns
    -------
    r : numpy.ndarray (2d)
        Rotation tensor (for given rotation angle theta, active transformation
        (+ theta) and passive transformation (- theta)).
    """
    # Convert euler angles to radians
    euler_rad = tuple(np.radians(x) for x in euler_deg)
    # Compute convenient sins and cosines
    s1 = np.sin(euler_rad[0])
    s2 = np.sin(euler_rad[1])
    s3 = np.sin(euler_rad[2])
    c1 = np.cos(euler_rad[0])
    c2 = np.cos(euler_rad[1])
    c3 = np.cos(euler_rad[2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize rotation tensor
    r = np.zeros((3, 3))
    # Build rotation tensor
    r[0, 0] = c1*c3 - c2*s1*s3
    r[1, 0] = c3*s1 + c1*c2*s3
    r[2, 0] = s2*s3
    r[0, 1] = -c1*s3 - c2*c3*s1
    r[1, 1] = c1*c2*c3 - s1*s3
    r[2, 1] = c3*s2
    r[0, 2] = s1*s2
    r[1, 2] = -c1*s2
    r[2, 2] = c2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return r
    