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
import random
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
# Local
from projects.gnn_material_patch.material_patch.patch import \
    FiniteElementPatch
from projects.gnn_material_patch.discretization.finite_element import \
    FiniteElement
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================
class FiniteElementPatchGenerator:
    """Finite element material patch generator.
    
    Attributes
    ----------
    _n_dim : int
        Number of spatial dimensions.
    _patch_dims : tuple[float]
        Patch size in each dimension.
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
        indexes of the two corresponding corners. Corners are stored such that
        (sorted in asceding order of
        coordinate).
    _orth_edge_dir : dict
        2D: For each dimension (key, str[int]), stores the edges
        (item, tuple[tuple]) outward orthogonal edge direction euler angles
        with respect to edge connectivities order.
    _edges_mapping : dict
        For each edge label (key, str[int]), store the edge orientation
        dimension and internal index consistent with `_edges_per_dim`
        (tuple(2)). Edges are labeled from 1 to number of edges.

    Methods
    -------
    generate_deformed_patch(self, elem_type, n_elems_per_dim, \
                            corners_lab_bc=None, corners_lab_disp_range=None, \
                            edges_lab_def_order=None, \
                            edges_lab_disp_range=None, \
                            translation_range=None, \
                            rotation_angles_range=None, \
                            max_iter=10, is_verbose=False)
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
    _get_patch_centroid(self, corners_coords, edges_coords)
        Compute patch centroid.
    _build_boundary_coords_array(self, edges_coords, is_close_polygon)
        Build patch boundary nodes coordinates array.
    _is_admissible_simulation(self, edges_coords)
        Check whether simulation of patch is physically admissible.
    _get_orthogonal_dims(self, dim)
        Get orthogonal dimensions to given dimension.
    _rotation_tensor_deformed_edge(self, edge_dim, edge_index, init_node_def, \
                                   end_node_def)
        Set rotation tensor to deformed boundary edge local coordinates.
    _transform_to_edge_local_coordinates(self, init_node_def, end_node_def, \
                                         nodes_coords_ref, \
                                         translation = None, rotation = None)
        Transform from patch coordinates to deformed edge local coordinates.
    _transform_from_edge_local_coordinates(self, local_nodes_coords_def, \
                                           translation=None, rotation=None)
        Transform from deformed edge local coordinates to patch coordinates.
    _rotate_coords_array(coords_array, r)
        Rotate coordinates array.
    _generate_finite_element_mesh(self, elem_type, n_elems_per_dim)
        Generate patch finite element mesh (reference configuration).
    _get_elem_node_index(self, elem_type, n_elems_per_dim, global_index)
        Get element node local index from global mesh index.
    _get_mesh_boundary_nodes_disps(self, edges_coords_ref, edges_coords_def, \
                                   mesh_nodes_coords_ref)
        Compute finite element patch boundary displacements.
    _get_node_label_from_coords(mesh_nodes_coords_ref, node_coords)
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
        elif self._n_dim == 3:
            self._n_corners = 8
            self._n_edges = 24
        else:
            raise RuntimeError('Invalid number of spatial dimensions.')
        # Set corners attributes
        self._set_corners_attributes()                                         # 3D STATUS: CHECK
        # Set patch corners coordinates (reference configuration)
        self._set_corners_coords_ref()                                         # 3D STATUS: CHECK
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edges attributes
        self._set_edges_attributes()                                           # 3D STATUS: CHECK
    # -------------------------------------------------------------------------  
    def generate_deformed_patch(self, elem_type, n_elems_per_dim,
                                corners_lab_bc=None,
                                corners_lab_disp_range=None,
                                edges_lab_def_order=None,
                                edges_lab_disp_range=None,
                                translation_range=None,
                                rotation_angles_range=None,
                                is_remove_rbm=False,
                                deformation_noise=0.0,
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
            corners. Unspecified corners are assumed free by default.
        corners_lab_disp_range : dict, default=None
            Displacement range along each dimension (item, tuple[tuple(2)]) for
            each corner label (key, str[int]). Corners are labeled from 1 to
            number of corners. Range is specified as tuple(min, max) for each
            dimension. If None, a null displacement range is set by default.
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
            range is orthogonal and relative to the edge (defined by limiting
            corner nodes in the deformed configuration), (2) positive/negative
            displacement corresponds to outward/inward direction with respect
            to the patch. Null displacement range is assumed for unspecified
            edges. If None, null displacement range is set by default.
        translation_range : dict, default=None
            Translational displacement range (item, tuple[float](2)) along each
            dimension (key, str[int]). Range is specified as tuple(min, max)
            for each dimension. Null range is assumed for unspecified
            dimensions. If None, then there is no translational motion.
        rotation_angles_range : dict, default=None
            Rotational angle range (item, tuple[float](2)) for each Euler angle
            (key, str). Euler angles follow Bunge convention (Z1-X2-Z3) and are
            labelled ('alpha', 'beta', 'gamma'), respectively. Null range is
            assumed for unspecified angles. If None, then there is no
            rotational motion.
        is_remove_rbm : bool, default=False
            Remove rigid body motions. Deprecated.
        deformation_noise : float, default=0.0
            Parameter that controls the normally distributed noise superimposed
            to the boundary edges interior nodes coordinates in the deformed
            configuration. Defines the noise standard deviation along given
            dimension after being multiplied by the corresponding patch size.
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
        corners_bc = self._build_corners_bc(corners_lab_bc=corners_lab_bc)     # 3D STATUS: CHECK
        # Build corners displacement range
        corners_disp_range = self._build_corners_disp_range(
            corners_lab_disp_range=corners_lab_disp_range,
            corners_bc=corners_bc)                                             # 3D STATUS: CHECK
        # Build edges deformation polynomials orders and displacement range
        edges_poly_orders = self._build_edges_poly_orders(
            edges_lab_def_order=edges_lab_def_order)                           # 3D STATUS: CHECK
        edges_disp_range = self._build_edges_disp_range(
            edges_lab_disp_range=edges_lab_disp_range)                         # 3D STATUS: CHECK
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get corners coordinates (reference configuration)
        corners_coords_ref = self._corners_coords_ref
        # Get edges attributes
        edges_per_dim = self._edges_per_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = \
            self._get_n_edge_nodes_per_dim(elem_type, n_elems_per_dim)         # 3D STATUS: CHECK
        # Set edges nodes coordinates (reference configuration)
        edges_coords_ref = {}
        for i in range(self._n_dim):
            # Get number of patch edge nodes
            n_edge_nodes = n_edge_nodes_per_dim[i]
            # Initialize edges nodes coordinates along dimension
            edges_coords_ref[str(i)] = []
            # Loop over edges
            for (cid_init, cid_end) in edges_per_dim[str(i)]:
                # Build edge nodes coordinates assuming a regular
                # discretization (evenly spaced nodes)
                coords = np.zeros((n_edge_nodes, self._n_dim))
                # Loop over dimensions
                for j in range(self._n_dim):
                    coords[:, j] = np.linspace(corners_coords_ref[cid_init, j],
                                               corners_coords_ref[cid_end, j],
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
                    corners_disp_range, edges_poly_orders=edges_poly_orders)   # 3D STATUS: CHECK (CHECK 0 ORDER LOGIC)
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
                for j, (cid_init, cid_end) in enumerate(edges_per_dim[str(i)]):
                    # Get edge nodes coordinates (reference configuration)
                    nodes_coords_ref = edges_coords_ref[str(i)][j]
                    # Get edge corners coordinates (deformed configuration)
                    init_node_def = corners_coords_def[cid_init, :]
                    end_node_def = corners_coords_def[cid_end, :]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get rotation tensor from patch coordinates to deformed
                    # boundary edge local coordinates
                    rotation = self._rotation_tensor_deformed_edge(
                        i, j, init_node_def, end_node_def)[:self._n_dim,
                                                            :self._n_dim]
                    # Get translation from patch coordinates to deformed
                    # boundary edge local coordinates
                    translation = np.matmul(rotation, init_node_def)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Transform from patch coordinates to deformed edge local
                    # coordinates
                    local_init_node_def, local_end_node_def, \
                        local_nodes_coords_ref = \
                        self._transform_to_edge_local_coordinates(
                            init_node_def, end_node_def, nodes_coords_ref,
                            translation, rotation)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get edge deformation polynomial order
                    poly_order = edges_poly_orders[str(i)][j]
                    # Get edge displacement range
                    disp_amp = edges_disp_range[str(i)][j]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Discard out-of-deformation-plane spatial dimension
                    if self._n_dim == 3:
                        local_init_node_def = local_init_node_def[:2]
                        local_end_node_def = local_end_node_def[:2]
                        local_nodes_coords_ref = local_nodes_coords_ref[:, :2]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Generate randomly deformed boundary edge node coordinates
                    # in the edge local coordinates (deformation plane)
                    local_nodes_coords_def, local_nodes_disp = \
                        self._get_deformed_boundary_edge(
                            local_nodes_coords_ref, local_init_node_def,
                            local_end_node_def, poly_order,
                            poly_bounds_range=disp_amp,
                            noise=deformation_noise)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Restore out-of-deformation-plane spatial dimension
                    if self._n_dim == 3:
                        n_nodes = local_nodes_coords_def.shape[0]
                        local_nodes_coords_def = np.hstack(
                            (local_nodes_coords_def, np.zeros((n_nodes, 1))))
                        local_nodes_disp = np.hstack(
                            (local_nodes_disp, np.zeros((n_nodes, 1))))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get translation from deformed boundary edge local
                    # coordinates to patch coordinates
                    translation = -init_node_def
                    # Get rotation tensor from deformed boundary edge local
                    # coordinates to patch coordinates
                    rotation = np.transpose(rotation)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Transform from deformed edge local coordinates to patch
                    # coordinates
                    nodes_coords_def = \
                        self._transform_from_edge_local_coordinates(
                            local_nodes_coords_def, translation, rotation)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store edge nodes coordinates (deformed configuration)
                    edges_coords_def[str(i)].append(nodes_coords_def)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
            # Compute random rigid body motions
            rbm_translation, rbm_rotation = self._get_random_rigid_motions(
                translation_range, rotation_angles_range)                      # 3D STATUS: CHECK
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get patch centroid (deformed configuration)
            centroid_def = self._get_patch_centroid(corners_coords_def,
                                                    edges_coords_def)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Superimpose rigid body motions (translation and rotation)
            for i in range(self._n_dim):
                # Loop over edges
                for coords_array in edges_coords_def[str(i)]:
                    # Get number of boundary edge nodes
                    n_nodes = coords_array.shape[0]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rigid body rotation (around deformed configuration
                    # centroid)
                    if centroid_def is not None and rotation is not None:
                        # Build centroid tile array (local)
                        centroid_def_tile = np.tile(centroid_def, (n_nodes, 1))
                        # Superimpose rigid body rotation around centroid
                        # (in-place update)
                        coords_array[:, :] = \
                            centroid_def_tile + self._rotate_coords_array(
                                coords_array - centroid_def_tile, rbm_rotation)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rigid body translation
                    if translation is not None:
                        # Build translation tile array (local)
                        translation_tile = \
                            np.tile(rbm_translation, (n_nodes, 1))
                        # Superimpose rigid body translation (in-place update)
                        coords_array += translation_tile
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
            # Check whether simulation of patch is physically admissible
            is_admissible = self._is_admissible_simulation(edges_coords_def)   # 3D STATUS: PENDING
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
                                                       n_elems_per_dim)        # 3D STATUS: CHECK
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print('    > Computing boundary nodes displacements...')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute finite element mesh boundary nodes displacements
                mesh_boundary_nodes_disps = \
                    self._get_mesh_boundary_nodes_disps(
                        edges_coords_ref, edges_coords_def,
                        mesh_nodes_coords_ref)                                 # 3D STATUS: PENDING
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
            Patch corners displacements range
            (numpy.ndarray(n_corners, n_dim, k), where k=0 (min) and k=1
            (max)).
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
                        elif len(range_dims[j]) != 2:
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
            n_edges_per_dim = 8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize edges deformation polynomials orders
        edge_poly_orders = {str(i): n_edges_per_dim*[0,]
                            for i in range(self._n_dim)}
        # Build edges deformation polynomials orders
        if edges_lab_def_order is None:
            # Set zero order deformation polynomial by default
            for i in range(self._n_dim):
                edge_poly_orders[str(i)] = n_edges_per_dim*(0,)
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
            # Convert to tuple
            for i in range(self._n_dim):
                edge_poly_orders[str(i)] = tuple(edge_poly_orders[str(i)])
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
            range is orthogonal and relative to the edge (defined by limiting
            corner nodes in the deformed configuration), (2) positive/negative
            displacement corresponds to outward/inward direction with respect
            to the patch. Null displacement range is assumed for unspecified
            edges. If None, null displacement range is set by default.
            
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
            n_edges_per_dim = 8
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set outward orthogonal edge direction euler angles with respect
            # to edge connectivities order
            orth_edge_dir = {'0': ((-90, 0, 0), (90, 0, 0)),
                             '1': ((90, 0, 0), (-90, 0, 0))}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set mapping between edges labels and internal indexing
            edges_mapping = {'1': ('0', 0), '2': ('0', 1),
                             '3': ('1', 0), '4': ('1', 1)}
        else:
            # Set edges connectities with respect to corners
            edges_per_dim = {'0': ((0, 1), (3, 2), (4, 5), (7, 6),
                                   (0, 1), (4, 5), (3, 2), (7, 6)),
                             '1': ((0, 3), (1, 2), (4, 7), (5, 6),
                                   (0, 3), (4, 7), (1, 2), (5, 6)),
                             '2': ((0, 4), (3, 7), (1, 5), (2, 6),
                                   (0, 4), (1, 5), (3, 7), (2, 6))}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set outward orthogonal edge direction euler angles with respect
            # to edge connectivities order
            orth_edge_dir = {
                '0': ((-90, 0, 0), (90, 0, 0), (-90, 0, 0), (90, 0, 0),
                      (0, 90, -90), (0, 90, 90), (0, 90, -90), (0, 90, 90)),
                '1': ((90, 0, 0), (-90, 0, 0), (90, 0, 0), (-90, 0, 0),
                      (90, -90, 0), (90, 90, 0), (90, -90, 0), (90, 90, 0)),
                '2': ((0, 90, 0), (0, -90, 0), (0, 90, 0), (0, -90, 0),
                      (90, -90, 0), (90, 90, 0), (90, -90, 0), (90, 90, 0))}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set mapping between edges labels and internal indexing
            edges_mapping = {
                '1': ('0', 0), '2': ('0', 1), '3': ('1', 0), '4': ('1', 1),
                '5': ('0', 2), '6': ('0', 3), '7': ('1', 2), '8': ('1', 3),
                '9': ('1', 4), '10': ('1', 5), '11': ('2', 0), '12': ('2', 1),
                '13': ('1', 6), '14': ('1', 7), '15': ('2', 2), '16': ('2', 3),
                '17': ('0', 4), '18': ('0', 5), '19': ('2', 4), '20': ('2', 5),
                '21': ('0', 6), '22': ('0', 7), '23': ('2', 6), '24': ('2', 7)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._edges_per_dim = edges_per_dim
        self._orth_edge_dir = orth_edge_dir
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
            Patch corners displacements range
            (numpy.ndarray(n_corners, n_dim, k), where k=0 (min) and k=1
            (max)).
        edges_poly_orders : dict, default=None
            For each dimension (key, str[int]), store the corresponding edges
            deformation polynomials orders (item, tuple[int]).
              
        Returns
        -------
        corners_disp : numpy.ndarray(2d)
            Patch corners displacements (numpy.ndarray(n_corners, n_dim)).
        corners_disp_range : numpy.ndarray(3d)
            Patch corners displacements range
            (numpy.ndarray(n_corners, n_dim, k), where k=0 (min) and k=1
            (max)).
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
                for j, (cid_init, cid_end) in enumerate(
                        self._edges_per_dim[str(i)]):
                    # If edge zero order deformation polynomial
                    if edges_poly_orders[str(i)][j] == 0:
                        # Loop over orthogonal dimensions
                        for k in orth_dims:
                            # Get displacement ranges
                            range_l = (corners_disp_range[cid_init, k, 0],
                                       corners_disp_range[cid_init, k, 1])
                            range_r = (corners_disp_range[cid_end, k, 0],
                                       corners_disp_range[cid_end, k, 1])
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Check displacement range intersection
                            min_max = np.min((range_l[1], range_r[1]))
                            max_min = np.max((range_l[0], range_r[0]))
                            if min_max > max_min:
                                # If non-null intersection range, then sample
                                # random displacement along dimension on that
                                # range: uniform distribution
                                bounds = (max_min, min_max)
                                disp = np.random.uniform(low=bounds[0],
                                                         high=bounds[1])
                            else:
                                # If null intersection range, then sample
                                # random displacement along dimension on each
                                # range (uniform distribution) and take the
                                # average value
                                disp_l = np.random.uniform(low=range_l[0],
                                                           high=range_l[1])
                                disp_r = np.random.uniform(low=range_r[0],
                                                           high=range_r[1])
                                disp = np.average((disp_l, disp_r))
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                          
                            # Enforce the same displacement on both corners
                            corners_disp[cid_init, k] = disp
                            corners_disp[cid_end, k] = disp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        return corners_disp, corners_disp_range
    # -------------------------------------------------------------------------
    def _get_deformed_boundary_edge(self, nodes_coords_ref, left_node_def,
                                    right_node_def, poly_order,
                                    poly_bounds_range=None, noise=0.0,
                                    is_plot=False):
        """Get randomly deformed boundary edge node coordinates in 2D plane.
        
        The boundary edge nodes (reference configuration) must be sorted
        along the first dimension (either ascending or descending order).
        
        The deformed configuration of the boundary edge is computed by sampling
        a random polynomial along the second dimension of the deformed boundary
        edge local coordinates.
        
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
        noise : float, default=0.0
            Parameter that controls the normally distributed noise superimposed
            to the boundary edges interior nodes coordinates in the deformed
            configuration. Defines the noise standard deviation along given
            dimension after being multiplied by the corresponding patch size.
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
        # Check if boundary edge nodes coordinates in the reference
        # configuration are sorted
        is_sorted_ascend = lambda arr: np.all(arr[:-1] <= arr[1:])
        is_sorted_descend = lambda arr: np.all(arr[:-1] >= arr[1:])
        if not is_sorted_ascend(nodes_coords_ref[:, 0]) \
                and not is_sorted_descend(nodes_coords_ref[:, 0]):
            raise RuntimeError('Boundary edge nodes coordinates must be '
                               'sorted in ascending or descending order along '
                               'the first dimension.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize flip boundary edge nodes order flag
        is_flipped = False
        # Check if boundary edge nodes are sorted in ascending order along
        # first dimension
        if not is_sorted_ascend(nodes_coords_ref[:, 0]):
            # Set flip boundary edge nodes order flag
            is_flipped = True
            # Flip boundary edge nodes order (reference configuration)
            nodes_coords_ref = np.flipud(nodes_coords_ref.copy())
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
        # Initialize node deformed coordinates noise
        nodes_coords_noise = np.zeros_like(nodes_coords_def)
        # Loop over dimensions
        for i in range(2):
            # Set noise standard deviation
            noise_std = np.abs(noise)*self._patch_dims[i]
            # Sample coordinates noise along dimension: normal distribution
            nodes_coords_noise[:, i] = np.random.normal(
                loc=0.0, scale=noise_std, size=nodes_coords_def.shape[0])
        # Add noise to interior nodes deformed coordinates
        nodes_coords_def[1:-1, :] += nodes_coords_noise[1:-1, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert boundary edge nodes to original order (reference
        # configuration)
        if is_flipped:
            nodes_coords_ref = np.flipud(nodes_coords_ref)
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

        Returns
        -------
        coefficients : tuple[float]
            Polynomial coefficients sorted by increasing order terms.
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
        coefficients = tuple(np.linalg.solve(matrix, rhs))
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
    def _get_random_rigid_motions(self, translation_range,
                                  rotation_angles_range):
        """Get random rigid body motion tensors (translation and rotation).
        
        Parameters
        ----------
        translation_range : dict, default=None
            Translational displacement range (item, tuple[float](2)) along each
            dimension (key, str[int]). Range is specified as tuple(min, max)
            for each dimension. Null range is assumed for unspecified
            dimensions. If None, then there is no translational motion.
        rotation_angles_range : dict, default=None
            Rotational angle range (item, tuple[float](2)) for each Euler angle
            (key, str). Euler angles follow Bunge convention (Z1-X2-Z3) and are
            labelled ('alpha', 'beta', 'gamma'), respectively. Null range is
            assumed for unspecified angles. If None, then there is no
            rotational motion.
        
        Returns
        -------
        translation : numpy.ndarray(1d)
            Translation array (numpy.ndarray(n_dim)).
        rotation : numpy.ndarray(2d)
            Rotation tensor (for given rotation angle theta, active
            transformation (+ theta) and passive transformation (- theta)).
        """
        # Set translation array
        if translation_range is None:
            translation = None
        else:
            # Initialize translation array
            translation = np.zeros(self._n_dim)
            # Loop over dimensions
            for i in range(self._n_dim):
                # Sample translation along dimension
                if str(i + 1) in translation_range.keys():
                    # Get translation bounds
                    bounds = (translation_range[str(i + 1)][0],
                              translation_range[str(i + 1)][1])
                    # Sample translation along dimension: uniform distribution
                    translation[i] = \
                        np.random.uniform(low=bounds[0], high=bounds[1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set rotation tensor
        if rotation_angles_range is None:
            rotation = None
        else:
            # Initialize Euler angles
            rotation_angles = np.zeros(3)
            # Loop over Euler angles
            for i, angle in enumerate(('alpha', 'beta', 'gamma')):
                # Process only first Euler angle under two dimensions
                if self._n_dim == 2 and angle != 'alpha':
                    continue
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Sample Euler angle
                if angle in rotation_angles_range.keys():
                    # Get angle bounds
                    bounds = (rotation_angles_range[angle][0],
                              rotation_angles_range[angle][1])
                    # Sample angle: uniform distribution
                    rotation_angles[i] = \
                        np.random.uniform(low=bounds[0], high=bounds[1])
            # Compute rotation tensor
            rotation = rotation_tensor_from_euler_angles(
                tuple(rotation_angles))[:self._n_dim, :self._n_dim]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return translation, rotation
    # -------------------------------------------------------------------------
    def _get_patch_centroid(self, corners_coords, edges_coords):
        """Compute patch centroid.
        
        Parameters
        ----------
        corners_coords : numpy.ndarray(2d)
            Patch corners coordinates (numpy.ndarray(n_corners, n_dim)).
        edges_coords : dict[list[numpy.ndarray(2d)]]
            For each dimension (key, str[int]), store the corresponding edges
            coordinates (item, list[numpy.ndarray(2d)]). Each edge coordinates
            are stored as a numpy.ndarray(n_edge_nodes, n_dim). Corner nodes
            are assumed part of the edge.

        Returns
        -------
        centroid : numpy.ndarray(2d)
            Patch centroid.
        """
        # Get patch (geometrical) centroid
        if self._n_dim == 2:
            # Build boundary nodes coordinates array (close polygon)
            boundary_coords_array = self._build_boundary_coords_array(
                edges_coords, is_close_polygon=True)
            # Generate boundary polygon
            polygon = shapely.geometry.Polygon(boundary_coords_array)
            # Get boundary polygon centroid
            centroid = np.array(polygon.centroid.coords).reshape(-1)
        else:
            # Set patch (geometrical) centroid as unknown                      # Idea: Approximate with average of corners coordinates?
            centroid = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return centroid
    # -------------------------------------------------------------------------
    def _build_boundary_coords_array(self, edges_coords,
                                     is_close_polygon=False):
        """Build patch boundary nodes coordinates array.
        
        In the two-dimensional case, boundary nodes are sorted in clockwise
        order. A closed polygon is obtained by setting is_close_polygon to
        True.
        
        Parameters
        ----------
        edges_coords : dict
            For each dimension (key, str[int]), store the corresponding edges
            coordinates (item, list[numpy.ndarray(2d)]). Each edge coordinates
            are stored as a numpy.ndarray(n_edge_nodes, n_dim). Corner nodes
            are assumed part of the edge.
        is_close_polygon : bool, default=False
            If True, then close the polygon by adding a copy of the first node
            to the end of the boundary nodes coordinates array.
        
        Returns
        -------
        boundary_coords_array : numpy.ndarray(2d)
            Boundary nodes coordinates array (numpy.ndarray(n_points, n_dim)).
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
                    is_target_edge = set(target) == set(corners)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Append edge 
                    if is_target_edge:
                        # Get edge nodes coordinates
                        edge_coords = edges_coords[dim][index]                        
                        # Set nodes sorting
                        is_flip = target != corners
                        # Sort edge nodes according to clockwise order
                        if is_flip:
                            edge_coords = np.flipud(edge_coords)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Append edge nodes to polygon coordinates
                        coords_array = np.append(coords_array,
                                                 edge_coords[:-1, :], axis=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Close polygon coordinates
            if is_close_polygon:
                coords_array = \
                    np.append(coords_array, coords_array[0:1, :], axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Missing 3D implementation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return coords_array
    # -------------------------------------------------------------------------
    def _is_admissible_simulation(self, edges_coords_def):
        """Check whether simulation of patch is physically admissible.
        
        Parameters
        ----------
        edges_coords_def : dict[list[numpy.ndarray(2d)]]
            For each dimension (key, str[int]), store the corresponding edges
            coordinates (item, list[numpy.ndarray(2d)]) (deformed
            configuration). Each edge coordinates are stored as a
            numpy.ndarray(n_edge_nodes, n_dim). Corner nodes are assumed part
            of the edge.
            
        Returns
        -------
        is_admissible : bool
            If True, the patch simulation is physically admissible.
        """
        # Check patch simulation physical admissibility
        if self._n_dim == 2:
            # Build boundary nodes coordinates array (close polygon)
            coords_array = self._build_boundary_coords_array(
                edges_coords_def, is_close_polygon=True)
            # Generate polygon
            polygon = shapely.geometry.Polygon(coords_array)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if polygon geometry is valid
            is_geometry_valid = polygon.is_valid
            # Check if polygon is sorted counter-clockwise
            is_counterclockwise = polygon.exterior.is_ccw
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set whether simulation of patch is physically admissible
            is_admissible = is_geometry_valid and is_counterclockwise
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Unavailable checking procedure
            is_admissible = True
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
    def _rotation_tensor_deformed_edge(self, edge_dim, edge_index,
                                       init_node_def, end_node_def):
        """Set rotation tensor to deformed boundary edge local coordinates.
        
        Parameters
        ----------
        edge_dim : int
            Dimension along which edge is oriented in the reference
            configuration.
        edge_index : int
            Edge index with respect to edges oriented along the corresponding
            dimension.
        init_node_def : numpy.ndarray(2d)
            Boundary edge initial corner node coordinates (deformed
            configuration) stored as numpy.ndarray(n_dim).
        end_node_def : numpy.ndarray(2d)
            Boundary edge ending corner node coordinates (deformed
            configuration) stored as numpy.ndarray(n_dim).
            
        Returns
        -------
        rotation : numpy.ndarray(2d)
            Rotation tensor (for given rotation angle theta, active
            transformation (- theta) and passive transformation (+ theta)).
        """
        # Initialize rotation tensor
        rotation = np.eye(3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute edge direction unit vector
        edge_dir = np.array(end_node_def) - np.array(init_node_def)
        edge_dir = (1.0/np.linalg.norm(edge_dir))*edge_dir
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get outward orthogonal rotation euler angles
        rotation_angle = self._orth_edge_dir[str(edge_dim)][edge_index]
        # Get corresponding rotation tensor
        orth_rotation = rotation_tensor_from_euler_angles(rotation_angle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute edge orthogonal direction unit vector 
        orth_dir = \
            np.matmul(orth_rotation[:self._n_dim, :self._n_dim], edge_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build rotation tensor
        if self._n_dim == 2:
            # Assemble rotation tensor from deformed boundary edge local
            # coordinates unit vectors
            rotation[0, 0:2] = edge_dir
            rotation[1, 0:2] = orth_dir
        else:
            # Assemble rotation tensor from deformed boundary edge local
            # coordinates unit vectors
            rotation[0, :] = edge_dir
            rotation[1, :] = orth_dir
            rotation[2, :] = np.cross(edge_dir, orth_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        return rotation
    # -------------------------------------------------------------------------
    def _transform_to_edge_local_coordinates(self, init_node_def, end_node_def,
                                             nodes_coords_ref,
                                             translation = None,
                                             rotation = None):
        """Transform from patch coordinates to deformed edge local coordinates.
        
        Parameters
        ----------
        init_node_def : numpy.ndarray(2d)
            Boundary edge initial corner node coordinates (deformed
            configuration) stored as numpy.ndarray(n_dim).
        end_node_def : numpy.ndarray(2d)
            Boundary edge ending corner node coordinates (deformed
            configuration) stored as numpy.ndarray(n_dim).
        nodes_coords_ref : numpy.ndarray(2d)
            Boundary edge nodes coordinates (reference configuration) stored as
            numpy.ndarray(n_edge_nodes, n_dim).
        translation : numpy.ndarray(1d), default=None
            Translation from patch coordinates to deformed boundary edge local
            coordinates stored as numpy.ndarray(n_dim).
        rotation : numpy.ndarray(2d), default=None
            Rotation tensor from patch coordinates to deformed boundary edge
            local coordinates stored as numpy.ndarray(n_dim, n_dim).
            
        Returns
        -------
        local_init_node_def : numpy.ndarray(1d)
            Boundary edge initial corner node coordinates (deformed
            configuration) in deformed boundary edge local coordinates stored
            as numpy.ndarray(n_dim).
        local_end_node_def : numpy.ndarray(1d)
            Boundary edge ending corner node coordinates (deformed
            configuration) in deformed boundary edge local coordinates stored
            as numpy.ndarray(n_dim).
        local_nodes_coords_ref : numpy.ndarray(2d)
            Boundary edge nodes coordinates (reference configuration) in
            deformed boundary edge local coordinates stored as
            numpy.ndarray(n_edge_nodes, n_dim).
        """
        # Set default translation and rotation
        if translation is None:
            translation = np.zeros(self._n_dim)
        if rotation is None:
            rotation = np.eye(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transform boundary edge initial and ending nodes to deformed edge
        # local coordinates    
        local_init_node_def = np.matmul(rotation, init_node_def) - translation
        local_end_node_def = np.matmul(rotation, end_node_def) - translation
        # Transform boundary nodes coordinates to deformed edge local
        # coordinates 
        local_nodes_coords_ref = \
            type(self)._rotate_coords_array(nodes_coords_ref, rotation) \
            -1.0*np.tile(translation, (nodes_coords_ref.shape[0], 1))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return local_init_node_def, local_end_node_def, local_nodes_coords_ref
    # -------------------------------------------------------------------------
    def _transform_from_edge_local_coordinates(self, local_nodes_coords_def,
                                               translation=None,
                                               rotation=None):
        """Transform from deformed edge local coordinates to patch coordinates.
        
        Parameters
        ----------
        local_nodes_coords_def : numpy.ndarray(2d)
            Boundary edge nodes coordinates (deformed configuration) in
            deformed boundary edge local coordinates and stored as
            numpy.ndarray(n_edge_nodes, n_dim).
        translation : numpy.ndarray(1d), default=None
            Translation from deformed boundary edge local coordinates to patch
            coordinates stored as numpy.ndarray(n_dim).
        rotation : numpy.ndarray(2d), default=None
            Rotation tensor from deformed boundary edge local coordinates to
            patch coordinates stored as numpy.ndarray(n_dim, n_dim).
            
        Returns
        -------
        nodes_coords_def : numpy.ndarray(2d)
            Boundary edge nodes coordinates (deformed configuration) stored as
            numpy.ndarray(n_edge_nodes, n_dim).
        """
        # Set default translation and rotation
        if translation is None:
            translation = np.zeros(self._n_dim)
        if rotation is None:
            rotation = np.eye(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transform boundary nodes deformed edge local coordinates to patch
        # coordinates
        nodes_coords_def = \
            type(self)._rotate_coords_array(local_nodes_coords_def, rotation) \
            -1.0*np.tile(translation, (local_nodes_coords_def.shape[0], 1))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return nodes_coords_def
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
        # Get number of spatial dimensions
        n_dim = coords_array.shape[1]
        # Compute rotated coordinates array
        rot_coords_array = np.zeros_like(coords_array)
        for i in range(n_points):
            rot_coords_array[i, :] = np.matmul(r[:n_dim, :n_dim],
                                               coords_array[i, :])
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
        else:
            for k in range(n_edge_nodes_per_dim[2]):
                for j in range(n_edge_nodes_per_dim[1]):
                    for i in range(n_edge_nodes_per_dim[0]):
                        # Get element node local index
                        elem_node_index = self._get_elem_node_index(
                            elem_type, n_elems_per_dim, (i, j, k))
                        # Set mesh node
                        if nodes_matrix[elem_node_index] != 0:
                            # Set mesh node label
                            mesh_nodes_matrix[i, j, k] = node_label
                            # Set mesh node coordinates
                            mesh_nodes_coords[str(node_label)] = \
                                np.array([i*coord_step_per_dim[0],
                                          j*coord_step_per_dim[1],
                                          k*coord_step_per_dim[2]])
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
        n_edge_nodes_per_dim = \
            self._get_n_edge_nodes_per_dim(elem_type, n_elems_per_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get global mesh finite element index
        elem_index = tuple([np.max((1, int(np.ceil(
            (global_index[k]/(n_edge_nodes_per_dim[k] - 1))
            *n_elems_per_dim[k])))) for k in range(self._n_dim)])
        # Get finite element node local index
        local_index = tuple([global_index[k] - (
            (elem_index[k] - 1)*n_edge_nodes_elem - (elem_index[k] - 1))
            for k in range(self._n_dim)])
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
                    label = type(self)._get_node_label_from_coords(
                        mesh_nodes_coords_ref, coord_ref)
                    # Store node displacement
                    if str(label) not in mesh_boundary_nodes_disps.keys():
                        mesh_boundary_nodes_disps[label] = disp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
        return mesh_boundary_nodes_disps
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_node_label_from_coords(mesh_nodes_coords_ref, node_coords):
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
    r : numpy.ndarray(2d)
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