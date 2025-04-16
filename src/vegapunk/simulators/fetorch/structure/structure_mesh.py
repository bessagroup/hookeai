"""FETorch: Structure finite element mesh.

Classes
-------
StructureMesh
    FETorch structure finite element mesh.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import torch
# Local
from simulators.fetorch.element.type.interface import ElementType
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class StructureMesh:
    """FETorch structure finite element mesh.
    
    Attributes
    ----------
    _n_dim : int
        Number of spatial dimensions.
    _n_node_mesh : int
        Number of nodes of finite element mesh.
    _n_elem : int
        Number of elements of finite element mesh.
    _elements_type : dict
        FETorch element type (item, ElementType) of each finite element mesh
        element (str[int]). Elements are labeled from 1 to n_elem.
    _connectivities : dict
        Nodes (item, tuple[int]) of each finite element mesh element
        (key, str[int]). Nodes are labeled from 1 to n_node_mesh. Elements are
        labeled from 1 to n_elem.
    _nodes_coords_mesh_init : torch.Tensor(2d)
        Initial coordinates of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    _nodes_coords_mesh : torch.Tensor(2d)
        Coordinates of finite element mesh nodes stored as torch.Tensor(2d) of
        shape (n_node_mesh, n_dim).
    _nodes_coords_mesh_old : torch.Tensor(2d)
        Last converged coordinates of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    _nodes_disps_mesh : torch.Tensor(2d)
        Displacements of finite element mesh nodes stored as torch.Tensor(2d)
        of shape (n_node_mesh, n_dim).
    _nodes_disps_mesh_old : torch.Tensor(2d)
        Last converged displacements of finite element mesh nodes stored as
        torch.Tensor(2d) of shape (n_node_mesh, n_dim).
    _dirichlet_bool_mesh : torch.Tensor(2d)
        Degrees of freedom of finite element mesh subject to Dirichlet boundary
        conditions. Stored as torch.Tensor(2d) of shape (n_node_mesh, n_dim)
        where constrained degrees of freedom are labeled 1, otherwise 0.

    Methods
    -------
    get_n_dim(self)
        Get number of spatial dimensions.
    get_n_node_mesh(self)
        Get number of nodes of finite element mesh.
    get_n_elem(self)
        Number of elements of finite element mesh.
    get_elements_type(self)
        Get type of elements of finite element mesh.
    get_n_element_type(self)
        Get the number of elements types of finite element mesh.
    get_connectivities(self)
        Get finite element mesh elements connectivities.
    get_connectivities_tensor(self)
        Get finite element mesh elements connectivities stored (tensor).
    get_dirichlet_bool_mesh(self)
        Get degrees of freedom subject to Dirichlet boundary conditions.
    get_n_dirichlet_dof(self)
        Get number of degrees of freedom subject to Dirichlet conditions.
    get_element_configuration(self, element_id, time='current')
        Get element nodes coordinates and displacements.
    get_mesh_configuration(self, time='current')
        Get finite element mesh configuration.
    update_mesh_configuration(self, nodes_disps_mesh, \
                              nodes_disps_mesh_old=None, is_update_coords=True)
    element_assembler(self, elements_array)
        Assemble element level arrays into mesh level counterparts.
    _element_assembler_1d(self, elements_array_1d)
        Assemble element level 1D array into mesh level counterpart.
    element_extractor(self, mesh_array, element_id)
        Extract element level array from mesh level counterpart.
    _element_extractor_1d(self, mesh_array, element_id)
        Extract element level 1D array from mesh level counterpart.
    build_elements_mesh_indexing(self, n_dof_node)
        Build elements nodes degrees of freedom mesh indexes.
    build_element_mesh_indexing(cls, element_nodes, n_dof_node)
        Get element nodes degrees of freedom mesh indexes.
    _check_mesh_initialization(self, nodes_coords_mesh_init, elements_type,
                               connectivities)
        Check finite element mesh initialization.
    """
    def __init__(self, nodes_coords_mesh_init, elements_type, connectivities,
                 dirichlet_bool_mesh):
        """Constructor.
        
        Parameters
        ----------
        nodes_coords_mesh_init : torch.Tensor(2d)
            Initial coordinates of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        elements_type : dict
            FETorch element type (item, ElementType) of each finite element
            mesh element (str[int]). Elements labels must be within the range
            of 1 to n_elem (included).
        connectivities : dict
            Nodes (item, tuple[int]) of each finite element mesh element
            (key, str[int]). Nodes are labeled from 1 to n_node_mesh. Elements
            are labeled from 1 to n_elem.
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
        """
        # Check finite element mesh initialization
        self._check_mesh_initialization(nodes_coords_mesh_init, elements_type,
                                        connectivities)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes initial coordinates
        self._nodes_coords_mesh_init = nodes_coords_mesh_init
        # Set elements types
        self._elements_type = elements_type
        # Set elements connectivities
        self._connectivities = connectivities
        # Set Dirichlet boundary conditions (boolean)
        self._dirichlet_bool_mesh = dirichlet_bool_mesh
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of nodes
        self._n_node_mesh = nodes_coords_mesh_init.shape[0]
        # Set number of spatial dimensions
        self._n_dim = nodes_coords_mesh_init.shape[1]
        # Set number of elements
        self._n_elem = len(elements_type.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize coordinates
        self._nodes_coords_mesh = self._nodes_coords_mesh_init.clone()
        self._nodes_coords_mesh_old = self._nodes_coords_mesh_init.clone()
        # Initialize displacements
        self._nodes_disps_mesh = torch.zeros_like(self._nodes_coords_mesh)
        self._nodes_disps_mesh_old = self._nodes_disps_mesh.clone()
    # -------------------------------------------------------------------------
    def get_n_dim(self):
        """Get number of spatial dimensions.
        
        Returns
        -------
        n_dim : int
            Number of spatial dimensions.
        """
        return self._n_dim
    # -------------------------------------------------------------------------
    def get_n_node_mesh(self):
        """Get number of nodes of finite element mesh.
        
        Returns
        -------
        n_node_mesh : int
            Number of nodes of finite element mesh.
        """
        return self._n_node_mesh
    # -------------------------------------------------------------------------
    def get_n_elem(self):
        """Number of elements of finite element mesh.
        
        Returns
        -------
        n_elem : int
            Number of elements of finite element mesh.
        """
        return self._n_elem
    # -------------------------------------------------------------------------
    def get_elements_type(self):
        """Get type of elements of finite element mesh.
        
        Returns
        -------
        elements_type : dict
            FETorch element type (item, ElementType) of each finite element
            mesh element (str[int]). Elements are labeled from 1 to n_elem.
        """
        return copy.deepcopy(self._elements_type)
    # -------------------------------------------------------------------------
    def get_n_element_type(self):
        """Get the number of elements types of finite element mesh.
        
        Returns
        -------
        n_element_type : int
            Number of element types of finite element mesh.
        """
        return len({type(x) for x in self._elements_type.values()})
    # -------------------------------------------------------------------------
    def get_connectivities(self):
        """Get finite element mesh elements connectivities.

        Returns
        -------
        connectivities : dict
            Nodes (item, tuple[int]) of each finite element mesh element
            (key, str[int]). Nodes are labeled from 1 to n_node_mesh. Elements
            are labeled from 1 to n_elem.
        """
        return copy.deepcopy(self._connectivities)
    # -------------------------------------------------------------------------
    def get_connectivities_tensor(self):
        """Get finite element mesh elements connectivities (tensor).
        
        Storage of finite element mesh elements connectivities in tensor,
        i.e., batching over the element dimension, requires that all finite
        element mesh elements have the same number of nodes.
        
        Returns
        -------
        connectivities_tensor : torch.Tensor(2d)
            Finite element mesh elements connectitivities stored as
            torch.Tensor(2d) of shape (n_elem, n_node). Elements are sorted
            according with their labels (1 to n_elem). Nodes are labeled from
            1 to n_node_mesh.
        """
        # Build connectivities tensor
        try:
            connectivities_tensor = torch.stack(
                [torch.tensor(self._connectivities[str(x)], dtype=torch.int)
                 for x in sorted(self._connectivities.keys(), key=int)], dim=0)
        except RuntimeError as error:
            print(f'Error: {error}')
            print('Stacking operation over elements requires that all finite '
                  'element mesh elements share the same element type '
                  '(number of nodes).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return connectivities_tensor
    # -------------------------------------------------------------------------
    def get_dirichlet_bool_mesh(self):
        """Get degrees of freedom subject to Dirichlet boundary conditions.
        
        Returns
        -------
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
        """
        return self._dirichlet_bool_mesh.clone()
    # -------------------------------------------------------------------------
    def get_n_dirichlet_dof(self):
        """Get number of degrees of freedom subject to Dirichlet conditions.
        
        Returns
        -------
        n_dof_dirichlet : int
            Number of degrees of freedom subject to Dirichlet boundary
            conditions.
        """
        return torch.sum(self._dirichlet_bool_mesh == 1)
    # -------------------------------------------------------------------------
    def get_element_configuration(self, element_id, time='current'):
        """Get element nodes coordinates and displacements.
        
        Parameters
        ----------
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).
        time : {'init', 'last', 'current'}, default='current'
            Time where element configuration is returned: initial configuration
            ('init'), last converged configuration ('last'), or current
            configuration ('current').
        
        Returns
        -------
        nodes_coords : torch.Tensor(2d)
            Nodes coordinates stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        nodes_disps : torch.Tensor(2d)
            Nodes displacements stored as torch.Tensor(2d) of shape
            (n_node, n_dof_node).
        """
        # Get element nodes indexes
        elem_nodes_idxs = \
            [node - 1 for node in self._connectivities[str(element_id)]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element nodes coordinates and displacements
        if time == 'init':
            nodes_coords = self._nodes_coords_mesh_init[elem_nodes_idxs, :]
            nodes_disps = torch.zeros_like(nodes_coords)
        elif time == 'last':
            nodes_coords = self._nodes_coords_mesh_old[elem_nodes_idxs, :]
            nodes_disps = self._nodes_disps_mesh_old[elem_nodes_idxs, :]
        elif time == 'current':
            nodes_coords = self._nodes_coords_mesh[elem_nodes_idxs, :]
            nodes_disps = self._nodes_disps_mesh[elem_nodes_idxs, :]
        else:
            raise RuntimeError('Unknown time option.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return nodes_coords, nodes_disps
    # -------------------------------------------------------------------------
    def get_mesh_configuration(self, time='current'):
        """Get finite element mesh configuration.

        Parameters
        ----------
        time : {'init', 'last', 'current'}, default='current'
            Time where element configuration is returned: initial configuration
            ('init'), last converged configuration ('last'), or current
            configuration ('current').

        Returns
        -------
        nodes_coords_mesh : torch.Tensor(2d)
            Coordinates of finite element mesh nodes stored as torch.Tensor(2d)
            of shape (n_node_mesh, n_dim).
        nodes_disps_mesh : torch.Tensor(2d)
            Displacements of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        """
        # Get nodes coordinates and displacements
        if time == 'init':
            nodes_coords_mesh = self._nodes_coords_mesh_init.clone()
            nodes_disps_mesh = torch.zeros_like(nodes_coords_mesh)
        elif time == 'last':
            nodes_coords_mesh = self._nodes_coords_mesh_old.clone()
            nodes_disps_mesh = self._nodes_disps_mesh_old.clone()
        elif time == 'current':
            nodes_coords_mesh = self._nodes_coords_mesh.clone()
            nodes_disps_mesh = self._nodes_disps_mesh.clone()
        else:
            raise RuntimeError('Unknown time option.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return nodes_coords_mesh, nodes_disps_mesh
    # -------------------------------------------------------------------------
    def update_mesh_configuration(self, nodes_disps_mesh, time='current',
                                  is_update_coords=True):
        """Update finite element mesh configuration from nodes displacements.

        Parameters
        ----------
        nodes_disps_mesh : torch.Tensor(2d)
            Displacements of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        time : {'last', 'current'}, default='current'
            Time where update of element state variables is performed: last
            converged state variables ('last') or current state variables
            ('current').
        is_update_coords : bool, default=True
            If False, then only updates the displacements of the finite element
            mesh nodes, leaving the nodes coordinates unchanged. If True, then
            update both coordinates and displacements of finite element mesh
            nodes.
        """            
        # Update nodes coordinates and displacements
        if time == 'last':
            # Update last converged nodes displacements
            self._nodes_disps_mesh_old = nodes_disps_mesh.clone()
            # Update last converged nodes coordinates
            if is_update_coords:
                self._nodes_coords_mesh_old = \
                    self._nodes_coords_mesh_init + self._nodes_disps_mesh_old
        elif time == 'current':
            # Update current nodes displacements
            self._nodes_disps_mesh = nodes_disps_mesh.clone()
            # Update current nodes coordinates
            if is_update_coords:
                self._nodes_coords_mesh = \
                    self._nodes_coords_mesh_init + self._nodes_disps_mesh
        else:
            raise RuntimeError('Unknown time option.')
    # -------------------------------------------------------------------------
    def element_assembler(self, elements_array):
        """Assemble element level arrays into mesh level counterparts.

        Assumes similar number of dimensions per node in the whole finite
        element mesh.
        
        Parameters
        ----------
        elements_array : dict
            Generic array (item, torch.Tensor) associated with given finite
            element mesh element (key, str[int]). Elements labels must be
            within the range of 1 to n_elem (included).
            
        Returns
        -------
        mesh_array : torch.Tensor
            Mesh level array resulting from the assembly of the corresponding
            element level arrays. Mesh level array is always sorted and
            dimensioned accounting for the total number of nodes in the finite
            element mesh.
        """
        # Probe element array shape
        elem_id = tuple(elements_array.keys())[0]
        elem_array_shape = elements_array[elem_id].shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble element level arrays
        if len(elem_array_shape) == 1:
            mesh_array = self._element_assembler_1d(elements_array)
        else:
            raise RuntimeError(f'The element assembly procedure is not '
                               f'implemented for {len(elem_array_shape)}-'
                               f'dimensional array.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mesh_array
    # -------------------------------------------------------------------------
    def _element_assembler_1d(self, elements_array_1d):
        """Assemble element level 1D array into mesh level counterpart.
        
        Assumes similar number of dimensions per node in the whole finite
        element mesh.
        
        Parameters
        ----------
        elements_arrays_1d : dict
            Generic array (item, torch.Tensor(1d)) associated with given finite
            element mesh element (key, str[int]). Elements labels must be
            within the range of 1 to n_elem (included).
        
        Returns
        -------
        mesh_array : torch.Tensor(1d)
            Mesh level array resulting from the assembly of the corresponding
            element level arrays. Mesh level array is always sorted and
            dimensioned accounting for the total number of nodes in the finite
            element mesh.
        """
        # Infer number of dimensions per node
        elem_key = tuple(elements_array_1d.keys())[0]
        n_dim_per_node = int(elements_array_1d[elem_key].shape[0]
                             /len(self._connectivities[elem_key]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize mesh level array
        mesh_array = torch.zeros(self._n_node_mesh*n_dim_per_node,
                                 device=elements_array_1d[elem_key].device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over element arrays
        for elem_key, element_array in elements_array_1d.items():
            # Get element nodes
            elem_nodes = self._connectivities[elem_key]
            # Loop over element nodes
            for i, node in enumerate(elem_nodes):
                # Set node initial assembly index
                mesh_index = (node - 1)*n_dim_per_node
                # Set node initial element index
                elem_index = i*n_dim_per_node
                # Assemble element node contribution
                mesh_array[mesh_index:mesh_index + n_dim_per_node] += \
                    element_array[elem_index:elem_index + n_dim_per_node]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mesh_array
    # -------------------------------------------------------------------------
    def element_extractor(self, mesh_array, element_id):
        """Extract element level array from mesh level counterpart.

        Assumes similar number of dimensions per node in the whole finite
        element mesh.
        
        Parameters
        ----------
        mesh_array : torch.Tensor
            Mesh level array resulting from the assembly of the corresponding
            element level arrays. Mesh level array is always sorted and
            dimensioned accounting for the total number of nodes in the finite
            element mesh.
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).
            
        Returns
        -------
        element_array : torch.Tensor
            Generic element array.
        """
        # Extract element level array
        if len(mesh_array.shape) == 1:
            element_array = self._element_extractor_1d(mesh_array, element_id)
        else:
            raise RuntimeError(f'The element extraction procedure is not '
                               f'implemented for {len(mesh_array.shape)}-'
                               f'dimensional array.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_array
    # -------------------------------------------------------------------------
    def _element_extractor_1d(self, mesh_array, element_id):
        """Extract element level 1D array from mesh level counterpart.

        Assumes similar number of dimensions per node in the whole finite
        element mesh.
        
        Parameters
        ----------
        mesh_array : torch.Tensor(1d)
            Mesh level array resulting from the assembly of the corresponding
            element level arrays. Mesh level array is always sorted and
            dimensioned accounting for the total number of nodes in the finite
            element mesh.
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).
            
        Returns
        -------
        element_array : torch.Tensor(1d)
            Generic element array.
        """
        # Infer number of dimensions per node
        n_dim_per_node = int(mesh_array.shape[0]/self._n_node_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element nodes
        elem_nodes = self._connectivities[str(element_id)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element level array
        element_array = torch.zeros((len(elem_nodes)*n_dim_per_node),
                                    device=mesh_array.device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over element nodes
        for i, node in enumerate(elem_nodes):
            # Set node initial element index
            elem_index = i*n_dim_per_node
            # Set node initial assembly index
            mesh_index = (node - 1)*n_dim_per_node
            # Extract element node data
            element_array[elem_index:elem_index + n_dim_per_node] = \
                mesh_array[mesh_index:mesh_index + n_dim_per_node]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_array
    # -------------------------------------------------------------------------
    def build_elements_mesh_indexing(self, n_dof_node):
        """Build elements nodes degrees of freedom mesh indexes.
        
        Parameters
        ----------
        n_dof_node : int
            Number of degrees of freedom per element node.
        
        Returns
        -------
        elements_mesh_indexes : torch.Tensor(2d)
            Elements nodes degrees of freedom mesh indexes stored as
            torch.Tensor(2d) of shape (n_elem, n_node*n_dof_node).
        """
        # Build connectivities tensor
        connectivities_tensor = self.get_connectivities_tensor()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized element mesh indexing (batch along element)
        vmap_element_mesh_indexing = torch.vmap(
            self.build_element_mesh_indexing, in_dims=(0, None), out_dims=(0,))
        # Compute elements mesh indexing
        elements_mesh_indexes = \
            vmap_element_mesh_indexing(connectivities_tensor, n_dof_node)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elements_mesh_indexes
    # -------------------------------------------------------------------------
    @classmethod
    def build_element_mesh_indexing(cls, element_nodes, n_dof_node):
        """Get element nodes degrees of freedom mesh indexes.
        
        Parameters
        ----------
        element_nodes : torch.Tensor(1d)
            Finite element mesh element connectitivities stored as
            torch.Tensor(1d) of shape (n_node,). Nodes are labeled from
            1 to n_node_mesh.
        n_dof_node : int
            Number of degrees of freedom per element node.
        
        Returns
        -------
        element_mesh_indexes : torch.Tensor(1d)
            Element nodes degrees of freedom mesh indexes stored as
            torch.Tensor(1d) of shape (n_node*n_dof_node,).
        """
        # Build element nodes degrees of freedom mesh indexes
        element_mesh_indexes = \
            ((element_nodes - 1)*n_dof_node).repeat_interleave(n_dof_node) \
            + torch.arange(n_dof_node).repeat(len(element_nodes))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_mesh_indexes
    # -------------------------------------------------------------------------
    def _check_mesh_initialization(self, nodes_coords_mesh_init, elements_type,
                                   connectivities):
        """Check finite element mesh initialization.
        
        Parameters
        ----------
        nodes_coords_mesh_init : torch.Tensor(2d)
            Initial coordinates of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        elements_type : dict
            FETorch element type (item, ElementType) of each finite element
            mesh element (str[int]). Elements are labeled from 1 to n_elem.
        connectivities : dict
            Nodes (item, tuple[int]) of each finite element mesh element
            (key, str[int]). Elements are labeled from 1 to n_elem.
        """
        # Check initial coordinates
        if not isinstance(nodes_coords_mesh_init, torch.Tensor):
            raise RuntimeError('Initial coordinates of finite element mesh '
                               'nodes must be stored in torch.Tensor(2d).')
        elif len(nodes_coords_mesh_init.shape) != 2:
            raise RuntimeError('Initial coordinates of finite element mesh '
                               'nodes must be stored in torch.Tensor(2d) '
                               'of shape (n_node_mesh, n_dim).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set expected elements labels
        elements_labels = \
            tuple([str(i) for i in range(1, len(elements_type.keys()) + 1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Check elements labels
        if set(elements_type.keys()) != set(elements_labels):
            raise RuntimeError('All elements type were not specified and/or'
                               'elements are not labeled from 1 to n_elem.')
        # Check element types
        for element_type in elements_type.values():
            if not isinstance(element_type, ElementType):
                raise RuntimeError('All finite element mesh elements must be '
                                   'of instances of class ElementType.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check elements
        if set(connectivities.keys()) != set(elements_labels):
            raise RuntimeError('All elements connectivities were not '
                               'specified and/or elements are not labeled '
                               'from 1 to n_elem.')
        # Check elements connectivities
        for elem_key, element_nodes in connectivities.items():
            # Get element type
            element_type = elements_type[elem_key]
            # Get element type number of nodes
            n_node = element_type.get_n_node()
            # Check element connectivities
            if len(element_nodes) != n_node:
                raise RuntimeError(f'Element {elem_key} connectivities '
                                   f'{len(element_nodes)} do '
                                   f'not match the corresponding element '
                                   f'type number of nodes ({n_node}).')