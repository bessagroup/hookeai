"""FETorch: Structure finite element mesh.

Classes
-------
StructureMesh
    FETorch structure finite element mesh.
"""
#
#                                                                       Modules
# =============================================================================
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
        (key, str[int]). Elements are labeled from 1 to n_elem.
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
    _internal_forces_mesh : torch.Tensor(2d)
        Internal forces of finite element mesh nodes stored as torch.Tensor(2d)
        of shape (n_node_mesh, n_dim).
    _external_forces_mesh : torch.Tensor(2d)
        External forces of finite element mesh nodes stored as torch.Tensor(2d)
        of shape (n_node_mesh, n_dim).
    _reaction_forces_mesh : torch.Tensor(2d)
        Reaction forces (Dirichlet boundary conditions) of finite element mesh
        nodes stored as torch.Tensor(2d) of shape (n_node_mesh, n_dim).

    Methods
    -------
    element_assembler(self, elements_array)
        Assemble element level arrays into mesh level counterparts.
    _element_assembler_1d(self, elements_array_1d)
        Assemble element level 1D array into mesh level counterpart.
    element_extractor(self, mesh_array, element_id)
        Extract element level array from mesh level counterpart.
    _element_extractor_1d(self, mesh_array, element_id)
        Extract element level 1D array from mesh level counterpart.
    _check_mesh_initialization(self, nodes_coords_mesh_init, elements_type,
                               connectivities)
        Check finite element mesh initialization.
    """
    def __init__(self, nodes_coords_mesh_init, elements_type, connectivities):
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
            (key, str[int]). Elements labels must be within the range of
            1 to n_elem (included).
        """
        # Check finite element mesh initialization
        self._check_mesh_initialization(nodes_coords_mesh_init, elements_type,
                                        connectivities)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elements types
        self._elements_type = elements_type
        # Set elements connectivities
        self._connectivities = connectivities
        # Set nodes initial coordinates
        self._nodes_coords_mesh_init = nodes_coords_mesh_init
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
        # Initialize forces
        self._internal_forces_mesh = torch.zeros_like(self._nodes_coords_mesh)
        self._external_forces_mesh = torch.zeros_like(self._nodes_coords_mesh)
        self._reaction_forces_mesh = torch.zeros_like(self._nodes_coords_mesh)
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
        mesh_array = torch.zeros(self._n_node_mesh*n_dim_per_node)
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
        element_array = torch.zeros((len(elem_nodes)*n_dim_per_node))
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
                                   'of type ElementType.')
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

    