"""Finite element.

Classes
-------
class FiniteElement
    Quadrilateral/Hexahedral finite element.
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
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================
class FiniteElement:
    """Quadrilateral/Hexahedral finite element.
    
    Attributes
    ----------
    _elem_type : str
        Finite element type.
    _n_dim : int
        Number of spatial dimensions.
    _n_nodes : int
        Number of nodes.
    _n_edges : int
        Number of edges.
    _n_edge_nodes : int
        Number of nodes per edge.
    _nodes_matrix : numpy.ndarray(2d or 3d)
        Element nodes matrix (numpy.ndarray[int](n_dim*(n_edge_nodes,))) where
        each element corresponds to a given node position and whose value is
        set either as the node label (according to adopted numbering
        convention) or zero (if the node does not exist). Nodes are labeled
        from 1 to n_nodes.

    Methods
    -------
    _available_elem_type()
        Get available finite element types.
    _is_available_elem_type(elem_type)
        Check if finite element type is available.
    _set_elem_type_attributes(self)
        Set finite element type attributes.
    get_n_dim(self)
        Get number of spatial dimensions.   
    get_n_edge_nodes(self)
        Get number of nodes per edge.
    get_nodes_matrix(self)
        Get element nodes matrix.  
    get_node_label_index(self, label)
        Get node label index on element nodal matrix.
    """
    def __init__(self, elem_type):
        """Constructor.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
        """
        # Set finite element type
        if not self._is_available_elem_type(elem_type):
            raise RuntimeError(f'Finite element type ({elem_type}) is not '
                               f'available.')
        else:
            self._elem_type = elem_type
        # Set finite element type attributes
        self._set_elem_type_attributes()
    # -------------------------------------------------------------------------
    @staticmethod
    def _available_elem_type():
        """Get available finite element types.
        
        Returns
        -------
        available : tuple[str]
            Available finite element types.
        """
        # Set available finite element types
        available = ('SQUAD4', 'SQUAD8', 'SQUAD12', 'LQUAD4', 'LQUAD9',
                     'LQUAD16', 'SHEXA8')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_available_elem_type(elem_type):
        """Check if finite element type is available.
        
        Parameters
        ----------
        elem_type : str
            Finite element type.
            
        Returns
        -------
        is_available : bool
            True if the element type is available, False otherwise.
        """
        # Check if finite element type is available
        is_available = elem_type in FiniteElement._available_elem_type()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_available
    # -------------------------------------------------------------------------
    def _set_elem_type_attributes(self):
        """Set finite element type attributes."""
        # Set finite element type attributes for each element type
        if self._elem_type == 'SQUAD4':
            n_dim = 2
            n_nodes = 4
            n_edges = 4
            n_edge_nodes = 2
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[1, 1] = 3
            nodes_matrix[0, 1] = 4
        elif self._elem_type == 'SQUAD8':
            n_dim = 2
            n_nodes = 8
            n_edges = 4
            n_edge_nodes = 3
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[2, 0] = 3
            nodes_matrix[2, 1] = 4
            nodes_matrix[2, 2] = 5
            nodes_matrix[1, 2] = 6
            nodes_matrix[0, 2] = 7
            nodes_matrix[0, 1] = 8
        elif self._elem_type == 'SQUAD12':
            n_dim = 2
            n_nodes = 12
            n_edges = 4
            n_edge_nodes = 4
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[2, 0] = 3
            nodes_matrix[3, 0] = 4
            nodes_matrix[3, 1] = 5
            nodes_matrix[3, 2] = 6
            nodes_matrix[3, 3] = 7
            nodes_matrix[2, 3] = 8
            nodes_matrix[1, 3] = 9
            nodes_matrix[0, 3] = 10
            nodes_matrix[0, 2] = 11
            nodes_matrix[0, 1] = 12
        elif self._elem_type == 'LQUAD4':
            n_dim = 2
            n_nodes = 4
            n_edges = 4
            n_edge_nodes = 2
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[0, 1] = 3
            nodes_matrix[1, 1] = 4
        elif self._elem_type == 'LQUAD9':
            n_dim = 2
            n_nodes = 9
            n_edges = 4
            n_edge_nodes = 3
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[2, 0] = 3
            nodes_matrix[0, 1] = 4
            nodes_matrix[1, 1] = 5
            nodes_matrix[2, 1] = 6
            nodes_matrix[0, 2] = 7
            nodes_matrix[1, 2] = 8
            nodes_matrix[2, 2] = 9
        elif self._elem_type == 'LQUAD16':
            n_dim = 2
            n_nodes = 16
            n_edges = 4
            n_edge_nodes = 4
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0] = 1
            nodes_matrix[1, 0] = 2
            nodes_matrix[2, 0] = 3
            nodes_matrix[3, 0] = 4
            nodes_matrix[0, 1] = 5
            nodes_matrix[1, 1] = 6
            nodes_matrix[2, 1] = 7
            nodes_matrix[3, 1] = 8
            nodes_matrix[0, 2] = 9
            nodes_matrix[1, 2] = 10
            nodes_matrix[2, 2] = 11
            nodes_matrix[3, 2] = 12
            nodes_matrix[0, 3] = 13
            nodes_matrix[1, 3] = 14
            nodes_matrix[2, 3] = 15
            nodes_matrix[3, 3] = 16
        elif self._elem_type == 'SHEXA8':
            n_dim = 3
            n_nodes = 8
            n_edges = 12
            n_edge_nodes = 2
            nodes_matrix = np.zeros(n_dim*(n_edge_nodes,), dtype=int)
            nodes_matrix[0, 0, 0] = 1
            nodes_matrix[1, 0, 0] = 2
            nodes_matrix[1, 1, 0] = 3
            nodes_matrix[0, 1, 0] = 4
            nodes_matrix[0, 0, 1] = 5
            nodes_matrix[1, 0, 1] = 6
            nodes_matrix[1, 1, 1] = 7
            nodes_matrix[0, 1, 1] = 8
        else:
            raise RuntimeError('Unknown finite element type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set finite element type attributes
        self._n_dim = n_dim
        self._n_nodes = n_nodes
        self._n_edges = n_edges 
        self._n_edge_nodes = n_edge_nodes
        self._nodes_matrix = nodes_matrix
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
    def get_n_nodes(self):
        """Get number of nodes.
        
        Returns
        -------
        n_nodes : int
            Number of nodes.
        """
        return self._n_nodes
    # -------------------------------------------------------------------------
    def get_n_edge_nodes(self):
        """Get number of nodes per edge.
        
        Returns
        -------
        n_edge_nodes : int
            Number of nodes per edge.
        """
        return self._n_edge_nodes
    # -------------------------------------------------------------------------
    def get_nodes_matrix(self):
        """Get element nodes matrix.
        
        Returns
        -------
        nodes_matrix : numpy.ndarray(2d or 3d)
            Element nodes matrix (numpy.ndarray[int](n_dim*(n_edge_nodes,)))
            where each element corresponds to a given node position and whose
            value is set either as the node label (according to adopted
            numbering convention) or zero (if the node does not exist). Nodes
            are labeled from 1 to n_nodes.
        """
        return copy.deepcopy(self._nodes_matrix)
    # -------------------------------------------------------------------------
    def get_node_label_index(self, label):
        """Get node label index on element nodal matrix.
        
        Parameters
        ----------
        label : int
            Node label.
            
        Returns
        -------
        index : tuple[int]
            Index of node label on element nodal matrix.
        """
        # Search for node label
        index_arrays = np.where(self._nodes_matrix==label)
        # Check if node label was found
        if np.any([len(index_arrays[i]) != 1 for i in range(self._n_dim)]):
            raise RuntimeError('Node label not found.')
        # Get nodel label index
        index = tuple([int(index_arrays[i][0]) for i in range(self._n_dim)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return index