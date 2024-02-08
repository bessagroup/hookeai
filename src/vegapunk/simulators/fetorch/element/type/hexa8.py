"""FETorch: 8-Node Hexahedral Finite Element.

Classes
-------
FEHexa8(Element)
    FETorch finite element: 8-Node Hexahedral.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.type.interface import Element
from simulators.fetorch.element.type.quadratures import gauss_quadrature
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class FEHexa8(Element):
    """FETorch finite element: 8-Node Hexahedral.
    
    Attributes
    ----------
    _name : str
        Name.
    _n_node : int
        Number of nodes.
    _n_dof_node : int
        Number of degrees of freedom per node.
    _node_local_coord : torch.Tensor(2d)
        Nodes local coordinates stored as torch.Tensor(2d) of shape
        (n_node, n_dof_node).
    _n_gauss : int
        Number of Gauss integration points.
    _gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, tuple). Gauss integration points are labeled from 1 to n_gauss.
    _gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
        
    Methods
    -------
    _set_node_local_coords(self)
        Set nodes local coordinates.
    eval_shapefun(self, local_coord)
        Evaluate shape functions at given local coordinates.
    eval_shapefun_local_deriv(self, local_coord)
        Evaluate shape functions local derivates at given local coordinates.
    _admissible_gauss_quadratures()
        Get admissible Gauss integration quadratures.
    """
    def __init__(self, n_gauss=8):
        """Constructor.
        
        Parameters
        ----------
        n_gauss : int, default=8
            Number of Gauss integration points.
        """
        # Set name
        self._name = 'hexa8'
        # Set number of nodes
        self._n_node = 8
        # Set number of degrees of freedom per node
        self._n_dof_node = 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes local coordinates
        self._set_nodes_local_coords()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of Gauss integration points
        if int(n_gauss) not in self._admissible_gauss_quadratures():
            raise RuntimeError(f'The 3D {n_gauss}-point Gauss quadrature is '
                               f'not available for element \'{self._name}\'.')
        else:
            self._n_gauss = int(n_gauss)
        # Get Gaussian quadrature points local coordinates and weights
        self._gp_coords, self._gp_weights = \
            gauss_quadrature(n_gauss, domain='hexahedral')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_nodes_local_coords(self):
        """Set nodes local coordinates."""
        # Initialize local coordinates
        node_local_coord = torch.zeros((self._n_node, 3), dtype=torch.float)
        # Set local coordinates
        node_local_coord[0, :] = torch.tensor((-1.0, -1.0, -1.0))
        node_local_coord[1, :] = torch.tensor((1.0, -1.0, -1.0))
        node_local_coord[2, :] = torch.tensor((1.0, 1.0, -1.0))
        node_local_coord[3, :] = torch.tensor((-1.0, 1.0, -1.0))
        node_local_coord[4, :] = torch.tensor((-1.0, -1.0, 1.0))
        node_local_coord[5, :] = torch.tensor((1.0, -1.0, 1.0))
        node_local_coord[6, :] = torch.tensor((1.0, 1.0, 1.0))
        node_local_coord[7, :] = torch.tensor((-1.0, 1.0, 1.0))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store nodes local coordinates
        self._node_local_coord = node_local_coord
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eval_shapefun(self, local_coord):
        """Evaluate shape functions at given local coordinates.
        
        Parameters
        ----------
        local_coord : torch.Tensor(1d)
            Local coordinates of point where shape functions are evaluated.
            
        Returns
        -------
        shape_functions : torch.Tensor(1d)
            Shape functions evaluated at given local coordinates, sorted
            according with element nodes.
        """
        # Unpack local coordinates
        c1, c2, c3 = local_coord
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize shape functions
        shape_functions = torch.zeros((self._n_node), dtype=torch.float)
        # Compute shape functions at given local coordinates
        shape_functions[0] = (1.0/8.0)*(1.0 - c1)*(1.0 - c2)*(1.0 - c3)
        shape_functions[1] = (1.0/8.0)*(1.0 + c1)*(1.0 - c2)*(1.0 - c3)
        shape_functions[2] = (1.0/8.0)*(1.0 + c1)*(1.0 + c2)*(1.0 - c3)
        shape_functions[3] = (1.0/8.0)*(1.0 - c1)*(1.0 + c2)*(1.0 - c3)
        shape_functions[4] = (1.0/8.0)*(1.0 - c1)*(1.0 - c2)*(1.0 + c3)
        shape_functions[5] = (1.0/8.0)*(1.0 + c1)*(1.0 - c2)*(1.0 + c3)
        shape_functions[6] = (1.0/8.0)*(1.0 + c1)*(1.0 + c2)*(1.0 + c3)
        shape_functions[7] = (1.0/8.0)*(1.0 - c1)*(1.0 + c2)*(1.0 + c3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return shape_functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eval_shapefun_local_deriv(self, local_coord):
        """Evaluate shape functions local derivates at given local coordinates.
        
        Parameters
        ----------
        local_coord : torch.Tensor(1d)
            Local coordinates of point where shape functions local derivatives
            are evaluated.
            
        Returns
        -------
        shape_function_deriv : torch.Tensor(2d)
            Shape functions local derivatives evaluated at given local
            coordinates, sorted according with element nodes. Derivative of the
            i-th shape function with respect to the j-th local coordinate is
            stored in shape_function_deriv[i, j].
        """
        # Unpack local coordinates
        c1, c2, c3 = local_coord
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize shape functions
        shape_function_deriv = \
            torch.zeros((self._n_node, 3), dtype=torch.float)
        # Compute shape functions at given local coordinates
        shape_function_deriv[0, :] = torch.tensor(
            (-(1.0/8.0)*(1.0 - c2)*(1.0 - c3),
             -(1.0/8.0)*(1.0 - c1)*(1.0 - c3),
             -(1.0/8.0)*(1.0 - c1)*(1.0 - c2)))
        shape_function_deriv[1, :] = torch.tensor(
            ((1.0/8.0)*(1.0 - c2)*(1.0 - c3),
             -(1.0/8.0)*(1.0 + c1)*(1.0 - c3),
             -(1.0/8.0)*(1.0 + c1)*(1.0 - c2)))
        shape_function_deriv[2, :] = torch.tensor(
            ((1.0/8.0)*(1.0 + c2)*(1.0 - c3),
             (1.0/8.0)*(1.0 + c1)*(1.0 - c3),
             -(1.0/8.0)*(1.0 + c1)*(1.0 + c2)))
        shape_function_deriv[3, :] = torch.tensor(
            (-(1.0/8.0)*(1.0 + c2)*(1.0 - c3),
             (1.0/8.0)*(1.0 - c1)*(1.0 - c3),
             -(1.0/8.0)*(1.0 - c1)*(1.0 + c2)))
        shape_function_deriv[4, :] = torch.tensor(
            (-(1.0/8.0)*(1.0 - c2)*(1.0 + c3),
             -(1.0/8.0)*(1.0 - c1)*(1.0 + c3),
             (1.0/8.0)*(1.0 - c1)*(1.0 - c2)))
        shape_function_deriv[5, :] = torch.tensor(
            ((1.0/8.0)*(1.0 - c2)*(1.0 + c3),
             -(1.0/8.0)*(1.0 + c1)*(1.0 + c3),
             (1.0/8.0)*(1.0 + c1)*(1.0 - c2)))
        shape_function_deriv[6, :] = torch.tensor(
            ((1.0/8.0)*(1.0 + c2)*(1.0 + c3),
             (1.0/8.0)*(1.0 + c1)*(1.0 + c3),
             (1.0/8.0)*(1.0 + c1)*(1.0 + c2)))
        shape_function_deriv[7, :] = torch.tensor(
            (-(1.0/8.0)*(1.0 + c2)*(1.0 + c3),
             (1.0/8.0)*(1.0 - c1)*(1.0 + c3),
             (1.0/8.0)*(1.0 - c1)*(1.0 + c2)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return shape_function_deriv
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _admissible_gauss_quadratures():
        """Get admissible Gauss integration quadratures.
        
        Returns
        -------
        admissible_n_gauss : tuple[int]
            Admissible Gauss integration quadratures (number of Gauss
            integration points).
        """
        # Set admissible Gauss integration quadratures
        admissible_n_gauss = (1, 8)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return admissible_n_gauss