"""FETorch: 4-Node Quadrilateral Finite Element.

Classes
-------
FEQuad4(Element)
    FETorch finite element: 4-Node Quadrilateral.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.element.interface import Element
from simulators.fetorch.element.quadratures import gauss_quadrature
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class FEQuad4(Element):
    """FETorch finite element: 4-Node Quadrilateral.
    
    Attributes
    ----------
    _name : str
        Name.
    _n_node : int
        Number of nodes.
    _n_dof_node : int
        Number of degrees of freedom per node.
    _node_local_coord : dict
        Nodes (key, str[int]) local coordinates (item, torch.Tensor(1d)). Nodes
        are labeled from 1 to n_node.
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
    eval_shapefun_deriv(self, local_coord)
        Evaluate shape functions derivates at given local coordinates.
    _admissible_gauss_quadratures()
        Get admissible Gauss integration quadratures.
    """
    def __init__(self, n_gauss=4):
        """Constructor.
        
        Parameters
        ----------
        n_gauss : int, default=4
            Number of Gauss integration points.
        """
        # Set name
        self._name = 'quad4'
        # Set number of nodes
        self._n_node = 4
        # Set number of degrees of freedom per node
        self._n_dof_node = 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes local coordinates
        self._set_nodes_local_coords()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of Gauss integration points
        if int(n_gauss) not in self._admissible_gauss_quadratures():
            raise RuntimeError(f'The 2D {n_gauss}-point Gauss quadrature is '
                               f'not available for element \'{self._name}\'.')
        else:
            self._n_gauss = int(n_gauss)
        # Get Gaussian quadrature points local coordinates and weights
        self._gp_coords, self._gp_weights = \
            gauss_quadrature(n_gauss, domain='quadrilateral')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_nodes_local_coords(self):
        """Set nodes local coordinates."""
        # Initialize local coordinates
        node_local_coord = torch.zeros((self._n_node, 2), dtype=torch.float)
        # Set local coordinates
        node_local_coord[0, :] = torch.tensor((-1.0, -1.0))
        node_local_coord[1, :] = torch.tensor((1.0, -1.0))
        node_local_coord[2, :] = torch.tensor((1.0, 1.0))
        node_local_coord[3, :] = torch.tensor((-1.0, 1.0))
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
        c1, c2 = local_coord
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize shape functions
        shape_functions = torch.zeros((self._n_node), dtype=torch.float)
        # Compute shape functions at given local coordinates
        shape_functions[0] = 0.25*(1.0 - c1)*(1.0 - c2)
        shape_functions[1] = 0.25*(1.0 + c1)*(1.0 - c2)
        shape_functions[2] = 0.25*(1.0 + c1)*(1.0 + c2)
        shape_functions[3] = 0.25*(1.0 - c1)*(1.0 + c2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return shape_functions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eval_shapefun_deriv(self, local_coord):
        """Evaluate shape functions derivates at given local coordinates.
        
        Parameters
        ----------
        local_coord : torch.Tensor(1d)
            Local coordinates of point where shape functions are evaluated.
            
        Returns
        -------
        shape_function_deriv : torch.Tensor(2d)
            Shape functions derivatives evaluated at given local coordinates,
            sorted according with element nodes. Derivative of the i-th shape
            function with respect to the j-th local coordinate is stored in
            shape_function_deriv[i, j].
        """
        # Unpack local coordinates
        c1, c2 = local_coord
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize shape functions
        shape_function_deriv = \
            torch.zeros((self._n_node, 2), dtype=torch.float)
        # Compute shape functions at given local coordinates
        shape_function_deriv[0, :] = \
            torch.tensor((-0.25*(1.0 - c2), -0.25*(1.0 - c1)))
        shape_function_deriv[1, :] = \
            torch.tensor((0.25*(1.0 - c2), -0.25*(1.0 + c1)))
        shape_function_deriv[2, :] = \
            torch.tensor((0.25*(1.0 + c2), 0.25*(1.0 + c1)))
        shape_function_deriv[3, :] = \
            torch.tensor((-0.25*(1.0 + c2), 0.25*(1.0 - c1)))
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
        admissible_n_gauss = (1, 4)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return admissible_n_gauss