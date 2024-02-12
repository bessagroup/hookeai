"""FETorch: Finite Element Interface.

Classes
-------
Element(ABC)
    FETorch finite element interface.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class ElementType(ABC):
    """FETorch finite element interface.
    
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
        Number of Gauss quadrature integration points.
    _gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    _gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
        
    Methods
    -------
    _set_nodes_local_coords(self)
        *abstract*: Set nodes local coordinates.
    eval_shapefun(self, local_coords)
        *abstract*: Evaluate shape functions at given local coordinates.
    eval_shapefun_local_deriv(self, local_coords)
        *abstract*: Evaluate shape functions local derivates at given local
        coordinates.
    _admissible_gauss_quadratures()
        *abstract*: Get admissible Gauss integration quadratures.
    get_n_dof_node(self)
        Get number of degrees of freedom per node.
    check_shape_functions_properties(self)
        Check if element shape functions satisfy known properties.
    """
    @abstractmethod
    def __init__(self, n_gauss=None):
        """Constructor.
        
        Parameters
        ----------
        n_gauss : int, default=None
            Number of Gauss quadrature integration points.
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def _set_nodes_local_coords(self):
        """Set nodes local coordinates."""
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def eval_shapefun(self, local_coords):
        """Evaluate shape functions at given local coordinates.
        
        Parameters
        ----------
        local_coords : torch.Tensor(1d)
            Local coordinates of point where shape functions are evaluated.
            
        Returns
        -------
        shape_fun : torch.Tensor(1d)
            Shape functions evaluated at given local coordinates, sorted
            according with element nodes.
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def eval_shapefun_local_deriv(self, local_coords):
        """Evaluate shape functions local derivates at given local coordinates.
        
        Parameters
        ----------
        local_coords : torch.Tensor(1d)
            Local coordinates of point where shape functions local derivatives
            are evaluated.
            
        Returns
        -------
        shape_fun_local_deriv : torch.Tensor(2d)
            Shape functions local derivatives evaluated at given local
            coordinates, sorted according with element nodes. Derivative of the
            i-th shape function with respect to the j-th local coordinate is
            stored in shape_fun_local_deriv[i, j].
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    @abstractmethod
    def _admissible_gauss_quadratures():
        """Get admissible Gauss integration quadratures.
        
        Returns
        -------
        admissible_n_gauss : tuple[int]
            Admissible Gauss integration quadratures (number of Gauss
            integration points).
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_n_node(self):
        """Get number of nodes.
        
        Returns
        -------
        n_node : int
            Number of nodes.
        """
        return self._n_node
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_n_dof_node(self):
        """Get number of degrees of freedom per node.
        
        Returns
        -------
        n_dof_node : int
            Number of degrees of freedom per node.
        """
        return self._n_dof_node
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_n_gauss(self):
        """Get number of Gauss quadrature integration points.
        
        Returns
        -------
        n_gauss : int
            Number of Gauss quadrature integration points.
        """
        return self._n_gauss
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_gauss_integration_points(self):
        """Get Gaussian quadrature points local coordinates and weights.
        
        Returns
        -------
        gp_coords : dict
            Gauss quadrature integration points (key, str[int]) local
            coordinates (item, torch.Tensor(1d)). Gauss integration points are
            labeled from 1 to n_gauss.
        gp_weights : dict
            Gauss quadrature integration points (key, str[int]) weights
            (item, float). Gauss integration points are labeled from
            1 to n_gauss.
        """
        return copy.deepcopy(self._gp_coords), copy.deepcopy(self._gp_weights)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_shape_functions_properties(self):
        """Check if element shape functions satisfy known properties."""
        # Display
        print(f'\nTesting element shape functions and derivatives'
              f'\n-----------------------------------------------')
        print(f'Element: {self._name}')
        print(f'\nShape functions properties:')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Property: At a given element node, shape function evaluates to 1 if
        #           corresponding node, otherwise evaluates to 0
        #
        # Loop over element nodes
        for i in range(self._n_node):
            # Get node local coordinates
            local_coords = self._nodes_local_coords[i, :]
            # Evaluate shape functions
            shape_fun = self.eval_shapefun(local_coords)
            # Loop over shape functions
            for j in range(self._n_node):
                # Check property
                if i == j and not torch.isclose(shape_fun[j],
                                                torch.tensor(1.0)):
                    raise RuntimeError(f'Shape function of node {j} does '
                                       f'not evaluate to 1 at node {i}.')
                elif i != j and not torch.isclose(shape_fun[j],
                                                  torch.tensor(0.0)):
                    raise RuntimeError(f'Shape function of node {j} does '
                                       f'not evaluate to 0 at node {i}.')
        # Display
        print(f'  > PASS: Shape functions Delta Dirac\'s property')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Property: Sum of shape functions at any point must equal 1
        #
        # Set random point
        local_coords = torch.rand(size=(self._n_dof_node,))
        # Compute shape functions sum
        sum_shape_fun = torch.sum(self.eval_shapefun(local_coords))
        # Check property
        if not torch.isclose(sum_shape_fun, torch.tensor(1.0)):
            raise RuntimeError(f'Sum of shape functions evaluated at point '
                               f'{local_coords} does not equal 1.')
        # Display
        print(f'  > PASS: Sum of shape functions equals 1')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Property : Sum of shape functions derivatives at any point must
        #            equal 0
        #
        # Set random point
        local_coords = torch.rand(size=(self._n_dof_node,))
        # Compute shape functions derivatives sum
        sum_shape_fun_local_deriv = \
            torch.sum(self.eval_shapefun_local_deriv(local_coords))
        # Check property
        if not torch.isclose(sum_shape_fun_local_deriv, torch.tensor(0.0),
                             atol=1e-05):
            raise RuntimeError(f'Sum of shape functions derivatives evaluated '
                               f'at point {local_coords} does not equal 0.')
        # Display
        print(f'  > PASS: Sum of shape functions derivatives equals 0')