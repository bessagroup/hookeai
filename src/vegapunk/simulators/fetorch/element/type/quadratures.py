"""FETorch: Gaussian Quadratures.

Functions
---------
gauss_quadrature
    Get Gaussian quadrature points coordinates and weights.
gauss_quadrature_1d
    Set 1D Gauss quadrature points coordinates and weights.
gauss_quadrature_2d
    Set 2D Gauss quadrature points coordinates and weights.
gauss_quadrature_3d
    Set 3D Gauss quadrature points coordinates and weights.
uniform_grid_quadrature
    Set n-dimensional uniform grid Gauss quadrature coordinates and weights.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import torch
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
def gauss_quadrature(n_gauss, domain):
    """Get Gaussian quadrature points local coordinates and weights.
    
    Parameters
    ----------
    n_gauss : int
        Number of Gauss quadrature integration points.
    domain : {'linear', quadrilateral', 'triangular', 'hexahedral',
              'tetrahedral'}
        Integration domain geometry type.

    Returns
    -------
    gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
    """
    # Get Gauss quadrature points coordinates and weights
    if domain == 'linear':
        gp_coords, gp_weights = gauss_quadrature_1d(n_gauss)
    elif domain in ('quadrilateral', 'triangular'):
        gp_coords, gp_weights = gauss_quadrature_2d(n_gauss, domain)
    elif domain in ('hexahedral', 'tetrahedral'):
        gp_coords, gp_weights = gauss_quadrature_3d(n_gauss, domain)
    else:
        raise RuntimeError('Unknown integration domain geometry type. '
                           'Available options:'
                           '\n1D: \'linear\''
                           '\n2D: \'quadrilateral\', \'triangular\''
                           '\n3D: \'hexahedral\', \'tetrahedral\'')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gp_coords, gp_weights
# =============================================================================
def gauss_quadrature_1d(n_gauss):
    """Set 1D Gauss quadrature points local coordinates and weights.
    
    Parameters
    ----------
    n_gauss : int
        Number of Gauss quadrature integration points.
        
    Returns
    -------
    gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
    """
    # Get 1D Gauss quadrature points coordinates and weights
    if n_gauss == 1:
        # Complete integration: Polynomial Order 1
        gp_coords = {'1': torch.tensor((0.0,))}
        gp_weights = {'1': 2.0}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif n_gauss == 2:
        # Complete integration: Polynomial Order 3
        gp_coords = {'1': torch.tensor((-1.0/np.sqrt(3.0),)),
                     '2': torch.tensor((1.0/np.sqrt(3.0),))}
        gp_weights = {'1': 1.0,
                      '2': 1.0}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif n_gauss == 3:
        # Complete integration: Polynomial Order 5
        gp_coords = {'1': torch.tensor((-np.sqrt(3.0/5.0),)),
                     '2': torch.tensor((0.0,)),
                     '3': torch.tensor((np.sqrt(3.0/5.0),))}
        gp_weights = {'1': 5.0/9.0,
                      '2': 8.0/9.0,
                      '3': 5.0/9.0}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError(f'The 1D {n_gauss}-point Gauss quadrature has not '
                           f'been implemented.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gp_coords, gp_weights
# =============================================================================
def gauss_quadrature_2d(n_gauss, domain):
    """Set 2D Gauss quadrature points local coordinates and weights.
    
    Parameters
    ----------
    n_gauss : int
        Number of Gauss quadrature integration points.
    domain : {'quadrilateral', 'triangular'}
        Integration domain geometry type.
        
    Returns
    -------
    gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
    """
    # Get 2D Gauss quadrature points coordinates and weights
    if domain == 'quadrilateral':
        # Set number of Gauss integration points per dimension
        if n_gauss == 1:
            n_gauss_dim = 1
        elif n_gauss == 4:
            n_gauss_dim = 2
        elif n_gauss == 9:
            n_gauss_dim = 3
        else:
            raise RuntimeError(f'The 2D quadrilateral {n_gauss}-point Gauss '
                               f'quadrature has not been implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Gauss quadrature points coordinates and weights
        gp_coords, gp_weights = uniform_grid_quadrature(
            n_dim=2, n_gauss=n_gauss, n_gauss_dim=n_gauss_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif domain == 'triangular':
        if n_gauss == 1:
            gp_coords = {'1': torch.tensor((1.0/3.0, 1.0/3.0))}
            gp_weights = {'1': 1.0/2.0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif n_gauss == 3:
            gp_coords = {'1': torch.tensor((1.0/6.0, 1.0/6.0)),
                         '2': torch.tensor((2.0/3.0, 1.0/6.0)),
                         '3': torch.tensor((1.0/6.0, 2.0/3.0))}
            gp_weights = {'1': 1.0/6.0,
                          '2': 1.0/6.0,
                          '3': 1.0/6.0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError(f'The 2D triangular {n_gauss}-point Gauss '
                               f'quadrature has not been implemented.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown integration domain geometry type. '
                           'Must be \'quadrilateral\' or \'triangular\'.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gp_coords, gp_weights
# =============================================================================
def gauss_quadrature_3d(n_gauss, domain):
    """Set 3D Gauss quadrature points local coordinates and weights.
    
    Parameters
    ----------
    n_gauss : int
        Number of Gauss quadrature integration points.
    domain : {'hexahedral', 'tetrahedral'}
        Integration domain geometry type.
        
    Returns
    -------
    gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
    """
    # Get 2D Gauss quadrature points coordinates and weights
    if domain == 'hexahedral':
        # Set number of Gauss integration points per dimension
        if n_gauss == 1:
            n_gauss_dim = 1
        elif n_gauss == 8:
            n_gauss_dim = 2
        elif n_gauss == 27:
            n_gauss_dim = 3
        else:
            raise RuntimeError(f'The 3D hexahedral {n_gauss}-point Gauss '
                               f'quadrature has not been implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Gauss quadrature points coordinates and weights
        gp_coords, gp_weights = uniform_grid_quadrature(
            n_dim=3, n_gauss=n_gauss, n_gauss_dim=n_gauss_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif domain == 'tetrahedral':
        if n_gauss == 1:
            gp_coords = {'1': (1.0/4.0, 1.0/4.0, 1.0/4.0)}
            gp_weights = {'1': 1.0/6.0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif n_gauss == 4:
            gp_coords = {'1': torch.tensor(((5.0 - np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0)),
                         '2': torch.tensor(((5.0 + 3.0*np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0)),
                         '3': torch.tensor(((5.0 - np.sqrt(5))/20.0,
                                            (5.0 + 3.0*np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0)),
                         '4': torch.tensor(((5.0 - np.sqrt(5))/20.0,
                                            (5.0 - np.sqrt(5))/20.0,
                                            (5.0 + 3.0*np.sqrt(5))/20.0))}
            gp_weights = {'1': 1.0/24.0,
                          '2': 1.0/24.0,
                          '3': 1.0/24.0,
                          '4': 1.0/24.0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError(f'The 3D tetrahedral {n_gauss}-point Gauss '
                               f'quadrature has not been implemented.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown integration domain geometry type. '
                           'Must be \'hexahedral\' or \'tetrahedral\'.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gp_coords, gp_weights
# =============================================================================
def uniform_grid_quadrature(n_dim, n_gauss, n_gauss_dim):
    """Set nD uniform grid Gauss quadrature local coordinates and weights.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    n_gauss : int
        Number of Gauss quadrature integration points.
    n_gauss_dim : int
        Number of Gauss integration points per dimension.

    Returns
    -------
    gp_coords : dict
        Gauss quadrature integration points (key, str[int]) local coordinates
        (item, torch.Tensor(1d)). Gauss integration points are labeled from
        1 to n_gauss.
    gp_weights : dict
        Gauss quadrature integration points (key, str[int]) weights
        (item, float). Gauss integration points are labeled from
        1 to n_gauss.
    """
    # Get 1D Gauss quadrature points coordinates and weights
    gp_coords_1d, gp_weights_1d = gauss_quadrature_1d(n_gauss_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize 1D to nD mapping (sorted by dimension)
    gauss_map = np.zeros((n_dim, n_gauss), dtype=int)
    # Loop over spatial dimensions
    for i in range(n_dim):
        # Loop over Gauss quadrature points
        for j in range(n_gauss):
            # Set 1D Gauss point mapping
            gauss_map[i, j] = 1 + np.floor(j/n_gauss_dim**i) % n_gauss_dim
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Gauss quadrature points coordinates and weights
    gp_coords = {}
    gp_weights = {}
    # Loop over Gauss quadrature points
    for j in range(n_gauss):
        # Set Gauss point coordinates
        gp_coords[str(j + 1)] = tuple(
            [gp_coords_1d[str(gauss_map[i, j])][0] for i in range(n_dim)])
        # Set Gauss point weight
        gp_weights[str(j + 1)] = np.prod(
            [gp_weights_1d[str(gauss_map[i, j])] for i in range(n_dim)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gp_coords, gp_weights