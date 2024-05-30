"""Strain hardening laws.

This module includes the definition of several types of strain hardening laws
and the suitable processing of the associated parameters.

Classes
-------
IsotropicHardeningLaw(ABC)
    Isotropic strain hardening law interface.
PiecewiseLinearIHL(IsotropicHardeningLaw)
    Piecewise linear isotropic strain hardening law.
LinearIHL(IsotropicHardeningLaw)
    Linear isotropic strain hardening law.
NadaiLudwikIHL(IsotropicHardeningLaw)
    Nadai-Ludwik isotropic strain hardening law.

Functions
---------
get_available_hardening_types
    Get available isotropic hardening laws.
get_hardening_law
    Get hardening law to compute yield stress and hardening slope.
torch_interp
    1D linear interpolation for monotonically increasing data points.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
# Third-party
import numpy as np
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def get_available_hardening_types():
    """Get available isotropic hardening laws.

    Available isotropic hardening laws:

    * Piecewise linear hardening

        .. math::

           \\{\\bar{\\varepsilon}^{p}_{i}, \\, \\sigma_{y, i}\\},
           \\quad i = 1, \\dots, n_{\\text{points}}

        Parameters:
        
        - 'hardening_points' : Hardening points (torch.Tensor(2d) of shape \
                               (n_points, 2), where the i-th hardening point \
                               is stored in tensor[i, :] as \
                               [accumulated plastic strain, yield stress]).

    ----

    * Linear hardening

        .. math::

           \\sigma_{y}(\\bar{\\varepsilon}^{p}) = \\sigma_{y, 0}
           + a \\bar{\\varepsilon}^{p}

        Parameters:
        
        - 's0' - Initial yielding stress (:math:`\\sigma_{y, 0}`) (float);
        - 'a'  - Hardening law parameter (:math:`a`) (float);

    ----

    * Nadai-Ludwik hardening:

        .. math::

           \\sigma_{y}(\\bar{\\varepsilon}^{p}) = \\sigma_{y, 0}
           + a (\\bar{\\varepsilon}^{p}_{0} + \\bar{\\varepsilon}^{p})^{b}

        Parameters:
        
        - 's0'  - Initial yielding stress (:math:`\\sigma_{y, 0}`) (float);
        - 'a'   - Hardening law parameter (:math:`a`) (float);
        - 'b'   - Hardening law parameter (:math:`a`) (float);
        - 'ep0' - Accumulated plastic strain corresponding to initial \
                  yielding stress (:math:`\\bar{\\varepsilon}^{p}_{0}`) \
                  (float);

    ----

    Returns
    -------
    available_hardening_types : tuple[str]
        List of available isotropic hardening laws (str).
    """
    # Set available isotropic hardening types
    available_hardening_types = ('piecewise_linear', 'linear', 'nadai_ludwik')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return available_hardening_types
# =============================================================================
def get_hardening_law(hardening_type):
    """Get hardening law to compute yield stress and hardening slope.

    Parameters
    ----------
    hardening_type : str
        Type of hardening law.

    Returns
    -------
    hardening_law : function
        Hardening law.
    """
    # Get hardening class
    if hardening_type == 'piecewise_linear':
        # Piecewise linear hardening isotropic strain hardening
        hardening_class = PiecewiseLinearIHL
    elif hardening_type == 'linear':
        # Linear isotropic strain hardening
        hardening_class = LinearIHL
    elif hardening_type == 'nadai_ludwik':
        # Nadai-Ludwik isotropic hardening
        hardening_class = NadaiLudwikIHL
    else:
        # Unknown isotropic hardening type
        raise RuntimeError('Unknown type of isotropic hardening law.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get hardening law
    hardening_law = hardening_class.hardening_law
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hardening_law
# =============================================================================
class IsotropicHardeningLaw(ABC):
    """Isotropic strain hardening law interface.
    
    Methods
    -------
    hardening_law(hardening_parameters, acc_p_strain)
        Compute yield stress and hardening slope for given plastic strain.
    """
    @staticmethod
    @abstractmethod
    def hardening_law(hardening_parameters, acc_p_strain):
        """Compute yield stress and hardening slope for given plastic strain.
        
        Parameters
        ----------
        hardening_parameters : dict
            Hardening law parameters.
        acc_p_strain : float
            Accumulated plastic strain.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        hard_slope : float
            Material hardening slope.
        """
        pass
# =============================================================================
class PiecewiseLinearIHL(IsotropicHardeningLaw):
    """Piecewise linear isotropic strain hardening law.
    
    Methods
    -------
    hardening_law(hardening_parameters, acc_p_strain)
        Compute yield stress and hardening slope for given plastic strain.
    """
    @staticmethod
    def hardening_law(hardening_parameters, acc_p_strain):
        """Compute yield stress and hardening slope for given plastic strain.
        
        Parameters
        ----------
        hardening_parameters : dict
            Hardening law parameters.
        acc_p_strain : float
            Accumulated plastic strain.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        hard_slope : float
            Material hardening slope.
        """
        # Get hardening curve points array
        hardening_points = hardening_parameters['hardening_points']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build lists with the accumulated plastic strain points and
        # associated yield stress values
        a = hardening_points[:, 0]
        b = hardening_points[:, 1]
        # Check if the accumulated plastic strain list is correctly sorted
        # in ascending order
        if not np.all(np.diff(a) > 0):
            raise RuntimeError('Points of piecewise linear isotropic '
                               'hardening law must be specified in '
                               'ascending order of accumulated '
                               'plastic strain.')
        elif not np.all([i >= 0 for i in a]):
            raise RuntimeError('Points of piecewise linear isotropic '
                               'hardening law must be associated with '
                               'non-negative accumulated plastic strain '
                               'values.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the value of the accumulated plastic strain is either below or
        # above the provided hardening curve points, then simply assume the
        # yield stress associated with the first or last provided points,
        # respectively. Otherwise, perform linear interpolation to compute
        # the yield stress        
        yield_stress = torch_interp(acc_p_strain, a, b, left=b[0], right=b[-1])        
        # Get hardening slope
        if acc_p_strain < a[0] or acc_p_strain >= a[-1]:
            hard_slope = 0
        else:
            # Get hardening curve bracket index
            idx = torch.sum(torch.ge(acc_p_strain, a)) - 1
            # Get hardening curve bracket points
            x0 = a[idx]
            y0 = b[idx]
            x1 = a[idx + 1]
            y1 = b[idx + 1]
            # Compute hardening slope
            hard_slope = (y1 - y0)/(x1 - x0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return yield_stress, hard_slope
# =============================================================================
class LinearIHL(IsotropicHardeningLaw):
    """Linear isotropic strain hardening law.
    
    Methods
    -------
    hardening_law(hardening_parameters, acc_p_strain)
        Compute yield stress and hardening slope for given plastic strain.
    """
    @staticmethod
    def hardening_law(hardening_parameters, acc_p_strain):
        """Compute yield stress and hardening slope for given plastic strain.
        
        Parameters
        ----------
        hardening_parameters : dict
            Hardening law parameters.
        acc_p_strain : float
            Accumulated plastic strain.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        hard_slope : float
            Material hardening slope.
        """
        # Get initial yield stress and hardening slope
        yield_stress_init = hardening_parameters['s0']
        hard_slope = hardening_parameters['a']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute yield stress
        yield_stress = yield_stress_init + hard_slope*acc_p_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return yield_stress, hard_slope
# =============================================================================
class NadaiLudwikIHL(IsotropicHardeningLaw):
    """Nadai-Ludwik isotropic strain hardening law.
    
    Methods
    -------
    hardening_law(hardening_parameters, acc_p_strain)
        Compute yield stress and hardening slope for given plastic strain.
    """
    @staticmethod
    def hardening_law(hardening_parameters, acc_p_strain):
        """Compute yield stress and hardening slope for given plastic strain.
        
        Parameters
        ----------
        hardening_parameters : dict
            Hardening law parameters.
        acc_p_strain : float
            Accumulated plastic strain.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        hard_slope : float
            Material hardening slope.
        """
        # Get initial yield stress and parameters
        yield_stress_init = hardening_parameters['s0']
        a = hardening_parameters['a']
        b = hardening_parameters['b']
        ep0 = hardening_parameters['ep0']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check non-negative accumulated plastic strain
        if acc_p_strain < 0.0:
            raise RuntimeError(f'Expecting non-negative accumulated plastic '
                               f'strain but got {acc_p_strain}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute yield stress
        yield_stress = yield_stress_init + a*((acc_p_strain + ep0)**b)
        # Compute hardening slope
        hard_slope = a*b*(acc_p_strain + ep0)**(b - 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return yield_stress, hard_slope
# =============================================================================
def torch_interp(x, xp, fp, left=None, right=None):
    """1D linear interpolation for monotonically increasing data points.
    
    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp), evaluated at x.
    
    This function is essentially a torch-based implementation of numpy.interp.

    Parameters
    ----------
    x : torch.Tensor(1d)
        The x-coordinates at which to evaluate the interpolated values, sorted
        by increasing order.
    xp : torch.Tensor(1d)
        The x-coordinates of the discrete data points, sorted by increasing
        order.
    fp : torch.Tensor(1d)
        The y-coordinates of the discrete data points.
    left : float, default=None
        Value to return for x < xp[0]. Defaults to fp[0].
    right : float, default=None
        Value to return for x > xp[-1]. Defaults to fp[-1].

    Returns
    -------
    y : torch.Tensor(1d)
        The y-coordinates of the interpolated data points.
    """
    # Check if data points are sorted
    if not torch.all(x == torch.sort(x)[0]):
        raise RuntimeError('The interpolated data points x-coordinates must '
                           'be sorted in ascending order.')
    if not torch.all(xp == torch.sort(xp)[0]):
        raise RuntimeError('The discrete data points x-coordinates must be '
                           'sorted in ascending order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute linear interpolation parameters
    m = (fp[1:] - fp[:-1])/(xp[1:] - xp[:-1])
    b = fp[:-1] - m*xp[:-1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute data points interpolation brackets
    idxs = torch.sum(torch.ge(x.reshape(-1, 1), xp.reshape(1, -1)), 1) - 1
    # Assign out-of-bounds data points to limits interpolation brackets
    idxs = torch.clamp(idxs, min=0, max=len(m) - 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute interpolated data points
    y = m[idxs]*x + b[idxs]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce out-of-bounds left value
    if left is not None:
        y[x < xp[0]] = left
    else:
        y[x < xp[0]] = fp[0]
    # Enforce out-of-bounds right value
    if right is not None:
        y[x > xp[-1]] = right
    else:
        y[x > xp[-1]] = fp[-1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return y