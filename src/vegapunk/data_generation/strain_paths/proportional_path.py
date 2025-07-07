"""Generation of synthetic strain loading paths: Proportional generator.

Classes
-------
ProportionalStrainPathGenerator(StrainPathGenerator)
    Proportional strain loading path generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import random
# Third-party
import numpy as np
# Local
from data_generation.strain_paths.interface import StrainPathGenerator
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class ProportionalStrainPathGenerator(StrainPathGenerator):
    """Proportional strain loading path generator.
    
    Attributes
    ----------
    _strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : tuple[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : tuple[str]
        Strain/Stress components nonsymmetric order.
    
    Methods
    -------
    generate_strain_path(self, strain_bounds, n_time, \
                         time_init=0.0, time_end=1.0, \
                         inc_strain_norm=None, strain_noise_std=None, \
                         n_cycle=None, random_seed=None)
        Generate strain path.
    """
    def generate_strain_path(self, strain_bounds, n_time,
                             time_init=0.0, time_end=1.0,
                             inc_strain_norm=None, strain_noise_std=None,
                             n_cycle=None, random_seed=None):
        """Generate strain path.
        
        Parameters
        ----------
        strain_bounds : dict
            Lower and upper sampling bounds (item, tuple(lower, upper)) for
            each independent strain component (key, str).
        n_time : int
            Number of discrete time points.
        time_init : float, default=0.0
            Initial time.
        time_end : float, default=1.0
            Final time.
        inc_strain_norm : float, default=None
            Enforce given incremental strain norm in all time steps of the
            strain path.
        strain_noise_std : float, default=None
            For each discrete time, add noise to the strain components sampled
            from a Gaussian distribution with zero mean and given standard
            deviation.
        n_cycle : int, default=None
            Number of strain path (similar) loading/reverse-loading cycles.
            Last time step (corresponding to the initial strain state) is
            replicated until the prescribed number of discrete time points is
            met.
        random_seed : int, default=None
            Seed used to initialize the random number generator of Python and
            other libraries to preserve reproducibility.

        Returns
        -------
        strain_comps_order : tuple[str]
            Strain components order.
        time_hist : numpy.ndarray(1d)
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        """
        # Set random number generators initialization for reproducibility
        if isinstance(random_seed, int):
            random.seed(random_seed)
            np.random.seed(random_seed)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain components
        if self._strain_formulation == 'infinitesimal':
            strain_comps_order = self._comp_order_sym
        else:
            strain_comps_order = self._comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check strain components
        if set(strain_bounds.keys()) != set(strain_comps_order):
            raise RuntimeError('The sampling bounds must be provided for all '
                               'independent strain components: '
                               '\n\n', strain_comps_order)
        # Check sampling bounds
        for key, val in strain_bounds.items():
            if val[0] > val[1]:
                raise RuntimeError(f'The lower bound of strain component '
                                   f'{key} is greater than the upper bound.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set discrete time history
        time_hist = np.linspace(time_init, time_end, n_time,
                                endpoint=True, dtype=float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of discrete time points in the forward direction
        if isinstance(n_cycle, int) and n_cycle > 0:
            # Check minimum number of time points
            if (n_time < 2*n_cycle + 1):
                raise RuntimeError(f'In order to generate {n_cycle} '
                                   f'loading/reverse-loading cycles, '
                                   f'the minimum required number of discrete '
                                   f'time points is {2*n_cycle + 1}.')
            # Set number of forward time points
            n_time_forward = int(np.floor(((n_time - 1)/(2*n_cycle)) + 1))
            # Set cyclic loading flag
            is_cyclic_loading = True
        else:
            # Set number of forward time points
            n_time_forward = n_time
            # Set cyclic loading flag
            is_cyclic_loading = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize trial strain path
        strain_path_trial = np.zeros((n_time_forward, len(strain_comps_order)))
        # Generate trial strain path (normalized)
        for j, comp in enumerate(strain_comps_order):
            # Get strain component sampling bounds
            lbound, ubound = strain_bounds[comp]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate strain component strain path
            if np.isclose(lbound, ubound):
                # Enforce constant strain
                strain_comp_hist = np.linspace(lbound, lbound,
                                               num=n_time_forward)
            else:
                # Initialize control strains (normalized)
                control_strains_normalized = np.zeros(2)
                # Enforce initial null strain (normalized)
                control_strains_normalized[0] = \
                    -(ubound + lbound)/(ubound - lbound)
                # Sample final strain (normalized)
                control_strains_normalized[1] = \
                    np.random.uniform(low=-1.0, high=1.0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Predict with linear regression model
                strain_mean_normalized = \
                    np.linspace(control_strains_normalized[0],
                                control_strains_normalized[1],
                                num=n_time_forward)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Denormalize strain component path
                strain_comp_hist = \
                    np.array([lbound + 0.5*(ubound - lbound)*(x + 1.0)
                              for x in strain_mean_normalized])
            # Assemble strain component path
            strain_path_trial[:, j] = strain_comp_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain path
        strain_path = strain_path_trial[:, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over time
        for i in range(1, n_time_forward):
            # Enforce increment strain norm
            if inc_strain_norm is not None:
                # Check increment strain norm
                if inc_strain_norm <= 0:
                    raise RuntimeError('Prescribed incremental strain norm '
                                       'must be positive value.')
                # Get strain tensors
                strain = StrainPathGenerator.build_strain_tensor(
                    self._n_dim, strain_path[i, :], strain_comps_order,
                    is_symmetric=self._strain_formulation == 'infinitesimal')
                strain_old = StrainPathGenerator.build_strain_tensor(
                    self._n_dim, strain_path[i - 1, :], strain_comps_order,
                    is_symmetric=self._strain_formulation == 'infinitesimal')
                # Compute strain increment
                if strain_formulation == 'infinitesimal':
                    inc_strain = strain - strain_old
                else:
                    raise RuntimeError('Not implemented.')
                # Compute strain increment unit vector
                inc_strain_direction = inc_strain/np.linalg.norm(inc_strain)
                # Enforce strain increment norm
                strain = strain_old + inc_strain_norm*inc_strain_direction
                # Update strain path
                strain_path[i, :] = [strain[int(x[0])-1, int(x[1])-1]
                                    for x in strain_comps_order]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add noise to strain components
            if strain_noise_std is not None:
                # Check noise standard deviation
                if strain_noise_std < 0:
                    raise RuntimeError('Prescribed strain noise standard '
                                       'deviation must be non-negative value.')
                # Sample strain components noise
                strain_comps_noise = \
                    np.random.normal(loc=0.0, scale=strain_noise_std,
                                        size=len(strain_comps_order))
                # Add strain components noise
                strain_path[i, :] += strain_comps_noise
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate cyclic strain loading path
        if is_cyclic_loading:
            # Get initial strain state
            initial_strain = strain_path[0, :].reshape(1, -1)
            # Get forward strain path
            forward_strain_path = strain_path[:, :]
            # Get backward strain path
            backward_strain_path = strain_path[-2:0:-1, :]
            # Stack backward strain path
            strain_path = \
                np.concatenate((strain_path, backward_strain_path), axis=0)
            # Loop over additional cycles
            for i in range(1, n_cycle):
                # Stack cycle (forward and backward strain path)
                strain_path = np.concatenate(
                    (strain_path, forward_strain_path, backward_strain_path),
                    axis=0)
            # Set strain path last time step
            strain_path = np.concatenate((strain_path, initial_strain), axis=0)
            # Replicate last time step until prescribed number of discrete time
            # points is met
            for i in range(n_time - strain_path.shape[0]):
                strain_path = \
                    np.concatenate((strain_path, initial_strain), axis=0)        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_comps_order, time_hist, strain_path