"""Generation of synthetic strain loading paths: Random generator.

Classes
-------
RandomStrainPathGenerator(StrainPathGenerator)
    Random strain loading path generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import random
# Third-party
import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
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
class RandomStrainPathGenerator(StrainPathGenerator):
    """Random strain loading path generator.
    
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
    generate_strain_path(self, n_control, strain_bounds, n_time, \
                         generative_type='polynomial', \
                         time_init=0.0, time_end=1.0, \
                         inc_strain_norm=None, strain_noise_std=None, \
                         n_cycle=None, random_seed=None)
        Generate strain path.
    """
    def generate_strain_path(self, n_control, strain_bounds, n_time,
                             generative_type='polynomial', time_init=0.0,
                             time_end=1.0, inc_strain_norm=None,
                             strain_noise_std=None, n_cycle=None,
                             random_seed=None):
        """Generate strain path.
        
        Parameters
        ----------
        n_control : {int, tuple[int]}
            Number of strain control points or number of strain control points
            lower and upper sampling bounds stored as tuple(lower, upper).
        strain_bounds : dict
            Lower and upper sampling bounds (item, tuple(lower, upper)) for
            each independent strain component (key, str).
        n_time : int
            Number of discrete time points.
        generative_type : {'polynomial', 'gaussian_process'}, \
                          default='polynomial'
            Regression model employed to generate strain loading path by
            fitting the randomly sampled strain control points.
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
        # Check number of strain control points
        if isinstance(n_control, tuple) and len(n_control) == 2:
            n_control_bounds = n_control
        elif isinstance(n_control, int):
            n_control_bounds = (n_control, n_control)
        else:
            raise RuntimeError('Invalid specification of number of strain '
                               'control points. Must be either int or '
                               'tuple(lower, upper).')
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
        # Set discrete time history (normalized)
        time_hist_normalized = np.linspace(-1.0, 1.0, n_time_forward,
                                           endpoint=True, dtype=float)
        
        # Generate trial strain path (normalized) by fitting generative
        # regression model
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
                # Sample number of strain control points
                n_control = np.random.randint(n_control_bounds[0],
                                              n_control_bounds[1] + 1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set strain control times (normalized)
                control_times_normalized = \
                    np.linspace(-1.0, 1.0, n_control, endpoint=True,
                                dtype=float)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Sample control strains (normalized)
                control_strains_normalized = \
                    np.random.uniform(low=-1.0, high=1.0, size=n_control)
                # Enforce initial null strain (normalized)
                control_strains_normalized[0] = \
                    -(ubound + lbound)/(ubound - lbound)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Fit polynomial regression model
                if generative_type == 'polynomial':
                    # Set polynomial degree
                    polynomial_degree = n_control - 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Fit polynomial model
                    polynomial_coefficients = \
                        np.polyfit(control_times_normalized,
                                   control_strains_normalized,
                                   polynomial_degree)
                    # Get polynomial model
                    polynomial_model = np.poly1d(polynomial_coefficients)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Predict with polynomial model
                    strain_mean_normalized = np.array(
                        [polynomial_model(x) for x in time_hist_normalized])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Fit Gaussian process regression model
                elif generative_type == 'gaussian_process':
                    # Set constant kernel (hyperparameter: variance)
                    constant_kernel = \
                        sklearn.gaussian_process.kernels.ConstantKernel(
                            1.0, (0.1, 10))
                    # Set RBF kernel (hyperparameter: length scale)
                    rbf_kernel = sklearn.gaussian_process.kernels.RBF(
                            1.0, (0.1, 10))
                    # Set kernel function
                    kernel = constant_kernel*rbf_kernel
                    # Set homoscedastic noise
                    constant_noise = 1e-5
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Initialize Gaussian Processes regression model
                    gp_model = \
                        sklearn.gaussian_process.GaussianProcessRegressor(
                            kernel=kernel, alpha=constant_noise,
                            optimizer='fmin_l_bfgs_b', n_restarts_optimizer=20)
                    # Fit Gaussian Processes regression model
                    gp_model.fit(control_times_normalized.reshape(-1, 1),
                                control_strains_normalized)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Predict with Gaussian Processes model
                    strain_mean_normalized, _ = gp_model.predict(
                        time_hist_normalized.reshape(-1, 1), return_std=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Denormalize strain component path
                strain_comp_hist = \
                    np.array([lbound + 0.5*(ubound - lbound)*(x + 1.0)
                            for x in strain_mean_normalized])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set strict bound scaling
            is_scrict_bound_scaling = True
            # Scale strain component history to strictly enforce bounds
            if (is_scrict_bound_scaling
                    and np.max(np.abs(strain_comp_hist)) > 0):
                # Get minimum and maximum values
                min_strain = np.min(strain_comp_hist)
                max_strain = np.max(strain_comp_hist)
                # Check if bounds are satisfied
                if min_strain < lbound or max_strain > ubound:
                    # Compute maximum deviation to lower bound
                    if min_strain < lbound:
                        l_dist = np.abs(lbound - min_strain)
                    else:
                        l_dist = 0
                    # Compute maximum deviation to upper bound
                    if max_strain > ubound:
                        u_dist = np.abs(ubound - max_strain)
                    else:
                        u_dist = 0
                    # Compute linear scale factor
                    if l_dist > u_dist:
                        scale_factor = np.abs(lbound/min_strain)
                    else:
                        scale_factor = np.abs(ubound/max_strain)
                    # Scale strain component history
                    strain_comp_hist = scale_factor*strain_comp_hist
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                if self._strain_formulation == 'infinitesimal':
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