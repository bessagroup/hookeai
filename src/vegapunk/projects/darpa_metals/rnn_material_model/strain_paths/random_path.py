"""DARPA METALS PROJECT: Strain deformation paths numerical data.

Classes
-------
RandomStrainPathGenerator(StrainPathGenerator)
    Random strain deformation path generator.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import random
# Third-party
import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
# Local
from projects.darpa_metals.rnn_material_model.strain_paths.interface import \
    StrainPathGenerator
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class RandomStrainPathGenerator(StrainPathGenerator):
    """Random strain path generator.
    
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
                         is_cyclic_loading=False, random_seed=None)
        Generate strain path.
    """
    def generate_strain_path(self, n_control, strain_bounds, n_time,
                             generative_type='polynomial', time_init=0.0,
                             time_end=1.0, inc_strain_norm=None,
                             strain_noise_std=None, is_cyclic_loading=False,
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
        is_cyclic_loading : bool, default=False
            If True, then the strain loading path is reversed after half of the
            prescribed discrete time steps (rounding-up) are generated. In the
            case of a even number of prescribed discrete time points, the
            strain loading path is appended with a last time point equal to the
            initial strain state.
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
        if is_cyclic_loading:
            n_time_forward = int(np.floor(0.5*(n_time + 1)))
        else:
            n_time_forward = n_time
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
            # Reverse forward strain path to complete cyclic loading
            strain_path = np.vstack((strain_path, strain_path[-2::-1, :]))
            # Append initial strain state to comply with prescribed even number
            # of discrete time points
            if n_time % 2 == 0:
                strain_path = np.vstack((strain_path, strain_path[0, :]))        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_comps_order, time_hist, strain_path
# =============================================================================
if __name__ == '__main__':
    # Set strain parameters
    strain_formulation = 'infinitesimal'
    n_dim = 2
    # Initialize strain path generator
    strain_path_generator = \
        RandomStrainPathGenerator(strain_formulation, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain control points
    n_control = 5
    # Set strain components bounds
    strain_bounds = {'11': (-1.0, 1.0),
                     '22': (-1.0, 1.0),
                     '12': (-1.0, 1.0)}
    # Set number of discrete times
    n_time = 100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain paths
    n_path = 1
    # Initialize strain paths data
    time_hists = []
    strain_paths = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over strain paths
    for i in range(n_path):
        # Generate strain path
        strain_comps_order, time_hist, strain_path = \
            strain_path_generator.generate_strain_path(
                n_control, strain_bounds, n_time,
                generative_type='polynomial',
                time_init=0.0, time_end=1.0,
                inc_strain_norm=None, strain_noise_std=None,
                is_cyclic_loading=False, random_seed=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path
        strain_path_generator.plot_strain_path(
            strain_formulation, n_dim,
            strain_comps_order, time_hist, strain_path,
            is_plot_strain_path=True,
            is_plot_strain_comp_hist=False,
            is_plot_strain_norm=False,
            is_plot_strain_norm_hist=False,
            is_plot_inc_strain_norm=False,
            is_plot_inc_strain_norm_hist=False,
            is_plot_strain_path_pairs=False,
            is_plot_strain_pairs_hist=False,
            is_plot_strain_pairs_marginals=False,
            is_plot_strain_comp_box=False,
            is_stdout_display=True,
            is_latex=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store strain path data
        time_hists.append(time_hist)
        strain_paths.append(strain_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot global strain paths data
    if n_path > 1:
        # Concatenate strain paths data
        global_strain_path = np.vstack(strain_paths)
        global_time_hist = np.concatenate(time_hists)
        # Plot strain paths data
        strain_path_generator.plot_strain_path(
            strain_formulation, n_dim,
            strain_comps_order, global_time_hist, global_strain_path,
            is_plot_strain_comp_hist=False,
            is_plot_strain_norm_hist=False,
            is_plot_inc_strain_norm_hist=False,
            is_plot_strain_pairs_hist=True,
            is_plot_strain_pairs_marginals=True,
            is_plot_strain_comp_box=True,
            is_stdout_display=True,
            is_latex=True)