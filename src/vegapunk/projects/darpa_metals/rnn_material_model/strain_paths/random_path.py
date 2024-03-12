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
# Third-party
import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import matplotlib.pyplot as plt
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
    """Random strain deformation path generator.
    
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
    generate_strain_path(self, n_control, strain_bounds, n_time,
                         time_init=0.0, time_end=1.0):
        Generate strain deformation path.
    """
    def generate_strain_path(self, n_control, strain_bounds, n_time,
                             time_init=0.0, time_end=1.0):
        """Generate strain deformation path.
        
        Parameters
        ----------
        n_control : int
            Number of control points.
        strain_bounds : dict
            Lower and upper sampling bounds (item, tuple(lower, upper)) for
            each independent strain component (key, str).
        n_time : int
            Number of discrete time points.
        time_init : float, default=0.0
            Initial time.
        time_end : float, default=1.0
            Final time.
        
        Returns
        -------
        strain_comps : tuple
            Strain components order.
        time_hist : tuple
            Discrete time history.
        strain_path : torch.Tensor(2d)
            Strain path history stored as torch.Tensor(2d) of shape
            (sequence_length, n_strain_comps).
        """
        # Get strain components
        if self._strain_formulation == 'infinitesimal':
            strain_comps = self._comp_order_sym
        else:
            strain_comps = self._comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check strain components
        if set(strain_bounds.keys()) != set(strain_comps):
            raise RuntimeError('The sampling bounds must be provided for all '
                               'independent strain components: '
                               '\n\n', strain_comps)
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
        # Initialize strain path
        strain_path = np.zeros((n_time, len(strain_comps)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain control times (normalized)
        control_times_norm = np.linspace(-1.0, 1.0, n_control,
                                         endpoint=True, dtype=float)
        # Set discrete time history (normalized)
        time_hist_norm = np.linspace(-1.0, 1.0, n_time,
                                     endpoint=True, dtype=float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over strain components
        for j, comp in enumerate(strain_comps):
            # Sample control strains (normalized)
            control_strains_norm = \
                np.random.uniform(low=-1.0, high=1.0, size=n_control)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set constant kernel
            constant_kernel = \
                sklearn.gaussian_process.kernels.ConstantKernel(1.0,
                                                                (1e-1, 1e1))
            # Set RBF kernel
            rbf_kernel = sklearn.gaussian_process.kernels.RBF(1.0,
                                                              (1e-1, 1e1))
            # Set kernel function
            kernel = constant_kernel*rbf_kernel
            # Set homoscedastic noise
            constant_noise = 1e-5
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Gaussian Processes regression model
            gp_model = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel, alpha=constant_noise, optimizer='fmin_l_bfgs_b',
                n_restarts_optimizer=20)
            # Train Gaussian Processes model
            gp_model.fit(control_times_norm.reshape(-1, 1),
                         control_strains_norm)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Predict with Gaussian Processes model
            strain_norm_mean, _ = gp_model.predict(
                time_hist_norm.reshape(-1, 1), return_std=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get strain component sampling bounds
            lbound, ubound = strain_bounds[comp]
            # Denormalize strain component path
            strain_comp_hist = \
                np.array([lbound + 0.5*(ubound - lbound)*(x - lbound)
                          for x in strain_norm_mean])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble strain component path
            strain_path[:, j] = strain_comp_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_comps, time_hist, strain_path
# =============================================================================
if __name__ == '__main__':
    # Set strain parameters
    strain_formulation = 'infinitesimal'
    n_dim = 2
    # Initialize strain deformation path generator
    strain_path_generator = \
        RandomStrainPathGenerator(strain_formulation, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of strain control points
    n_control = 6
    # Set strain components bounds
    strain_bounds = {'11': (-1.0, 1.0),
                     '22': (-1.0, 1.0),
                     '12': (-1.0, 1.0)}
    # Set number of discrete times
    n_time = 100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate strain deformation path
    strain_comps, time_hist, strain_path = \
        strain_path_generator.generate_strain_path(n_control, strain_bounds,
                                                   n_time)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot strain deformation path
    strain_path_generator.plot_strain_path(strain_comps, time_hist,
                                           strain_path,
                                           is_stdout_display=True,
                                           is_latex=True)