# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Pruning procedure of time series data set 
# =============================================================================
def convexity_return_mapping(yield_c, yield_d):
    """Perform convexity return-mapping.
    
    For a given set (c, d), the convexity return-mapping works as follows:
    
    (1) If the yield parameters (c, d) lie inside the convexity domain
        (yield surface is convex), then they are kept unchanged;
        
    (2) If the yield parameters (c, d) lie outside the convexity domain
        (yield surface is not convex), then they are updated to the convexity
        domain boundary point along the same angular direction.

    Parameters
    ----------
    yield_c : torch.Tensor(0d)
        Yield parameter.
    yield_d : torch.Tensor(0d)
        Yield parameter.

    Returns
    -------
    is_convex : bool
        If True, then yield surface is convex, False otherwise.
    yield_c : torch.Tensor(0d)
        Yield parameter.
    yield_d : torch.Tensor(0d)
        Yield parameter.
    """
    # Check yield surface convexity
    is_convex = check_yield_surface_convexity(yield_c, yield_d)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform convexity return-mapping
    if not is_convex:
        # Compute angular direction
        theta = torch.atan2(yield_d, yield_c)
        # Compute convexity boundary point
        yield_c, yield_d = directional_convex_boundary(theta)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_convex, yield_c, yield_d
# =============================================================================
def compute_convex_boundary(n_theta=360):
    """Compute convexity domain boundary.
    
    Parameters
    ----------
    n_theta : int, default=360
        Number of discrete angular coordinates to discretize the convexity
        boundary domain.

    Returns
    -------
    convex_boundary : torch.Tensor(2d)
        Convexity domain boundary stored as torch.Tensor(2d) of shape
        (n_point, 2), where each point is stored as (yield_c, yield_d).
    """
    # Set discrete angular coordinates
    thetas = torch.linspace(0, 2.0*torch.pi, steps=n_theta)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize convexity domain boundary
    convex_boundary = torch.zeros(n_theta, 2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete angular coordinates
    for i, theta in enumerate(thetas):
        # Compute directional convexity domain boundary
        yield_c, yield_d = directional_convex_boundary(theta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store convexity domain boundary point
        convex_boundary[i, :] = torch.tensor((yield_c, yield_d)) 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return convex_boundary
# =============================================================================
def directional_convex_boundary(theta, r_lower=0.0, r_upper=4.0,
                                search_tol=1e-6):
    """Compute convexity domain boundary along given angular direction.
    
    Parameters
    ----------
    theta : torch.Tensor(0d)
        Angular coordinate in yield parameters domain (radians).
    r_lower : float, default=0.0
        Initial searching radius lower bound.
    r_upper : float, default=4.0
        Initial searching radius upper bound.
    search_tol : float, default = 1e-6
        Searching window tolerance.
    
    Return
    ------
    yield_c : torch.Tensor(0d)
        Yield parameter.
    yield_d : torch.Tensor(0d)
        Yield parameter.
    """
    # Store input angle type
    input_dtype = theta.dtype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize searching window
    r_window = r_upper - r_lower
    # Initialize mean searching radius
    r_mean = (r_upper + r_lower)/2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convexity boundary searching loop
    while (r_window > search_tol):
        # Compute yield parameters
        yield_c = r_mean*torch.cos(theta)
        yield_d = r_mean*torch.sin(theta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield surface convexity
        is_convex = check_yield_surface_convexity(yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update searching bounds
        if is_convex:
            r_lower = r_mean
        else:
            r_upper = r_mean
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update mean searching radius
        r_mean = (r_upper + r_lower)/2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update searching window
        r_window = r_upper - r_lower
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce consistent output type
    yield_c = yield_c.to(input_dtype)
    yield_d = yield_d.to(input_dtype)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return yield_c, yield_d
# =============================================================================
def check_yield_surface_convexity(yield_c, yield_d):
    """Check yield surface convexity.
    
    Parameters
    ----------
    yield_c : torch.Tensor(0d)
        Yield parameter.
    yield_d : torch.Tensor(0d)
        Yield parameter.

    Returns
    -------
    is_convex : bool
        If True, then yield surface is convex, False otherwise.
    """
    def get_dev_stress(lode_angle):
        """Compute deviatoric stress from Lode angle.
        
        Parameters
        ----------
        lode_angle : torch.Tensor(0d)
            Lode angle (radians).
            
        Returns
        -------
        dev_stress : torch.Tensor(2d)
            Deviatoric stress.
        """
        # Compute principal deviatoric stresses
        s1 = (2/3)*torch.cos(lode_angle)
        s2 = (2/3)*torch.cos((2*torch.pi/3) - lode_angle)
        s3 = (2/3)*torch.cos((4*torch.pi/3) - lode_angle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build deviatoric stress tensor
        dev_stress = torch.diag(torch.stack([s1, s2, s3]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dev_stress
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def convexity_function(dev_stress, yield_c, yield_d):
        """Function to evaluate convexity.
        
        Parameters
        ----------
        dev_stress : torch.Tensor(2d)
            Deviatoric stress.
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
            
        Returns
        -------
        val : torch.Tensor(0d)
            Convexity function value.
        """
        # Compute second invariant of deviatoric stress tensor
        j2 = 0.5*torch.sum(dev_stress*dev_stress)
        # Compute third invariant of deviatoric stress tensor
        j3 = torch.det(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute convexity function
        val = ((j2**3 - yield_c*j3**2)**(1/2) - yield_d*j3)**(1/3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return val
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def evaluate_convexity_lode(lode_angle, yield_c, yield_d, d_lode=None):
        """Evaluate convexity function for given Lode angle.
        
        Parameters
        ----------
        lode_angle : torch.Tensor(0d)
            Lode angle (radians).
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
        d_lode : torch.Tensor(0d), default=None
            Infinitesimal Lode angle (radians).
        
        Returns
        -------
        convex_fun_val : torch.Tensor(0d)
            Convexity function value.
        """
        # Enforce double precision
        lode_angle = lode_angle.double()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set infinitesimal Lode angle
        if d_lode is None:
            lode_small = torch.deg2rad(torch.tensor(0.001))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute arc point A deviatoric stress
        lode_angle_a = lode_angle
        dev_stress_a = get_dev_stress(lode_angle_a)
        dev_stress_a = \
            dev_stress_a/convexity_function(dev_stress_a, yield_c, yield_d)
        # Compute arc point B deviatoric stress
        lode_angle_b = lode_angle + lode_small
        dev_stress_b = get_dev_stress(lode_angle_b)
        dev_stress_b = \
            dev_stress_b/convexity_function(dev_stress_b, yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute midpoint C deviatoric stress
        dev_stress_c = (dev_stress_a + dev_stress_b)/2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate convexity function
        convex_fun_val = convexity_function(dev_stress_c, yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return convex_fun_val
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete Lode angles
    lode_angles = torch.deg2rad(torch.linspace(0, 360, steps=1000))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set vectorized convexity function computation (batch along Lode angles)
    vmap_evaluate_convexity_lode = \
        torch.vmap(evaluate_convexity_lode,
                   in_dims=(0, None, None), out_dims=(0,))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute convexity function values
    convex_fun_vals = \
        vmap_evaluate_convexity_lode(lode_angles, yield_c, yield_d)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check yield surface convexity
    is_convex = torch.all(convex_fun_vals <= 1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_convex
# =============================================================================
def plot_convexity_boundary(convex_boundary, parameters_paths=None,
                            is_plot_legend=False, save_dir=None,
                            is_save_fig=False, is_stdout_display=False,
                            is_latex=False):
    """Plot convexity domain boundary.
    
    Parameters
    ----------
    convex_boundary : torch.Tensor(2d)
        Convexity domain boundary stored as torch.Tensor(2d) of shape
        (n_point, 2), where each point is stored as (yield_c, yield_d).
    parameters_paths : dict, default=None
        For each yield parameters path (key, str), store a torch.Tensor(2d)
        (item, torch.Tensor) of shape (n_point, 2), where each point is stored
        as (yield_c, yield_d).
    is_plot_legend : bool, default=False
        If True, then plot legend.
    save_dir : str, default=None
        Directory where data set plots are saved.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Set data array
    data_xy = convex_boundary.numpy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    if is_plot_legend:
        data_labels = ['Convexity boundary',]
    else:
        data_labels = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Yield parameter $c$'
    y_label = 'Yield parameter $d$'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot convexity domain boundary
    figure, axes = plot_xy_data(
        data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
        x_scale='linear', y_scale='linear', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot yield parameters paths
    if isinstance(parameters_paths, dict):
        # Loop over paths
        for path_label, path_points in parameters_paths.items():
            # Convert parameters path
            path_points = path_points.numpy()
            # Plot parameters path points
            (line, ) = axes.plot(path_points[:, 0], path_points[:, 1], lw=0,
                                 marker='o', ms=3, label=path_label)
            # Plot parameters path directional arrows
            if path_points.shape[0] > 1:
                axes.quiver(path_points[:-1, 0], path_points[:-1, 1],
                            np.diff(path_points[:, 0]),
                            np.diff(path_points[:, 1]),
                            angles="xy", color=line.get_color(),
                            scale_units="xy", scale=1, width=0.005)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot legend
    if is_plot_legend:
        legend = axes.legend(loc='best', frameon=True, fancybox=True,
                             facecolor='inherit', edgecolor='inherit',
                             fontsize=8, framealpha=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set filename
    filename = f'lou_yield_convexity_domain'
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close('all')
# =============================================================================
if __name__ == '__main__':
    # Compute convexity domain boundary
    convex_boundary = compute_convex_boundary()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize parameters paths
    parameters_paths = {}
    # Set trial yield parameters
    parameters_trials = ((-2.5, 1.0), (-3.0, -0.25), (2.5, -1.0), (1.5, 1.5),
                         (1.0, 0.5), (-1.0, 0.0))
    # Loop over trial yield parameters
    for i, trial in enumerate(parameters_trials):
        # Set trial yield parameters
        yield_c_trial = torch.tensor(trial[0])
        yield_d_trial = torch.tensor(trial[1])
        # Perform convexity return-mapping
        is_convex, yield_c, yield_d = \
            convexity_return_mapping(yield_c_trial, yield_d_trial)
        # Store convexity return-mapping
        parameters_paths[f'return-mapping-{i}'] = \
            torch.tensor(([[yield_c_trial, yield_d_trial],
                           [yield_c, yield_d]]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot convexity domain boundary
    plot_convexity_boundary(convex_boundary, parameters_paths=parameters_paths,
                            is_plot_legend=False, is_stdout_display=True,
                            is_latex=True)