# Standard
import os
# Third-party
import numpy as np
import pyvista as pv
# =============================================================================
# Solving: AttributeError: module 'numpy' has no attribute 'bool'
# Cause: vtk 9.0.3 requires numpy < 1.24
np.bool = np.bool_
# =============================================================================
# Summary: Plot yield surface of generic constitutive material model
# =============================================================================
def yield_function(s1, s2, s3, sy, model_name, model_parameters,
                   is_normalize_sy=False):
    """Compute model yield function value from principal stresses.
    
    Parameters
    ----------
    s1 : float
        First principal stress.
    s2 : float
        Second principal stress.
    s3 : float
        Third principal stress.
    sy : float
        Yield stress.
    model_name : str
        Constitutive model for which yield function is computed.
    model_parameters : dict
        Constitutive model parameters required to compute yield function.
    is_normalize_sy : bool, default=False
        Normalize yield function value with yield stress.
        
    Returns
    -------
    phi : float
        Yield function value.
    """
    # Get model yield function
    if 'von_mises' in model_name:
        yf = yf_von_mises
    elif 'drucker_prager' in model_name:
        yf = yf_drucker_prager
    elif 'lou_zhang_yoon' in model_name:
        yf = yf_lou_zhang_yoon
    else:
        raise RuntimeError('Unknown constitutive model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute yield function value
    phi = yf(s1, s2, s3, sy, model_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize yield function value
    if is_normalize_sy:
        phi = phi/sy
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return phi
# =============================================================================
def yf_von_mises(s1, s2, s3, sy, model_parameters):
    """Yield function: Von Mises model with isotropic hardening.
    
    Parameters
    ----------
    s1 : float
        First principal stress.
    s2 : float
        Second principal stress.
    s3 : float
        Third principal stress.
    sy : float
        Yield stress.
    model_parameters : dict
        Constitutive model parameters required to compute yield function.
    
    Returns
    -------
    phi : float
        Yield function value.
    """
    # Build stress tensor
    stress = np.diag((s1, s2, s3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stress invariants
    _, _, _, _, j2, _ = get_stress_invariants(stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add small perturbation to avoid numerical issues for null stress
    j2 += 1e-10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute effective stress
    effective_stress = np.sqrt(3.0*j2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute yield function
    phi = effective_stress - sy
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return phi
# =============================================================================
def yf_drucker_prager(s1, s2, s3, sy, model_parameters):
    """Yield function: Drucker-Prager model with isotropic hardening.
    
    Parameters
    ----------
    s1 : float
        First principal stress.
    s2 : float
        Second principal stress.
    s3 : float
        Third principal stress.
    sy : float
        Yield stress.
    model_parameters : dict
        Constitutive model parameters required to compute yield function.
    
    Returns
    -------
    phi : float
        Yield function value.
    """
    # Get frictional angle
    friction_angle = model_parameters['friction_angle']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build stress tensor
    stress = np.diag((s1, s2, s3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stress invariants
    i1, _, _, _, j2, _ = get_stress_invariants(stress)
    # Compute hydrostatic stress
    p = (1/3)*i1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add small perturbation to avoid numerical issues for null stress
    j2 += 1e-10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute yield surface cohesion parameter
    xi = (2.0/np.sqrt(3))*np.cos(friction_angle)
    # Set yield pressure parameter
    eta = (3.0/np.sqrt(3))*np.sin(friction_angle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce matching yield surface with von Mises model for pi-plane
    # (assuming both models share the same yield stress)
    cy = sy/(np.sqrt(3)*xi)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute effective stress
    effective_stress = np.sqrt(j2) + eta*p
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute yield function
    phi = effective_stress - xi*cy
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return phi
# =============================================================================
def yf_lou_zhang_yoon(s1, s2, s3, sy, model_parameters):
    """Yield function: Lou-Zhang-Yoon model with isotropic strain hardening.
    
    Parameters
    ----------
    s1 : float
        First principal stress.
    s2 : float
        Second principal stress.
    s3 : float
        Third principal stress.
    sy : float
        Yield stress.
    model_parameters : dict
        Constitutive model parameters required to compute yield function.
    
    Returns
    -------
    phi : float
        Yield function value.
    """
    # Get yielding parameters
    yield_a = model_parameters['yield_a']
    yield_b = model_parameters['yield_b']
    yield_c = model_parameters['yield_c']
    yield_d = model_parameters['yield_d']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build stress tensor
    stress = np.diag((s1, s2, s3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stress invariants
    i1, _, _, _, j2, j3 = get_stress_invariants(stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add small perturbation to avoid numerical issues for null stress
    j2 += 1e-10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute auxiliary terms
    aux_1 = yield_b*i1
    aux_2 = j2**3 - yield_c*(j3**2)
    aux_3 = yield_d*j3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute effective stress
    effective_stress = yield_a*(aux_1 + (aux_2**(1/2) - aux_3)**(1/3))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute yield function
    phi = effective_stress - sy
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return phi
# =============================================================================
def get_stress_invariants(stress):
    """Compute invariants of stress and deviatoric stress.
    
    Parameters
    ----------
    stress : torch.Tensor(2d)
        Stress.
        
    Returns
    -------
    i1 : torch.Tensor(0d)
        First (principal) invariant of stress tensor.
    i2 : torch.Tensor(0d)
        Second (principal) invariant of stress tensor.
    i3 : torch.Tensor(0d)
        Third (principal) invariant of stress tensor.
    j1 : torch.Tensor(0d)
        First invariant of deviatoric stress tensor.
    j2 : torch.Tensor(0d)
        Second invariant of deviatoric stress tensor.
    j3 : torch.Tensor(0d)
        Third invariant of deviatoric stress tensor.
    """
    # Compute first (principal) invariant of stress tensor.
    i1 = np.trace(stress)
    # Compute second (principal) invariant of stress tensor.
    i2 = 0.5*(np.trace(stress)**2
                - np.trace(np.matmul(stress, stress)))
    # Compute third (principal) invariant of stress tensor.
    i3 = np.linalg.det(stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute first invariant of deviatoric stress tensor
    j1 = i1
    # Compute second invariant of deviatoric stress tensor
    j2 = (1/3)*(i1**2) - i2
    # Compute third invariant of deviatoric stress tensor
    j3 = (2/27)*(i1**3) - (1/3)*i1*i2 + i3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return i1, i2, i3, j1, j2, j3
# =============================================================================
def pi_to_principal(pi1, pi2, pi3):
    """Convert pi-stress to principal stress.
    
    Parameters
    ----------
    pi1 : float
        First pi-stress.
    pi2 : float
        Second pi-stress.
    pi3 : float
        Third pi-stress.

    Returns
    -------
    s1 : float
        First principal stress.
    s2 : float
        Second principal stress.
    s3 : float
        Third principal stress.
    """
    # Set rotation matrix (from principal stress coordinates to pi-stress
    # coordinates)
    rotation_matrix = \
        np.array([[np.sqrt(2/3), -np.sqrt(1/6), -np.sqrt(1/6)],
                  [0, np.sqrt(1/2), -np.sqrt(1/2)],
                  [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute principal stresses
    s1, s2, s3 =  np.matmul(rotation_matrix.T, np.array((pi1, pi2, pi3)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return s1, s2, s3
# =============================================================================
def batched_yield_function(pi_stress_points, sy, model_name, model_parameters):
    """Compute model yield function from batch of stress points.
    
    Parameters
    ----------
    pi_stress_points : np.ndarray(2d)
        Pi-stress points stored as numpy.ndarray(2d) of shape (n_point, 3).
    sy : float
        Yield stress.
    model_name : str
        Constitutive model for which yield function is computed.
    model_parameters : dict
        Constitutive model parameters required to compute yield function.

    Returns
    -------
    phi_points : np.ndarray(1d)
        Yield function values for each pi-stress point stored as
        numpy.ndarray(1d) of shape (n_point,).
    """
    # Get number of stress points
    n_point = pi_stress_points.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize yield function values
    phi_points = []
    # Loop over stress points
    for i in range(n_point):
        # Get pi-stress coordinates
        pi1, pi2, pi3 = pi_stress_points[i, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute principal stress coordinates
        s1, s2, s3 = pi_to_principal(pi1, pi2, pi3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute yield function value
        phi = yield_function(s1, s2, s3, sy, model_name, model_parameters,
                             is_normalize_sy=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store yield function value
        phi_points.append(phi)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert to array
    phi_points = np.array(phi_points)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return phi_points
# =============================================================================
def plot_yield_surface(models_names, models_parameters, models_sy,
                       models_labels=None, is_null_planes=False,
                       is_pi_plane_only=False,
                       filename='yield_surfaces', save_dir=None,
                       is_save_fig=False, is_stdout_display=False):
    """Plot constitutive material models yield surface.
    
    Parameters
    ----------
    models_names : tuple[str]
        Constitutive models for which the yield surface is plotted.
    models_parameters : dict
        Constitutive model parameters (item, dict) required to compute each
        model (key, str) yield surface.
    models_sy : dict
        Constitutive model yield stress (item, float) required to compute each
        model (key, str) yield surface.
    models_labels : dict, default=None
        Constitutive model label (item, str) for each model (key, str).
    is_null_planes : bool, default=False
        If True, then plot null pi-stress planes.
    is_pi_plane_only : bool, default=False
        If True, then plot only yield surfaces slice at the pi-plane.
    filename : str, default='yield_surfaces'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False
        otherwise.
    """
    # Set pi-stress range factor (bounding box)
    pi_stress_factor = 2
    # Get maximum yield stress among all models
    max_sy = max(models_sy.values())
    # Set pi-stress range
    pi_min = -pi_stress_factor*max_sy
    pi_max = pi_stress_factor*max_sy
    pi_range = pi_max - pi_min
    # Set bounding box padding
    bbox_padding = 0.05*pi_range
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of points along each uniform grid dimension
    dimensions=(50, 50, 50)
    # Set period along each uniform grid dimension
    spacing = tuple([pi_range/x for x in dimensions])
    # Set uniform grid origin (minimum starting point of grid)
    origin = (pi_min, pi_min, pi_min)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create uniform grid
    grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get grid bounds
    grid_bounds = grid.bounds
    # Extract grid bounds along each dimension
    grid_bounds_dim = tuple(
        [(grid_bounds[i], grid_bounds[i + 1]) for i in range(3)])
    # Get grid range along each dimension
    grid_range_dim = tuple(
        [grid_bounds_dim[i][1] - grid_bounds_dim[i][0] for i in range(3)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect grid points (pi-stress points)
    pi_stress_points = grid.points
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize plotter
    plotter = pv.Plotter(lighting='light kit', polygon_smoothing=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set light
    light = pv.Light(position=(pi_max, pi_max, pi_max), focal_point=(0, 0, 0),
                     intensity=1.0)
    # Add light
    plotter.add_light(light)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over constitutive models
    for i, model_name in enumerate(models_names):
        # Get model parameters
        model_parameters = models_parameters[model_name]
        # Get model yield stress
        sy = models_sy[model_name]
        # Get model label
        model_label = models_labels[model_name]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute model yield function
        phi_points = batched_yield_function(
            pi_stress_points, sy=sy, model_name=model_name,
            model_parameters=model_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set isosurface threshold
        isosurfaces = [0.0]
        # Set VTK filter to create isosurface
        contour_method = 'marching_cubes'
        # Build yield surface
        surface_mesh = \
            grid.contour(isosurfaces, phi_points, method=contour_method)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set mesh label
        mesh_label = model_label
        # Set mesh style
        mesh_style = ('surface', 'wireframe', 'points')[0]
        # Set mesh color
        mesh_color = ('#4477AA', '#EE6677', '#CCBB44', '#228833', '#66CCEE',
                      '#AA3377', '#BBBBBB', '#EE7733', '#009988', '#CC3311',
                      '#DDAA33', '#999933', '#DDCC77', '#882255')[i]
        # Set show edges and edges color
        mesh_show_edges = False
        # Set surface opacity
        mesh_opacity = 1.0
        # Set smooth shading
        mesh_smooth_shading = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add yield surface to plot
        if is_pi_plane_only:
            # Get yield surface slice at pi-plane
            pi_plane = \
                surface_mesh.slice(normal=(0, 0, 1), origin=(0.0, 0.0, 0.0))
            # Add yield surface slice mesh to plot
            _ = plotter.add_mesh(pi_plane, label=mesh_label, color=mesh_color,
                                 line_width=4.0)
        else:
            # Clip yield surface
            #surface_mesh = surface_mesh.clip(
            #   normal=(1, 0, 0), origin=(0, 0, 0), invert=True)
            # Add yield surface mesh to plot
            _ = plotter.add_mesh(surface_mesh, label=mesh_label,
                                 color=mesh_color, style=mesh_style,
                                 show_edges=mesh_show_edges,
                                 opacity=mesh_opacity,
                                 smooth_shading=mesh_smooth_shading)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot null pi-stress planes
    if is_null_planes:
        # Loop over null pi-stress planes
        for i in range(3):
            # Get plane normal direction
            direction = tuple(1 if j == i else 0 for j in range(3))
            # Get plane size indexes
            size_indexes = tuple(set((0, 1, 2)) - set((i,)))
            # Get plane color
            plane_color = ('lightgrey', 'lightgrey', 'lightgrey')[i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute null pi-stress plane
            null_pi_plane = pv.Plane(
                direction=direction, i_size=grid_range_dim[size_indexes[0]],
                j_size=grid_range_dim[size_indexes[1]])
            # Add null pi-stress plane
            plotter.add_mesh(null_pi_plane, color=plane_color, opacity=0.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add legend to plot
    plotter.add_legend(border=True, size=(0.2, 0.2), loc='upper right',
                       face='o', font_family='arial')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes titles
    xtitle = 'Pi-Stress 1'
    ytitle = 'Pi-Stress 2'
    ztitle = 'Pi-Stress 3'
    # Set axes tick labels
    show_xlabels = True
    show_ylabels = True
    show_zlabels = True
    # Plot bounding box
    plotter.show_bounds(grid="back", location="outer", bounds=grid.bounds,
                        font_family='arial', use_3d_text=False, font_size=16,
                        xtitle=xtitle, ytitle=ytitle, ztitle=ztitle,
                        show_xlabels=show_xlabels, show_ylabels=show_ylabels,
                        show_zlabels=show_zlabels, padding=bbox_padding)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display plot
    if is_stdout_display:
        plotter.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save plot
    if is_save_fig:
        # Set plot file path
        plot_file_path = \
            os.path.join(os.norm.path(save_dir), filename + '.png')
        # Save figure
        plotter.screenshot(plot_file_path)
# =============================================================================
if __name__ == '__main__':
    # Set models names
    models_names = ('von_mises', 'drucker_prager', 'lou_zhang_yoon')
    models_names = ('lou_zhang_yoon',)
    # Set models parameters
    models_parameters = {}
    models_parameters['von_mises'] = {}
    models_parameters['drucker_prager'] = {'friction_angle': np.deg2rad(10)}
    models_parameters['lou_zhang_yoon'] = {'yield_a': 1.5838,
                                           'yield_b': 0.05,
                                           'yield_c': -0.2669,
                                           'yield_d': 0.8161}
    models_parameters['lou_zhang_yoon'] = {'yield_a': 1.5838,
                                           'yield_b': 0.05,
                                           'yield_c': -1,
                                           'yield_d': 0.25*np.sqrt(3)}

    # Set models yield stress
    models_sy = {'von_mises': 0.5,
                 'drucker_prager': 0.5,
                 'lou_zhang_yoon': 0.5}
    # Set models labels
    models_labels = {'von_mises': 'Von Mises',
                     'drucker_prager': 'Drucker-Prager',
                     'lou_zhang_yoon': 'Lou-Zhang-Yoon'}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot constitutive material models yield surface
    plot_yield_surface(models_names, models_parameters, models_sy,
                       models_labels, is_null_planes=False,
                       is_pi_plane_only=False, is_stdout_display=True)