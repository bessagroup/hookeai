"""Discretize scalar-valued scalar function."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
# Set independent variable bounds
x_lbound = 0.0
x_ubound = 2.0
# Set number of discretization points
n_point = 200
# Set display data flag
is_stdout_display_data = False
# Set display figure flag
is_stdout_display_fig = True
# Set save figure flag
is_save_fig = True
# Set save file flag
is_save_file = True
# Set save directory
save_dir = '/home/bernardoferreira/Downloads'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set function option
fun_option = ('custom', 'linear_hardening', 'nadai_ludwik_hardening')[0]
# Set function
if fun_option == 'linear_hardening':
    pass
elif fun_option == 'nadai_ludwik_hardening':
    # Set parameters
    s0 = 900
    a = 700
    b = 0.5
    ep0 = 1e-5
    # Set function
    def fun(x):
        return s0 + a*((x + ep0)**b)
else:
    # Set frictional angle
    friction_angle = np.deg2rad(5)
    # Set dilatancy angle
    dilatancy_angle = friction_angle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute angle-related material parameters
    # (matching with Mohr-Coulomb under uniaxial tension and
    # compression)
    # Set yield surface cohesion parameter
    yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
    # Set yield pressure parameter
    yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
    # Set plastic flow pressure parameter
    flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set parameters
    s0 = 900/yield_cohesion_parameter
    a = 700/yield_cohesion_parameter
    b = 0.5
    ep0 = 1e-5
    # Set function
    def fun(x):
        return s0 + a*((x + ep0)**b)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize discretized data array
data_array = np.zeros((n_point, 2))
# Set independent variables discrete points
x = np.linspace(x_lbound, x_ubound, n_point)
# Compute function discrete values
y = fun(x)
# Assemble discretized data
data_array[:, 0] = x
data_array[:, 1] = y
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Print discretized function to stdout
if is_stdout_display_data:
    print('\n' + f'Discretizing function: {fun_option}')
    print('\n' + 'Number of discretization points: ' + f'{n_point:d}')
    print('\n' + 'Discrete values:' + '\n')
    print(f'{"x":^15s} {"y":^15s}' + '\n' + f'{32*"-":32s}')
    print(''.join(f'{data_array[i, 0]:15.8e} {data_array[i, 1]:15.8e}' + '\n'
          for i in range(n_point)))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot discretized function data
if is_stdout_display_fig:
    # Plot discretized function
    figure, axes = plot_xy_data(data_array,
                                x_label='Accumulated plastic strain',
                                y_label='Yield stress (MPa)',
                                x_lims=(0, 2.0), y_lims=(800, 2000),
                                is_latex=True)
    # Display figure
    plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save figure
if is_save_fig:
    # Set figure name
    filename = 'discretized_function'
    # Save figure
    save_figure(figure, filename, format='pdf', save_dir=save_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save discretized function to file
if is_save_file:
    # Set file path
    file_path = os.path.join(save_dir, 'discretized_function.dat')
    # Open data file
    data_file = open(file_path, 'w')
    # Write data file
    data_file.writelines(
        [f'Discretizing function: {fun_option}' + '\n\n',
         'Number of discretization points: ' + f'{n_point:d}' + '\n\n',
         'Discrete values:' + '\n'] \
        + [''.join(f'{data_array[i, 0]:15.8e} {data_array[i, 1]:15.8e}' + '\n'
           for i in range(n_point))])
    # Close data file
    data_file.close()