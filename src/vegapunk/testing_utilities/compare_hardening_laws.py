# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Compare strain hardening laws
# =============================================================================
def compare_hardening_laws(hardening_laws, save_dir=None, is_save_fig=False,
                           is_stdout_display=False):
    """Compare strain hardening laws.
    
    Parameters
    ----------
    hardening_laws : dict
        Strain hardening laws to be compared. Each hardening law (key, str) is
        stored as dictionary (item, dict) with the corresponding
        'hardening_law' and 'hardening_parameters'.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Set comparison bounds
    acc_p_strain_min = 0.0
    acc_p_strain_max = 1.0
    # Set number of comparison points
    n_point = 10000
    # Set comparison points
    acc_p_strain_points = \
        torch.linspace(acc_p_strain_min, acc_p_strain_max, n_point)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of hardening laws
    n_law = len(hardening_laws.keys())
    # Initialize hardening laws comparison data
    comparison_data = np.zeros((n_point, 3*n_law))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over hardening laws
    for i, name in enumerate(hardening_laws.keys()):
        # Get hardening law and parameters
        hardening_law = hardening_laws[name]['hardening_law']
        hardening_parameters = hardening_laws[name]['hardening_parameters']
        # Compute hardening law data
        for j in range(n_point):
            # Get accumulated plastic strain
            acc_p_strain = acc_p_strain_points[j]
            # Compute yield stress and hardening slope
            yield_stress, hardening_slope = \
                hardening_law(hardening_parameters, acc_p_strain)
            # Store hardening law data
            comparison_data[j, 3*i] = acc_p_strain
            comparison_data[j, 3*i+1] = yield_stress
            comparison_data[j, 3*i+2] = hardening_slope
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    np.set_printoptions(linewidth=1000)
    print(comparison_data[:10, :])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot hardening laws
    if is_stdout_display:
        # Get labels
        data_labels = list(hardening_laws.keys())
        # Set data array
        col_indices = np.arange(comparison_data.shape[1])[
            np.arange(comparison_data.shape[1]) % 3 != 2]
        data_array = comparison_data[:, col_indices]
        # Plot hardening laws
        figure, axes = plot_xy_data(
            data_array, data_labels=data_labels, is_reference_data=False,
            x_label='Accumulated plastic strain', y_label='Yield stress (MPa)',
            x_lims=(acc_p_strain_min, acc_p_strain_max),
            y_lims=(None, None), is_latex=True)
        # Display figure
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        # Set figure name
        filename = 'comparison_hardening_laws'
        # Save figure
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
# =============================================================================
if __name__ == "__main__":
    # Set float precision
    torch.set_default_dtype(torch.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hardening laws
    hardening_laws = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hardening law
    name = 'nadai_ludwik'
    hardening_law = get_hardening_law('nadai_ludwik')
    hardening_parameters = {'s0': math.sqrt(3)*900,
                            'a': math.sqrt(3)*700,
                            'b': 0.5,
                            'ep0': 1e-5}
    # Store hardening law
    hardening_laws[name] = {'hardening_law': hardening_law,
                            'hardening_parameters': hardening_parameters}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set independent variable bounds
    x_lbound = 0.0
    x_ubound = 2.0
    # Set number of discretization points
    n_point = 10000
    # Set parameters
    s0 = np.sqrt(3)*900
    a = np.sqrt(3)*700
    b = 0.5
    ep0 = 1e-5
    # Set function
    def fun(x):
        return s0 + a*((x + ep0)**b)
    # Initialize discretized data array
    data_array = np.zeros((n_point, 2))
    # Set independent variables discrete points
    x = np.linspace(x_lbound, x_ubound, n_point)
    # Compute function discrete values
    y = fun(x)
    # Assemble discretized data
    data_array[:, 0] = x
    data_array[:, 1] = y
    # Set hardening law
    name = 'piecewise_linear'
    hardening_law = get_hardening_law('piecewise_linear')
    hardening_parameters = {'hardening_points': torch.tensor(data_array)}
    # Store hardening law
    hardening_laws[name] = {'hardening_law': hardening_law,
                            'hardening_parameters': hardening_parameters}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare hardening laws
    compare_hardening_laws(hardening_laws, is_stdout_display=True,
                           save_dir=None, is_save_fig=False)