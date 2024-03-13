"""DARPA METALS PROJECT: Strain paths numerical data.

Classes
-------
StrainPathGenerator(ABC)
    Strain paths generator interface.
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
from abc import ABC, abstractmethod
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
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class StrainPathGenerator(ABC):
    """Strain path generator interface.
    
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
    generate_strain_path(self):
        Generate strain path.
    plot_strain_path(strain_comps, time_hist, strain_path, \
                     filename='strain_path_random', \
                     save_dir=None, is_save_fig=False, \
                     is_stdout_display=False, is_latex=False)
        Plot strain path.
    """
    def __init__(self, strain_formulation, n_dim):
        """Constructor.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
        """
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._n_dim = n_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain components order
        if n_dim == 2:
            self._comp_order_sym = ('11', '22', '12')
            self._comp_order_nsym = ('11', '21', '12', '22')
        else:
            self._comp_order_sym = ('11', '22', '33', '12', '23', '13')
            self._comp_order_nsym = ('11', '21', '31', '12', '22', '32',
                                     '13', '23', '33')
    # -------------------------------------------------------------------------
    @abstractmethod
    def generate_strain_path(self):
        """Generate strain path.

        Returns
        -------
        strain_comps : tuple[str]
            Strain components order.
        time_hist : tuple
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    def plot_strain_path(strain_comps, time_hist, strain_path,
                         strain_axis_lims = None,
                         filename='strain_path_random',
                         save_dir=None, is_save_fig=False,
                         is_stdout_display=False, is_latex=False):
        """Plot strain path.
        
        Parameters
        ----------
        strain_comps : tuple[str]
            Strain components order.
        time_hist : tuple
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        strain_axis_lims : tuple, default=None
            Enforce the limits of the plot strain axis, stored as
            tuple(min, max).
        filename : str, default='strain_path_random'
            Figure name.
        save_dir : str, default=None
            Directory where figure is saved. If None, then figure is saved in
            current working directory.
        is_save_fig : bool, default=False
            Save figure.
        is_stdout_display : bool, default=False
            True if displaying figure to standard output device, False
            otherwise.
        is_latex : bool, default=False
            If True, then render all strings in LaTeX. If LaTex is not
            available, then this option is silently set to False and all input
            strings are processed to remove $(...)$ enclosure.
        """
        # Set data array
        data_xy = np.zeros((len(time_hist), 2*len(strain_comps)))
        for j in range(len(strain_comps)):
            data_xy[:, 2*j] = time_hist
            data_xy[:, 2*j + 1] = strain_path[:, j]
        # Set data labels
        data_labels = [f'Strain {x}' for x in strain_comps]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain axis limits
        y_lims = (None, None)
        if isinstance(strain_axis_lims, tuple):
            y_lims = strain_axis_lims
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path
        figure, _ = plot_xy_data(data_xy, data_labels=data_labels,
                                 x_lims=(time_hist[0], time_hist[-1]),
                                 y_lims=y_lims,
                                 title='Strain path',
                                 x_label='Time', y_label='Strain',
                                 is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(figure)