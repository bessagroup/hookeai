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
from ioput.plots import plot_xy_data, plot_histogram, save_figure
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
    generate_strain_path(self)
        Generate strain path.
    build_strain_tensor(strain_comps, n_dim, comp_order, is_symmetric=False)
        Build second-order strain tensor from strain components.
    plot_strain_path(self, strain_comps_order, time_hist, strain_path,
                     strain_axis_lims = None, is_plot_strain_norm=False,
                     is_plot_inc_strain_norm=False, filename='strain_path',
                     save_dir=None, is_save_fig=False,
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
        strain_comps_order : tuple[str]
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
    def build_strain_tensor(strain_comps, n_dim, comp_order,
                            is_symmetric=False):
        """Build second-order strain tensor from strain components.
        
        All the components required to build the complete second-order strain
        tensor must be provided.

        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        strain_comps : np.ndarray(1d)
            Strain tensor components sorted according with given components
            order.
        comp_order : tuple[str]
            Strain components order.
        is_symmetric : bool, default=False
            If True, then assembles off-diagonal strain components from
            symmetric component.

        Returns
        -------
        strain : np.ndarray(2d)
            Strain tensor.
        """
        # Check strain tensor components
        if not isinstance(strain_comps, np.ndarray):
            raise RuntimeError('Strain tensor components must be provided as '
                               'a np.ndarray(1d).')
        elif len(strain_comps.shape) != 1:
            raise RuntimeError('Strain tensor components must be provided as '
                               'a np.ndarray(1d).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input arguments
        if any([int(x) not in range(1, n_dim + 1)
                for x in list(''.join(comp_order))]):
            raise RuntimeError('Invalid component in strain components order.')
        elif any([len(comp) != 2 for comp in comp_order]):
            raise RuntimeError('Invalid component in strain order.')
        elif len(set(comp_order)) != len(comp_order):
            raise RuntimeError('Duplicated component in strain components '
                               'order.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check strain components completeness
        if is_symmetric and len(comp_order) != 0.5*n_dim*(n_dim + 1):
            raise RuntimeError(f'Expecting {0.5*n_dim(n_dim + 1)} independent '
                               f'strain components under symmetry, but '
                               f'{len(comp_order)} were provided.')
        elif not is_symmetric and len(comp_order) != n_dim**2:
            raise RuntimeError(f'Expecting {n_dim**2} independent strain '
                               f'components, but {len(comp_order)} were '
                               f'provided.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set second-order and components indexes
        so_indexes = list()
        comp_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            comp_indexes.append(comp_order.index(comp_order[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain tensor
        strain = np.zeros((n_dim, n_dim))
        # Build strain tensor from components
        for i in range(len(comp_indexes)):
            comp_idx = comp_indexes[i]
            so_idx = tuple(so_indexes[i])
            if is_symmetric and so_idx[0] != so_idx[1]:
                strain[so_idx[::-1]] = strain_comps[comp_idx]
            strain[so_idx] = strain_comps[comp_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain
    # -------------------------------------------------------------------------
    def plot_strain_path(self, strain_comps_order, time_hist, strain_path,
                         strain_axis_lims = None, is_plot_strain_norm=False,
                         is_plot_inc_strain_norm=False,
                         filename='strain_path',
                         save_dir=None, is_save_fig=False,
                         is_stdout_display=False, is_latex=False):
        """Plot strain path.
        
        Parameters
        ----------
        strain_comps_order : tuple[str]
            Strain components order.
        time_hist : tuple
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        strain_axis_lims : tuple, default=None
            Enforce the limits of the plot strain axis, stored as
            tuple(min, max).
        is_plot_strain_norm : bool, default=False
            If True, then plot strain norm path.
        is_plot_inc_strain_norm : bool, default=False
            If True, then plot incremental strain norm path and distribution.
        filename : str, default='strain_path'
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
        # Set strain data array
        strain_data_xy = np.zeros((len(time_hist), 2*len(strain_comps_order)))
        for j in range(len(strain_comps_order)):
            strain_data_xy[:, 2*j] = time_hist
            strain_data_xy[:, 2*j + 1] = strain_path[:, j]
        # Set strain data labels
        data_labels = [f'Strain {x}' for x in strain_comps_order]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain axis limits
        y_lims = (None, None)
        if isinstance(strain_axis_lims, tuple):
            y_lims = strain_axis_lims
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path
        figure, _ = plot_xy_data(data_xy=strain_data_xy,
                                 data_labels=data_labels,
                                 x_lims=(time_hist[0], time_hist[-1]),
                                 y_lims=y_lims,
                                 title='Strain path',
                                 x_label='Time', y_label='Strain',
                                 is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path norm
        if is_plot_strain_norm:
            # Set strain norm data array
            strain_norm_data_xy = np.zeros((len(time_hist), 2))
            for i in range(len(time_hist)):
                strain_norm_data_xy[i, 0] = time_hist[i]
                # Get strain tensor
                strain = StrainPathGenerator.build_strain_tensor(
                    strain_path[i, :], self._n_dim, strain_comps_order,
                    is_symmetric=self._strain_formulation == 'infinitesimal')
                # Compute strain norm
                strain_norm_data_xy[i, 1] = np.linalg.norm(strain_path[i, :])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain path norm
            figure, _ = plot_xy_data(data_xy=strain_norm_data_xy,
                                     x_lims=(time_hist[0], time_hist[-1]),
                                     y_lims=(0, None),
                                     title='Strain norm path',
                                     x_label='Time', y_label='Strain norm',
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_norm', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path increment norm
        if is_plot_inc_strain_norm:
            # Set strain increment norm data array
            inc_strain_norm_data_xy = np.zeros((len(time_hist), 2))
            for i in range(1, len(time_hist)):
                inc_strain_norm_data_xy[i, 0] = time_hist[i]
                # Get strain tensors
                strain = StrainPathGenerator.build_strain_tensor(
                    strain_path[i, :], self._n_dim, strain_comps_order,
                    is_symmetric=self._strain_formulation == 'infinitesimal')
                strain_old = StrainPathGenerator.build_strain_tensor(
                    strain_path[i - 1, :], self._n_dim, strain_comps_order,
                    is_symmetric=self._strain_formulation == 'infinitesimal')
                # Compute strain increment
                if self._strain_formulation == 'infinitesimal':
                    inc_strain = strain - strain_old
                else:
                    raise RuntimeError('Not implemented.')
                # Compute strain increment norm
                inc_strain_norm_data_xy[i, 1] = np.linalg.norm(inc_strain)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain path increment norm
            figure, _ = plot_xy_data(data_xy=inc_strain_norm_data_xy,
                                     x_lims=(time_hist[0], time_hist[-1]),
                                     y_lims=(0, None),
                                     title='Strain increment norm path',
                                     x_label='Time',
                                     y_label='Strain increment norm',
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_inc_norm', format='pdf',
                            save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain increment norm histogram
            figure, _ = plot_histogram(
                (inc_strain_norm_data_xy[:, 1],), bins=20, density=True,
                title='Strain increment norm distribution',
                x_label='Strain increment norm',
                y_label='Probability density',
                is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_inc_norm_hist', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(figure)