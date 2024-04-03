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
from ioput.plots import plot_xy_data, scatter_xy_data, plot_histogram, \
    plot_histogram_2d, plot_boxplots, save_figure
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
    build_strain_tensor(n_dim, strain_comps, strain_comp_order, \
                        is_symmetric=False)
        Build second-order strain tensor from strain components.
    plot_strain_path(strain_formulation, n_dim, strain_comps_order, \
                     time_hist, strain_path, strain_axis_lims = None, \
                     is_plot_strain_norm=False, \
                     is_plot_inc_strain_norm=False, \
                     filename='strain_path', save_dir=None, \
                     is_save_fig=False, is_stdout_display=False, \
                     is_latex=False)
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
        # Set problem strain formulation and number of spatial dimensions
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
        time_hist : numpy.ndarray(1d)
            Discrete time history.
        strain_path : numpy.ndarray(2d)
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps).
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    def build_strain_tensor(n_dim, strain_comps, strain_comp_order,
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
        strain_comp_order : tuple[str]
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
                for x in list(''.join(strain_comp_order))]):
            raise RuntimeError('Invalid component in strain components order.')
        elif any([len(comp) != 2 for comp in strain_comp_order]):
            raise RuntimeError('Invalid component in strain order.')
        elif len(set(strain_comp_order)) != len(strain_comp_order):
            raise RuntimeError('Duplicated component in strain components '
                               'order.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check strain components completeness
        if is_symmetric and len(strain_comp_order) != 0.5*n_dim*(n_dim + 1):
            raise RuntimeError(f'Expecting {0.5*n_dim(n_dim + 1)} independent '
                               f'strain components under symmetry, but '
                               f'{len(strain_comp_order)} were provided.')
        elif not is_symmetric and len(strain_comp_order) != n_dim**2:
            raise RuntimeError(f'Expecting {n_dim**2} independent strain '
                               f'components, but {len(strain_comp_order)} were '
                               f'provided.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain tensor
        strain = np.zeros((n_dim, n_dim))
        # Loop over components
        for k, comp in enumerate(strain_comp_order):
            # Get component indexes
            i, j = [int(x) - 1 for x in comp]
            # Assemble tensor component
            strain[i, j] = strain_comps[k]
            # Assemble symmetric tensor component
            if is_symmetric and i != j:
                strain[j, i] = strain_comps[k]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain
    # -------------------------------------------------------------------------
    @staticmethod
    def plot_strain_path(strain_formulation, n_dim, strain_comps_order,
                         time_hist, strain_path,
                         is_plot_strain_path=False,
                         is_plot_strain_comp_hist=False,
                         is_plot_strain_norm=False,
                         is_plot_strain_norm_hist=False,
                         is_plot_inc_strain_norm=False,
                         is_plot_inc_strain_norm_hist=False,
                         is_plot_strain_path_pairs=False,
                         is_plot_strain_pairs_hist=False,
                         is_plot_strain_pairs_marginals=False,
                         is_plot_strain_comp_box=False,
                         strain_label='Strain',
                         strain_units='',
                         filename='strain_path',
                         save_dir=None, is_save_fig=False,
                         is_stdout_display=False, is_latex=False):
        """Plot strain path.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
        strain_comps_order : tuple[str]
            Strain components order.
        time_hist : {numpy.ndarray(1d), list[numpy.ndarray(1d)]}
            Discrete time history or list of multiple discrete time histories.
        strain_path : {numpy.ndarray(2d), list[numpy.ndarray(2d)]}
            Strain path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_strain_comps) or list of multiple strain path
            histories.
        is_plot_strain_path : bool, default=False
            Plot the strain components path.
        is_plot_strain_comp_hist : bool, default=False
            Plot a histogram for each strain component.
        is_plot_strain_norm : bool, default=False
            Plot strain norm path. and distribution.
        is_plot_strain_norm_hist : bool, default=False
            Plot strain norm distribution.
        is_plot_inc_strain_norm : bool, default=False
            Plot incremental strain norm path.
        is_plot_inc_strain_norm_hist : bool, default=False
            Plot incremental strain norm distribution.
        is_plot_strain_path_pairs : bool, default=False
            Plot the strain path for pairs of strain components in the strain
            space.
        is_plot_strain_pairs_hist : bool, default=False
            Plot the distribution for pairs of strain components in the strain
            space.
        is_plot_strain_pairs_marginals : bool, default=False
            Plot the pairs of strain components in the strain space together
            with the marginal distributions for each component.
        is_plot_strain_comp_box : bool, default=False
            If True, then plot a box plot including the different strain
            components.
        strain_label : str, default='Strain'
            Strain label.
        strain_units : str, default=''
            Strain units label.
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
        # Check strain paths data
        if isinstance(time_hist, list) or isinstance(strain_path, list):
            # Check multiple strain paths
            if (not isinstance(time_hist, list)
                    and not isinstance(strain_path, list)):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'strain path histories when providing '
                                   'multiple strain paths.')
            elif len(time_hist) != len(strain_path):
                raise RuntimeError('Inconsistent discrete time histories and '
                                   'strain path histories when providing '
                                   'multiple strain paths.')
            # Get number of strain paths
            n_path = len(strain_path)
            # Get longest strain path
            n_time_max = max([len(x) for x in time_hist])
            # Get minimum and maximum discrete times
            time_min = min([x[0] for x in time_hist])
            time_max = max([x[-1] for x in time_hist])
        else:
            # Set number of strain paths
            n_path = 1
            # Set longest strain path
            n_time_max = len(time_hist)
            # Set minimum and maximum discrete times
            time_min = time_hist[0]
            time_max = time_hist[-1]
            # Store discrete time history and stress path in list
            time_hist = [time_hist,]
            strain_path = [strain_path,]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize figure
        figure = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path
        if is_plot_strain_path and n_path == 1:
            # Set strain data array
            strain_data_xy = \
                np.zeros((n_time_max, 2*len(strain_comps_order)))
            for j in range(len(strain_comps_order)):
                strain_data_xy[:, 2*j] = time_hist[0].reshape(-1)
                strain_data_xy[:, 2*j+1] = strain_path[0][:, j]
            # Set strain data labels
            data_labels = [f'{strain_label} {x}' for x in strain_comps_order]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain path
            figure, _ = plot_xy_data(data_xy=strain_data_xy,
                                     data_labels=data_labels,
                                     x_lims=(time_min, time_max),
                                     x_label='Time',
                                     y_label=strain_label + strain_units,
                                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        # Plot strain component distribution
        if is_plot_strain_comp_hist:
            # Loop over strain components
            for j, comp in enumerate(strain_comps_order):
                # Set strain data array
                strain_paths = tuple([path[:, j] for path in strain_path])
                # Plot strain component distribution
                figure, _ = plot_histogram(
                     strain_paths, bins=20, density=True,
                     x_label=f'{strain_label} {comp}' + strain_units,
                     y_label='Probability density',
                     is_latex=is_latex)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_hist_{comp}', format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain norm (path and distribution)
        if (is_plot_strain_norm or is_plot_strain_norm_hist):
            # Initialize strain norm data array
            strain_norm_data_xy = np.full((n_time_max, 2*n_path),
                                            fill_value=np.nan)
            # Loop over strain paths
            for k in range(n_path):     
                # Loop over time steps
                for i in range(len(time_hist[k])):
                    # Assemble discrete time history
                    strain_norm_data_xy[i, 2*k] = time_hist[k][i]
                    # Get strain tensor
                    strain = StrainPathGenerator.build_strain_tensor(
                        n_dim, strain_path[k][i, :], strain_comps_order,
                        is_symmetric=strain_formulation == 'infinitesimal')
                    # Compute strain norm
                    strain_norm_data_xy[i, 2*k+1] = np.linalg.norm(strain)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain norm path
            if is_plot_strain_norm:
                figure, _ = plot_xy_data(data_xy=strain_norm_data_xy,
                                         x_lims=(time_min, time_max),
                                         y_lims=(0, None),
                                         x_label='Time',
                                         y_label=(f'{strain_label} norm'
                                                  + strain_units),
                                         is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + '_norm', format='pdf',
                                save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain norm distribution
            if is_plot_strain_norm_hist:
                # Set strain data array
                strain_paths_norm = tuple([strain_norm_data_xy[:, 1],])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                figure, _ = plot_histogram(
                    strain_paths_norm, bins=20, density=True,
                    x_label=f'{strain_label} norm' + strain_units,
                    y_label='Probability density',
                    is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + '_norm_hist', format='pdf',
                                save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain increment norm (path and distribution)
        if (is_plot_inc_strain_norm or is_plot_inc_strain_norm_hist):
            # Initialize strain increment norm data array
            inc_strain_norm_data_xy = np.zeros((n_time_max, 2*n_path))
            # Loop over strain paths
            for k in range(n_path):     
                # Loop over time steps
                for i in range(1, len(time_hist[k])):
                    # Assemble discrete time history
                    inc_strain_norm_data_xy[i, 2*k] = time_hist[k][i]
                    # Get strain tensors
                    strain = StrainPathGenerator.build_strain_tensor(
                        n_dim, strain_path[k][i, :], strain_comps_order,
                        is_symmetric=strain_formulation == 'infinitesimal')
                    strain_old = StrainPathGenerator.build_strain_tensor(
                        n_dim, strain_path[k][i-1, :], strain_comps_order,
                        is_symmetric=strain_formulation == 'infinitesimal')
                    # Compute strain increment
                    if strain_formulation == 'infinitesimal':
                        inc_strain = strain - strain_old
                    else:
                        raise RuntimeError('Not implemented.')
                    # Compute strain increment norm
                    inc_strain_norm_data_xy[i, 2*k+1] = \
                        np.linalg.norm(inc_strain)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain path increment norm
            if is_plot_inc_strain_norm:
                figure, _ = plot_xy_data(
                    data_xy=inc_strain_norm_data_xy,
                    x_lims=(time_min, time_max),
                    y_lims=(0, None),
                    x_label='Time',
                    y_label=f'{strain_label} increment norm' + strain_units,
                    is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + '_inc_norm', format='pdf',
                                save_dir=save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain increment norm distribution
            if is_plot_inc_strain_norm_hist:
                # Set strain data array
                inc_strain_paths_norm = tuple(
                    [inc_strain_norm_data_xy[:len(time_hist[k]), 2*k+1]
                        for k in range(n_path)])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                figure, _ = plot_histogram(
                    inc_strain_paths_norm, bins=20, density=True,
                    x_label=f'{strain_label} increment norm' + strain_units,
                    y_label='Probability density',
                    is_latex=is_latex)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save figure
                if is_save_fig:
                    save_figure(figure, filename + '_inc_norm_hist',
                                format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain path in strain space (path and distribution)
        if (is_plot_strain_path_pairs or is_plot_strain_pairs_hist
                or is_plot_strain_pairs_marginals):
            # Set strain component pairs according with strain formulation and
            # number of spatial dimensions
            if strain_formulation == 'infinitesimal':
                if n_dim == 2:
                    strain_pairs = (('11', '22'), ('11', '12'), ('22', '12'))
                else:
                    strain_pairs = (('11', '22'), ('11', '33'), ('22', '33'),
                                    ('11', '12'), ('11', '23'), ('11', '13'))
            else:
                raise RuntimeError('Not implemented.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over strain component pairs
            for strain_pair in strain_pairs:
                # Get strain components indexes
                j_x = strain_comps_order.index(strain_pair[0])
                j_y = strain_comps_order.index(strain_pair[1])
                # Initialize strain data array
                strain_data_xy = np.full((n_time_max, 2*n_path),
                                            fill_value=np.nan)
                # Loop over strain paths
                for k in range(n_path):
                    # Set strain data array
                    strain_data_xy[:len(time_hist[k]), 2*k] = \
                        strain_path[k][:, j_x]
                    strain_data_xy[:len(time_hist[k]), 2*k + 1] = \
                        strain_path[k][:, j_y]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot strain path for strain components pair
                if is_plot_strain_path_pairs:
                    figure, _ = plot_xy_data(
                        data_xy=strain_data_xy,
                        x_label=(f'{strain_label} {strain_pair[0]}'
                                 + strain_units),
                        y_label=(f'{strain_label} {strain_pair[1]}'
                                 + strain_units),
                        marker='o', is_latex=is_latex)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save figure
                    if is_save_fig:
                        save_figure(figure, filename
                                    + f'_{strain_pair[0]}v{strain_pair[1]}',
                                    format='pdf', save_dir=save_dir)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot strain components pair with marginal distributions
                if is_plot_strain_pairs_marginals:
                    figure, _ = scatter_xy_data(
                        data_xy=strain_data_xy,
                        x_label=(f'{strain_label} {strain_pair[0]}'
                                 + strain_units),
                        y_label=(f'{strain_label} {strain_pair[1]}'
                                 + strain_units),
                        is_marginal_dists = True,
                        is_latex=is_latex)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save figure
                    if is_save_fig:
                        save_figure(figure, filename + '_marginals'
                                    + f'_{strain_pair[0]}v{strain_pair[1]}',
                                    format='pdf', save_dir=save_dir)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot strain components pair distribution
                if is_plot_strain_pairs_hist:
                    # Concatenate strain paths
                    strain_data_xy = np.vstack(
                        [np.stack((x[:, j_x], x[:, j_y]), axis=1)
                            for x in strain_path])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    figure, _ = plot_histogram_2d(
                        strain_data_xy, bins=20, density=False,
                        x_label=(f'{strain_label} {strain_pair[0]}'
                                 + strain_units),
                        y_label=(f'{strain_label} {strain_pair[1]}'
                                 + strain_units),
                        is_latex=True)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Save figure
                    if is_save_fig:
                        save_figure(figure, filename + '_hist'
                                    + f'_{strain_pair[0]}v{strain_pair[1]}',
                                    format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot strain components box plot
        if is_plot_strain_comp_box:
            # Set strain data labels
            data_labels = [f'{x}' for x in strain_comps_order]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate strain paths
            strain_data = np.vstack(strain_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain components box plot
            figure, _ = plot_boxplots(strain_data, data_labels,
                                      x_label=f'{strain_label} components',
                                      y_label=f'{strain_label}' + strain_units,
                                      is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                save_figure(figure, filename + '_boxplot'
                            + f'_{strain_pair[0]}v{strain_pair[1]}',
                            format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        if figure is not None:
            plt.close(figure)