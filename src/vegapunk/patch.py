"""Finite element material patch.

Classes
-------
FiniteElementPatch
    Finite element material patch.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from finite_element import FiniteElement
from ioput.iostandard import new_file_path_with_int
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class FiniteElementPatch:
    """Finite element material patch.
    
    Material patch is assumed quadrilateral (2d) or parallelepipedic (3D)
    and discretized in a regular finite element mesh of quadrilateral (2d) /
    hexahedral (3d) finite elements.
    
    Attributes
    ----------
    n_dim : int
        Number of spatial dimensions.
    patch_dims : tuple[float]
        Patch size in each dimension.
    elem_type : str
        Finite element type.
    n_elems_per_dim : tuple[int]
        Number of finite elements per dimension.
    mesh_nodes_matrix : numpy.ndarray(2d or 3d)
        Finite element mesh nodes matrix
        (numpy.ndarray[int](n_edge_nodes_per_dim) where each element
        corresponds to a given node position and whose value is set either
        as the global node label or zero (if the node does not exist).
        Nodes are labeled from 1 to n_nodes.
    mesh_nodes_coords_ref : dict
        Coordinates (item, numpy.ndarray(n_dim)) of each finite element
        mesh node (key, str[int]) in the reference configuration. Nodes are
        labeled from 1 to n_nodes.
    mesh_boundary_nodes_disps : dict
        Displacements (item, numpy.ndarray(n_dim)) prescribed on each
        finite element mesh boundary node (key, str[int]). Free degrees of
        freedom must be set as None.
    n_edge_nodes_per_dim : tuple[int]
        Number of patch edge nodes along each dimension.
    
    Methods
    -------
    _set_n_edge_nodes_per_dim(self):
        Set number of patch edge nodes per dimension.
    _get_boundary_nodes_labels(self):
        Get finite element mesh boundary nodes labels.
    plot_deformed_patch(self, is_show_plot=None, is_save_plot=False,
                        save_directory=None, plot_name=None,
                        is_overwrite_file=False)
        Generate plot of material patch.
    """
    def __init__(self, n_dim, patch_dims, elem_type, n_elems_per_dim,
                 mesh_nodes_matrix, mesh_nodes_coords_ref,
                 mesh_boundary_nodes_disps):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        patch_dims : tuple[float]
            Patch size in each dimension.
        elem_type : str
            Finite element type.
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension.
        mesh_nodes_matrix : numpy.ndarray(2d or 3d)
            Finite element mesh nodes matrix
            (numpy.ndarray[int](n_edge_nodes_per_dim) where each element
            corresponds to a given node position and whose value is set either
            as the global node label or zero (if the node does not exist).
            Nodes are labeled from 1 to n_nodes.
        mesh_nodes_coords_ref : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]) in the reference configuration. Nodes are
            labeled from 1 to n_nodes.
        mesh_boundary_nodes_disps : dict
            Displacements (item, numpy.ndarray(n_dim)) prescribed on each
            finite element mesh boundary node (key, str[int]). Free degrees of
            freedom must be set as None.
        """
        self._n_dim = n_dim
        self._patch_dims = copy.deepcopy(patch_dims)
        self._elem_type = elem_type
        self._n_elems_per_dim = copy.deepcopy(n_elems_per_dim)
        self._mesh_nodes_matrix = copy.deepcopy(mesh_nodes_matrix)
        self._mesh_nodes_coords_ref = copy.deepcopy(mesh_nodes_coords_ref)
        self._mesh_boundary_nodes_disps = \
            copy.deepcopy(mesh_boundary_nodes_disps)
        # Check if only boundary nodes have prescribed displacements
        boundary_nodes_labels = self._get_boundary_nodes_labels()
        if np.any([int(label) not in boundary_nodes_labels
                   for label in self._mesh_boundary_nodes_disps.keys()]):
            raise RuntimeError('Displacements can only be prescribed on '
                               'finite element mesh boundary nodes.')
        # Set number of patch edge nodes per dimension.
        self._set_n_edge_nodes_per_dim()
    # -------------------------------------------------------------------------
    def get_elem_type(self):
        """Get finite element type.
        
        Returns
        -------
        elem_type : str
            Finite element type.
        """
        return copy.deepcopy(self._elem_type)
    # -------------------------------------------------------------------------
    def get_n_elems_per_dim(self):
        """Get number of finite elements per dimension.
        
        Returns
        -------
        n_elems_per_dim : tuple[int]
            Number of finite elements per dimension.
        """
        return copy.deepcopy(self._n_elems_per_dim)
    # -------------------------------------------------------------------------
    def get_n_edge_nodes_per_dim(self):
        """Get number of patch edge nodes along each dimension.
        
        Returns
        -------
        n_edge_nodes_per_dim : tuple[int]
            Number of patch edge nodes along each dimension.
        """
        return copy.deepcopy(self._n_edge_nodes_per_dim)
    # -------------------------------------------------------------------------
    def get_mesh_nodes_matrix(self):
        """Get finite element mesh nodes matrix.
        
        Returns
        -------
        mesh_nodes_matrix : numpy.ndarray(2d or 3d)
           Finite element mesh nodes matrix
           (numpy.ndarray[int](n_edge_nodes_per_dim)) where each element
           corresponds to a given node position and whose value is set either
           as the global node label or zero (if the node does not exist).
           Nodes are labeled from 1 to n_nodes.
        """
        return copy.deepcopy(self._mesh_nodes_matrix)
    # -------------------------------------------------------------------------
    def get_mesh_nodes_coords_ref(self):
        """Get reference coordinates of each finite element mesh node.
        
        Returns
        -------
        mesh_nodes_coords_ref : dict
            Coordinates (item, numpy.ndarray(n_dim)) of each finite element
            mesh node (key, str[int]) in the reference configuration. Nodes are
            labeled from 1 to n_nodes.
        """
        return copy.deepcopy(self._mesh_nodes_coords_ref)
    # -------------------------------------------------------------------------
    def get_mesh_boundary_nodes_disps(self):
        """Get displacements prescribed on finite element mesh boundary nodes.
        
        Returns
        -------
        mesh_boundary_nodes_disps : dict
            Displacements (item, numpy.ndarray(n_dim)) prescribed on each
            finite element mesh boundary node (key, str[int]). Free degrees of
            freedom must be set as None.
        """
        return copy.deepcopy(self._mesh_boundary_nodes_disps)
    # -------------------------------------------------------------------------
    def get_mesh_connected_nodes(self):
        """Get finite element mesh connected nodes pairs.
        
        Returns
        -------
        connected_nodes : tuple[tuple(2)]
            A set containing all pairs of nodes that are connected by a finite
            element edge. Each connection is stored a single time as a
            tuple(node[int], node[int]) and is independent of the corresponding
            nodes storage order.
        """
        # Initialize node connectivities
        connected_nodes = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find node connectivities
        if self._n_dim == 2:
            # Get connectivities along first spatial dimension
            for j in range(self._mesh_nodes_matrix.shape[1]):
                for i in range(self._mesh_nodes_matrix.shape[0] - 1):
                    node_1 = self._mesh_nodes_matrix[i, j]
                    node_2 = self._mesh_nodes_matrix[i + 1, j]
                    if node_1 != 0 and node_2 != 0:
                        connected_nodes.append((node_1, node_2))
            # Get connectivities along second spatial dimension
            for i in range(self._mesh_nodes_matrix.shape[0]):
                for j in range(self._mesh_nodes_matrix.shape[1] - 1):
                    node_1 = self._mesh_nodes_matrix[i, j]
                    node_2 = self._mesh_nodes_matrix[i, j + 1]
                    if node_1 != 0 and node_2 != 0:
                        connected_nodes.append((node_1, node_2))
        else:
            raise RuntimeError('Missing 3D implementation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(connected_nodes)
    # -------------------------------------------------------------------------
    def _set_n_edge_nodes_per_dim(self):
        """Set number of patch edge nodes per dimension."""
        # Get finite element
        elem = FiniteElement(self._elem_type)
        # Get number of edge nodes per finite element
        n_edge_nodes_elem = elem.get_n_edge_nodes()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute number of patch edge nodes along each dimension
        n_edge_nodes_per_dim = []
        # Loop over each dimension
        for i in range(self._n_dim):
            # Compute number of edge nodes
            n_edge_nodes_per_dim.append(
                self._n_elems_per_dim[i]*n_edge_nodes_elem
                - (self._n_elems_per_dim[i] - 1))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._n_edge_nodes_per_dim = tuple(n_edge_nodes_per_dim)
    # -------------------------------------------------------------------------  
    def _get_boundary_nodes_labels(self):
        """Get finite element mesh boundary nodes labels.
        
        Returns
        -------
        boundary_nodes_labels : tuple[int]
            Finite element mesh boundary nodes labels.
        """
        # Get finite element mesh boundary nodes labels 
        if self._n_dim == 2:
            boundary_nodes_labels = tuple([
                label for label in list(self._mesh_nodes_matrix[:, 0])\
                + list(self._mesh_nodes_matrix[:, -1]) \
                + list(self._mesh_nodes_matrix[0, :]) \
                + list(self._mesh_nodes_matrix[-1, :])])
        else:
            raise RuntimeError('Missing 3D implementation.') 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return boundary_nodes_labels    
    # -------------------------------------------------------------------------
    def plot_deformed_patch(self, is_show_plot=False, is_save_plot=False,
                            save_directory=None, plot_name=None,
                            is_overwrite_file=False):
        """Generate plot of finite element material patch.
        
        Deformed configuration is only plotted if all boundary degrees of
        freedom displacements are prescribed.
        
        Parameters
        ----------
        is_show_plot : bool, default=False
            Display plot of finite element material patch if True.
        is_save_plot : bool, default=False
            Save plot of finite element material patch. Plot is only saved if
            `save_directory` is provided and exists.
        save_directory : str, default=None
            Directory where plot of finite element material patch is stored.
        plot_name : str, default=None
            Filename of finite element material patch plot.
        is_overwrite_file : bool, default=False
            Overwrite plot of finite element material patch if True, otherwise
            generate non-existent file path by extending the original file path
            with an integer.
        """
        # Generate plot
        fig, ax = plt.subplots()
        if self._n_dim == 2:
            # Get finite element
            elem = FiniteElement(self._elem_type)
            # Get number of edge nodes per finite element
            n_edge_nodes_elem = elem.get_n_edge_nodes()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get finite element mesh labels and reference coordinates
            mesh_nodes_labels = self._mesh_nodes_coords_ref.keys()
            mesh_bnd_nodes_labels = self._get_boundary_nodes_labels()
            mesh_nodes_coords_ref = self._mesh_nodes_coords_ref
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over dimensions
            for i in range(self._n_dim):
                # Loop over nodes
                for j in range(0, self._n_edge_nodes_per_dim[i],
                               n_edge_nodes_elem - 1):
                    # Get limit nodes mesh indexes
                    index_1 = tuple([j if x == i else 0 for x in range(2)])
                    index_2 = tuple([j if x == i else -1 for x in range(2)])
                    # Get limit nodes labels
                    node_1_label = self._mesh_nodes_matrix[index_1]
                    node_2_label = self._mesh_nodes_matrix[index_2]
                    # Plot finite elements countours
                    ax.plot([mesh_nodes_coords_ref[str(label)][0]
                        for label in (node_1_label, node_2_label)],
                        [mesh_nodes_coords_ref[str(label)][1]
                        for label in (node_1_label, node_2_label)],
                        '-', color='k')
            # Plot finite element mesh reference configuration
            ax.plot([mesh_nodes_coords_ref[str(label)][0]
                     for label in mesh_nodes_labels],
                    [mesh_nodes_coords_ref[str(label)][1]
                     for label in mesh_nodes_labels],
                    'o', color='k')
            ax.plot([], [], '-', color='k', label='Reference configuration')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get finite element mesh boundary nodes labels with prescribed
            # displacements
            presc_bnd_nodes_labels = [int(label) for label in
                                       self._mesh_boundary_nodes_disps.keys()]
            # Set deformed boundary configuration plot flag
            is_plot_deformed_boundary = False
            if set(presc_bnd_nodes_labels) == set(mesh_bnd_nodes_labels):
                # Plot deformed boundary configuration only if all boundary
                # degrees of freedom displacements are known                
                is_plot_deformed_boundary = not np.any([
                    None in self._mesh_boundary_nodes_disps[str(label)]
                    for label in mesh_bnd_nodes_labels])                
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot finite element mesh boundary deformed configuration
            if is_plot_deformed_boundary:
                # Loop over finite element mesh boundary nodes
                mesh_bnd_nodes_coords_def = {}
                for label in mesh_bnd_nodes_labels:
                    mesh_bnd_nodes_coords_def[str(label)] = \
                        mesh_nodes_coords_ref[str(label)] \
                        + self._mesh_boundary_nodes_disps[str(label)]
                # Get finite element mesh boundary nodes labels sorted in
                # clockwise order (closed boundary)
                nodes_labels_clockwise = tuple([
                    str(label) for label in
                    list(self._mesh_nodes_matrix[:, 0])\
                    + list(self._mesh_nodes_matrix[-1, 1:]) \
                    + list(self._mesh_nodes_matrix[-2::-1, -1]) \
                    + list(self._mesh_nodes_matrix[0, -2::-1])])
                # Plot finite element mesh boundary deformed configuration
                ax.plot([mesh_bnd_nodes_coords_def[str(label)][0]
                        for label in nodes_labels_clockwise],
                        [mesh_bnd_nodes_coords_def[str(label)][1]
                        for label in nodes_labels_clockwise],
                        'o-', color='#d62728', label='Deformed configuration')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
            # Set fixed degrees of freedom markers
            markers_dim = ('>', '^')
            # Plot fixed boundary degrees of freedom
            for label in presc_bnd_nodes_labels:
                # Get boundary node coordinates (reference configuration)
                coord = mesh_nodes_coords_ref[str(label)]
                # Get displacement
                disp = self._mesh_boundary_nodes_disps[str(label)]
                # Loop over dimensions
                for i in range(self._n_dim):
                    if isinstance(disp[i], float) and np.isclose(disp[i], 0.0):
                        ax.plot([coord[0],], [coord[1],],
                                marker=markers_dim[i],
                                markersize=15,
                                markerfacecolor='None',
                                markeredgecolor='#1f77b4',
                                markeredgewidth=2,
                                zorder=15)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set plot legend
            ax.legend(loc='center', ncol=2, numpoints=1, frameon=True,
                      fancybox=True, facecolor='inherit', edgecolor='inherit',
                      fontsize=10, framealpha=1.0,
                      bbox_to_anchor=(0, 1.05, 1.0, 0.1),
                      borderaxespad=0.0, markerscale=0.0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set axes properties
            ax.set_aspect('equal', adjustable='box') 
            # Set plot properties
            fig.set_figheight(8, forward=True)
            fig.set_figwidth(8, forward=True)
        else:
            raise RuntimeError('Missing 3D implementation.') 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_show_plot:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure (pdf format)
        if is_save_plot and os.path.exists(str(save_directory)):
            # Set figure path
            if plot_name is None:
                plot_name = 'finite_element_material_patch'
            fig_path = \
                os.path.join(os.path.normpath(save_directory), plot_name) \
                    + '.pdf'
            if os.path.isfile(fig_path) and not is_overwrite_file:
                fig_path = new_file_path_with_int(fig_path)
            # Set figure size (inches)
            fig.set_figheight(3.6, forward=False)
            fig.set_figwidth(3.6, forward=False)
            # Save figure file
            fig.savefig(fig_path, transparent=False, dpi=300,
                        bbox_inches='tight')