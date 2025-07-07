"""Finite element material patch.

Classes
-------
FiniteElementPatch
    Finite element material patch.

Functions
---------
rotation_angle_2d
    Compute the rotation angle between two vectors in 2D.
mean_rotation_angle_2d
    Compute mean rotation angle between pairs of vectors in 2D.
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
from simulators.links.discretization.finite_element import FiniteElement
from ioput.iostandard import new_file_path_with_int
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
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
    _n_dim : int
        Number of spatial dimensions.
    _patch_dims : tuple[float]
        Patch size in each dimension.
    _elem_type : str
        Finite element type.
    _n_elems_per_dim : tuple[int]
        Number of finite elements per dimension.
    _mesh_nodes_matrix : numpy.ndarray(2d or 3d)
        Finite element mesh nodes matrix
        (numpy.ndarray[int](n_edge_nodes_per_dim) where each element
        corresponds to a given node position and whose value is set either
        as the global node label or zero (if the node does not exist).
        Nodes are labeled from 1 to n_nodes.
    _mesh_nodes_coords_ref : dict
        Coordinates (item, numpy.ndarray(n_dim)) of each finite element
        mesh node (key, str[int]) in the reference configuration. Nodes are
        labeled from 1 to n_nodes.
    _mesh_boundary_nodes_disps : dict
        Displacements (item, numpy.ndarray(n_dim)) prescribed on each
        finite element mesh boundary node (key, str[int]). Free degrees of
        freedom must be set as None.
    _n_edge_nodes_per_dim : tuple[int]
        Number of patch edge nodes along each dimension.
    
    Methods
    -------
    get_n_dim(self)
        Get number of spatial dimensions.
    get_elem_type(self)
        Get finite element type.
    get_n_elems_per_dim(self)
        Get number of finite elements per dimension.
    get_n_edge_nodes_per_dim(self)
        Get number of patch edge nodes along each dimension.
    get_mesh_nodes_matrix(self)
        Get finite element mesh nodes matrix.
    get_mesh_nodes_coords_ref(self)
        Get reference coordinates of each finite element mesh node.
    get_mesh_boundary_nodes_disps(self)
        Get displacements prescribed on finite element mesh boundary nodes.
    get_elem_size_dims(self)
        Get finite element size along each dimension.
    get_mesh_connected_nodes(self)
        Get finite element mesh connected nodes pairs.
    _set_n_edge_nodes_per_dim(self)
        Set number of patch edge nodes per dimension.
    get_boundary_nodes_labels(self)
        Get finite element mesh boundary nodes labels.
    get_boundary_edges_nodes_labels(self)
        Get finite element mesh boundary edges nodes labels.
    get_patch_attributes(self)
        Get finite element material patch attributes.
    plot_deformed_patch(self, is_hide_axes=False, is_show_fixed_dof=False,
                        is_hide_deformed_faces=True, is_show_plot=False,
                        is_save_plot=False, save_directory=None,
                        plot_name=None, is_overwrite_file=False)
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
        boundary_nodes_labels = self.get_boundary_nodes_labels()
        if np.any([int(label) not in boundary_nodes_labels
                   for label in self._mesh_boundary_nodes_disps.keys()]):
            raise RuntimeError('Displacements can only be prescribed on '
                               'finite element mesh boundary nodes.')
        # Set number of patch edge nodes per dimension.
        self._set_n_edge_nodes_per_dim()
    # -------------------------------------------------------------------------
    def get_n_dim(self):
        """Get number of spatial dimensions.
        
        Returns
        -------
        n_dim : int
            Number of spatial dimensions.
        """
        return copy.deepcopy(self._n_dim)
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
    def get_elem_size_dims(self):
        """Get finite element size along each dimension.
        
        Returns
        -------
        elem_size_dims : tuple
            Finite element size along each dimension.
        """
        return tuple([self._patch_dims[i]/self._n_elems_per_dim[i]
                      for i in range(self._n_dim)])
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
    def get_boundary_nodes_labels(self):
        """Get finite element mesh boundary nodes labels.
        
        Returns
        -------
        boundary_nodes_labels : tuple[int]
            Finite element mesh boundary nodes labels.
        """
        # Get finite element mesh boundary nodes labels 
        if self._n_dim == 2:
            boundary_nodes_labels = tuple(set([
                label for label in list(self._mesh_nodes_matrix[:, 0]) \
                + list(self._mesh_nodes_matrix[:, -1]) \
                + list(self._mesh_nodes_matrix[0, :]) \
                + list(self._mesh_nodes_matrix[-1, :])]))
        else:
            boundary_nodes_labels = tuple(set([
                label for label in
                list(self._mesh_nodes_matrix[:, :, 0].flatten()) \
                + list(self._mesh_nodes_matrix[:, :, -1].flatten()) \
                + list(self._mesh_nodes_matrix[0, :, :].flatten()) \
                + list(self._mesh_nodes_matrix[-1, :, :].flatten()) \
                + list(self._mesh_nodes_matrix[:, 0, :].flatten()) \
                + list(self._mesh_nodes_matrix[:, -1, :].flatten())
                if label != 0]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return boundary_nodes_labels
    # -------------------------------------------------------------------------
    def get_boundary_edges_nodes_labels(self):
        """Get finite element mesh boundary edges nodes labels.
        
        Returns
        -------
        boundary_edges_nodes_labels : tuple[int]
            Finite element mesh boundary edges nodes labels.
        """
        # Get finite element mesh boundary edges nodes labels 
        if self._n_dim == 2:
            boundary_edges_nodes_labels = tuple(set([
                label for label in list(self._mesh_nodes_matrix[:, 0]) \
                + list(self._mesh_nodes_matrix[:, -1]) \
                + list(self._mesh_nodes_matrix[0, :]) \
                + list(self._mesh_nodes_matrix[-1, :])]))
        else:
            boundary_edges_nodes_labels = tuple(set([
                label for label in
                list(self._mesh_nodes_matrix[:, 0, 0].flatten()) \
                + list(self._mesh_nodes_matrix[:, 0, -1].flatten()) \
                + list(self._mesh_nodes_matrix[:, -1, 0].flatten()) \
                + list(self._mesh_nodes_matrix[:, -1, -1].flatten()) \
                + list(self._mesh_nodes_matrix[0, :, 0].flatten()) \
                + list(self._mesh_nodes_matrix[0, :, -1].flatten()) \
                + list(self._mesh_nodes_matrix[-1, :, 0].flatten()) \
                + list(self._mesh_nodes_matrix[-1, :, -1].flatten()) \
                + list(self._mesh_nodes_matrix[0, 0, :].flatten()) \
                + list(self._mesh_nodes_matrix[0, -1, :].flatten()) \
                + list(self._mesh_nodes_matrix[-1, 0, :].flatten()) \
                + list(self._mesh_nodes_matrix[-1, -1, :].flatten())]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return boundary_edges_nodes_labels
    # -------------------------------------------------------------------------
    def get_patch_attributes(self):
        """Get finite element material patch attributes.
        
        Returns
        -------
        patch_attributes : dict
            Material patch attributes.
        """
        # Initialize material patch attributes
        patch_attributes = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble material patch attributes
        patch_attributes['n_dim'] = self._n_dim
        patch_attributes['patch_dims'] = self._patch_dims
        patch_attributes['elem_type'] = self._elem_type
        patch_attributes['n_elems_per_dim'] = self._n_elems_per_dim
        patch_attributes['mesh_nodes_matrix'] = self._mesh_nodes_matrix
        patch_attributes['mesh_nodes_coords_ref'] = self._mesh_nodes_coords_ref
        patch_attributes['mesh_boundary_nodes_disps'] = \
            self._mesh_boundary_nodes_disps
        patch_attributes['n_edge_nodes_per_dim'] = self._n_edge_nodes_per_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return patch_attributes
    # -------------------------------------------------------------------------
    def plot_deformed_patch(self, is_hide_axes=False, is_show_fixed_dof=False,
                            is_hide_deformed_faces=True, is_show_plot=False,
                            is_save_plot=False, save_directory=None,
                            plot_name=None, is_overwrite_file=False):
        """Generate plot of finite element material patch.
        
        Deformed configuration is only plotted if all boundary degrees of
        freedom displacements are prescribed.
        
        Parameters
        ----------
        is_hide_axes : bool, default=False
            If True, then hide all visual components of axes.
        is_show_fixed_dof : bool, default=False
            If True, then signal fixed boundary degrees of freedom.
        is_hide_deformed_faces : bool, default=True
            If True, then hide boundary faces nodes deformed configuration
            when available. Only effective for three-dimensional patch.
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
        # Get finite element mesh nodes reference coordinates
        mesh_nodes_coords_ref = self._mesh_nodes_coords_ref
        # Get finite element mesh nodes labels
        mesh_bnd_nodes_labels = self.get_boundary_nodes_labels()
        mesh_bnd_edges_nodes_labels = self.get_boundary_edges_nodes_labels()
        mesh_bnd_faces_nodes_labels = tuple(
            set(mesh_bnd_nodes_labels) - set(mesh_bnd_edges_nodes_labels))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get finite element mesh boundary nodes labels with prescribed
        # displacements
        presc_bnd_nodes_labels = \
            [int(label) for label in self._mesh_boundary_nodes_disps.keys()]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize multi-directional connectivities node labels
        limit_nodes_labels = []
        # Loop over dimensions
        for i in range(self._n_dim):
            # Initialize slicer
            slicer = self._n_dim*[slice(None)]
            # Loop over nodes
            for j in range(self._mesh_nodes_matrix.shape[i] - 1):
                # Set slices along dimension
                slicer_1 = slicer[:]
                slicer_1[i] = j
                slicer_2 = slicer[:]
                slicer_2[i] = j + 1
                # Get connectivities node labels
                limit_nodes_labels += list(zip(
                    self._mesh_nodes_matrix[tuple(slicer_1)].flatten(),
                    self._mesh_nodes_matrix[tuple(slicer_2)].flatten()))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove connectivities involving unexistent nodes
        limit_nodes_labels = [pair_label for pair_label in limit_nodes_labels
                              if 0 not in pair_label]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes marker size
        nodes_ms = 4
        # Set reference and deformed configuration colors
        ref_color = '#a9a9a9'
        def_color = '#d62728'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Default LaTeX Computer Modern Roman
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'serif',
                          'serif': ['Computer Modern Roman']}) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self._n_dim == 2:
            # Generate plot
            fig, ax = plt.subplots()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove non-boundary connectivities nodes labels
            masked_limit_nodes_labels = []
            for pair_labels in limit_nodes_labels:
                # Store boundary connectivity nodes labels
                if set(pair_labels).issubset(mesh_bnd_nodes_labels):
                    masked_limit_nodes_labels.append(pair_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot finite element mesh (reference configuration)
            for pair_labels in masked_limit_nodes_labels:
                ax.plot([mesh_nodes_coords_ref[str(label)][0]
                         for label in pair_labels],
                        [mesh_nodes_coords_ref[str(label)][1]
                         for label in pair_labels],
                        'o-', color=ref_color, ms=nodes_ms)
            ax.plot([], [], '-', color=ref_color, label='Reference')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                    # Compute finite element mesh boundary nodes coordinates
                    # (deformed configuration)
                    mesh_bnd_nodes_coords_def[str(label)] = \
                        mesh_nodes_coords_ref[str(label)] \
                        + self._mesh_boundary_nodes_disps[str(label)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove non-edge connectivities nodes labels
                masked_limit_nodes_labels = []
                for pair_labels in limit_nodes_labels:
                    # Store edge-to-edge connectivity nodes labels
                    if set(pair_labels).issubset(mesh_bnd_edges_nodes_labels):
                        masked_limit_nodes_labels.append(pair_labels)    
                # Plot finite element mesh (deformed configuration)
                for pair_labels in masked_limit_nodes_labels:
                    ax.plot([mesh_bnd_nodes_coords_def[str(label)][0]
                            for label in pair_labels],
                            [mesh_bnd_nodes_coords_def[str(label)][1]
                            for label in pair_labels],
                            'o-', color=def_color, ms=nodes_ms)
                ax.plot([], [], '-', color=def_color, label='Deformed')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set axes labels  
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
        else:
            # Generate plot
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove non-boundary connectivities nodes labels
            masked_limit_nodes_labels = []
            for pair_labels in limit_nodes_labels:
                # Store boundary connectivity nodes labels
                if set(pair_labels).issubset(mesh_bnd_nodes_labels):
                    masked_limit_nodes_labels.append(pair_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot finite element mesh (reference configuration)
            for pair_labels in masked_limit_nodes_labels:
                ax.plot([mesh_nodes_coords_ref[str(label)][0]
                         for label in pair_labels],
                        [mesh_nodes_coords_ref[str(label)][1]
                         for label in pair_labels],
                        [mesh_nodes_coords_ref[str(label)][2]
                         for label in pair_labels],
                        'o-', color=ref_color, ms=nodes_ms)
            ax.plot([], [], '-', color=ref_color, label='Reference')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set boundary deformed configuration plot flags
            is_plot_deformed_boundary_edges = False
            is_plot_deformed_boundary_faces = False
            if set(mesh_bnd_edges_nodes_labels).issubset(
                    presc_bnd_nodes_labels):
                # Plot boundary edges deformed configuration only if all
                # degrees of freedom displacements are known                
                is_plot_deformed_boundary_edges = not np.any([
                    None in self._mesh_boundary_nodes_disps[str(label)]
                    for label in mesh_bnd_edges_nodes_labels])
            if set(mesh_bnd_faces_nodes_labels).issubset(
                    presc_bnd_nodes_labels):
                # Plot boundary faces deformed configuration only if all
                # degrees of freedom displacements are known                
                is_plot_deformed_boundary_faces = not np.any([
                    None in self._mesh_boundary_nodes_disps[str(label)]
                    for label in mesh_bnd_faces_nodes_labels])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot finite element mesh boundary deformed configuration
            if is_plot_deformed_boundary_edges:
                # Loop over finite element mesh boundary nodes
                mesh_bnd_nodes_coords_def = {}
                for label in presc_bnd_nodes_labels:
                    # Compute finite element mesh boundary nodes coordinates
                    # (deformed configuration)
                    mesh_bnd_nodes_coords_def[str(label)] = \
                        mesh_nodes_coords_ref[str(label)] \
                        + self._mesh_boundary_nodes_disps[str(label)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove non-boundary connectivities nodes labels
                masked_limit_nodes_labels = []
                for pair_labels in limit_nodes_labels:
                    # Store edge-to-edge connectivity nodes labels
                    if is_plot_deformed_boundary_edges:
                        # Store edge connectivity nodes labels
                        if set(pair_labels).issubset(
                                mesh_bnd_edges_nodes_labels):
                            masked_limit_nodes_labels.append(pair_labels)
                    # Store face-to-face and face-to-edge connectivity nodes
                    # labels
                    if (not is_hide_deformed_faces
                            and is_plot_deformed_boundary_faces):
                        # Store edge connectivity nodes labels
                        if (set(pair_labels).issubset(mesh_bnd_nodes_labels)
                                and not set(pair_labels).issubset(
                                mesh_bnd_edges_nodes_labels)):
                            masked_limit_nodes_labels.append(pair_labels)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot finite element mesh (deformed configuration)
                for pair_labels in masked_limit_nodes_labels:
                    ax.plot([mesh_bnd_nodes_coords_def[str(label)][0]
                            for label in pair_labels],
                            [mesh_bnd_nodes_coords_def[str(label)][1]
                            for label in pair_labels],
                            [mesh_bnd_nodes_coords_def[str(label)][2]
                            for label in pair_labels],
                            'o-', color=def_color, ms=nodes_ms)
                ax.plot([], [], '-', color=def_color, label='Deformed')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set axes labels  
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_zlabel('Dim 3')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set grid off
                ax.grid(False)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Adjust view angle
                ax.view_init(elev=33, azim=36)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fixed degrees of freedom markers
        if self._n_dim == 2:
            markers_dim = ('>', '^')
        else:
            markers_dim = ('>', '<', '^')
        # Plot fixed boundary degrees of freedom
        if is_show_fixed_dof:
            for label in presc_bnd_nodes_labels:
                # Get boundary node coordinates (reference configuration)
                coord = mesh_nodes_coords_ref[str(label)]
                # Get displacement
                disp = self._mesh_boundary_nodes_disps[str(label)]
                # Loop over dimensions
                for i in range(self._n_dim):
                    if isinstance(disp[i], float) and np.isclose(disp[i], 0.0):
                        ax.plot(*list(coord),
                                marker=markers_dim[i],
                                markersize=nodes_ms + 4,
                                markerfacecolor='None',
                                markeredgecolor='#1f77b4',
                                markeredgewidth=1,
                                zorder=15)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Hide axes components
        if is_hide_axes:
            ax.set_axis_off()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plot legend
        ax.legend(loc='center', ncol=2, numpoints=1, frameon=True,
                    fancybox=True, facecolor='inherit', edgecolor='inherit',
                    fontsize=10, framealpha=1.0,
                    bbox_to_anchor=(0, 1.03, 1.0, 0.1),
                    borderaxespad=0.0, markerscale=0.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes properties
        ax.set_aspect('equal', adjustable='box') 
        # Set plot properties
        fig.set_figheight(8, forward=True)
        fig.set_figwidth(8, forward=True)
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
            # Set padding
            if self._n_dim == 3 and not is_hide_axes:
                pad_inches = 0.25
            else:
                pad_inches = None
            # Save figure file
            fig.savefig(fig_path, transparent=False, dpi=300,
                        bbox_inches='tight', pad_inches=pad_inches)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(fig)
# =============================================================================
def rotation_angle_2d(x1, x2):
    """Compute the rotation angle between two vectors in 2D.
    
    The rotation angle is computed from x1 to x2 array.
    
    Parameters
    ----------
    x1 : np.ndarray(1d)
        1D array.
    x2 : np.ndarray(1d)
        1D array.
        
    Returns
    -------
    angle_deg : float
        Angle (degrees) from x1 to x2, contained between -180 and +180 degrees.
    """
    # Check 1D arrays
    for x in (x1, x2):
        if (not isinstance(x, np.ndarray)) or (len(x.shape) != 1):
            raise RuntimeError(f'Input arrays must be 1D numpy.ndarray.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute angle (radians)
    angle = np.arctan2(np.linalg.det([x1, x2]),np.dot(x1, x2))
    # Compute angle (degrees)
    angle_deg = np.degrees(angle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return angle_deg
# =============================================================================
def mean_rotation_angle_2d(x1_arrays, x2_arrays):
    """Compute mean rotation angle between pairs of vectors in 2D.
    
    The i-th rotation angle is computed from x1 to x2 arrays, stored in
    x1_arrays[i, :] and x2_arrays[i, :], respectively.
    
    Parameters
    ----------
    x1_arrays : numpy.ndarray(2d)
        1D arrays stored as numpy.ndarray(n_arrays, 2).
    x2_arrays : numpy.ndarray(2d)
        1D arrays stored as numpy.ndarray(n_arrays, 2).
        
    Returns
    -------
    mean_angle_deg : float
        Mean rotation angle (degrees) from x1 to x2, contained between -180 and
        +180 degrees.
    """
    # Check 2D arrays
    for x in (x1_arrays, x2_arrays):
        if (not isinstance(x, np.ndarray)) or (len(x.shape) != 2):
            raise RuntimeError(f'Input arrays must be 2D numpy.ndarray.')
    if x1_arrays.shape[0] != x2_arrays.shape[0]:
        raise RuntimeError(f'The number of arrays in x1_arrays '
                           f'({x1_arrays.shape[0]}) does not match the '
                           f'number of arrays in x2_arrays '
                           f'({x2_arrays.shape[0]}).')
    else:
        n_arrays = x1_arrays.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute mean rotation angle (degrees)
    mean_angle_deg = \
        np.mean([rotation_angle_2d(x1_arrays[i, :], x2_arrays[i, :])
                 for i in range(n_arrays)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mean_angle_deg