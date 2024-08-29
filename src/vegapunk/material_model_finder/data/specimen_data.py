"""DARPA METALS PROJECT: Specimen numerical data.

Classes
-------
SpecimenNumericalData
    Specimen numerical data translated from experimental results.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
# Local
from simulators.fetorch.structure.structure_mesh import StructureMesh
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class SpecimenNumericalData:
    """Specimen numerical data translated from experimental results.
    
    After discretizing the specimen in a suitable finite element mesh, the
    experimental results are translated to numerical data, namely the
    nodes displacement history and reaction forces history.
    
    Attributes
    ----------
    specimen_mesh : StructureMesh
        Specimen finite element mesh.
    nodes_disps_mesh_hist : torch.Tensor(3d)
        Displacements history of finite element mesh nodes stored as
        torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
    reaction_forces_mesh_hist : torch.Tensor(3d)
        Reaction forces (Dirichlet boundary conditions) history of finite
        element mesh nodes stored as torch.Tensor(3d) of shape
        (n_node_mesh, n_dim, n_time).
    time_hist : torch.Tensor(1d)
        Discrete time history.

    Methods
    -------
    set_specimen_mesh(self, nodes_coords_mesh_init, elements_type, \
                      connectivities)
        Set the specimen finite element mesh.
    set_specimen_data(self, nodes_disps_mesh_hist, reaction_forces_mesh_hist)
        Set specimen numerical data translated from experimental results.
    update_specimen_mesh_configuration(self, time_idx, is_update_coords=True)
        Update the specimen mesh configuration for given discrete time.
    get_batched_mesh_configuration_hist(self, is_update_coords=True)
        Get batched finite element mesh configuration history.
    get_element_nodes_field_hist(self, element_nodes, nodes_field_mesh_hist)
        Get field history of finite element nodes.
    get_n_dim(self)
        Get number of spatial dimensions.
    """
    def __init__(self):
        """Constructor."""
        # Initialize specimen finite element mesh
        self.specimen_mesh = None
        # Initialize specimen data
        self.nodes_disps_mesh_hist = None
        self.reaction_forces_mesh_hist = None
        self.time_hist = None
    # -------------------------------------------------------------------------
    def set_specimen_mesh(self, nodes_coords_mesh_init, elements_type,
                          connectivities, dirichlet_bool_mesh):
        """Set the specimen finite element mesh.
        
        Parameters
        ----------
        nodes_coords_mesh_init : torch.Tensor(2d)
            Initial coordinates of finite element mesh nodes stored as
            torch.Tensor(2d) of shape (n_node_mesh, n_dim).
        elements_type : dict
            FETorch element type (item, ElementType) of each finite element
            mesh element (str[int]). Elements labels must be within the range
            of 1 to n_elem (included).
        connectivities : dict
            Nodes (item, tuple[int]) of each finite element mesh element
            (key, str[int]). Nodes must be within the range of 1 to n_node_mesh
            (included). Elements labels must be within the range of 1 to n_elem
            (included).
        dirichlet_bool_mesh : torch.Tensor(2d)
            Degrees of freedom of finite element mesh subject to Dirichlet
            boundary conditions. Stored as torch.Tensor(2d) of shape
            (n_node_mesh, n_dim) where constrained degrees of freedom are
            labeled 1, otherwise 0.
        """
        # Set specimen finite element mesh
        self.specimen_mesh = StructureMesh(
            nodes_coords_mesh_init, elements_type, connectivities,
            dirichlet_bool_mesh)
    # -------------------------------------------------------------------------
    def set_specimen_data(self, nodes_disps_mesh_hist,
                          reaction_forces_mesh_hist, time_hist):
        """Set specimen numerical data translated from experimental results.
        
        Parameters
        ----------
        nodes_disps_mesh_hist : torch.Tensor(3d)
            Displacements history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        reaction_forces_mesh_hist : torch.Tensor(3d)
            Reaction forces (Dirichlet boundary conditions) history of finite
            element mesh nodes stored as torch.Tensor(3d) of shape
            (n_node_mesh, n_dim, n_time).
        time_hist : torch.Tensor(1d)
            Discrete time history.
        """
        # Check specimen finite element mesh
        if self.specimen_mesh is None:
            raise RuntimeError('The experimental specimen finite element '
                               'mesh must be set before setting the '
                               'corresponding experimental data.')
        else:
            # Get number of spatial dimensions
            n_dim = self.specimen_mesh.get_n_dim()
            # Get number of nodes of finite element mesh
            n_node_mesh = self.specimen_mesh.get_n_node_mesh()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        data_labels = ('nodes displacement history',
                       'nodes reaction forces history')
        # Check specimen data
        for i, tensor in enumerate((nodes_disps_mesh_hist,
                                    reaction_forces_mesh_hist)):
            if (not isinstance(tensor, torch.Tensor)
                or (len(tensor.shape) != 3)):
                raise RuntimeError(f'The {data_labels[i]} must be provided as '
                                   f' a torch.Tensor(3d) of shape '
                                   f'(n_node_mesh, n_dim, n_time).')
            elif tensor.shape[0] != n_node_mesh:
                raise RuntimeError(f'The {data_labels[i]} ({tensor.shape[0]}) '
                                   f'is not consistent with the number of '
                                   f'nodes of the specimen mesh '
                                   f'({n_node_mesh}).')
            elif tensor.shape[1] != n_dim:
                raise RuntimeError(f'The {data_labels[i]} ({tensor.shape[1]}) '
                                   f'is not consistent with the number of '
                                   f'spatial dimensions of the specimen mesh '
                                   f'({n_dim}).')
        if not isinstance(time_hist, torch.Tensor):
            raise RuntimeError(f'The discrete time history must be provided '
                               f'as a torch.Tensor(1d).')
        elif (nodes_disps_mesh_hist.shape[2] != time_hist.shape[0]
              or reaction_forces_mesh_hist.shape[2] != time_hist.shape[0]):
            raise RuntimeError(
                f'The time history length of the {data_labels[0]} '
                f'({nodes_disps_mesh_hist.shape[2]}) or the {data_labels[1]} '
                f'is not consistent the discrete time history '
                f'({time_hist.shape[0]}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store nodes displacements history
        self.nodes_disps_mesh_hist = nodes_disps_mesh_hist
        # Store nodes reaction forces history
        self.reaction_forces_mesh_hist = reaction_forces_mesh_hist
        # Store discrete time history
        self.time_hist = time_hist
    # -------------------------------------------------------------------------
    def update_specimen_mesh_configuration(self, time_idx,
                                           is_update_coords=True):
        """Update the specimen mesh configuration for given discrete time.
        
        For a given discrete time, the known nodes displacement history is used
        to update the specimen finite element mesh nodes coordinates and
        displacements. A similar update is performed for the last converged
        mesh configuration.
        
        Parameters
        ----------
        time_idx : int
            Discrete time index.
        is_update_coords : bool, default=True
            If False, then only updates the displacements of the finite element
            mesh nodes, leaving the nodes coordinates unchanged. If True, then
            update both coordinates and displacements of finite element mesh
            nodes.
        """
        # Check discrete time index
        if time_idx < 0 or time_idx >= self.time_hist.shape[0]:
            raise RuntimeError(f'The discrete time index ({time_idx}) '
                               f'must be non-negative integer lower than '
                               f'the discrete time history length '
                               f'({self.time_hist.shape[0]}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set last converged time index
        time_idx_old = max((0, time_idx - 1))
        # Update specimen last converged mesh configuration
        self.specimen_mesh.update_mesh_configuration(
            self.nodes_disps_mesh_hist[:, :, time_idx_old], time='last',
            is_update_coords=is_update_coords)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update specimen mesh current configuration
        self.specimen_mesh.update_mesh_configuration(
            self.nodes_disps_mesh_hist[:, :, time_idx], time='current',
            is_update_coords=is_update_coords)
    # -------------------------------------------------------------------------
    def get_batched_mesh_configuration_hist(self, is_update_coords=True):
        """Get batched finite element mesh configuration history.
        
        Batching operation over elements requires that all finite element mesh
        elements share the same element type (number of nodes).
        
        Parameters
        ----------
        is_update_coords : bool, default=True
            If False, then finite element mesh nodes coordinates are kept fixed
            throughout history. If True, then finite element mesh nodes
            coordinates are computed from nodes displacements history.
        
        Returns
        -------
        elements_coords_hist : torch.Tensor(4d)
            Coordinates history of finite element mesh elements nodes stored
            as torch.Tensor(4d) of shape (n_elem, n_node, n_dim, n_time).
        elements_disps_hist : torch.Tensor(4d)
            Displacements history of finite element mesh elements nodes stored
            as torch.Tensor(4d) of shape (n_elem, n_node, n_dim, n_time).        
        """
        # Build connectivities tensor
        connectivities_tensor = self.specimen_mesh.get_connectivities_tensor()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized element field history computation (batch along
        # element)
        vmap_get_element_nodes_field_hist = \
            torch.vmap(self.get_element_nodes_field_hist,
                       in_dims=(0, None), out_dims=(0,))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get batched finite element mesh elements displacement history
        elements_disps_hist = vmap_get_element_nodes_field_hist(
            connectivities_tensor, self.nodes_disps_mesh_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get finite element mesh nodes initial coordinates
        nodes_coords_mesh_init, _ = \
            self.specimen_mesh.get_mesh_configuration(time='init')
        # Build finite element mesh nodes coordinates history
        nodes_coords_mesh_hist = nodes_coords_mesh_init.unsqueeze(2).expand(
            -1, -1, len(self.time_hist))
        if is_update_coords:
            nodes_coords_mesh_hist = \
                nodes_coords_mesh_hist + self.nodes_disps_mesh_hist
        # Get batched finite element mesh elements coordinates history
        elements_coords_hist = vmap_get_element_nodes_field_hist(
            connectivities_tensor, nodes_coords_mesh_hist)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elements_coords_hist, elements_disps_hist
    # -------------------------------------------------------------------------
    def get_element_nodes_field_hist(self, element_nodes,
                                     nodes_field_mesh_hist):
        """Get field history of finite element nodes.
        
        Parameters
        ----------
        element_nodes : torch.Tensor(1d)
            Finite element mesh element connectitivities stored as
            torch.Tensor(1d) of shape (n_node,). Nodes are labeled from
            1 to n_node_mesh.
        nodes_field_mesh_hist : torch.Tensor(3d)
            Field history of finite element mesh nodes stored as
            torch.Tensor(3d) of shape (n_node_mesh, n_dim, n_time).
        
        Returns
        -------
        element_nodes_hist : torch.Tensor(3d)
            Field history of finite element nodes stored as
            torch.Tensor(3d) of shape (n_node, n_dim, n_time).
        """
        # Get finite element nodes field history
        element_nodes_hist = nodes_field_mesh_hist[element_nodes - 1, :, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_nodes_hist
    # -------------------------------------------------------------------------
    def get_n_dim(self):
        """Get number of spatial dimensions.
        
        Returns
        -------
        n_dim : int
            Number of spatial dimensions.
        """
        # Get number of spatial dimensions
        n_dim = self.specimen_mesh.get_n_dim()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return n_dim
        