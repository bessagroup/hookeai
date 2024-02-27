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