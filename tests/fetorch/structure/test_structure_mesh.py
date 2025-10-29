"""Test StructureMesh class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.hookeai.simulators.fetorch.structure.structure_mesh import StructureMesh
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def test_update_mesh_configuration(toy_uniaxial_specimen_2d_quad4,
                                   toy_uniaxial_specimen_3d_hexa8):
    """Test update of finite element mesh configuration from displacements."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set structures finite element meshes
    structures_meshes = (toy_uniaxial_specimen_2d_quad4,
                         toy_uniaxial_specimen_3d_hexa8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over structures
    for structure_mesh in structures_meshes:
        # Get structure initial configuration
        nodes_coords_mesh_init, _ = \
            structure_mesh.get_mesh_configuration(time='init')
        # Set random displacements
        random_nodes_disps_mesh = torch.rand(nodes_coords_mesh_init.shape)
        # Update finite element mesh configuration (last converged)
        structure_mesh.update_mesh_configuration(random_nodes_disps_mesh,
                                                 time='last',
                                                 is_update_coords=True)
        # Get finite element mesh configuration (last converged)
        nodes_coords_mesh_old, nodes_disps_mesh_old = \
            structure_mesh.get_mesh_configuration(time='last')
        # Check finite element mesh configuration update (last converged)
        if not torch.allclose(nodes_coords_mesh_old,
                            nodes_coords_mesh_init + random_nodes_disps_mesh):
            raise RuntimeError('Finite element mesh last converged nodes '
                            'coordinates were not properly updated.')
        if not torch.allclose(nodes_disps_mesh_old, random_nodes_disps_mesh):
            raise RuntimeError('Finite element mesh last converged nodes '
                            'coordinates were not properly updated.')
        # Update finite element mesh configuration (current)
        structure_mesh.update_mesh_configuration(random_nodes_disps_mesh,
                                                 time='current',
                                                 is_update_coords=True)
        # Get finite element mesh configuration (current)
        nodes_coords_mesh, nodes_disps_mesh = \
            structure_mesh.get_mesh_configuration(time='current')
        # Check finite element mesh configuration update (current)
        if not torch.allclose(nodes_coords_mesh,
                            nodes_coords_mesh_init + random_nodes_disps_mesh):
            raise RuntimeError('Finite element mesh current nodes '
                            'coordinates were not properly updated.')
        if not torch.allclose(nodes_disps_mesh, random_nodes_disps_mesh):
            raise RuntimeError('Finite element mesh current nodes '
                            'coordinates were not properly updated.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_element_assembler_1d(toy_uniaxial_specimen_2d_quad4):
    """Test finite element mesh assembly of 1D element arrays."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set structure finite element mesh
    structure_mesh = toy_uniaxial_specimen_2d_quad4
    # Get number of elements
    n_elem = structure_mesh.get_n_elem()
    # Get elements type
    elements_type = structure_mesh.get_elements_type()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize random 1D elements arrays
    random_elements_array = {}
    # Set random 1D elements arrays
    for i in range(n_elem):
        # Get element type
        element_type = elements_type[str(i + 1)]
        # Get element properties
        n_node = element_type.get_n_node()
        n_dof_node = element_type.get_n_dof_node()
        # Generate random 1D element array
        random_elements_array[str(i + 1)] = torch.rand(n_node*n_dof_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble element level arrays
    mesh_array = structure_mesh.element_assembler(random_elements_array)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check assembly by probing non-connected nodes
    for mnode, elem_id, enode in ((1, 1, 1), (5, 4, 2), (6, 1, 4), (10, 4, 3)):
        # Loop over degrees of freedom
        for j in range(n_dof_node):
            # Get mesh and local indexes
            midx = (mnode - 1)*n_dof_node + j
            lidx = (enode - 1)*n_dof_node + j
            # Check assemble degree of freedom
            if not torch.isclose(mesh_array[midx],
                                 random_elements_array[str(elem_id)][lidx]):
                errors.append(f'Incorrect assembly of 1D element arrays in '
                              f'finite element mesh node {mnode} '
                              f'(dim {j + 1}).')
    # Check assembly by probing connected nodes
    for mnode, elem1_id, e1node, elem2_id, e2node in (
        (2, 1, 2, 2, 1), (3, 2, 2, 3, 1), (4, 3, 2, 4, 1), (7, 1, 3, 2, 4),
        (8, 2, 3, 3, 4), (9, 3, 3, 4, 4)):
        # Loop over degrees of freedom
        for j in range(n_dof_node):
            # Get mesh and local indexes
            midx = (mnode - 1)*n_dof_node + j
            l1idx = (e1node - 1)*n_dof_node + j
            l2idx = (e2node - 1)*n_dof_node + j
            # Check assemble degree of freedom
            if not torch.isclose(
                mesh_array[midx],
                (random_elements_array[str(elem1_id)][l1idx]
                 + random_elements_array[str(elem2_id)][l2idx])):
                errors.append(f'Incorrect assembly of 1D element arrays in '
                              f'finite element mesh node {mnode} '
                              f'(dim {j + 1}).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_element_extractor_1d(toy_uniaxial_specimen_2d_quad4):
    """Test finite element mesh extraction of 1D element arrays."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set structure finite element mesh
    structure_mesh = toy_uniaxial_specimen_2d_quad4
    # Get number of nodes
    n_node_mesh = structure_mesh.get_n_node_mesh()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random 1D mesh array
    random_mesh_array = torch.rand(n_node_mesh*2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over elements
    for elem_id, idx1, idx2, idx3, idx4 in (
        (1, 0, 2, 12, 10), (2, 2, 4, 14, 12), (3, 4, 6, 16, 14),
        (4, 6, 8, 18, 16)):
        # Extract 1D element array
        element_array = structure_mesh.element_extractor(random_mesh_array,
                                                         elem_id)
        # Set expected 1D element array
        element_array_sol = torch.cat((random_mesh_array[idx1:idx1 + 2],
                                       random_mesh_array[idx2:idx2 + 2],
                                       random_mesh_array[idx3:idx3 + 2],
                                       random_mesh_array[idx4:idx4 + 2]))
        # Check extracted 1D element array
        if not torch.allclose(element_array, element_array_sol):
            errors.append(f'Incorrect extraction of 1D element array of '
                          f'element {elem_id}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
    