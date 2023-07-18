"""Test FiniteElement class."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
# Local
from src.vegapunk.finite_element import FiniteElement
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
@pytest.mark.parametrize('elem_type', ['unknown_type',])
def test_unavailable_elem_type(elem_type):
    """Test unavailable finite element type initialization."""
    with pytest.raises(RuntimeError):
        FiniteElement(elem_type=elem_type)
# -----------------------------------------------------------------------------
def test_is_available_elem_type(available_elem_type):
    """Test finite element type availability."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for elem_type in available_elem_type:
        if not FiniteElement._is_available_elem_type(elem_type) == True:
            errors.append('Available finite element type (' + str(elem_type)
                          + ') not recognized.')
    if not FiniteElement._is_available_elem_type('unknown_type') == False:
        errors.append('Unavailable finite element type (' + str(elem_type)
                      + ') not detected.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_elem_type_attributes(available_elems):
    """Test finite element type attributes."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over available finite elements
    for elem in available_elems:
        # Element type identifier
        elem_str = str(elem._elem_type) + ': '
        # Test finite element type attributes
        if not isinstance(elem._n_dim, int):
            errors.append(elem_str
                          + 'Number of spatial dimensions is not an int.')
        if not isinstance(elem._n_nodes, int):
            errors.append(elem_str + 'Number of nodes is not an int.')
        if not isinstance(elem._n_edges, int):
            errors.append(elem_str + 'Number of edges is not an int.')
        if not isinstance(elem._n_edge_nodes, int):
            errors.append(elem_str + 'Number of nodes per edge is not an int.')
        if not isinstance(elem._nodes_matrix, np.ndarray):
            errors.append(elem_str
                          + 'Element nodes matrix is not a numpy array.')
        else:
            if elem._nodes_matrix.shape != elem._n_dim*(elem._n_edge_nodes,):
                errors.append(elem_str
                              + 'Invalid shape of element nodes matrix.')
            if elem._nodes_matrix.dtype != int:
                errors.append(elem_str
                              + 'Invalid type of element nodes matrix.')
            if not set(elem._nodes_matrix.flatten()).issubset(
                    set([x for x in range(elem._n_nodes + 1)])):
                errors.append(elem_str
                              + 'Element nodes matrix does not include all '
                              'element nodes.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test unknown finite element type
    elem = available_elems[0]
    elem._elem_type = 'unknown_type'
    with pytest.raises(RuntimeError):
        elem._set_elem_type_attributes()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_get_n_dim_type(available_elems):
    """Test number of spatial dimensions getter type."""
    assert all([isinstance(elem.get_n_dim(), int)
                for elem in available_elems]), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_n_nodes_type(available_elems):
    """Test number of nodes getter type."""
    assert all([isinstance(elem.get_n_nodes(), int)
                for elem in available_elems]), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_n_edge_nodes_type(available_elems):
    """Test number of nodes per edge getter type."""
    assert all([isinstance(elem.get_n_edge_nodes(), int)
                for elem in available_elems]), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_nodes_matrix_type(available_elems):
    """Test element nodes matrix getter type."""
    assert all([isinstance(elem.get_nodes_matrix(), np.ndarray)
                for elem in available_elems]), 'Unexpected return type.'
    assert all([elem.get_nodes_matrix().dtype == int
                for elem in available_elems]), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_node_label_index_type(available_elems):
    """Test nodel label index getter type."""
    assert all([isinstance(elem.get_node_label_index(1), tuple)
                for elem in available_elems]), 'Unexpected return type.'
# -----------------------------------------------------------------------------
def test_get_node_label_index_label_not_found(available_elems):
    """Test nodel label index getter label not found."""
    with pytest.raises(RuntimeError):
        for elem in available_elems:
            elem.get_node_label_index(elem._n_nodes + 1)