"""Test file and directory operations."""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import pytest
# Local
from src.vegapunk.ioput.iostandard import make_directory, \
    new_file_path_with_int
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
def test_make_directory(tmp_path):
    """Test directory creation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_directory = os.path.join(os.path.normpath(tmp_path), 'test_directory')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    directory = make_directory(test_directory)
    if not os.path.isdir(directory):
        errors.append('Failed creation of new directory.')
    directory = make_directory(test_directory, is_overwrite=False)
    if not os.path.isdir(directory):
        errors.append('Failed creation of new directory (existing directory, '
                      'no overwrite)')
    directory = make_directory(test_directory, is_overwrite=True)
    if not os.path.isdir(directory):
        errors.append('Failed creation of new directory (existing directory, '
                      'overwrite)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_new_file_path_with_int(tmp_path):
    """Test generation of new and non-existent file path."""
    file_path, new_file_path = ('test_path.dat', 'test_path.dat')
    test_new_file_path = new_file_path_with_int(
        os.path.join(os.path.normpath(tmp_path), file_path))
    target_new_file_path = os.path.join(os.path.normpath(tmp_path),
                                        new_file_path)
    assert test_new_file_path == target_new_file_path, 'Failed to generate ' \
        'non-existent file path.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    open(test_new_file_path, 'w').close()
    file_path, new_file_path = ('test_path.dat', 'test_path_1.dat')
    test_new_file_path = new_file_path_with_int(
        os.path.join(os.path.normpath(tmp_path), file_path))
    target_new_file_path = os.path.join(os.path.normpath(tmp_path),
                                        new_file_path)
    assert test_new_file_path == target_new_file_path, 'Failed to generate ' \
        'non-existent file path (extension with integer).'

