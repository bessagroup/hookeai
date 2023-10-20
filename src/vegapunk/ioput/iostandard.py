"""File and directory operations.

This module includes standard operations to handle files and directories.

Functions
---------
make_directory
    Create a directory.
new_file_path_with_int
    Generate new and non-existent file path by extending with an integer.
write_summary_file
    Write summary data file with provided keyword-based parameters.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Alpha'
# =============================================================================
def make_directory(directory, is_overwrite=False):
    """Create a directory.

    Parameters
    ----------
    directory : str
        Directory path (without trailing slash).
    is_overwrite : bool, default=False
        If True, then directory overwrites existing directory with the same
        name. If False, then directory is extended with '_int' until
        non-existing directory is found, where int is an integer starting as 1.

    Returns
    -------
    directory : str
        Created directory path.
    """
    # Get absolute path
    directory = os.path.abspath(directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if os.path.isdir(directory):
        if is_overwrite:
            # Remove existing directory
            shutil.rmtree(directory)
        else:
            # Search for available directory name by extending it with an
            # integer
            new_directory = directory
            i = 1
            while os.path.isdir(new_directory):
                new_directory = directory + '_' + str(i)
                i += 1
            # Set new directory
            directory = new_directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create directory
    os.makedirs(directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return directory
# =============================================================================
def new_file_path_with_int(file_path):
    """Generate new and non-existent file path by extending with an integer.
    
    Parameters
    ----------
    file_path : str
        File path.
    
    Returns
    -------
    new_file_path : str
        File path extended with an integer.
    """
    # Get file path root and extension
    root, ext = os.path.splitext(file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize new file path
    new_file_path = file_path
    # Initialize extension integer
    i = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find non-existent file path
    while os.path.isfile(new_file_path):
        new_file_path = root + '_' + str(i) + ext
        i += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return new_file_path
# =============================================================================
def write_summary_file(summary_directory, filename='summary',
                       summary_title=None, **kwargs):
    """Write summary data file with provided keyword-based parameters.
    
    Parameters
    ----------
    summary_directory : str
        Directory where the summary file is written.
    filename : str, default='summary'
        Summary file name.
    summary_title : str, default=None
        Header of summary file.
    kwargs: dict
        Keyword-based parameters to be written in summary file.
    """
    # Check summary file directory
    if not os.path.isdir(summary_directory):
        raise RuntimeError('The summary file directory has not been '
                           'found:\n\n' + summary_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set summary file path
    summary_file_path = os.path.join(os.path.normpath(summary_directory),
                                     str(filename) + '.dat')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize summary file content
    summary = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set summary file title
    if isinstance(summary_title, str):
        summary += [f'{summary_title}\n', len(summary_title)*'-' + '\n']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over summary parameters
    for key1, val1 in kwargs.items():
        # Append parameter
        if isinstance(val1, dict):
            summary.append(f'\n"{str(key1)}":\n')
            for key2, val2 in val1.items():
                if isinstance(val2, dict):
                    summary.append(f'\n  "{str(key2)}":\n')
                    for key3, val3 in val2.items():
                        summary.append(f'    "{str(key3)}": {str(val3)}\n')
                else:
                    summary.append(f'  "{str(key2)}": {str(val2)}\n')
        else:
            summary.append(f'\n"{str(key1)}":  {str(val1)}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    open(summary_file_path, 'w').writelines(summary)