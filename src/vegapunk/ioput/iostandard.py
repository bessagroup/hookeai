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
find_unique_file_with_regex(directory, regex)
    Find file in directory based on regular expression.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
import re
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
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
        elif isinstance(val1, np.ndarray):
            summary.append(f'\n"{str(key1)}":\n  {str(val1)}\n')
        elif isinstance(val1, str) and '\n' in val1:
            summary.append(f'\n"{str(key1)}":\n\n{str(val1)}\n')
        else:
            summary.append(f'\n"{str(key1)}":  {str(val1)}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    open(summary_file_path, 'w').writelines(summary)
# =============================================================================
def find_unique_file_with_regex(directory, regex):
    """Find file in directory based on regular expression.
    
    Returns first occurrence if file matching regular expression is found.
    
    Parameters
    ----------
    directory : str
        Directory where file is searched.
    regex : {str, tuple[str]}
        Regular expression to search for file. If tuple of regular expressions
        is provided, then search procedure loops over them.
        
    Returns
    -------
    is_file_found : bool
        True if file is found, False otherwise.
    file_path : {str, None}
        File path. Set to None if file is not found.
    """
    # Check directory
    if not os.path.isdir(directory):
        raise RuntimeError('The searching directory has not been found:\n\n'
                           + directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check regular expression
    if not isinstance(regex, str) and not isinstance(regex, tuple):
        raise RuntimeError('Regular expression(s) must be provided as str or '
                           'tuple[str].')
    elif (isinstance(regex, tuple)
          and not all([isinstance(x, str) for x in regex])):
        raise RuntimeError('Regular expression(s) must be provided as str or '
                           'tuple[str].')
    else:
        if isinstance(regex, str):
            # Convert to tuple
            regex = (regex,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check regular expression pattern
    try:
        for pattern in regex:
            re.compile(pattern)
    except re.error:
        raise RuntimeError('Invalid regular expression.')    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get files in directory
    directory_list = os.listdir(directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize file found flag
    is_file_found = False
    # Loop over training regex
    for pattern in regex:
        # Loop over files
        for filename in directory_list:
            # Check if file meeting pattern has been found
            is_file_found = bool(re.search(pattern, filename))
            # Leave searching loop when file is found
            if is_file_found:
                break
        # Leave searching loop when file is found
        if is_file_found:
            break
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set file path
    if is_file_found:
        dataset_file_path = os.path.join(os.path.normpath(directory),
                                         filename)
    else:
        dataset_file_path = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_file_found, dataset_file_path