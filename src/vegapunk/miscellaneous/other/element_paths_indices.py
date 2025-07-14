"""Get finite element local paths indices.

Functions
---------
get_element_local_paths_idx
    Get finite element local paths global indices.
"""
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def get_element_local_paths_idx(element_id, n_gauss):
    """Get finite element local paths global indices.
    
    Parameters
    ----------
    element_id : int
        Element label. Elements labels must be within the range of
        1 to n_elem (included).
    n_gauss : int
        Number of Gauss quadrature integration points (per element).
        
    Returns
    -------
    local_paths_idx : tuple[int]
        Element local paths indices.
    """
    # Compute element local paths indices
    local_paths_idx = \
        tuple(range((element_id - 1)*n_gauss, element_id*n_gauss))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return local_paths_idx
# =============================================================================
if __name__ == "__main__":
    # Set element label
    element_id = 6
    # Set number of Gauss points per element
    n_gauss = 8
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get element local paths indices
    local_paths_idx = get_element_local_paths_idx(element_id, n_gauss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display results
    print(f'\nElement label:          {element_id}')
    print(f'\nNumber of Gauss points: {n_gauss}')
    print(f'\nLocal paths indices:    {local_paths_idx}')
    print()