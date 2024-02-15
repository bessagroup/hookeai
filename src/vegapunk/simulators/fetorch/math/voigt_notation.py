"""FETorch: Strain/Stress tensors Voigt notation matricial storage.

This module contains standard finite element procedures associated with the
storage of strain/stress tensorial quantities following the Voigt notation.

Functions
---------
get_strain_from_vfm
    Recover strain tensor from associated Voigt matricial form.
get_stress_vfm
    Get stress tensor Voigt matricial form.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def get_strain_from_vfm(strain_vmf, n_dim, comp_order_sym):
    """Recover strain tensor from associated Voigt matricial form.

    Parameters
    ----------
    strain_vmf : torch.Tensor(1d)
        Strain tensor stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.

    Returns
    -------
    strain : torch.Tensor(2d)
        Strain tensor recovered from Voigt matricial form.
    """
    # Check strain tensor Voigt matricial form
    if not isinstance(strain_vmf, torch.Tensor):
        raise RuntimeError('Strain tensor Voigt matricial form must be '
                           'torch.Tensor.')
    elif len(strain_vmf.shape) != 1:
        raise RuntimeError('Strain tensor Voigt matricial form must be '
                           'torch.Tensor(1d).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if any([int(x) not in range(1, n_dim + 1)
            for x in list(''.join(comp_order_sym))]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif any([len(comp) != 2 for comp in comp_order_sym]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif len(list(dict.fromkeys(comp_order_sym))) != len(comp_order_sym):
        raise RuntimeError('Duplicated component in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set second-order and matricial form indexes
    so_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order_sym)):
        so_indexes.append([int(x) - 1 for x in list(comp_order_sym[i])])
        mf_indexes.append(comp_order_sym.index(comp_order_sym[i]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain tensor
    strain = torch.zeros((n_dim, n_dim), dtype=torch.float)
    # Get strain tensor from matricial form
    for i in range(len(mf_indexes)):
        mf_idx = mf_indexes[i]
        so_idx = tuple(so_indexes[i])
        factor = 1.0
        if so_idx[0] != so_idx[1]:
            factor = 2.0
            strain[so_idx[::-1]] = (1.0/factor)*strain_vmf[mf_idx]
        strain[so_idx] = (1.0/factor)*strain_vmf[mf_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain
# =============================================================================
def get_stress_vfm(stress, n_dim, comp_order_sym):
    """Get stress tensor Voigt matricial form.

    Parameters
    ----------
    stress : torch.Tensor(2d)
        Stress tensor to be stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.

    Returns
    -------
    stress_vmf : torch.Tensor(1d)
        Stress tensor stored in Voigt matricial form.
    """
    # Check stress tensor
    if not isinstance(stress, torch.Tensor):
        raise RuntimeError('Stress tensor must be torch.Tensor.')
    elif len(stress.shape) != 2:
        raise RuntimeError('Stress tensor must be torch.Tensor(2d).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if any([int(x) not in range(1, n_dim + 1)
            for x in list(''.join(comp_order_sym))]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif any([len(comp) != 2 for comp in comp_order_sym]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif len(list(dict.fromkeys(comp_order_sym))) != len(comp_order_sym):
        raise RuntimeError('Duplicated component in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set second-order and matricial form indexes
    so_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order_sym)):
        so_indexes.append([int(x) - 1 for x in list(comp_order_sym[i])])
        mf_indexes.append(comp_order_sym.index(comp_order_sym[i]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize stress tensor Voigt matricial form
    stress_vmf = torch.zeros(len(comp_order_sym), dtype=torch.float)
    # Store stress tensor in Voigt matricial form
    for i in range(len(mf_indexes)):
        mf_idx = mf_indexes[i]
        so_idx = tuple(so_indexes[i])
        stress_vmf[mf_idx] = stress[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_vmf