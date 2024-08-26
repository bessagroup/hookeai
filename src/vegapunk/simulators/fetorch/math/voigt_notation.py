"""FETorch: Strain/Stress tensors Voigt notation matricial storage.

This module contains standard finite element procedures associated with the
storage of strain/stress tensorial quantities following the Voigt notation.

Functions
---------
get_strain_from_vfm
    Recover strain tensor from associated Voigt matricial form.
vget_strain_from_vmf
    Recover strain tensor from associated Voigt matricial form (vectorized).
get_stress_vfm
    Get stress tensor Voigt matricial form.
vget_stress_vmf
    Get stress tensor Voigt matricial form (vectorized).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools as it
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
def get_strain_from_vfm(strain_vmf, n_dim, comp_order_sym, device=None):
    """Recover strain tensor from associated Voigt matricial form.

    Parameters
    ----------
    strain_vmf : torch.Tensor(1d)
        Strain tensor stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    strain : torch.Tensor(2d)
        Strain tensor recovered from Voigt matricial form.
    """
    # Get device from input strain tensor
    if device is None:
        device = strain_vmf.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    strain = torch.zeros((n_dim, n_dim), dtype=torch.float, device=device)
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
def vget_strain_from_vmf(strain_vmf, n_dim, comp_order_sym, device=None):
    """Recover strain tensor from associated Voigt matricial form.

    Parameters
    ----------
    strain_vmf : torch.Tensor(1d)
        Strain tensor stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    strain : torch.Tensor(2d)
        Strain tensor recovered from Voigt matricial form.
    """
    # Get device from input strain tensor
    if device is None:
        device = strain_vmf.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set row major components order     
    row_major_order = tuple(f'{i + 1}{j + 1}'
                            for i, j in it.product(range(n_dim), repeat=2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing inverse Voigt factor
    index_voigt_inv = torch.tensor(
        [1.0/2.0 if x[0] != x[1] else 1.0 for x in comp_order_sym],
        dtype=torch.float, device=device)
    # Build indexing mapping
    index_map = [comp_order_sym.index(x) if x in comp_order_sym
                 else comp_order_sym.index(x[::-1]) for x in row_major_order]
    # Get tensor from matricial form
    strain = \
        torch.mul(strain_vmf, index_voigt_inv)[index_map].view(n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain
# =============================================================================
def get_stress_vfm(stress, n_dim, comp_order_sym, device=None):
    """Get stress tensor Voigt matricial form.

    Parameters
    ----------
    stress : torch.Tensor(2d)
        Stress tensor to be stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    stress_vmf : torch.Tensor(1d)
        Stress tensor stored in Voigt matricial form.
    """
    # Get device from input stress tensor
    if device is None:
        device = stress.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    stress_vmf = torch.zeros(len(comp_order_sym),
                             dtype=torch.float, device=device)
    # Store stress tensor in Voigt matricial form
    for i in range(len(mf_indexes)):
        mf_idx = mf_indexes[i]
        so_idx = tuple(so_indexes[i])
        stress_vmf[mf_idx] = stress[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_vmf
# =============================================================================
def vget_stress_vmf(stress, n_dim, comp_order_sym, device=None):
    """Get stress tensor Voigt matricial form (vectorized).

    Parameters
    ----------
    stress : torch.Tensor(2d)
        Stress tensor to be stored in Voigt matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    stress_vmf : torch.Tensor(1d)
        Stress tensor stored in Voigt matricial form.
    """
    # Get device from input stress tensor
    if device is None:
        device = stress.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = \
        tuple([int(x[i]) - 1 for x in comp_order_sym] for i in range(2))
    # Compute tensor Voigt matricial form
    stress_vmf = stress[index_map]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_vmf