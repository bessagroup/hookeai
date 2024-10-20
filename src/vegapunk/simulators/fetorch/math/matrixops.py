"""FETorch: Strain/Stress tensors matricial storage and procedures.

This module contains fundamental procedures associated with the matricial
storage of strain/stress tensorial quantities and related manipulations.

Functions
---------
get_problem_type_parameters
    Get parameters dependent on the problem type.
get_tensor_mf
    Get tensor matricial form.
vget_tensor_mf
    Get tensor matricial form.
get_tensor_from_mf
    Recover tensor from associated matricial form.
vget_tensor_from_mf
    Recover tensor from associated matricial form.
kelvin_factor
    Get Kelvin notation coefficient of given strain/stress component.
get_state_3Dmf_from_2Dmf
    Build 3D counterpart of 2D strain/stress second-order tensor.
vget_state_3Dmf_from_2Dmf
    Build 3D counterpart of 2D second-order tensor.
get_state_2Dmf_from_3Dmf
    Build 2D counterpart of 3D strain/stress second- or fourth-order tensor.
vget_state_2Dmf_from_3Dmf
    Build 2D counterpart of 3D second- or fourth-order tensor.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import numpy as np
import itertools as it
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                       Problem type parameters
# =============================================================================
def get_problem_type_parameters(problem_type):
    """Get parameters dependent on the problem type.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric
        (3) and 3D (4).

    Returns
    -------
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : tuple
        Strain/Stress components symmetric order.
    comp_order_nsym : tuple
        Strain/Stress components nonsymmetric order.
    """
    # Set problem number of spatial dimensions and strain/stress components
    # symmetric and nonsymmetric order
    if problem_type == 1:
        n_dim = 2
        comp_order_sym = ('11', '22', '12')
        comp_order_nsym = ('11', '21', '12', '22')
    elif problem_type == 4:
        n_dim = 3
        comp_order_sym = ('11', '22', '33', '12', '23', '13')
        comp_order_nsym = ('11', '21', '31', '12', '22', '32', '13', '23',
                           '33')
    else:
        raise RuntimeError('Unavailable problem type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_dim, comp_order_sym, comp_order_nsym
#
#                                        Tensorial - Matricial forms conversion
# =============================================================================
def get_tensor_mf(tensor, n_dim, comp_order, is_kelvin_notation=True,
                  device=None):
    """Get tensor matricial form.

    Store a given second-order or fourth-order tensor in matricial form for a
    given number of problem spatial dimensions and given ordered strain/stress
    components list. If the second-order tensor is symmetric or the
    fourth-order tensor has minor symmetry (component list only contains
    independent components), then the Kelvin notation[#]_ is employed to
    perform the storage by default.

    .. [#] Nagel, T., Görke, U.-J., Moerman, K. M., and Kolditz, O. (2016). On
           advantages of the Kelvin mapping in finite element implementations
           of deformation processes. Environmental Earth Sciences, 75(11):937
           (see `here <https://dspace.mit.edu/handle/1721.1/105251>`_)

    ----

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : tuple
        Strain/Stress components order associated to matricial form.
    is_kelvin_notation : bool, default=True
        If True, then Kelvin notation is employed to store symmetric tensors in
        matricial form. If False, then tensor components are stored unchanged.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    tensor_mf : torch.Tensor
        Matricial form of input tensor.
    """
    # Get device from input tensor
    if device is None:
        device = tensor.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor order
    tensor_order = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if tensor_order not in (2, 4):
        raise RuntimeError('Matricial form storage is only available for '
                           'second-order or fourth-order tensors.')
    elif any([int(x) not in range(1, n_dim + 1)
              for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif any([tensor.shape[i] != n_dim for i in range(len(tensor.shape))]):
        raise RuntimeError('Invalid tensor dimensions.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        pass
    else:
        raise RuntimeError('Invalid number of components in strain/stress '
                           'components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor matricial form
        if tensor.dtype == torch.cfloat:
            tensor_mf = torch.zeros(len(comp_order), dtype=torch.cfloat,
                                    device=device)
        else:
            tensor_mf = torch.zeros(len(comp_order), dtype=torch.float,
                                    device=device)
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if is_kelvin_notation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
            tensor_mf[mf_idx] = factor*tensor[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in
                               list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in [comp_order.index(comps[i][0]),
                                           comp_order.index(comps[i][1])]])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor matricial form
        if tensor.dtype == torch.cfloat:
            tensor_mf = torch.zeros((len(comp_order), len(comp_order)),
                                    dtype=torch.cfloat, device=device)
        else:
            tensor_mf = torch.zeros((len(comp_order), len(comp_order)),
                                    dtype=torch.float, device=device)
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if is_kelvin_notation and not (fo_idx[0] == fo_idx[1]
                                           and fo_idx[2] == fo_idx[3]):
                factor = \
                    factor*np.sqrt(2) if fo_idx[0] != fo_idx[1] else factor
                factor = \
                    factor*np.sqrt(2) if fo_idx[2] != fo_idx[3] else factor
            tensor_mf[mf_idx] = factor*tensor[fo_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor_mf
# =============================================================================
def vget_tensor_mf(tensor, n_dim, comp_order, is_kelvin_notation=True,
                   device=None):
    """Get tensor matricial form.

    Compatible with vectorized mapping.

    Store a given second-order or fourth-order tensor in matricial form for a
    given number of problem spatial dimensions and given ordered strain/stress
    components list. If the second-order tensor is symmetric or the
    fourth-order tensor has minor symmetry (component list only contains
    independent components), then the Kelvin notation[#]_ is employed to
    perform the storage by default.

    .. [#] Nagel, T., Görke, U.-J., Moerman, K. M., and Kolditz, O. (2016). On
        advantages of the Kelvin mapping in finite element implementations
        of deformation processes. Environmental Earth Sciences, 75(11):937
        (see `here <https://dspace.mit.edu/handle/1721.1/105251>`_)

    ----

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : tuple
        Strain/Stress components order associated to matricial form.
    is_kelvin_notation : bool, default=True
        If True, then Kelvin notation is employed to store symmetric tensors in
        matricial form. If False, then tensor components are stored unchanged.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    tensor_mf : torch.Tensor
        Matricial form of input tensor.
    """
    # Get device from input tensor
    if device is None:
        device = tensor.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor order
    tensor_order = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        pass
    else:
        raise RuntimeError('Invalid number of components in strain/stress '
                            'components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of components
    n_comps = len(comp_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store tensor according to tensor order
    if tensor_order == 2:
        # Build indexing mapping
        index_map = tuple(
            [int(x[i]) - 1 for x in comp_order] for i in range(2))
        # Build indexing Kelvin factor
        if is_kelvin_notation:
            index_kelvin = torch.tensor(
                [np.sqrt(2) if x[0] != x[1] else 1.0 for x in comp_order],
                dtype=torch.float, device=device)
        else:
            index_kelvin = torch.ones(n_comps, dtype=torch.float,
                                      device=device)
        # Compute tensor matricial form
        tensor_mf = torch.mul(tensor[index_map], index_kelvin)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif tensor_order == 4:
        # Build index mapping
        index_map = tuple(
            [sum([[int(x[0]) - 1,]*n_comps for x in comp_order], []),
             sum([[int(x[1]) - 1,]*n_comps for x in comp_order], []),
             [int(x[0]) - 1 for x in comp_order]*n_comps,
             [int(x[1]) - 1 for x in comp_order]*n_comps])
        # Build indexing Kelvin factor
        if is_kelvin_notation:
            index_kelvin_1d = torch.tensor(
                [np.sqrt(2) if x[0] != x[1] else 1.0 for x in comp_order],
                dtype=torch.float, device=device)
            index_kelvin = torch.outer(index_kelvin_1d, index_kelvin_1d)
        else:
            index_kelvin = torch.ones((n_comps, n_comps), dtype=torch.float,
                                      device=device)
        # Compute tensor matricial form
        tensor_mf = \
            torch.mul(tensor[index_map].view(-1, n_comps), index_kelvin)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor_mf
# =============================================================================
def get_tensor_from_mf(tensor_mf, n_dim, comp_order, is_kelvin_notation=True,
                       device=None):
    """Recover tensor from associated matricial form.

    Recover a given second-order or fourth-order tensor from the associated
    matricial form, given the problem number of spatial dimensions and given a
    (compatible) ordered strain/stress components list. If the second-order
    tensor is symmetric or the fourth-order tensor has minor symmetry
    (component list only contains independent components), then matricial form
    is assumed to follow the Kelvin notation [#]_ by default.

    .. [#] Nagel, T., Görke, U.-J., Moerman, K. M., and Kolditz, O. (2016). On
           advantages of the Kelvin mapping in finite element implementations
           of deformation processes. Environmental Earth Sciences, 75(11):937
           (see `here <https://dspace.mit.edu/handle/1721.1/105251>`_)

    ----

    Parameters
    ----------
    tensor_mf : torch.Tensor
        Tensor stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : tuple
        Strain/Stress components order associated to matricial form.
    is_kelvin_notation : bool, default=True
        If True, then Kelvin notation is employed to store symmetric tensors in
        matricial form. If False, then tensor components are stored unchanged.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    tensor : torch.Tensor
        Tensor recovered from matricial form.
    """
    # Get device from input tensor
    if device is None:
        device = tensor_mf.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
        if tensor_mf.shape[0] != n_dim**2 and \
                tensor_mf.shape[0] != sum(range(n_dim + 1)):
            raise RuntimeError('Invalid number of components in tensor '
                               'matricial form.')
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
        if tensor_mf.shape[0] != tensor_mf.shape[1]:
            raise RuntimeError('Fourth-order tensor matricial form must be a'
                               'square matrix.')
        elif tensor_mf.shape[0] != n_dim**2 and \
                tensor_mf.shape[0] != sum(range(n_dim + 1)):
            raise RuntimeError('Invalid number of components in tensor '
                               'matricial form.')
    else:
        raise RuntimeError('Tensor matricial form must be a vector or a '
                           'matrix.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if any([int(x) not in range(1, n_dim + 1)
            for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components '
                           'order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        pass
    else:
        raise RuntimeError('Invalid number of components in strain/stress '
                           'components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor
        if tensor_mf.dtype == torch.cfloat:
            tensor = torch.zeros(tensor_order*(n_dim,), dtype=torch.cfloat,
                                 device=device)
        else:
            tensor = torch.zeros(tensor_order*(n_dim,), dtype=torch.float,
                                 device=device)
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if is_kelvin_notation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
                tensor[so_idx[::-1]] = (1.0/factor)*tensor_mf[mf_idx]
            tensor[so_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        mf_indexes = list()
        fo_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1
                               for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in [comp_order.index(comps[i][0]),
                                           comp_order.index(comps[i][1])]])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize tensor
        if tensor_mf.dtype == torch.cfloat:
            tensor = torch.zeros(tensor_order*(n_dim,), dtype=torch.cfloat,
                                 device=device)
        else:
            tensor = torch.zeros(tensor_order*(n_dim,), dtype=torch.float,
                                 device=device)
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if is_kelvin_notation and not (fo_idx[0] == fo_idx[1]
                                           and fo_idx[2] == fo_idx[3]):
                factor = \
                    factor*np.sqrt(2) if fo_idx[0] != fo_idx[1] else factor
                factor = \
                    factor*np.sqrt(2) if fo_idx[2] != fo_idx[3] else factor
                if fo_idx[0] != fo_idx[1] and fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[1::-1]+fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[:2]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[1::-1]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[0] != fo_idx[1]:
                    tensor[tuple(fo_idx[1::-1]+fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[:2]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
            tensor[fo_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor
# =============================================================================
def vget_tensor_from_mf(tensor_mf, n_dim, comp_order, is_kelvin_notation=True,
                        device=None):
    """Recover tensor from associated matricial form.

    Compatible with vectorized mapping.

    Recover a given second-order or fourth-order tensor from the associated
    matricial form, given the problem number of spatial dimensions and given a
    (compatible) ordered strain/stress components list. If the second-order
    tensor is symmetric or the fourth-order tensor has minor symmetry
    (component list only contains independent components), then matricial form
    is assumed to follow the Kelvin notation [#]_ by default.

    .. [#] Nagel, T., Görke, U.-J., Moerman, K. M., and Kolditz, O. (2016). On
           advantages of the Kelvin mapping in finite element implementations
           of deformation processes. Environmental Earth Sciences, 75(11):937
           (see `here <https://dspace.mit.edu/handle/1721.1/105251>`_)

    ----

    Parameters
    ----------
    tensor_mf : torch.Tensor
        Tensor stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : tuple
        Strain/Stress components order associated to matricial form.
    is_kelvin_notation : bool, default=True
        If True, then Kelvin notation is employed to store symmetric tensors in
        matricial form. If False, then tensor components are stored unchanged.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    tensor : torch.Tensor
        Tensor recovered from matricial form.
    """
    # Get device from input tensor
    if device is None:
        device = tensor_mf.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
    else:
        raise RuntimeError('Tensor matricial form must be a vector or a '
                           'matrix.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        pass
    else:
        raise RuntimeError('Invalid number of components in strain/stress '
                           'components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of components
    n_comps = len(comp_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set row major components order
    if tensor_order == 2:        
        row_major_order = tuple(
            f'{i + 1}{j + 1}' for i, j
            in it.product(range(n_dim), repeat=2))
    else:
        row_major_order = tuple(
            f'{i + 1}{j + 1}{k + 1}{l + 1}' for i, j, k, l
            in it.product(range(n_dim), repeat=4))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor according to tensor order
    if tensor_order == 2:
        # Build indexing inverse Kelvin factor
        if is_kelvin_notation:
            index_kelvin_inv = torch.tensor(
                [1.0/np.sqrt(2) if x[0] != x[1] else 1.0 for x in comp_order],
                dtype=torch.float, device=device)
        else:
            index_kelvin_inv = torch.ones(n_comps, dtype=torch.float,
                                          device=device)
        # Build indexing mapping
        index_map = [comp_order.index(x) if x in comp_order
                     else comp_order.index(x[::-1]) for x in row_major_order]
        # Get tensor from matricial form
        tensor = torch.mul(tensor_mf, index_kelvin_inv)[index_map].view(
            n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif tensor_order == 4:
        # Build indexing inverse Kelvin factor
        if is_kelvin_notation:
            index_kelvin_inv_1d = torch.tensor(
                [1.0/np.sqrt(2) if x[0] != x[1] else 1.0 for x in comp_order],
                dtype=torch.float, device=device)
            index_kelvin_inv = torch.outer(index_kelvin_inv_1d,
                                           index_kelvin_inv_1d)
        else:
            index_kelvin_inv = torch.ones((n_comps, n_comps),
                                          dtype=torch.float, device=device)
        # Build indexing mapping
        index_map = ([[comp_order.index(x[:2]) if x[:2] in comp_order
                       else comp_order.index(x[:2][::-1])
                       for x in row_major_order],
                      [comp_order.index(x[2:]) if x[2:] in comp_order
                       else comp_order.index(x[2:][::-1])
                       for x in row_major_order]])
        # Get tensor from matricial form
        tensor = torch.mul(tensor_mf, index_kelvin_inv)[index_map].view(
            n_dim, n_dim, n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor
# =============================================================================
def kelvin_factor(idx, comp_order):
    """Get Kelvin notation coefficient of given strain/stress component.

    The Kelvin notation [#]_ is a particular way of building the matricial form
    of tensorial quantities.

    .. [#] Nagel, T., Görke, U.-J., Moerman, K. M., and Kolditz, O. (2016). On
           advantages of the Kelvin mapping in finite element implementations
           of deformation processes. Environmental Earth Sciences, 75(11):937
           (see `here <https://dspace.mit.edu/handle/1721.1/105251>`_)

    ----

    Parameters
    ----------
    idx : int (or tuple[int])
        Index of strain/stress component. Alternatively, a pair of
        strain/stress components indexes (associated to a given fourth-order
        tensor matricial form element) can also be provided.
    comp_order : tuple
        Strain/Stress components order associated to matricial form.

    Returns
    -------
    factor : float
        Kelvin notation coefficient.
    """
    if len(comp_order) == 4 or len(comp_order) == 9:
        # Set Kelvin coefficient associated to a non-symmetric tensor matricial
        # storage
        factor = 1.0
    else:
        # Set Kelvin coefficient associated to symmetric tensor matricial
        # storage
        if isinstance(idx, int) or isinstance(idx, np.integer):
            # Set Kelvin coefficient associated with single strain/stress
            # component
            if int(list(comp_order[idx])[0]) == int(list(comp_order[idx])[1]):
                factor = 1.0
            else:
                factor = np.sqrt(2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif isinstance(idx, list) and len(idx) == 2:
            # Set Kelvin coefficient associated with pair of strain/stress
            # components
            factor = 1.0
            for i in idx:
                if int(list(comp_order[i])[0]) != int(list(comp_order[i])[1]):
                    factor = factor*np.sqrt(2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Invalid strain/stress component(s) index(es).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return factor
#
#                              Strain/Stress 2D - 3D matricial form conversions
# =============================================================================
def get_state_3Dmf_from_2Dmf(problem_type, mf_2d, comp_33, device=None):
    """Build 3D counterpart of 2D strain/stress second-order tensor.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    mf_2d : torch.Tensor(1d)
        Matricial form of 2D strain/stress second-order tensor.
    comp_33 : float
        Out-of-plane strain/stress component.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    mf_3d : torch.Tensor(1d)
        Matricial form of 3D strain/stress second-order tensor.
    """
    # Get device from input tensor
    if device is None:
        device = mf_2d.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = \
        get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor
    # symmetry
    if len(mf_2d) == len(comp_order_sym_2d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 3D strain/stress second-order tensor (matricial form)
    mf_3d = torch.zeros(len(comp_order_3d), dtype=torch.float, device=device)
    if problem_type in (3, 4):
        raise RuntimeError('Unavailable problem type.')
    else:
        # Include out-of-plane strain/stress component under 2D plane
        # strain/stress conditions
        for i in range(len(comp_order_2d)):
            comp = comp_order_2d[i]
            mf_3d[comp_order_3d.index(comp)] = mf_2d[i]
        mf_3d[comp_order_3d.index('33')] = comp_33
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mf_3d
# =============================================================================
def vget_state_3Dmf_from_2Dmf(mf_2d, comp_33, device=None):
    """Build 3D counterpart of 2D second-order tensor.
    
    Compatible with vectorized mapping.

    Parameters
    ----------
    mf_2d : torch.Tensor(1d)
        Matricial form of 2D strain/stress second-order tensor.
    comp_33 : float
        Out-of-plane strain/stress component.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    mf_3d : torch.Tensor(1d)
        Matricial form of 3D strain/stress second-order tensor.
    """
    # Get device from input tensor
    if device is None:
        device = mf_2d.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = \
        get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor
    # symmetry
    if len(mf_2d) == len(comp_order_sym_2d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 3D strain/stress second-order tensor (matricial form)
    mf_3d = torch.cat([mf_2d[comp_order_2d.index(x)].view(1)
                       if x in comp_order_2d
                       else torch.tensor([0.0], dtype=torch.float,
                                         device=device)
                       for x in comp_order_3d]) \
        + comp_33*torch.eye(len(comp_order_3d), dtype=torch.float,
                            device=device)[comp_order_3d.index('33')]    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mf_3d
# =============================================================================
def get_state_2Dmf_from_3Dmf(problem_type, mf_3d, device=None):
    """Build 2D counterpart of 3D strain/stress second- or fourth-order tensor.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    mf_3d : torch.Tensor (1d or 2d)
        Matricial form of 3D strain/stress related tensor.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    mf_2d : torch.Tensor (1d or 2d)
        Matricial form of 2D strain/stress related tensor.
    """
    # Get device from input tensor
    if device is None:
        device = mf_3d.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = \
        get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor
    # symmetry
    if len(mf_3d) == len(comp_order_sym_3d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 2D strain/stress related tensor (matricial form)
    mf_2d = torch.zeros(len(mf_3d.shape)*(len(comp_order_2d),),
                        dtype=torch.float, device=device)
    if len(mf_3d.shape) == 1:
        for i in range(len(comp_order_2d)):
            comp = comp_order_2d[i]
            mf_2d[i] = mf_3d[comp_order_3d.index(comp)]
    elif len(mf_3d.shape) == 2:
        for j in range(len(comp_order_2d)):
            comp_j = comp_order_2d[j]
            for i in range(len(comp_order_2d)):
                comp_i = comp_order_2d[i]
                mf_2d[i, j] = mf_3d[comp_order_3d.index(comp_i),
                                    comp_order_3d.index(comp_j)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mf_2d
# =============================================================================
def vget_state_2Dmf_from_3Dmf(mf_3d, device=None):
    """Build 2D counterpart of 3D second- or fourth-order tensor.
    
    Compatible with vectorized mapping.

    Parameters
    ----------
    mf_3d : torch.Tensor (1d or 2d)
        Matricial form of 3D strain/stress related tensor.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    mf_2d : torch.Tensor (1d or 2d)
        Matricial form of 2D strain/stress related tensor.
    """
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = \
        get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = \
        get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor
    # symmetry
    if len(mf_3d) == len(comp_order_sym_3d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = [comp_order_3d.index(x) for x in comp_order_2d]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 2D strain/stress related tensor (matricial form)
    if len(mf_3d.shape) == 1:
        mf_2d = mf_3d[index_map]
    elif len(mf_3d.shape) == 2:
        index_map = list(zip(*it.product(index_map, repeat=2)))
        mf_2d = mf_3d[index_map].view(len(comp_order_2d), len(comp_order_2d))
    else:
        RuntimeError('The 3D matricial form must correspond to 1d or 2d '
                     'torch.Tensor (second- or fourth-order strain/stress '
                     'tensor).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mf_2d