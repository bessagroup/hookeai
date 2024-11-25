"""FETorch: Algebraic tensorial operations and standard tensorial operators.

This module is essentially a toolkit containing the definition of several
standard tensorial operators (e.g., Kronecker delta, second- and fourth-order
identity tensors, rotation tensor) and tensorial operations (e.g., tensorial
product, tensorial contraction, spectral decomposition) arising in
computational mechanics.

Apart from a conversion to a PyTorch framework and some additional procedures,
most of the code is taken from the package cratepy [#]_.

.. [#] Ferreira, B.P., Andrade Pires, F.M., and Bessa, M.A. (2023). CRATE: A
       Python package to perform fast material simulations. The Journal of Open
       Source Software, 8(87)
       (see `here <https://joss.theoj.org/papers/10.21105/joss.05594>`_)

Functions
---------
dyad11
    Dyadic product: :math:`i \\otimes j \\rightarrow ij`.
dyad22_1
    Dyadic product: :math:`ij \\otimes kl \\rightarrow ijkl`.
dyad22_2
    Dyadic product: :math:`ik \\otimes jl \\rightarrow ijkl`.
dyad22_3
    Dyadic product: :math:`il \\otimes jk \\rightarrow ijkl`.
dot21_1
    Single contraction: :math:`ij \\cdot j \\rightarrow i`.
dot12_1
    Single contraction: :math:`i \\cdot ij \\rightarrow j`.
dot42_1
    Single contraction: :math:`ijkm \\cdot lm \\rightarrow ijkl`.
dot42_2
    Single contraction: :math:`ipkl \\cdot jp \\rightarrow ijkl`.
dot42_3
    Single contraction: :math:`ijkm \\cdot ml \\rightarrow ijkl`.
dot24_1
    Single contraction: :math:`im \\cdot mjkl \\rightarrow ijkl`.
dot24_2
    Single contraction: :math:`jm \\cdot imkl \\rightarrow ijkl`.
dot24_3
    Single contraction: :math:`km \\cdot ijml \\rightarrow ijkl`.
dot24_4
    Single contraction: :math:`lm \\cdot ijkm \\rightarrow ijkl`.
ddot22_1
    Double contraction: :math:`ij : ij \\rightarrow \\text{scalar}`.
ddot42_1
    Double contraction: :math:`ijkl : kl \\rightarrow ij`.
ddot44_1
    Double contraction: :math:`ijmn : mnkl \\rightarrow ijkl`.
dd
    Kronecker delta function.
get_id_operators
    Set common second- and fourth-order identity operators.
fo_dinv_sym(inv)
    Derivative of inverse of symmetric second-order tensor w.r.t. to itself.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools
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
#
#                                                          Tensorial operations
# =============================================================================
# Tensorial products
dyad11 = lambda a1, b1: torch.einsum('i,j -> ij', a1, b1)
dyad22_1 = lambda a2, b2: torch.einsum('ij,kl -> ijkl', a2, b2)
dyad22_2 = lambda a2, b2: torch.einsum('ik,jl -> ijkl', a2, b2)
dyad22_3 = lambda a2, b2: torch.einsum('il,jk -> ijkl', a2, b2)
# Tensorial single contractions
dot21_1 = lambda a2, b1: torch.einsum('ij,j -> i', a2, b1)
dot12_1 = lambda a1, b2: torch.einsum('i,ij -> j', a1, b2)
dot42_1 = lambda a4, b2: torch.einsum('ijkm,lm -> ijkl', a4, b2)
dot42_2 = lambda a4, b2: torch.einsum('ipkl,jp -> ijkl', a4, b2)
dot42_3 = lambda a4, b2: torch.einsum('ijkm,ml -> ijkl', a4, b2)
dot24_1 = lambda a2, b4: torch.einsum('im,mjkl -> ijkl', a2, b4)
dot24_2 = lambda a2, b4: torch.einsum('jm,imkl -> ijkl', a2, b4)
dot24_3 = lambda a2, b4: torch.einsum('km,ijml -> ijkl', a2, b4)
dot24_4 = lambda a2, b4: torch.einsum('lm,ijkm -> ijkl', a2, b4)
# Tensorial double contractions
ddot22_1 = lambda a2, b2: torch.einsum('ij,ij', a2, b2)
ddot42_1 = lambda a4, b2: torch.einsum('ijkl,kl -> ij', a4, b2)
ddot44_1 = lambda a4, b4: torch.einsum('ijmn,mnkl -> ijkl', a4, b4)
ddot24_1 = lambda a2, b4: torch.einsum('kl,klij -> ij', a2, b4)
#
#                                                                     Operators
# =============================================================================
def dd(i, j):
    """Kronecker delta function.

    .. math::

       \\delta_{ij} =
           \\begin{cases}
                   1, &         \\text{if } i=j, \\\\
                   0, &         \\text{if } i\\neq j.
           \\end{cases}

    ----

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    value : int (0 or 1)
        Kronecker delta.
    """
    if (not isinstance(i, int) and not isinstance(i, torch.int)) or \
            (not isinstance(j, int) and not isinstance(j, torch.int)):
        raise RuntimeError('The Kronecker delta function only accepts two '
                           + 'integer indexes as arguments.')
    value = 1 if i == j else 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return value
# =============================================================================
def get_id_operators(n_dim, device=None):
    """Set common second- and fourth-order identity operators.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.

    Returns
    -------
    soid : torch.Tensor(2d)
        Second-order identity tensor:

        .. math::

           I_{ij} = \\delta_{ij}
    foid : torch.Tensor(4d)
        Fourth-order identity tensor:

        .. math::
           I_{ijkl} = \\delta_{ik}\\delta_{jl}
    fotransp : torch.Tensor(4d)
        Fourth-order transposition tensor:

        .. math::

           I_{ijkl} = \\delta_{il}\\delta_{jk}
    fosym : torch.Tensor(4d)
        Fourth-order symmetric projection tensor:

        .. math::

           I_{ij} = 0.5(\\delta_{ik}\\delta_{jl} +
                    \\delta_{il}\\delta_{jk})
    fodiagtrace : torch.Tensor(4d)
        Fourth-order 'diagonal trace' tensor:

        .. math::

           I_{ijkl} = \\delta_{ij}\\delta_{kl}
    fodevproj : torch.Tensor(4d)
        Fourth-order deviatoric projection tensor:

        .. math::

           I_{ijkl} = \\delta_{ik}\\delta_{jl}
                      - \\dfrac{1}{3} \\delta_{ij}\\delta_{kl}
    fodevprojsym : torch.Tensor(4d)
        Fourth-order deviatoric projection tensor (second-order symmetric
        tensors):

        .. math::

           I_{ijkl} = 0.5(\\delta_{ik}\\delta_{jl}
                      + \\delta_{il}\\delta_{jk})
                      - \\dfrac{1}{3} \\delta_{ij}\\delta_{kl}
    """
    # Set second-order identity tensor
    soid = torch.eye(n_dim, device=device)
    # Set fourth-order identity tensor and fourth-order transposition tensor
    foid = torch.zeros((n_dim, n_dim, n_dim, n_dim), device=device)
    fotransp = torch.zeros((n_dim, n_dim, n_dim, n_dim), device=device)
    for i in range(n_dim):
        for j in range(n_dim):
            foid[i, j, i, j] = 1.0
            fotransp[i, j, j, i] = 1.0
    # Set fourth-order symmetric projection tensor
    fosym = 0.5*(foid + fotransp)
    # Set fourth-order 'diagonal trace' tensor
    fodiagtrace = dyad22_1(soid, soid)
    # Set fourth-order deviatoric projection tensor
    fodevproj = foid - (1.0/3.0)*fodiagtrace
    # Set fourth-order deviatoric projection tensor (second order symmetric
    # tensors)
    fodevprojsym = fosym - (1.0/3.0)*fodiagtrace
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return soid, foid, fotransp, fosym, fodiagtrace, fodevproj, fodevprojsym
# =============================================================================
def fo_dinv_sym(inv):
    """Derivative of inverse of symmetric second-order tensor w.r.t. to itself.
    
    Parameters
    ----------
    inv : torch.Tensor(2d)
        Inverse of symmetric second-order tensor.
        
    Returns
    -------
    dinv : torch.Tensor(4d)
        Derivative of inverse of symmetric second-order tensor w.r.t itself.
    """
    # Get number of dimensions
    n_dim = inv.shape[0]
    # Set fourth-order tensor shape
    fo_shape = 4*(n_dim,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get fourth-order tensor components indices (cartesian product)
    fo_idxs = itertools.product(*[range(dim) for dim in fo_shape])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize derivative components
    dinv_list = []
    # Loop over components
    for i, j, k, l in fo_idxs:
        # Compute derivative (partial) component
        dinv_list.append((inv[i, k]*inv[l, j] + inv[i, l]*inv[k, j]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute fourth-order derivative
    dinv = -0.5*torch.tensor((dinv_list)).view(fo_shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dinv