"""PyTorch loss functions.

Functions
---------
get_pytorch_loss
    Get PyTorch loss function.
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
__status__ = 'Planning'
# =============================================================================
def get_pytorch_loss(loss_type, **kwargs):
    """Get PyTorch loss function.
   
    Parameters
    ----------
    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    **kwargs
        Arguments of torch.nn._Loss initializer.
        
    Returns
    -------
    loss_function : torch.nn._Loss
        PyTorch loss function.
    """
    # Set available PyTorch loss functions
    available = ('mse',)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyTorch loss function
    if loss_type == 'mse':
        loss_function = torch.nn.MSELoss(**kwargs)
    else:
        raise RuntimeError(f'Unknown or unavailable PyTorch loss function.'
                           f'\n\nAvailable: {available}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_function