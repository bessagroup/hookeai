"""PyTorch-based loss functions.

Functions
---------
get_pytorch_loss
    Get PyTorch-based loss function.
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
#
# =============================================================================
def get_pytorch_loss(loss_type, **kwargs):
    """Get PyTorch-based loss function.
   
    Includes both native and custom PyTorch-based loss functions.
   
    Parameters
    ----------
    loss_type : {'mse', 'mean_relative_error'}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
        'mre' : MRE (Mean Relative Error, custom)

    **kwargs
        Arguments of Pytorch-based loss function.
        
    Returns
    -------
    loss_function : torch.nn.Module
        PyTorch-based loss function.
    """
    # Set available PyTorch-based loss functions
    available = ('mse', 'mre')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get PyTorch loss function
    if loss_type == 'mse':
        loss_function = torch.nn.MSELoss(**kwargs)
    elif loss_type == 'mre':
        loss_function = MeanRelativeErrorLoss(**kwargs)
    else:
        raise RuntimeError(f'Unknown or unavailable PyTorch-based loss '
                           f'function. \n\nAvailable: {available}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_function
# =============================================================================
class MeanRelativeErrorLoss(torch.nn.Module):
    """Loss function: Mean relative error.
    
    Attributes
    ----------
    _zero_handling : {'absolute', 'clip', 'regularizer'}
        Strategy to handle target values below minimum threshold.
    _small : float
        Minimum threshold to handle target values close or equal to zero.
    
    Methods
    -------
    forward(self, input, target)
        Forward propagation.
    """
    def __init__(self, zero_handling='absolute', small=0.1):
        """Constructor.
        
        Parameters
        ----------
        zero_handling : {'absolute', 'clip', 'regularizer'}, default='absolute'
            Strategy to handle target values below minimum threshold:
            
            'absolute'    : Assign absolute error to target values below
                            minimum threshold
            
            'clip'        : Clip target values below minimum threshold
                            (denominator only)

            'regularizer' : Add minimum threshold to target values
                            (denominator only)
            
        small : float, default=0.1
            Minimum threshold to handle target values close or equal to zero.
        """
        # Initialize from base class
        super(MeanRelativeErrorLoss, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strategy to handle target values below minimum threshold
        if zero_handling not in ('clip', 'absolute', 'regularizer'):
            raise RuntimeError('Unknown strategy to handle target values '
                               'below minimum threshold.')
        else:
            self._zero_handling = zero_handling
        # Set minimum threshold
        self._small = small
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, input, target):
        """Forward propagation.
        
        Parameters
        ----------
        input : torch.Tensor
            Input tensor.
        target : torch.Tensor
            Target tensor.
        
        Returns
        -------
        loss : torch.Tensor(0d)
            Loss value.
        """
        # Get target tensor device
        tensor_device = target.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute absolute error
        absolute_error = torch.abs(input - target)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get minimum threshold
        small = self._small
        # Get zero target handling type
        zero_handling = self._zero_handling
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute relative error
        if zero_handling == 'absolute':
            # Compute relative error
            relative_error = absolute_error/target.abs()
            # Assign absolute error to target values below minimum threshold
            relative_error = \
                torch.where(target < small, absolute_error, relative_error)
        elif zero_handling == 'clip':
            # Clip target values below minimum threshold
            clipped_target = torch.maximum(
                target.abs(), torch.tensor(small, device=tensor_device))
            # Compute relative error
            relative_error = absolute_error/clipped_target
        elif zero_handling == 'regularizer':
            # Add regularizer to target values below minimum threshold
            regularized_target = \
                target.abs() + torch.tensor(small, device=tensor_device)
            # Compute relative error
            relative_error = absolute_error/regularized_target
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute mean relative error
        loss = relative_error.mean()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return loss
        