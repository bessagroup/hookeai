"""Get PyTorch model summary data.
 
Functions
---------
get_n_model_parameters
    Get number of parameters of PyTorch model.
get_model_summary
    Get summary of PyTorch model.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import torchinfo
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def get_n_model_parameters(model):
    """Get number of parameters of PyTorch model.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
        
    Returns
    -------
    n_total_params : int
        Total number of model parameters.
    n_train_params : int
        Number of trainable model parameters.
    n_nontrain_params : int
        Number of non-trainable model parameters.
    """
    # Check model
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError('Model is not a torch.nn.Module.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model total number of parameters
    n_total_params = sum(p.numel() for p in model.parameters())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model number of trainable parameters
    n_train_params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
    # Get model number of non-trainable parameters
    n_nontrain_params = n_total_params - n_train_params
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_total_params, n_train_params, n_nontrain_params
# =============================================================================
def get_model_summary(model, input_data=None, device_type='cpu',
                      is_verbose=False, **kwargs):
    """Get summary of PyTorch model.
    
    Wrapper: torchinfo (https://pypi.org/project/torchinfo/)
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    input_data : list[torch.Tensor], default=None
        Input data of PyTorch model forward propagation. If provided, then
        further summary data is computed and displayed (e.g., input/output
        shapes, number of operations, memory requirements). Must be list of one
        or more torch.Tensor to avoid unexpected behavior.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    kwargs : dict
        Other arguments of PyTorch model forward propagation.

    Returns
    -------
    model_statistics : torchinfo.model_statistics.ModelStatistics
        PyTorch model summary object.
    """   
    # Check PyTorch model input data
    if input_data is not None:
        if not isinstance(input_data, list):
            raise RuntimeError('If provided, input data of PyTorch model must '
                               'be list of torch.Tensor.')
        elif not all([isinstance(x, torch.Tensor) for x in input_data]):
            raise RuntimeError('If provided, input data of PyTorch model must '
                               'be list of torch.Tensor.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check device
    if device_type in ('cpu', 'cuda'):
        if device_type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('PyTorch with CUDA is not available.')
        device = torch.device(device_type)
    else:
        raise RuntimeError('Invalid device type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model forward propagation mode
    mode = 'eval'
    # Set maximum nesting model depth to display
    depth = 5
    # Set model verbose
    if is_verbose:
        verbose = 1
    else:
        verbose = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\ntorchinfo - Model summary in PyTorch'
              '\n------------------------------------')
        print('\n> Model summary:\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate Pytorch model summary
    model_statistics = torchinfo.summary(model, input_data=input_data,
                                         depth=depth, device=device,
                                         mode=mode, verbose=verbose,
                                         **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_statistics