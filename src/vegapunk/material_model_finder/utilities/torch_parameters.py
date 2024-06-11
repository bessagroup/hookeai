"""Utilities to handle Torch parameters.

Functions
---------
set_parameter_in_dict
    Store parameter in torch.nn.ParameterDict.
get_model_parameter_dict
    Store torch Module parameters in torch.nn.ParameterDict.
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
def set_parameter_in_dict(param_dict, split_name, value):
    """Store parameter in torch.nn.ParameterDict.
    
    Provided parameter dictionary is updated in-place. Nested dictionaries are
    created to handle parameters from nested modules or submodules.
    
    Parameters
    ----------
    param_dict : torch.nn.ParameterDict()
        Parameter dictionary.
    split_name : list[str]
        Parameter name splitted by dot delimiter (nested structure).
    value : torch.Tensor(0d)
        Parameter value.
    """
    # Get initial parameter key
    key = split_name[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store parameter if leaf module, otherwise create and process nested
    # dictionary
    if len(split_name) == 1:
        # Store parameter
        param_dict[key] = value
    else:
        # Create nested dictionary
        if key not in param_dict:
            param_dict[key] = torch.nn.ParameterDict()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store parameter in nested dictionary (recursive)
        set_parameter_in_dict(param_dict[key], split_name[1:], value)
# =============================================================================
def get_model_parameter_dict(model):
    """Store torch Module parameters in torch.nn.ParameterDict.
    
    Nested modules or submodules are stored as nested dictionaries.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
        
    Returns
    -------
    param_dict : torch.nn.ParameterDict()
        Parameter dictionary.
    """
    # Initialize parameter dictionary
    param_dict = torch.nn.ParameterDict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters    
    for name, value in model.named_parameters():
        # Split parameter name (nested structure)
        split_name = name.split('.')
        # Store parameter
        set_parameter_in_dict(param_dict, split_name, value)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return param_dict