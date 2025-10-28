"""Torch data types enforcement and conversion.

Functions
---------
convert_dict_to_tensor
    Convert all int, float and bool in dictionary to torch.Tensor.
convert_tensor_to_float32
    Convert floating point torch tensor to torch.float32.
convert_dict_to_float32
    Convert all floating point torch tensors in dictionary to torch.float32.
convert_tensor_to_float64
    Convert floating point torch tensor to torch.float64.
convert_dict_to_float64
    Convert all floating point torch tensors in dictionary to torch.float64.
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
def convert_dict_to_tensor(data_dict, is_inplace=True):
    """Convert all int, float and bool in dictionary to torch.Tensor.
    
    Torch default types are assumed for each variable input type.
    
    Torch tensors and non-listed types are kept unchanged.
    
    Nested dictionaries are processed recursively.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary.
    is_inplace : bool, default=True
        If True, then input dictionary is updated in-place.
        
    Returns
    -------
    data_dict : dict
        Dictionary.
    """
    # Perform dictionary conversion
    if is_inplace:
        # Loop over dictionary items
        for key, value in data_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                data_dict[key] = convert_dict_to_tensor(value)
            elif (isinstance(value, (int, float, bool))
                and not isinstance(value, torch.Tensor)):
                # Convert to torch tensor
                data_dict[key] = torch.tensor(value)
    else:
        # Initialize converted dictionary
        local_data_dict = {}
        # Loop over dictionary items
        for key, value in data_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                local_data_dict[key] = convert_dict_to_tensor(value)
            elif (isinstance(value, (int, float, bool))
                and not isinstance(value, torch.Tensor)):
                # Convert to torch tensor
                local_data_dict[key] = torch.tensor(value)
        # Assign pointer
        data_dict = local_data_dict
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_dict
# =============================================================================
def convert_tensor_to_float32(tensor):
    """Convert floating point torch tensor to torch.float32.
    
    Torch tensor with type torch.float32 or other non-float types is kept
    unchanged.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor.
        
    Returns
    -------
    tensor : torch.Tensor
        Tensor.
    """
    # Perform type conversion
    if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
        # Convert torch tensor to torch.float32
        tensor = tensor.to(torch.float32)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor
# =============================================================================
def convert_dict_to_float32(tensor_dict, is_inplace=True):
    """Convert all floating point torch tensors in dictionary to torch.float32.
    
    Torch tensors with type torch.float32 or other non-float types are kept
    unchanged.
    
    Nested dictionaries are processed recursively.
    
    Parameters
    ----------
    tensor_dict : dict
        Dictionary.
    is_inplace : bool, default=True
        If True, then input dictionary is updated in-place.
        
    Returns
    -------
    tensor_dict : dict
        Dictionary.
    """
    # Perform dictionary conversion
    if is_inplace:
        # Loop over dictionary items
        for key, value in tensor_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                tensor_dict[key] = convert_dict_to_float32(value)
            elif (isinstance(value, torch.Tensor)
                and torch.is_floating_point(value)):
                # Convert torch tensor to torch.float32
                tensor_dict[key] = value.to(torch.float32)
    else:
        # Initialize converted dictionary
        local_tensor_dict = {}
        # Loop over dictionary items
        for key, value in tensor_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                local_tensor_dict[key] = convert_dict_to_float32(value)
            elif (isinstance(value, torch.Tensor)
                and torch.is_floating_point(value)):
                # Convert torch tensor to torch.float32
                local_tensor_dict[key] = value.to(torch.float32)
            else:
                # Keep value unchanged
                local_tensor_dict[key] = value
        # Assign pointer
        tensor_dict = local_tensor_dict
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor_dict
# =============================================================================
def convert_tensor_to_float64(tensor):
    """Convert floating point torch tensor to torch.float64.
    
    Torch tensor with type torch.float64 or other non-float types is kept
    unchanged.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor.
        
    Returns
    -------
    tensor : torch.Tensor
        Tensor.
    """
    # Perform type conversion
    if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
        # Convert torch tensor to torch.float64
        tensor = tensor.to(torch.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor
# =============================================================================
def convert_dict_to_float64(tensor_dict, is_inplace=True):
    """Convert all floating point torch tensors in dictionary to torch.float64.
    
    Torch tensors with type torch.float64 or other non-float types are kept
    unchanged.
    
    Nested dictionaries are processed recursively.
    
    Parameters
    ----------
    tensor_dict : dict
        Dictionary.
    is_inplace : bool, default=True
        If True, then input dictionary is updated in-place.
        
    Returns
    -------
    tensor_dict : dict
        Dictionary.
    """
    # Perform dictionary conversion
    if is_inplace:
        # Loop over dictionary items
        for key, value in tensor_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                tensor_dict[key] = convert_dict_to_float64(value)
            elif (isinstance(value, torch.Tensor)
                and torch.is_floating_point(value)):
                # Convert torch tensor to torch.float64
                tensor_dict[key] = value.to(torch.float64)
    else:
        # Initialize converted dictionary
        local_tensor_dict = {}
        # Loop over dictionary items
        for key, value in tensor_dict.items():
            # Perform type conversion
            if isinstance(value, dict):
                # Process nested dictionary recursively
                local_tensor_dict[key] = convert_dict_to_float64(value)
            elif (isinstance(value, torch.Tensor)
                and torch.is_floating_point(value)):
                # Convert torch tensor to torch.float64
                local_tensor_dict[key] = value.to(torch.float64)
            else:
                # Keep value unchanged
                local_tensor_dict[key] = value
        # Assign pointer
        tensor_dict = local_tensor_dict
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return tensor_dict