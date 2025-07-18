"""Procedures associated to model data scalers and scaling operations.

Functions
---------
init_data_scalers
    Initialize model data scalers.
set_data_scalers
    Set fitted model data scalers.
set_fitted_data_scalers
    Set fitted model data scalers from given scaler type and parameters.
fit_data_scalers
    Fit model data scalers.
data_scaler_transform
    Perform data scaling operation on features PyTorch tensor.
load_model_data_scalers_from_file
    Load data scalers from model initialization file.
check_normalized_return
    Check if model data normalization is available.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
# Third-party
import torch
# Local
from utilities.data_scalers import TorchMinMaxScaler, TorchStandardScaler
from utilities.fit_data_scalers import fit_data_scaler_from_dataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def init_data_scalers(model):
    """Initialize model data scalers.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    """
    model._data_scalers = {}
    model._data_scalers['features_in'] = None
    model._data_scalers['features_out'] = None
# =============================================================================
def set_data_scalers(model, scaler_features_in, scaler_features_out):
    """Set fitted model data scalers.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    scaler_features_in : {TorchStandardScaler, TorchMinMaxScaler}
        Data scaler for input features.
    scaler_features_out : {TorchStandardScaler, TorchMinMaxScaler}
        Data scaler for output features.
    """
    # Initialize data scalers
    init_data_scalers(model)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set fitted data scalers
    model._data_scalers['features_in'] = scaler_features_in
    model._data_scalers['features_out'] = scaler_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update model initialization file with fitted data scalers
    if model._is_save_model_init_file:
        model.save_model_init_file()
# =============================================================================
def set_fitted_data_scalers(model, scaling_type, scaling_parameters):
    """Set fitted model data scalers from given scaler type and parameters.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    scaling_type : {'min-max', 'mean-std'}
        Type of data scaling. Min-Max scaling ('min-max') or
        standardization ('mean-std').
    scaling_parameters : dict
        Data scaling parameters (item, dict) for each features type
        (key, str). For 'min-max' data scaling, the parameters are the
        'minimum' and 'maximum' features normalization tensors, as well as
        the 'norm_minimum' and 'norm_maximum' normalization bounds. For
        'mean-std' data scaling, the parameters are the 'mean' and 'std'
        features normalization tensors.
    """
    # Initialize data scalers
    model._init_data_scalers()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input and output features scaling parameters
    scaling_parameters_in = scaling_parameters['features_in']
    scaling_parameters_out = scaling_parameters['features_out']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate data scalers
    if scaling_type == 'min-max':
        # Instantiate data scalers
        scaler_features_in = TorchMinMaxScaler(
            model._n_features_in,
            device_type=model._device_type
            **scaling_parameters_in)
        scaler_features_out = TorchMinMaxScaler(
            model._n_features_out,
            device_type=model._device_type
            **scaling_parameters_out)
    elif scaling_type == 'mean-std':
        scaler_features_in = TorchStandardScaler(
            model._n_features_in,
            device_type=model._device_type,
            **scaling_parameters_in)
        scaler_features_out = TorchStandardScaler(
            model._n_features_out,
            device_type=model._device_type,
            **scaling_parameters_out)
    else:
        raise RuntimeError('Unknown type of data scaling.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set fitted data scalers
    model._data_scalers['features_in'] = scaler_features_in
    model._data_scalers['features_out'] = scaler_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update model initialization file with fitted data scalers
    if model._is_save_model_init_file:
        model.save_model_init_file()
# =============================================================================
def fit_data_scalers(model, dataset, scaling_type='mean-std',
                     scaling_parameters={}, is_verbose=False):
    """Fit model data scalers.
    
    Data scaler normalization tensors are fitted from given data set,
    overriding provided data scaling parameters.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    scaling_type : {'min-max', 'mean-std'}, default='mean-std'
        Type of data scaling. Min-Max scaling ('min-max') or
        standardization ('mean-std').
    scaling_parameters : dict, default={}
        Data scaling parameters (item, dict) for each features type
        (key, str). For 'min-max' data scaling, the parameters are the
        'minimum' and 'maximum' features normalization tensors, as well as
        the 'norm_minimum' and 'norm_maximum' normalization bounds. For
        'mean-std' data scaling, the parameters are the 'mean' and 'std'
        features normalization tensors.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    if is_verbose:
        print('\nFitting model data scalers'
              '\n--------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data scalers
    init_data_scalers(model)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get scaling parameters and fit data scalers: input features
    scaler_features_in = fit_data_scaler_from_dataset(
        dataset, features_type='features_in',
        n_features=model._n_features_in, scaling_type=scaling_type,
        scaling_parameters=scaling_parameters)
    # Get scaling parameters and fit data scalers: output features
    scaler_features_out = fit_data_scaler_from_dataset(
        dataset, features_type='features_out',
        n_features=model._n_features_out, scaling_type=scaling_type,
        scaling_parameters=scaling_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Setting fitted standard scalers...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set fitted data scalers
    model._data_scalers['features_in'] = scaler_features_in
    model._data_scalers['features_out'] = scaler_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update model initialization file with fitted data scalers
    if model._is_save_model_init_file:
        model.save_model_init_file()
# =============================================================================
def get_fitted_data_scaler(model, features_type):
    """Get fitted model data scalers.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    features_type : str
        Features for which data scaler is required:
        
        'features_in'  : Input features

        'features_out' : Output features

    Returns
    -------
    data_scaler : {TorchStandardScaler, TorchMinMaxScaler}
        Data scaler.
    """
    # Get fitted data scaler
    if features_type not in model._data_scalers.keys():
        raise RuntimeError(f'Unknown data scaler for {features_type}.')
    elif model._data_scalers[features_type] is None:
        raise RuntimeError(f'Data scaler for {features_type} has not '
                            'been fitted.')
    else:
        data_scaler = model._data_scalers[features_type]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_scaler
# =============================================================================
def data_scaler_transform(model, tensor, features_type, mode='normalize'):
    """Perform data scaling operation on features PyTorch tensor.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    tensor : torch.Tensor
        Features PyTorch tensor.
    features_type : str
        Features for which data scaler is required:
        
        'features_in'  : Input features

        'features_out' : Output features

    mode : {'normalize', 'denormalize'}, default=normalize
        Data scaling transformation type.
        
    Returns
    -------
    transformed_tensor : torch.Tensor
        Transformed features PyTorch tensor.
    """
    # Check model data normalization
    check_normalized_return(model)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input features tensor
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError('Input tensor is not torch.Tensor.')
    # Get input features tensor data type
    input_dtype = tensor.dtype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get fitted data scaler for input features
    data_scaler = get_fitted_data_scaler(model, features_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform data scaling normalization/denormalization
    if mode == 'normalize':
        transformed_tensor = data_scaler.transform(tensor)
    elif mode == 'denormalize':
        transformed_tensor = data_scaler.inverse_transform(tensor)
    else:
        raise RuntimeError('Invalid data scaling transformation type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce same data type of input features tensor 
    transformed_tensor = transformed_tensor.to(input_dtype)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check transformed features tensor
    if not isinstance(transformed_tensor, torch.Tensor):
        raise RuntimeError('Transformed tensor is not torch.Tensor.') 
    elif not torch.equal(torch.tensor(transformed_tensor.size()),
                            torch.tensor(tensor.size())):
        raise RuntimeError('Input and transformed tensors do not have '
                            'the same shape.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return transformed_tensor
# =============================================================================
def load_model_data_scalers_from_file(model):
    """Load data scalers from model initialization file.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    """
    # Check model directory
    if not os.path.isdir(model.model_directory):
        raise RuntimeError('The model directory has not been found:'
                           '\n\n' + model.model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model initialization file path from model directory
    model_init_file_path = os.path.join(model.model_directory,
                                        'model_init_file' + '.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model initialization attributes from file
    if not os.path.isfile(model_init_file_path):
        raise RuntimeError('The model initialization file '
                           'has not been found:\n\n'
                           + model_init_file_path)
    else:
        with open(model_init_file_path, 'rb') as model_init_file:
            model_init_attributes = pickle.load(model_init_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model data scalers
    model_data_scalers = model_init_attributes['model_data_scalers']
    model._data_scalers = model_data_scalers
# =============================================================================
def check_normalized_return(model):
    """Check if model data normalization is available.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    """
    if model._data_scalers is None:
        raise RuntimeError('Data scalers for model features have not '
                           'been set or fitted.')
    if all([x is None for x in model._data_scalers.values()]):
        raise RuntimeError('Data scalers for model features have not '
                           'been set or fitted.')