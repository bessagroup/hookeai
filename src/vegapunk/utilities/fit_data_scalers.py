"""Fitting of data scalers.

Functions
---------
fit_data_scaler_from_dataset
    Get fitted data scaler for given data set features type.
mean_std_batch_fit
    Perform batch fitting of standardization data scaler.
min_max_batch_fit
    Perform batch fitting of min-max data scaler.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import tqdm
import sklearn.preprocessing
# Local
from time_series_data.time_dataset import get_time_series_data_loader
from utilities.data_scalers import TorchStandardScaler, TorchMinMaxScaler
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def fit_data_scaler_from_dataset(dataset, features_type, n_features,
                                 scaling_type='mean-std',
                                 scaling_parameters={}):
    """Fit features type data scaler from given data set.
    
    Data scaler normalization tensors are fitted from given data set,
    overriding provided data scaling parameters.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_type : str
        Features for which data scaler is fitted (e.g., 'features_in',
        'features_out'). Must be directly available from data set samples.
    n_features : int
        Number of features (dimensionality).
    scaling_type : {'min-max', 'mean-std'}, default='mean-std'
        Type of data scaling. Min-Max scaling ('min-max') or standardization
        ('mean-std').
    scaling_parameters : dict, default={}
        Data scaling parameters (item, dict) for each features type
        (key, str). For 'min-max' data scaling, the parameters are the
        'minimum' and 'maximum' features normalization tensors, as well as
        the 'norm_minimum' and 'norm_maximum' normalization bounds. For
        'mean-std' data scaling, the parameters are the 'mean' and 'std'
        features normalization tensors.
    
    Returns
    -------
    data_scaler : {TorchStandardScaler, TorchMinMaxScaler}
        Data scaler.
    """
    # Set available data scaling types
    available_scaling_types = ('mean-std', 'min-max')
    # Check data scaling type
    if scaling_type not in available_scaling_types:
        raise RuntimeError(f'Unknown data scaling type \'{scaling_type}\'.'
                           f'\n\nAvailable data scaling types: '
                           f'{available_scaling_types}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get scaling parameters
    if features_type in scaling_parameters.keys():
        scaling_parameters_type = scaling_parameters[features_type]
    else:
        scaling_parameters_type = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit data scaler
    if scaling_type == 'mean-std':
        # Initialize data scaler
        data_scaler = \
            TorchStandardScaler(n_features, **scaling_parameters_type)
        # Perform batch fitting of data scaler parameters
        mean, std = mean_std_batch_fit(dataset, features_type, n_features)
        # Set fitted data scaler
        data_scaler.set_mean_and_std(mean, std)
    elif scaling_type == 'min-max':
        # Initialize data scaler
        data_scaler = TorchMinMaxScaler(n_features, **scaling_parameters_type)
        # Perform batch fitting of data scaler parameters
        minimum, maximum = \
            min_max_batch_fit(dataset, features_type, n_features)
        # Set fitted data scaler
        data_scaler.set_minimum_and_maximum(minimum, maximum)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return data_scaler
# =============================================================================
def mean_std_batch_fit(dataset, features_type, n_features, is_verbose=False):
    """Perform batch fitting of standardization data scaler.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_type : str
        Features for which data scaler is fitted (e.g., 'features_in',
        'features_out'). Must be directly available from data set samples.
    n_features : int
        Number of features (dimensionality).
    is_verbose : bool, default=False
        If True, enable verbose output.
    
    Returns
    -------
    mean : torch.Tensor
        Features standardization mean tensor stored as a torch.Tensor with
        shape (n_features,).
    std : torch.Tensor
        Features standardization standard deviation tensor stored as a
        torch.Tensor with shape (n_features,).
        
    Notes
    -----
    A biased estimator is used to compute the standard deviation according with
    scikit-learn 1.3.2 documentation (sklearn.preprocessing.StandardScaler).
    """
    # Instantiate data scaler
    data_scaler = sklearn.preprocessing.StandardScaler()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    data_loader = get_time_series_data_loader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample in tqdm.tqdm(data_loader,
                            desc='> Processing data samples: ',
                            disable=not is_verbose):        
        # Check sample
        if not isinstance(sample, dict):
            raise RuntimeError('Time series sample must be dictionary where '
                               'each feature (key, str) data is a '
                               'torch.Tensor(2d) of shape '
                               '(sequence_length, n_featues).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check sample features tensor
        if features_type not in sample.keys():
            raise RuntimeError(f'Unavailable feature from sample.')
        else:
            features_tensor = sample[features_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process sample to fit data scaler
        if isinstance(features_tensor, torch.Tensor):
            # Check number of features
            if features_tensor.shape[-1] != n_features:
                raise RuntimeError(f'Mismatch between features tensor '
                                   f'({features_tensor.shape[-1]}) and '
                                   f'number of expected features '
                                   f'({n_features}) for features type: '
                                   f'\'{features_type}\'')
            # Process sample
            data_scaler.partial_fit(features_tensor[:, 0, :].clone())
        else:
            raise RuntimeError('Sample features tensor is not torch.Tensor.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get fitted mean and standard deviation tensors
    mean = torch.tensor(data_scaler.mean_)
    std = torch.sqrt(torch.tensor(data_scaler.var_))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check features standardization mean tensor
    if not isinstance(mean, torch.Tensor):
        raise RuntimeError('Features standardization mean tensor is not a '
                           'torch.Tensor.')
    elif len(mean) != features_tensor.shape[-1]:
        raise RuntimeError('Features standardization mean tensor is not a '
                           'torch.Tensor(1d) with shape (n_features,).')
    # Check features standardization standard deviation tensor
    if not isinstance(std, torch.Tensor):
        raise RuntimeError('Features standardization standard deviation '
                           'tensor is not a torch.Tensor.')
    elif len(std) != features_tensor.shape[-1]:
        raise RuntimeError('Features standardization standard deviation '
                           'tensor is not a torch.Tensor(1d) with shape '
                           '(n_features,).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mean, std
# =============================================================================
def min_max_batch_fit(dataset, features_type, n_features, is_verbose=False):
    """Perform batch fitting of min-max data scaler.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_type : str
        Features for which data scaler is fitted (e.g., 'features_in',
        'features_out'). Must be directly available from data set samples.
    n_features : int
        Number of features (dimensionality).
    is_verbose : bool, default=False
        If True, enable verbose output.
    
    Returns
    -------
    minimum : torch.Tensor
        Features normalization minimum tensor stored as a torch.Tensor with
        shape (n_features,).
    maximum : torch.Tensor
        Features normalization maximum tensor stored as a torch.Tensor with
        shape (n_features,).
    """
    # Instantiate data scaler
    data_scaler = sklearn.preprocessing.MinMaxScaler()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    data_loader = get_time_series_data_loader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample in tqdm.tqdm(data_loader,
                            desc='> Processing data samples: ',
                            disable=not is_verbose):        
        # Check sample
        if not isinstance(sample, dict):
            raise RuntimeError('Time series sample must be dictionary where '
                               'each feature (key, str) data is a '
                               'torch.Tensor(2d) of shape '
                               '(sequence_length, n_featues).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check sample features tensor
        if features_type not in sample.keys():
            raise RuntimeError(f'Unavailable feature from sample.')
        else:
            features_tensor = sample[features_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process sample to fit data scaler
        if isinstance(features_tensor, torch.Tensor):
            # Check number of features
            if features_tensor.shape[-1] != n_features:
                raise RuntimeError(f'Mismatch between features tensor '
                                   f'({features_tensor.shape[-1]}) and '
                                   f'number of expected features '
                                   f'({n_features}) for features type: '
                                   f'\'{features_type}\'')
            # Process sample
            data_scaler.partial_fit(features_tensor[:, 0, :].clone())
        else:
            raise RuntimeError('Sample features tensor is not torch.Tensor.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get fitted minimum and maximum tensors
    minimum = torch.tensor(data_scaler.data_min_)
    maximum = torch.tensor(data_scaler.data_max_)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check features minimum tensor
    if not isinstance(minimum, torch.Tensor):
        raise RuntimeError('Features minimum tensor is not a torch.Tensor.')
    elif len(minimum) != features_tensor.shape[-1]:
        raise RuntimeError('Features minimum tensor is not a torch.Tensor(1d) '
                           'with shape (n_features,).')
    # Check features maximum tensor
    if not isinstance(maximum, torch.Tensor):
        raise RuntimeError('Features maximum tensor is not a torch.Tensor.')
    elif len(maximum) != features_tensor.shape[-1]:
        raise RuntimeError('Features maximum tensor is not a torch.Tensor(1d) '
                           'with shape (n_features,).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return minimum, maximum