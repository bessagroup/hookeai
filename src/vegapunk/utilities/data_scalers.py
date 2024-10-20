"""Torch-based data scalers.

Classes
-------
TorchMinMaxScaler
    PyTorch tensor min-max data scaler.
TorchStandardScaler
    PyTorch tensor standardization data scaler.
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
class TorchMinMaxScaler:
    """PyTorch tensor min-max data scaler.
    
    Attributes
    ----------
    _n_features : int
        Number of features to normalize.
    _minimum : torch.Tensor
        Features normalization minimum tensor stored as a torch.Tensor with
        shape (n_features,).
    _maximum : torch.Tensor
        Features normalization maximum tensor stored as a torch.Tensor with
        shape (n_features,).
    _device : torch.device
        Device on which torch.Tensor is allocated.
    
    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    set_minimum(self, minimum)
        Set features normalization minimum tensor.
    set_maximum(self, maximum)
        Set features normalization maximum tensor.
    set_minimum_and_maximum(self, minimum, maximum)
        Set features normalization minimum and maximum tensors.
    get_minimum_and_maximum(self)
        Get features normalization minimum and maximum tensors.
    fit(self, tensor)
        Fit features normalization minimum and maximum tensors.
    transform(self, tensor, is_check_data=False)
        Normalize features tensor.
    inverse_transform(self, tensor, is_check_data=False)
        Denormalize features tensor.
    _check_minimum(self, minimum)
        Check features normalization minimum tensor.
    _check_maximum(self, maximum)
        Check features normalization maximum tensor.
    _check_tensor(self, tensor)
        Check features tensor to be transformed.
    """
    def __init__(self, n_features, minimum=None, maximum=None,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features : int
            Number of features to scale.
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        if not isinstance(n_features, int) or n_features <= 0:
            raise RuntimeError('Number of features is not positive int.')
        else:
            self._n_features = n_features
        if minimum is not None:
            self._minimum = self._check_minimum(minimum)
        else:
            self._minimum = None
        if maximum is not None:
            self._maximum = self._check_maximum(maximum)
        else:
            self._maximum = None
        self._device = torch.device(device_type)
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_minimum(self, minimum):
        """Set features normalization minimum tensor.
        
        Parameters
        ----------
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        self._minimum = self._check_minimum(minimum)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_maximum(self, maximum):
        """Set features normalization maximum tensor.
        
        Parameters
        ----------
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        self._maximum = self._check_maximum(maximum)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_minimum_and_maximum(self, minimum, maximum):
        """Set features normalization minimum and maximum tensors.
        
        Parameters
        ----------
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        self._minimum = self._check_minimum(minimum)
        self._maximum = self._check_maximum(maximum)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_minimum_and_maximum(self):
        """Get features normalization minimum and maximum tensors.
        
        Returns
        -------
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        return self._minimum, self._maximum
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, tensor):
        """Fit features normalization minimum and maximum tensors.
        
        If sequential data is provided, then all sequence times are
        concatenated and the normalization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(tensor.shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set normalization minimum and maximum
        self._minimum = self._check_minimum(torch.min(tensor, dim=0)[0])
        self._maximum = self._check_maximum(torch.max(tensor, dim=0)[0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def transform(self, tensor, is_check_data=False):
        """Normalize features tensor.

        If sequential data is provided, then all sequence times are
        concatenated and the normalization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        is_check_data : bool, default=False
            If True, then check transformed tensor data.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Normalized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # Store input tensor shape
        input_shape = tensor.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(input_shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build minimum and maximum tensors for normalization
        minimum = torch.tile(self._minimum, (n_samples, 1)).to(self._device)
        maximum = torch.tile(self._maximum, (n_samples, 1)).to(self._device)
        # Normalization features tensor
        transformed_tensor = \
            -1.0*torch.ones_like(tensor, device=self._device) \
            + torch.div(2.0*torch.ones_like(tensor, device=self._device),
                        maximum - minimum)*(tensor - minimum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if is_check_data:
            if not torch.equal(torch.tensor(transformed_tensor.shape),
                               torch.tensor(input_shape)):
                raise RuntimeError('Input and transformed tensors do not '
                                   'have the same shape.')
            elif torch.any(torch.isnan(transformed_tensor)):
                raise RuntimeError('One or more NaN elements were detected in '
                                   'the transformed tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def inverse_transform(self, tensor, is_check_data=False):
        """Denormalize features tensor.

        If sequential data is provided, then all sequence times are
        concatenated and the normalization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Normalized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        is_check_data : bool, default=False
            If True, then check transformed tensor data.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # Store input tensor shape
        input_shape = tensor.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(input_shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build minimum and maximum tensors for normalization
        minimum = torch.tile(self._minimum, (n_samples, 1)).to(self._device)
        maximum = torch.tile(self._maximum, (n_samples, 1)).to(self._device)
        # Denormalize features tensor
        transformed_tensor = minimum \
            + torch.div(maximum - minimum,
                        2.0*torch.ones_like(tensor, device=self._device))*(
                tensor + torch.ones_like(tensor, device=self._device))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if is_check_data:
            if not torch.equal(torch.tensor(transformed_tensor.shape),
                               torch.tensor(input_shape)):
                raise RuntimeError('Input and transformed tensors do not have '
                                   'the same shape.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # -------------------------------------------------------------------------
    def _check_minimum(self, minimum):
        """Check features normalization minimum tensor.
        
        Parameters
        ----------
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
            
        Returns
        -------
        minimum : torch.Tensor
            Features normalization minimum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        # Check features normalization minimum tensor        
        if not isinstance(minimum, torch.Tensor):
            raise RuntimeError('Features normalization minimum tensor is not '
                               'a torch.Tensor.')
        elif len(minimum) != self._n_features:
            raise RuntimeError('Features standardization minimum tensor is '
                               'not a torch.Tensor(1d) with shape '
                               '(n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return minimum
    # -------------------------------------------------------------------------
    def _check_maximum(self, maximum):
        """Check features normalization maximum tensor.
        
        Parameters
        ----------
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
            
        Returns
        -------
        maximum : torch.Tensor
            Features normalization maximum tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        # Check features normalization maximum tensor
        if not isinstance(maximum, torch.Tensor):
            raise RuntimeError('Features normalization maximum tensor is not '
                               'a torch.Tensor.')
        elif len(maximum) != self._n_features:
            raise RuntimeError('Features standardization maximum tensor is '
                               'not a torch.Tensor(1d) with shape '
                               '(n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return maximum
    # -------------------------------------------------------------------------
    def _check_tensor(self, tensor):
        """Check features tensor to be transformed.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Features tensor is not a torch.Tensor.')
        elif len(tensor.shape) not in (2, 3):
            raise RuntimeError('Features tensor is not a torch.Tensor with '
                               'shape (n_samples, n_features) or '
                               'torch.Tensor with shape '
                               '(sequence_length, n_samples, n_features).')
        elif tensor.shape[-1] != self._n_features:
            raise RuntimeError('Features tensor is not consistent with data'
                               'scaler number of features.')
# =============================================================================
class TorchStandardScaler:
    """PyTorch tensor standardization data scaler.
    
    Attributes
    ----------
    _n_features : int
        Number of features to standardize.
    _mean : torch.Tensor
        Features standardization mean tensor stored as a torch.Tensor with
        shape (n_features,).
    _std : torch.Tensor
        Features standardization standard deviation tensor stored as a
        torch.Tensor with shape (n_features,).
    _device : torch.device
        Device on which torch.Tensor is allocated.
    
    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    set_mean(self, mean)
        Set features standardization mean tensor.
    set_std(self, std)
        Set features standardization standard deviation tensor.
    set_mean_and_std(self, mean, std)
        Set features standardization mean and standard deviation tensors.
    get_mean_and_std(self)
        Get features standardization mean and standard deviation tensors.
    fit(self, tensor)
        Fit features standardization mean and standard deviation tensors.
    transform(self, tensor, is_check_data=False)
        Standardize features tensor.
    inverse_transform(self, tensor, is_check_data=False)
        Destandardize features tensor.
    _check_mean(self, mean):
        Check features standardization mean tensor.
    _check_std(self, std)
        Check features standardization standard deviation tensor.
    _check_tensor(self, tensor):
        Check features tensor to be transformed.
    """
    def __init__(self, n_features, mean=None, std=None, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features : int
            Number of features to standardize.
        mean : torch.Tensor, default=None
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        std : torch.Tensor, default=None
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        if not isinstance(n_features, int) or n_features <= 0:
            raise RuntimeError('Number of features is not positive int.')
        else:
            self._n_features = n_features
        if mean is not None:
            self._mean = self._check_mean(mean)
        else:
            self._mean = None
        if std is not None:
            self._std = self._check_std(std)
        else:
             self._std = None
        self._device = torch.device(device_type)
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_mean(self, mean):
        """Set features standardization mean tensor.
        
        Parameters
        ----------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        self._mean = self._check_mean(mean)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_std(self, std):
        """Set features standardization standard deviation tensor.
        
        Parameters
        ----------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        self._std = self._check_std(std)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_mean_and_std(self, mean, std):
        """Set features standardization mean and standard deviation tensors.
        
        Parameters
        ----------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        self._mean = self._check_mean(mean)
        self._std = self._check_std(std)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_mean_and_std(self):
        """Get features standardization mean and standard deviation tensors.
        
        Returns
        -------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        return self._mean, self._std
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, tensor, is_bessel=False):
        """Fit features standardization mean and standard deviation tensors.
        
        If sequential data is provided, then all sequence times are
        concatenated and the standardization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        is_bessel : bool, default=False
            If True, apply Bessel's correction to compute standard deviation,
            False otherwise.
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(tensor.shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set standardization mean and standard deviation
        self._mean = self._check_mean(torch.mean(tensor, dim=0))
        self._std = self._check_std(torch.std(tensor, dim=0,
                                              correction=int(is_bessel)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def transform(self, tensor, is_check_data=False):
        """Standardize features tensor.

        If sequential data is provided, then all sequence times are
        concatenated and the standardization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        is_check_data : bool, default=False
            If True, then check transformed tensor data.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # Store input tensor shape
        input_shape = tensor.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(input_shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build mean and standard deviation tensors for standardization
        mean = torch.tile(self._mean, (n_samples, 1)).to(self._device)
        std = torch.tile(self._std, (n_samples, 1)).to(self._device)
        # Set non-null standard deviation mask
        non_null_mask = (std != 0)
        # Standardize features tensor
        transformed_tensor = tensor - mean
        transformed_tensor[non_null_mask] = \
            torch.div(transformed_tensor[non_null_mask], std[non_null_mask])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if is_check_data:
            if not torch.equal(torch.tensor(transformed_tensor.shape),
                               torch.tensor(input_shape)):
                raise RuntimeError('Input and transformed tensors do not '
                                   'have the same shape.')
            elif torch.any(torch.isnan(transformed_tensor)):
                raise RuntimeError('One or more NaN elements were detected in '
                                   'the transformed tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def inverse_transform(self, tensor, is_check_data=False):
        """Destandardize features tensor.

        If sequential data is provided, then all sequence times are
        concatenated and the standardization procedures take into account the
        whole sequence length data for each feature.

        Parameters
        ----------
        tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        is_check_data : bool, default=False
            If True, then check transformed tensor data.
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features) or as torch.Tensor with shape
            (sequence_length, n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # Store input tensor shape
        input_shape = tensor.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate sequential data
        if len(input_shape) == 3:
            tensor = torch.reshape(tensor, (-1, self._n_features))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build mean and standard deviation tensors for standardization
        mean = torch.tile(self._mean, (n_samples, 1)).to(self._device)
        std = torch.tile(self._std, (n_samples, 1)).to(self._device)
        # Destandardize features tensor
        transformed_tensor = torch.mul(tensor, std) + mean
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if is_check_data:
            if not torch.equal(torch.tensor(transformed_tensor.shape),
                               torch.tensor(input_shape)):
                raise RuntimeError('Input and transformed tensors do not have '
                                   'the same shape.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # -------------------------------------------------------------------------
    def _check_mean(self, mean):
        """Check features standardization mean tensor.
        
        Parameters
        ----------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
            
        Returns
        -------
        mean : torch.Tensor
            Features standardization mean tensor stored as a torch.Tensor with
            shape (n_features,).
        """
        # Check features standardization mean tensor
        if not isinstance(mean, torch.Tensor):
            raise RuntimeError('Features standardization mean tensor is not a '
                                'torch.Tensor.')
        elif len(mean) != self._n_features:
            raise RuntimeError('Features standardization mean tensor is not a '
                               'torch.Tensor(1d) with shape (n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mean
    # -------------------------------------------------------------------------
    def _check_std(self, std):
        """Check features standardization standard deviation tensor.
        
        Parameters
        ----------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
            
        Returns
        -------
        std : torch.Tensor
            Features standardization standard deviation tensor stored as a
            torch.Tensor with shape (n_features,).
        """
        # Check features standardization standard deviation tensor
        if not isinstance(std, torch.Tensor):
            raise RuntimeError('Features standardization standard deviation '
                               'tensor is not a torch.Tensor.')
        elif len(std) != self._n_features:
            raise RuntimeError('Features standardization standard deviation '
                               'tensor is not a torch.Tensor(1d) with shape '
                               '(n_features,).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return std
    # -------------------------------------------------------------------------
    def _check_tensor(self, tensor):
        """Check features tensor to be transformed.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features).
        """
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Features tensor is not a torch.Tensor.')
        elif len(tensor.shape) not in (2, 3):
            raise RuntimeError('Features tensor is not a torch.Tensor with '
                               'shape (n_samples, n_features) or '
                               'torch.Tensor with shape '
                               '(sequence_length, n_samples, n_features).')
        elif tensor.shape[-1] != self._n_features:
            raise RuntimeError('Features tensor is not consistent with data'
                               'scaler number of features.')