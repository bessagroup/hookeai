import torch
# =============================================================================
# Summary: Implementation of PyTorch tensor min-max data scaler.
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
    set_minimum(self, minimum)
        Set features normalization minimum tensor.
    set_maximum(self, maximum)
        Set features normalization maximum tensor.
    set_minimum_and_maximum(self, minimum, maximum)
        Set features normalization minimum and maximum tensors.
    fit(self, tensor)
        Fit features normalization minimum and maximum tensors.
    transform(self, tensor)
        Normalize features tensor.
    inverse_transform(self, tensor)
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
    def transform(self, tensor):
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
        transformed_tensor = -1.0*torch.ones_like(tensor) \
            + torch.div(2.0*torch.ones_like(tensor),
                        maximum - minimum)*(tensor - minimum)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if not torch.equal(torch.tensor(transformed_tensor.shape),
                           torch.tensor(input_shape)):
            raise RuntimeError('Input and transformed tensors do not have the '
                               'same shape.')
        elif torch.any(torch.isnan(transformed_tensor)):
            raise RuntimeError('One or more NaN elements were detected in '
                               'the transformed tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def inverse_transform(self, tensor):
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
            + torch.div(maximum - minimum, 2.0*torch.ones_like(tensor))*(
                tensor + torch.ones_like(tensor))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert concatenation of sequential data
        if len(input_shape) == 3:
            transformed_tensor = torch.reshape(transformed_tensor, input_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if not torch.equal(torch.tensor(transformed_tensor.shape),
                           torch.tensor(input_shape)):
            raise RuntimeError('Input and transformed tensors do not have the '
                               'same shape.')
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
# Generate random torch tensor
tensor = torch.rand((5, 3))
print(f'\nOriginal tensor:\n{tensor}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get number of features
n_features = tensor.shape[1]
# Initialize data scaler
scaler = TorchMinMaxScaler(n_features)
# Fit features normalization mimum and maximum tensors
scaler.fit(tensor)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Normalize tensor
transformed_tensor = scaler.transform(tensor)
print(f'\nNormalized tensor:\n{transformed_tensor}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Denormalize tensor
recovered_tensor = scaler.inverse_transform(transformed_tensor)
print(f'\nDenormalized tensor:\n{recovered_tensor}')