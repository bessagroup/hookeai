"""Transfer-learning models for hybrid material model (HybridMaterialModel).

Classes
-------
PolynomialLinearRegressor(torch.nn.Module):
    Polynomial linear regression model.
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
class PolynomialLinearRegressor(torch.nn.Module):
    """Polynomial linear regression model.
    
    This model is implemented exclusively to be employed as a transfer-learning
    model of a hybrid material constitutive model (HybridMaterialModel). 
    
    Attributes
    ----------
    _n_features_in : int
        Number of input features.
    _poly_degree : int
        Degree of polynomial linear regression model.
    _is_bias : bool
        If True, then account for bias in polynomial expansion, False
        otherwise.
    _n_poly : int
        Effective number of polynomial expansion coefficients.
    _polynomial_layers : torch.nn.ModuleList
        Features polynomial linear regression layers.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_data_normalization : bool
        If True, then input features are normalized prior to forward
        propagation, False otherwise. Fitted data scalers need to be stored
        as model attributes to carry out data normalization procedures.
    _data_scalers : dict
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str).

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, features_in)
        Forward propagation.
    _init_data_scalers(self)
        Initialize model data scalers.
    set_data_scalers(self, scaler_features_in, scaler_features_out)
        Set fitted model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    data_scaler_transform(self, tensor, features_type, mode='normalize')
        Perform data scaling operation on features PyTorch tensor.
    check_normalized_return(self)
        Check if model data normalization is available.
    """
    def __init__(self, n_features_in, poly_degree=1, is_bias=True,
                 is_data_normalization=False, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features_in : int
            Number of input features.
        poly_degree : int, default=1
            Degree of polynomial linear regression model.
        is_bias : bool, default=True
            If True, then account for bias weight in polynomial regression,
            otherwise False.
        is_data_normalization : bool, default=False
            If True, then input features are normalized prior to forward
            propagation, False otherwise. Fitted data scalers need to be stored
            as model attributes to carry out data normalization procedures.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(PolynomialLinearRegressor, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_features_in = n_features_in
        # Set degree of polynomial
        self._poly_degree = poly_degree
        # Set bias
        self._is_bias = is_bias
        # Set normalization flag
        self.is_data_normalization = is_data_normalization
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of polynomial coefficients
        if self._is_bias:
            self._n_poly = self._poly_degree + 1
        else:
            self._n_poly = self._poly_degree
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize features polynomial linear regression layers
        self._polynomial_layers = torch.nn.ModuleList()
        # Loop over features
        for _ in range(self._n_features_in):
            # Set (independent) linear regression layer for each feature
            # (bias is explicitly accounted for in polynomial expansion)
            self._polynomial_layers.append(
                torch.nn.Linear(self._n_poly, 1, bias=False))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        self._data_scalers = None
        if self.is_data_normalization:
            self._init_data_scalers()
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Consistent update of data scalers device
            if (hasattr(self, '_data_scalers')
                    and self._data_scalers is not None):
                # Loop over data scalers
                for _, data_scaler in self._data_scalers.items():
                    # Update data scaler device
                    if data_scaler is not None:
                        data_scaler.set_device(self._device_type) 
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Returns
        -------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def forward(self, features_in, is_normalized_out=False):
        """Forward propagation.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        is_normalized_out : bool, default=False
            If True, get normalized output features, False otherwise.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Check input state features
        if not isinstance(features_in, torch.Tensor):
            raise RuntimeError('Input features were not provided as '
                               'torch.Tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check model data normalization
        if is_normalized_out:
            self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize input features data
        if self.is_data_normalization:
            features_in = \
                self.data_scaler_transform(tensor=features_in,
                                           features_type='features_in',
                                           mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if batched input
        is_batched = features_in.dim() == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batching dimension
        batch_dim = 1
        # Set batched dimension
        if not is_batched:
            features_in = features_in.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input shape
        sequence_length, batch_size, n_features_in = features_in.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate features sequences
        # (shape: sequence_length*batch_size x n_features_in)
        features_in = features_in.view(-1, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set polynomial expansion degree bounds
        ini_deg = self._n_poly - self._poly_degree
        end_deg = self._n_poly
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize features output data
        features_out_data = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over features
        for i in range(self._n_features_in):
            # Get feature input data
            feature_input = features_in[:, i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create feature polynomial expansion
            feature_polynomial = torch.stack(
                [feature_input**deg for deg in range(ini_deg, end_deg)], dim=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute polynomial linear regression
            feature_output = self._polynomial_layers[i](feature_polynomial)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store feature output data
            features_out_data.append(feature_output.squeeze(-1))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stack features output
        features_out = torch.stack(features_out_data, dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert features sequences concatenation
        # (shape: sequence_length x batch_size x n_features_in)
        features_out = \
            features_out.view(sequence_length, batch_size, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            features_out = features_out.squeeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features data
        if self.is_data_normalization and not is_normalized_out:
            features_out = \
                self.data_scaler_transform(tensor=features_out,
                                           features_type='features_out',
                                           mode='denormalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
    # -------------------------------------------------------------------------
    def _init_data_scalers(self):
        """Initialize model data scalers."""
        self._data_scalers = {}
        self._data_scalers['features_in'] = None
        self._data_scalers['features_out'] = None
    # -------------------------------------------------------------------------
    def set_data_scalers(self, scaler_features_in, scaler_features_out):
        """Set fitted model data scalers.
        
        Parameters
        ----------
        scaler_features_in : {TorchMinMaxScaler, TorchMinMaxScaler}
            Data scaler for input features.
        scaler_features_out : {TorchMinMaxScaler, TorchMinMaxScaler}
            Data scaler for output features.
        """
        # Set fitted data scalers
        self._data_scalers['features_in'] = scaler_features_in
        self._data_scalers['features_out'] = scaler_features_out
    # -------------------------------------------------------------------------
    def get_fitted_data_scaler(self, features_type):
        """Get fitted model data scalers.
        
        Parameters
        ----------
        features_type : str
            Features for which data scaler is required:
            
            'features_in'  : Input features

            'features_out' : Output features

        Returns
        -------
        data_scaler : TorchStandardScaler
            Fitted data scaler.
        """
        # Get fitted data scaler
        if features_type not in self._data_scalers.keys():
            raise RuntimeError(f'Unknown data scaler for {features_type}.')
        elif self._data_scalers[features_type] is None:
            raise RuntimeError(f'Data scaler for {features_type} has not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')
        else:
            data_scaler = self._data_scalers[features_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_scaler
    # -------------------------------------------------------------------------
    def data_scaler_transform(self, tensor, features_type, mode='normalize'):
        """Perform data scaling operation on features PyTorch tensor.
        
        Parameters
        ----------
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
        self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input features tensor
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Input tensor is not torch.Tensor.')
        # Get input features tensor data type
        input_dtype = tensor.dtype
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fitted data scaler for input features
        data_scaler = self.get_fitted_data_scaler(features_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform data scaling normalization/denormalization
        if mode == 'normalize':
            transformed_tensor = data_scaler.transform(tensor)
        elif mode == 'denormalize':
            transformed_tensor = data_scaler.inverse_transform(tensor)
        else:
            raise RuntimeError('Invalid data scaling transformation type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce same data type of input features tensor 
        transformed_tensor = transformed_tensor.to(input_dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed features tensor
        if not isinstance(transformed_tensor, torch.Tensor):
            raise RuntimeError('Transformed tensor is not torch.Tensor.') 
        elif not torch.equal(torch.tensor(transformed_tensor.size()),
                             torch.tensor(tensor.size())):
            raise RuntimeError('Input and transformed tensors do not have '
                               'the same shape.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # -------------------------------------------------------------------------
    def check_normalized_return(self):
        """Check if model data normalization is available."""
        if not self.is_data_normalization or self._data_scalers is None:
            raise RuntimeError('Data scalers for model features have not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')
        if all([x is None for x in self._data_scalers.values()]):
            raise RuntimeError('Data scalers for model features have not '
                               'been fitted. Fit data scalers by calling '
                               'method fit_data_scalers() before training '
                               'or predicting with the model.')