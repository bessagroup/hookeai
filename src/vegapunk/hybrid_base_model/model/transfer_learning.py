"""Transfer-learning models for hybrid material model (HybridMaterialModel).

Classes
-------
PolynomialLinearRegressor(torch.nn.Module):
    Polynomial linear regression model.
BatchedElasticModel(torch.nn.Module):
    Linear elastic constitutive model for batched time series data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
# Local
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_state_2Dmf_from_3Dmf
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
    _n_poly_coeff : int
        Effective number of polynomial expansion coefficients.
    _polynomial_layers : torch.nn.ModuleList
        Features polynomial linear regression layers.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_model_in_normalized : bool, default=False
        If True, then model input features are assumed to be normalized
        (normalized input data has been seen during model training).
    is_model_out_normalized : bool, default=False
        If True, then model output features are assumed to be normalized
        (normalized output data has been seen during model training).
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
                 is_model_in_normalized=False, is_model_out_normalized=False,
                 device_type='cpu'):
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
        is_model_in_normalized : bool, default=False
            If True, then model input features are assumed to be normalized
            (normalized input data has been seen during model training).
        is_model_out_normalized : bool, default=False
            If True, then model output features are assumed to be normalized
            (normalized output data has been seen during model training).
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
        # Set model input and output features normalization
        self.is_model_in_normalized = is_model_in_normalized
        self.is_model_out_normalized = is_model_out_normalized
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of polynomial coefficients
        if self._is_bias:
            self._n_poly_coeff = self._poly_degree + 1
        else:
            self._n_poly_coeff = self._poly_degree
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize features polynomial linear regression layers
        self._polynomial_layers = torch.nn.ModuleList()
        # Loop over features
        for _ in range(self._n_features_in):
            # Set (independent) linear regression layer for each feature
            # (bias is explicitly accounted for in polynomial expansion)
            self._polynomial_layers.append(
                torch.nn.Linear(self._n_poly_coeff, 1, bias=False))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
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
    def forward(self, features_in):
        """Forward propagation.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

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
        # Set degree bounds of polynomial expansion
        if self._n_poly_coeff == self._poly_degree:
            ini_deg = 1
        else:
            ini_deg = 0
        end_deg = ini_deg + self._n_poly_coeff
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
        scaler_features_in : {TorchStandardScaler, TorchMinMaxScaler}
            Data scaler for input features.
        scaler_features_out : {TorchStandardScaler, TorchMinMaxScaler}
            Data scaler for output features.
        """
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        if self._data_scalers is None:
            raise RuntimeError('Data scalers for model features have not '
                               'been set or fitted. Call set_data_scalers() '
                               'to make model normalization procedures '
                               'available.')
        if all([x is None for x in self._data_scalers.values()]):
            raise RuntimeError('Data scalers for model features have not '
                               'been set or fitted. Call set_data_scalers() '
                               'to make model normalization procedures '
                               'available.')
# =============================================================================
class BatchedElasticModel(torch.nn.Module):
    """Linear elastic constitutive model for batched time series data.
    
    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _elastic_properties : dict
        Elastic material properties (key, str) values (item, float).
        Expecting independent elastic moduli ('Eijkl') according to
        elastic symmetries. Young modulus ('E') and Poisson's ratio ('v')
        may alternatively be provided under elastic isotropy.
    _elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                         'orthotropic', 'monoclinic', 'triclinic'}, \
                        default='isotropic'
        Elastic symmetries:

        * 'triclinic':  assumes no elastic symmetries.

        * 'monoclinic': assumes plane of symmetry 12.

        * 'orthotropic': assumes planes of symmetry 12 and 13.

        * 'transverse_isotropic': assumes axis of symmetry 3

        * 'isotropic': assumes complete symmetry.

    _comp_order_sym : tuple[str]
        Strain/Stress components symmetric order.
    _n_strain_comp : int
        Number of strain/stress components.
    _elastic_tangent_mf : torch.Tensor(2d)
        Elasticity tensor (matricial form).
    _kelvin_matrix : torch.Tensor(1d)
        Kelvin factors transformation matrix associated with strain components
        symmetric order.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, features_in)
        Forward propagation.
    """
    def __init__(self, problem_type, elastic_properties, elastic_symmetry,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float).
            Expecting independent elastic moduli ('Eijkl') according to
            elastic symmetries. Young modulus ('E') and Poisson's ratio ('v')
            may alternatively be provided under elastic isotropy.
        elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                            'orthotropic', 'monoclinic', 'triclinic'}, \
                            default='isotropic'
            Elastic symmetries:

            * 'triclinic':  assumes no elastic symmetries.

            * 'monoclinic': assumes plane of symmetry 12.

            * 'orthotropic': assumes planes of symmetry 12 and 13.

            * 'transverse_isotropic': assumes axis of symmetry 3

            * 'isotropic': assumes complete symmetry.

        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(BatchedElasticModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem type
        self._problem_type = problem_type
        # Set elastic material properties
        self._elastic_properties = elastic_properties
        # Set elastic symmetry
        self._elastic_symmetry = elastic_symmetry
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # Set number of strain components
        self._n_strain_comp = len(self._comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute 3D elasticity tensor
        self._elastic_tangent_mf = Elastic.elastic_tangent_modulus(
            elastic_properties, elastic_symmetry='isotropic',
            device=self._device)
        # Extract 2D elasticity tensor
        if self._n_dim == 2:
            self._elastic_tangent_mf = vget_state_2Dmf_from_3Dmf(
                self._elastic_tangent_mf, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Kelvin factors transformation matrix
        self._kelvin_matrix = \
            torch.tensor([math.sqrt(2) if x[0] != x[1] else 1.0
                          for x in self._comp_order_sym],
                         dtype=torch.float, device=self._device)
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
    def forward(self, features_in):
        """Forward propagation.
        
        The tensor of input features can be one of two (infered from number of
        features): (1) the elastic strain or (2) the concatenated total strain
        and plastic strain. In the second case, the elastic strain is computed
        from the elastoplastic additive decomposition of the total strain.
        
        The tensor of output features is the stress.
        
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

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
        # Check if batched input
        is_batched = features_in.dim() == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batching dimension
        batch_dim = 1
        # Set batched dimension
        if not is_batched:
            features_in = features_in.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic strain
        if features_in.shape[-1] == self._n_strain_comp:
            # Input features are already the elastic strain
            pass
        elif features_in.shape[-1] == 2*self._n_strain_comp:
            # Input features are the concatenated total and plastic strains
            features_in = (features_in[:, :, :self._n_strain_comp]
                           - features_in[:, :, self._n_strain_comp:])
        else:
            raise RuntimeError(f'Number of input features '
                               f'({features_in.shape[-1]}) must be equal to '
                               f'the number of strain components (elastic '
                               f'strain) or equal to twice the number of '
                               f'strain components (concatenated total and '
                               f'plastic strain), {self._n_strain_comp} or '
                               f'{2*self._n_strain_comp}, respectively.')                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input shape
        sequence_length, batch_size, n_features_in = features_in.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate features sequences
        # (shape: sequence_length*batch_size x n_features_in)
        features_in = features_in.view(-1, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute output features
        features_out = ((1.0/self._kelvin_matrix.view(-1, 1))*torch.matmul(
            self._elastic_tangent_mf,
            self._kelvin_matrix.view(-1, 1)*(features_in.T))).T
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert features sequences concatenation
        # (shape: sequence_length x batch_size x n_features_in)
        features_out = features_out.view(
            sequence_length, batch_size, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            features_out = features_out.squeeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out