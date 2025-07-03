"""Multi-layer gated recurrent unit (GRU) recurrent neural network model.

Classes
-------
GRURNNModel(torch.nn.Module)
    Multi-layer gated recurrent unit (GRU) recurrent neural network model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
import pickle
# Third-party
import torch
# Local
from model_architectures.rnn_base_model.custom.gru_vmap import GRU
from model_architectures.procedures.model_data_scaling import init_data_scalers
from model_architectures.procedures.model_state_files import save_model_state
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class GRURNNModel(torch.nn.Module):
    """Multi-layer gated recurrent unit (GRU) recurrent neural network model.
    
    The model is composed of a multi-layer GRU followed by a (time distributed)
    linear layer sharing the same number of hidden state features.
    
    Attributes
    ----------
    model_directory : {str, None}
        Directory where model is stored. If None, then all methods that depend
        on an existent model directory become unavailable.
    model_name : {str, None}
        Name of model. If None, then all methods that depend on a valid model
        name become unavailable.
    _n_features_in : int
        Number of input features.
    _n_features_out : int
        Number of output features.
    _hidden_layer_size : int
        Number of hidden state features.
    _n_recurrent_layers : int
        Number of recurrent layers. A number of recurrent layers greater than 1
        results in a stacked GRU (output of GRU in each time t is the input of
        next GRU).
    _dropout : int
        Dropout probability. If non-zero, each GRU recurrent layer is followed
        by a dropout layer with the provided dropout probability.
    _gru_model_source : {'torch', 'custom'}, default='torch'
        GRU model source: 'torch' imports torch.nn.GRU model, while
        'custom' gets custom GRU implementation.
    _gru_rnn_model : torch.nn.Module
        Multi-layer gated recurrent unit (GRU) recurrent neural network model.
    _linear_layer : torch.nn.Module
        Linear layer.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_explicit_parameters : bool
        True if model learnable parameters are explicit, False otherwise.
    is_model_in_normalized : bool
        If True, then model expects normalized input features (normalized
        input data has been seen during model training).
    is_model_out_normalized : bool
        If True, then model expects normalized output features (normalized
        output data has been seen during model training).
    _is_save_model_init_file: bool, default=True
        If True, saves model initialization file when model is initialized
        (overwritting existent initialization file), False otherwise.
    _data_scalers : dict
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str).

    Methods
    -------
    init_model_from_file(model_directory=None, model_init_file_path=None)
        Initialize model from initialization file.
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, features_in, hidden_features_in=None)
        Forward propagation.
    save_model_init_file(self)
        Save model class initialization attributes.
    """
    def __init__(self, n_features_in, n_features_out, hidden_layer_size,
                 model_directory, model_name='gru_material_model',
                 is_model_in_normalized=False, is_model_out_normalized=False,
                 n_recurrent_layers=1, dropout=0, is_save_model_init_file=True,
                 gru_model_source='torch', device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features_in : int
            Number of input features.
        n_features_out : int
            Number of output features.
        hidden_layer_size : int
            Number of hidden state features.
        model_directory : {str, None}
            Directory where model is stored. If None, then all methods that
            depend on an existent model directory become unavailable.
        model_name : {str, None}, default='gru_material_model'
            Name of model. If None, then all methods that depend on a valid
            model name become unavailable.
        is_model_in_normalized : bool, default=False
            If True, then model expects normalized input features (normalized
            input data has been seen during model training).
        is_model_out_normalized : bool, default=False
            If True, then model expects normalized output features (normalized
            output data has been seen during model training).
        n_recurrent_layers : int, default=1
            Number of recurrent layers. A number of recurrent layers greater
            than 1 results in a stacked GRU (output of GRU in each time t is
            the input of next GRU).
        dropout : float, default=0
            Dropout probability. If non-zero, each GRU recurrent layer is
            followed by a dropout layer with the provided dropout probability.
        is_save_model_init_file: bool, default=True
            If True, saves model initialization file when model is initialized
            (overwritting existent initialization file), False otherwise. When
            initializing model from initialization file this option should be
            set to False to avoid updating the initialization file and preserve
            fitted data scalers.
        gru_model_source : {'torch', 'custom'}, default='torch'
            GRU model source: 'torch' imports torch.nn.GRU model, while
            'custom' gets custom GRU implementation.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(GRURNNModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_features_in = n_features_in
        self._n_features_out = n_features_out
        self._hidden_layer_size = hidden_layer_size
        # Set architecture parameters
        self._n_recurrent_layers = n_recurrent_layers
        self._dropout = dropout
        # Set model directory
        if model_directory is None:
            self.model_directory = model_directory
        elif os.path.isdir(model_directory):
            self.model_directory = str(model_directory)
        else:
            raise RuntimeError('The model directory has not been found.')
        # Set model name
        if model_name is None:
            self.model_name = model_name
        elif isinstance(model_name, str):
            self.model_name = model_name
        else:
            raise RuntimeError('The model name must be a string.')
        # Set model input and output features normalization
        self.is_model_in_normalized = is_model_in_normalized
        self.is_model_out_normalized = is_model_out_normalized
        # Set save initialization file flag
        self._is_save_model_init_file = is_save_model_init_file
        # Set multi-layer gated recurrent unit (GRU) model source
        self._gru_model_source = gru_model_source
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize multi-layer gated recurrent unit (GRU) recurrent neural
        # network
        if self._gru_model_source == 'custom':
            self._gru_rnn_model = \
                GRU(input_size=self._n_features_in,
                    hidden_size=self._hidden_layer_size,
                    num_layers=self._n_recurrent_layers,
                    bias=True,
                    device=self._device)
        else:
            self._gru_rnn_model = \
                torch.nn.GRU(input_size=self._n_features_in,
                            hidden_size=self._hidden_layer_size,
                            num_layers = self._n_recurrent_layers,
                            bias=True,
                            batch_first=False,
                            dropout=self._dropout,
                            device=self._device)
        # Initialize linear layer
        self._linear_layer = \
            torch.nn.Linear(in_features=self._hidden_layer_size,
                            out_features=n_features_out, bias=True,
                            device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set learnable parameters nature
        self.is_explicit_parameters = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        init_data_scalers(self)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model initialization file
        if self._is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    @staticmethod
    def init_model_from_file(model_directory=None, model_init_file_path=None):
        """Initialize model from initialization file.
        
        If model directory is provided, then (1) model initialization file is
        assumed to be stored in the model directory under the name
        model_init_file.pkl and (2) model initialization attributes are read
        from the stored model_init_file.pkl file.
        
        In model initialization file path is provided, then (1) model
        initialization attributes are read from the provided
        model_init_file.pkl file and (2) model directory is set as the
        corresponding directory.
        
        Parameters
        ----------
        model_directory : str, default=None
            Directory where model is stored.
        model_init_file_path : str, default=None
            Model initialization file path. Ignored if model_directory is
            provided.
        """
        # Get model directory or model initialization file path
        if model_directory is not None:
            # Check model directory
            if not os.path.isdir(model_directory):
                raise RuntimeError('The model directory has not been found:'
                                   '\n\n' + model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model initialization file path from model directory
            model_init_file_path = os.path.join(model_directory,
                                                'model_init_file' + '.pkl')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check model initialization file
            if not os.path.isfile(model_init_file_path):
                raise RuntimeError('The model initialization file has not '
                                   'been found:\n\n' + model_init_file_path)
        elif model_init_file_path is not None:
            # Check model initialization file
            if not os.path.isfile(model_init_file_path):
                raise RuntimeError('The model initialization file has not '
                                   'been found:\n\n' + model_init_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model directory from model initialization file path
            model_directory = os.path.dirname(model_init_file_path)
        else:
            raise RuntimeError('Either the model directory or the model '
                               'initialization file path must be provided in '
                               'order to initialize the model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        with open(model_init_file_path, 'rb') as model_init_file:
            model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization attributes
        model_init_args = model_init_attributes['model_init_args']
        # Update model directory
        model_init_args['model_directory'] = model_directory
        # Initialize model
        model = GRURNNModel(**model_init_args, is_save_model_init_file=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        model._data_scalers = model_data_scalers
        # Set model data scalers device
        if model._data_scalers is not None:
            # Loop over model data scalers
            for _, data_scaler in model._data_scalers.items():
                if data_scaler is not None:
                    data_scaler.set_device(model._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model
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
    def forward(self, features_in, hidden_features_in=None):
        """Forward propagation.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        hidden_features_in : torch.Tensor, default=None
            Tensor of initial hidden state features stored as torch.Tensor(2d)
            of shape (n_recurrent_layers, hidden_layer_size) for unbatched
            input or torch.Tensor(3d) of shape
            (n_recurrent_layers, batch_size, hidden_layer_size) for batched
            input. If None, initial hidden state features are initialized to
            zero.
            
        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        hidden_features_out : torch.Tensor
            Tensor of final multi-layer GRU hidden state features stored as
            torch.Tensor(2d) of shape (n_recurrent_layers, hidden_layer_size)
            for unbatched input or torch.Tensor(3d) of shape
            (n_recurrent_layers, batch_size, hidden_layer_size) for batched
            input.
        """
        # Check input state features
        if not isinstance(features_in, torch.Tensor):
            raise RuntimeError('Input features were not provided as '
                               'torch.Tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hidden state features
        if hidden_features_in is None:
            if len(features_in.shape) == 2:
                hidden_features_in = torch.zeros((self._n_recurrent_layers,
                                                  self._hidden_layer_size),
                                                 device=features_in.device)
            else:
                hidden_features_in = torch.zeros((self._n_recurrent_layers,
                                                  features_in.shape[1],
                                                  self._hidden_layer_size),
                                                 device=features_in.device)
        # Check hidden state features
        if not isinstance(hidden_features_in, torch.Tensor):
            raise RuntimeError('Initial hidden state features were not '
                               'provided as torch.Tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward propagation: Multi-layer GRU
        features_out, hidden_features_out = self._gru_rnn_model(
            features_in, hidden_features_in)
        # Forward propagation: Linear layer
        features_out = self._linear_layer(features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out, hidden_features_out
    # -------------------------------------------------------------------------
    def save_model_init_file(self):
        """Save model initialization file.
        
        Initialization file is stored in the model directory under the name
        model_init_file.pkl.
        
        Initialization file contains a dictionary model_init_attributes that
        includes:
        
        'model_init_args' - Model initialization parameters
        
        'model_data_scalers' - Model fitted data scalers
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model initialization attributes
        model_init_attributes = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initialization parameters
        model_init_args = {}
        model_init_args['n_features_in'] = self._n_features_in
        model_init_args['n_features_out'] = self._n_features_out
        model_init_args['hidden_layer_size'] = self._hidden_layer_size
        model_init_args['n_recurrent_layers'] = self._n_recurrent_layers
        model_init_args['dropout'] = self._dropout
        model_init_args['model_directory'] = self.model_directory
        model_init_args['model_name'] = self.model_name
        model_init_args['is_model_in_normalized'] = self.is_model_in_normalized
        model_init_args['is_model_out_normalized'] = \
            self.is_model_out_normalized
        model_init_args['gru_model_source'] = self._gru_model_source
        model_init_args['device_type'] = self._device_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble initialization parameters
        model_init_attributes['model_init_args'] = model_init_args
        # Assemble model data scalers
        model_init_attributes['model_data_scalers'] = self._data_scalers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model initialization file path
        model_init_file_path = os.path.join(self.model_directory,
                                            'model_init_file' + '.pkl')
        # Save model initialization file
        with open(model_init_file_path, 'wb') as init_file:
            pickle.dump(model_init_attributes, init_file)