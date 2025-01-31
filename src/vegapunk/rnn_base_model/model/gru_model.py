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
from rnn_base_model.custom.gru_vmap import GRU
from utilities.data_scalers import TorchMinMaxScaler, TorchStandardScaler
from utilities.fit_data_scalers import fit_data_scaler_from_dataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
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
    save_model_init_state(self)
        Save model initial state to file.
    save_model_state(self, epoch=None, is_best_state=False, \
                     is_remove_posterior=True)
    load_model_state(self, load_model_state=None, is_remove_posterior=True)
        Load model state from file.
    _check_state_file(self, filename)
        Check if file is model training epoch state file.
    _check_best_state_file(self, filename)
        Check if file is model training epoch best state file.
    _remove_posterior_state_files(self, epoch)
        Delete model training epoch state files posterior to given epoch.
    _remove_best_state_files(self)
        Delete existent model best state files.
    _init_data_scalers(self)
        Initialize model data scalers.
    set_data_scalers(self, scaler_features_in, scaler_features_out)
        Set fitted model data scalers.
    set_fitted_data_scalers(self, scaling_type, scaling_parameters)
        Set fitted model data scalers from given scaler type and parameters.
    fit_data_scalers(self, dataset, scaling_type='mean-std', \
                     scaling_parameters={}, is_verbose=False)
        Fit model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    data_scaler_transform(self, tensor, features_type, mode='normalize')
        Perform data scaling operation on features PyTorch tensor.
    load_model_data_scalers_from_file(self)
        Load data scalers from model initialization file.
    check_normalized_return(self)
        Check if model data normalization is available.
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
        self._init_data_scalers()
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
    # -------------------------------------------------------------------------
    def save_model_state(self, epoch=None, is_best_state=False,
                         is_remove_posterior=True):
        """Save model state to file.
        
        Model state file is stored in model_directory under the name
        < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt or
        < model_name >-< epoch >-best.pt if epoch is known.
        
        Parameters
        ----------
        epoch : int, default=None
            Training epoch corresponding to current model state.
        is_best_state : bool, default=False
            If True, save model state file corresponding to the best
            performance instead of regular state file.
        is_remove_posterior : bool, default=True
            Remove model and optimizer state files corresponding to training
            epochs posterior to the saved state file. Effective only if saved
            training epoch is known.
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        model_state_file = self.model_name
        # Append epoch
        if isinstance(epoch, int):
            model_state_file += '-' + str(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state corresponding to the best performance
        if is_best_state:
            # Append best performance
            model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove any existent best model state file
            self._remove_best_state_files()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Save model state
        torch.save(self.state_dict(), model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model epoch state files posterior to saved epoch
        if isinstance(epoch, int) and is_remove_posterior:
            self._remove_posterior_state_files(epoch)
    # -------------------------------------------------------------------------
    def save_model_init_state(self):
        """Save model initial state to file.
        
        Model state file is stored in model_directory under the name
        < model_name >-init.pt.

        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        model_state_file = self.model_name + '-init'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model state
        torch.save(self.state_dict(), model_path)
    # -------------------------------------------------------------------------
    def load_model_state(self, load_model_state=None,
                         is_remove_posterior=True):
        """Load model state from file.
        
        Model state file is stored in model_directory under the name
        < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt or
        < model_name >-< epoch >-best.pt if epoch if known.
        
        Model initial state file is stored in model directory under the name
        < model_name >-init.pt
        
        Parameters
        ----------
        load_model_state : {'best', 'last', int, 'init', None}, default=None
            Load available model state from the model directory.
            Options:
            
            'best'      : Model state corresponding to best performance
            
            'last'      : Model state corresponding to highest training epoch
            
            int         : Model state corresponding to given training epoch
            
            'init'      : Model initial state
            
            None        : Model default state file
        
        is_remove_posterior : bool, default=True
            Remove model state files corresponding to training epochs posterior
            to the loaded state file. Effective only if loaded training epoch
            is known.
            
        Returns
        -------
        epoch : int
            Loaded model state training epoch. Defaults to None if training
            epoch is unknown.
        """
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if load_model_state == 'best':
            # Get state files in model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model best state files epochs
            best_state_epochs = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in model directory
            for filename in directory_list:
                # Check if file is model epoch best state file
                is_best_state_file, best_state_epoch = \
                    self._check_best_state_file(filename)
                # Store model best state file training epoch
                if is_best_state_file:
                    best_state_epochs.append(best_state_epoch)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model best state file
            if not best_state_epochs:
                raise RuntimeError('Model best state file has not been found '
                                   'in directory:\n\n' + self.model_directory)
            elif len(best_state_epochs) > 1:
                raise RuntimeError('Two or more model best state files have '
                                   'been found in directory:'
                                   '\n\n' + self.model_directory)
            else:
                # Set best state epoch
                epoch = best_state_epochs[0]
                # Set model best state file
                model_state_file = self.model_name
                if isinstance(epoch, int):
                    model_state_file += '-' + str(epoch)      
                model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model epoch state files posterior to loaded epoch
            if isinstance(epoch, int) and is_remove_posterior:
                self._remove_posterior_state_files(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif load_model_state == 'last':
            # Get state files in model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model state files training epochs
            epochs = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in model directory
            for filename in directory_list:
                # Check if file is model epoch state file
                is_state_file, epoch = self._check_state_file(filename)
                # Store model state file training epoch
                if is_state_file:
                    epochs.append(epoch)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set highest epoch model state file
            if epochs:
                # Set highest epoch
                epoch = max(epochs)
                # Set highest epoch model state file
                model_state_file = self.model_name + '-' + str(epoch)
            else:
                raise RuntimeError('Model state files corresponding to epochs '
                                   'have not been found in directory:\n\n'
                                   + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        elif isinstance(load_model_state, int):
            # Get epoch
            epoch = load_model_state
            # Set model state filename with epoch
            model_state_file = self.model_name + '-' + str(int(epoch))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model epoch state files posterior to loaded epoch
            if is_remove_posterior:
                self._remove_posterior_state_files(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif load_model_state == 'init':
            # Set model initial state file
            model_state_file = self.model_name + '-init'
            # Set epoch as unknown
            epoch = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model epoch state files posterior to loaded epoch
            if is_remove_posterior:
                self._remove_posterior_state_files(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Set model state filename
            model_state_file = self.model_name
            # Set epoch as unknown
            epoch = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Check model state file
        if not os.path.isfile(model_path):
            raise RuntimeError('Model state file has not been found:\n\n'
                               + model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model state
        self.load_state_dict(torch.load(model_path,
                                        map_location=torch.device('cpu')))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return epoch
    # -------------------------------------------------------------------------
    def _check_state_file(self, filename):
        """Check if file is model training epoch state file.
        
        Model training epoch state file is stored in model_directory under the
        name < model_name >-< epoch >.pt.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_state_file : bool
            True if model training epoch state file, False otherwise.
        epoch : {None, int}
            Training epoch corresponding to model state file if
            is_state_file=True, None otherwise.
        """
        # Check if file is model epoch state file
        is_state_file = bool(re.search(r'^' + self.model_name + r'-[0-9]+'
                                       + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epoch = None
        if is_state_file:
            # Get model state epoch
            epoch = int(os.path.splitext(filename)[0].split('-')[-1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_state_file, epoch
    # -------------------------------------------------------------------------
    def _check_best_state_file(self, filename):
        """Check if file is model best state file.
        
        Model state file corresponding to the best performance is stored in
        model_directory under the name < model_name >-best.pt. or
        < model_name >-< epoch >-best.pt if the training epoch is known.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_best_state_file : bool
            True if model training epoch state file, False otherwise.
        epoch : {None, int}
            Training epoch corresponding to model state file if
            is_best_state_file=True and training epoch is known, None
            otherwise.
        """
        # Check if file is model epoch best state file
        is_best_state_file = bool(re.search(r'^' + self.model_name
                                            + r'-?[0-9]*' + r'-best' + r'\.pt',
                                            filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epoch = None
        if is_best_state_file:
            # Get model state epoch
            epoch = int(os.path.splitext(filename)[0].split('-')[-2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_best_state_file, epoch
    # -------------------------------------------------------------------------
    def _remove_posterior_state_files(self, epoch):
        """Delete model training epoch state files posterior to given epoch.
        
        Parameters
        ----------
        epoch : int
            Training epoch.
        """
        # Get files in model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model epoch state file
            is_state_file, file_epoch = self._check_state_file(filename)
            # Delete model epoch state file posterior to given epoch
            if is_state_file and file_epoch > epoch:
                os.remove(os.path.join(self.model_directory, filename))
    # -------------------------------------------------------------------------
    def _remove_best_state_files(self):
        """Delete existent model best state files."""
        # Get files in model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model best state file
            is_best_state_file, _ = self._check_best_state_file(filename)
            # Delete state file
            if is_best_state_file:
                os.remove(os.path.join(self.model_directory, filename))
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model initialization file with fitted data scalers
        if self._is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    def set_fitted_data_scalers(self, scaling_type, scaling_parameters):
        """Set fitted model data scalers from given scaler type and parameters.
        
        Parameters
        ----------
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
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input and output features scaling parameters
        scaling_parameters_in = scaling_parameters['features_in']
        scaling_parameters_out = scaling_parameters['features_out']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate data scalers
        if scaling_type == 'min-max':
            # Instantiate data scalers
            scaler_features_in = TorchMinMaxScaler(
                self._n_features_in,
                device_type=self._device_type
                **scaling_parameters_in)
            scaler_features_out = TorchMinMaxScaler(
                self._n_features_out,
                device_type=self._device_type
                **scaling_parameters_out)
        elif scaling_type == 'mean-std':
            scaler_features_in = TorchStandardScaler(
                self._n_features_in,
                device_type=self._device_type,
                **scaling_parameters_in)
            scaler_features_out = TorchStandardScaler(
                self._n_features_out,
                device_type=self._device_type,
                **scaling_parameters_out)
        else:
            raise RuntimeError('Unknown type of data scaling.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['features_in'] = scaler_features_in
        self._data_scalers['features_out'] = scaler_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model initialization file with fitted data scalers
        if self._is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    def fit_data_scalers(self, dataset, scaling_type='mean-std',
                         scaling_parameters={}, is_verbose=False):
        """Fit model data scalers.
        
        Data scaler normalization tensors are fitted from given data set,
        overriding provided data scaling parameters.
        
        Parameters
        ----------
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get scaling parameters and fit data scalers: input features
        scaler_features_in = fit_data_scaler_from_dataset(
            dataset, features_type='features_in',
            n_features=self._n_features_in, scaling_type=scaling_type,
            scaling_parameters=scaling_parameters)
        # Get scaling parameters and fit data scalers: output features
        scaler_features_out = fit_data_scaler_from_dataset(
            dataset, features_type='features_out',
            n_features=self._n_features_out, scaling_type=scaling_type,
            scaling_parameters=scaling_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Setting fitted standard scalers...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['features_in'] = scaler_features_in
        self._data_scalers['features_out'] = scaler_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model initialization file with fitted data scalers
        if self._is_save_model_init_file:
            self.save_model_init_file()
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
        data_scaler : {TorchStandardScaler, TorchMinMaxScaler}
            Data scaler.
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
    def load_model_data_scalers_from_file(self):
        """Load data scalers from model initialization file."""
        # Check model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(self.model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The model initialization file '
                               'has not been found:\n\n'
                                + model_init_file_path)
        else:
            with open(model_init_file_path, 'rb') as model_init_file:
                model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        self._data_scalers = model_data_scalers
    # -------------------------------------------------------------------------
    def check_normalized_return(self):
        """Check if model data normalization is available."""
        if self._data_scalers is None:
            raise RuntimeError('Data scalers for model features have not '
                               'been set or fitted. Call set_data_scalers() '
                               'or fit_data_scalers() to make model '
                               'normalization procedures available.')
        if all([x is None for x in self._data_scalers.values()]):
            raise RuntimeError('Data scalers for model features have not '
                               'been set or fitted. Call set_data_scalers() '
                               'or fit_data_scalers() to make model '
                               'normalization procedures available.')