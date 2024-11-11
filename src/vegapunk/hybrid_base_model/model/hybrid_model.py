"""Hybrid model.

Classes
-------
HybridModel(torch.nn.Module)
    Hybrid model.

Functions
---------
standard_partial_fit(dataset, features_type, n_features, is_verbose=False)
    Perform batch fitting of standardization data scalers.
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
import tqdm
import sklearn.preprocessing
# Local
from hybrid_base_model.model.hybridization import HybridizationModel
from rnn_base_model.data.time_dataset import get_time_series_data_loader
from utilities.data_scalers import TorchStandardScaler
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class HybridModel(torch.nn.Module):
    """Hybrid model.
    
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
    _hyb_models_dict : dict
        For each hybridized model (key, str), store dictionary (item, dict)
        with the hybridized model data.
    _hyb_models_names : list[str]
        Hybridized models names.
    _hyb_models : torch.nn.ModuleList
        Hybridized models.
    _hyb_channels : dict
        For each hybridization channel (key, str[int]), store the
        corresponding hybridized models module index (item, list[int]),
        sorted according with the associated hybridization indices.
        Hybridization channels are labeled between 0 and n_hyb_channel,
        where n_hyb_channel is the total number of channels required to
        place the hybridized models.
    _hyb_models_input_residual : dict
        For each hybridization model (key, str), store input residual
        connection flag (item, bool). If True, then input residual
        connection is assigned to hybridized model, False otherwise.
    _hybridization_type : str
        Hybridization model type.
    _hybridization_model : HybridizationModel
        Hybridization model.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_model_in_normalized : bool
        If True, then model expects normalized input features (normalized
        input data has been seen during model training).
    is_model_out_normalized : bool
        If True, then model expects normalized output features (normalized
        output data has been seen during model training).
    _data_scalers : dict
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str).

    Methods
    -------
    init_model_from_file(model_directory=None, model_init_file_path=None)
        Initialize model from initialization file.
    get_hybridization_channels(cls, hyb_models_dict)
        Get hybrid model hybridization channels.
    get_hybridized_model(self, hyb_model_name)
        Get hybridized model from name.
    get_hybridized_models(self)
        Get hybridized models.
    get_hybridized_models_names(self)
        Get hybridized models names.
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    get_detached_model_parameters(self, is_normalized_out=False)
        Get model parameters detached of gradients.
    get_model_parameters_bounds(self)
        Get model parameters bounds.
    forward(self, features_in)
        Forward propagation.
    check_model_in_normalized(cls, model)
        Check if generic model expects normalized input features.
    check_model_out_normalized(cls, model)
        Check if generic model expects normalized output features.
    features_out_extractor(cls, model_output)
        Extract output features from generic model output.
    save_model_init_file(self)
        Save model initialization file.
    sync_material_model_parameters(self)
        Synchronize model parameters with learnable parameters.
    check_hyb_models_data_scalers(self)
        Check hybridized models data scalers.
    save_model_init_state(self)
        Save model initial state to file.
    save_model_state(self, epoch=None, is_best_state=False, \
                     is_remove_posterior=True)
        Save model state to file.
    load_model_state(self, load_model_state=None, is_remove_posterior=True)
        Load model state from file.
    _check_state_file(self, filename)
        Check if file is model training epoch state file.
    _check_best_state_file(self, filename)
        Check if file is model best state file.
    _remove_posterior_state_files(self, epoch)
        Delete model training epoch state files posterior to given epoch.
    _remove_best_state_files(self)
        Delete existent model best state files.
    _init_data_scalers(self)
        Initialize model data scalers.
    set_data_scalers(self, scaler_features_in, scaler_features_out)
        Set fitted model data scalers.
    fit_data_scalers(self, dataset, is_verbose=False)
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
    def __init__(self, n_features_in, n_features_out, hyb_models_dict,
                 model_directory, model_name='hybrid_model',
                 hybridization_type='identity', is_model_in_normalized=False,
                 is_model_out_normalized=False, is_save_model_init_file=True,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_features_in : int
            Number of input features.
        n_features_out : int
            Number of output features.
        hyb_models_dict : dict
            For each hybridized model (key, str), store dictionary (item, dict)
            with the hybridized model data, namely:
            
            'hyb_model' : Hybridized model (torch.nn.Module)
            
            'hyb_indices' : Hybridization indices (tuple[int])
            
            'is_input_residual' Input residual connection (concatenation)

        model_directory : str
            Directory where model is stored. If None, then all methods that
            depend on an existent model directory become unavailable.
        model_name : str, default='hybrid_model'
            Name of model. If None, then all methods that depend on a valid
            model name become unavailable.
        hybridization_type : str, default='identity'
            Hybridization model type.
        is_model_in_normalized : bool, default=False
            If True, then model expects normalized input features (normalized
            input data has been seen during model training).
        is_model_out_normalized : bool, default=False
            If True, then model expects normalized output features (normalized
            output data has been seen during model training).
        is_save_model_init_file: bool, default=True
            If True, saves model initialization file when model is initialized
            (overwritting existent initialization file), False otherwise. When
            initializing model from initialization file this option should be
            set to False to avoid updating the initialization file and preserve
            fitted data scalers.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(HybridModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_features_in = n_features_in
        self._n_features_out = n_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        self._hyb_models_dict = hyb_models_dict
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridization channels
        hyb_models_names, hyb_models, hyb_channels, \
            hyb_models_input_residual = self.get_hybridization_channels(
                self._hyb_models_dict)
        # Set hybridized model names
        self._hyb_models_names = hyb_models_names
        # Set hybridized models
        self._hyb_models = torch.nn.ModuleList(hyb_models)
        # Set hybridization channels
        self._hyb_channels = hyb_channels
        # Set hybridization models input residual connections
        self._hyb_models_input_residual = hyb_models_input_residual
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Set hybridization model type
        self._hybridization_type = hybridization_type
        # Set model input and output features normalization
        self.is_model_in_normalized = is_model_in_normalized
        self.is_model_out_normalized = is_model_out_normalized
        # Set save initialization file flag
        self._is_save_model_init_file = is_save_model_init_file
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hybridized models data scalers
        self.check_hyb_models_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hybridization model
        self._hybridization_model = \
            HybridizationModel(hybridization_type=hybridization_type)
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model
        model = HybridModel(**model_init_args, is_save_model_init_file=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        model._data_scalers = model_data_scalers
        # Loop over model data scalers and set device
        if model._data_scalers is not None:
            for _, data_scaler in model._data_scalers.items():
                data_scaler.set_device(model._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hybridized models data scalers
        model.check_hyb_models_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model
    # -------------------------------------------------------------------------
    @classmethod
    def get_hybridization_channels(cls, hyb_models_dict):
        """Get hybrid model hybridization channels.
        
        Parameters
        ----------
        hyb_models_dict : dict
            For each hybridized model (key, str), store dictionary (item, dict)
            with the hybridized model data.
        
        Returns
        -------
        hyb_models_names : list[str]
            Hybridized models names.
        hyb_models : list[torch.nn.Module]
            Hybridized models.
        hyb_channels : dict
            For each hybridization channel (key, str[int]), store the
            corresponding hybridized models module index (item, list[int]),
            sorted according with the associated hybridization indices.
            Hybridization channels are labeled between 0 and n_hyb_channel,
            where n_hyb_channel is the total number of channels required to
            place the hybridized models.
        hyb_models_input_residual : dict
            For each hybridization model (key, str), store input residual
            connection flag (item, bool). If True, then input residual
            connection is assigned to hybridized model, False otherwise.
        """
        # Initialize hybridized models names
        hyb_models_names = []
        # Initialize hybridized models
        hyb_models = []
        # Initialize hybridization channels
        hyb_channels = {}
        # Initialize hybridized models input residual connection
        hyb_models_input_residual = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hybridized models indices
        hyb_models_indices = torch.zeros((0, 2), dtype=torch.int)
        # Loop over hybridized models
        for hyb_model_name, hyb_model_data in hyb_models_dict.items():
            # Get hybridized model indices
            hyb_indices = torch.tensor(hyb_model_data['hyb_indices'],
                                       dtype=torch.int).view(1, 2)
            # Store hybridized model name
            hyb_models_names.append(hyb_model_name)
            # Store hybridized model indices
            hyb_models_indices = \
                torch.cat((hyb_models_indices, hyb_indices), dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get sorted column indices (secondary)
        sorted_cols_indices = \
            torch.argsort(hyb_models_indices[:, 1], stable=True)
        # Sort hybridized models
        hyb_models_names = [hyb_models_names[i] for i in sorted_cols_indices]
        hyb_models_indices = hyb_models_indices[sorted_cols_indices, :]
        # Get sorted row indices (primary)
        sorted_rows_indices = \
            torch.argsort(hyb_models_indices[:, 0], stable=True)
        # Sort hybridized models
        hyb_models_names = [hyb_models_names[i] for i in sorted_rows_indices]
        hyb_models_indices = hyb_models_indices[sorted_rows_indices, :]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hybridization channel id
        channel_id = 0
        # Initialize hybridized model index
        model_idx = 0
        # Initialize hybridization channel
        hyb_channels[str(channel_id)] = []
        # Loop over (sorted) hybridized models
        for i, hyb_model_name in enumerate(hyb_models_names):
            # Get hybridized model
            hyb_model = hyb_models_dict[hyb_model_name]['hyb_model']
            # Store hybridized model
            hyb_models.append(hyb_model)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get hybridized model indices
            hyb_indices = hyb_models_indices[i, :]
            # Assemble hybridized model to channel
            if len(hyb_channels[str(channel_id)]) == 0:
                hyb_channels[str(channel_id)].append(model_idx)
            elif hyb_indices[0] == hyb_models_indices[i - 1, 0]:
                # Check for duplicate indices
                if hyb_indices[1] == hyb_models_indices[i - 1, 1]:
                    raise RuntimeError(
                        f'Two hybridization models were assigned similar '
                        f'hybridization indices ({hyb_indices.tolist()}). '
                        f'Check hybridization models \'{hyb_model_name}\' '
                        f'and \'{hyb_models_names[i - 1]}\'.')
                else:
                    hyb_channels[str(channel_id)].append(model_idx)
            else:
                # Increment hybridization channel id
                channel_id += 1
                # Initialize hybridized channel
                hyb_channels[str(channel_id)] = []
                # Assemble hybridized model to channel
                hyb_channels[str(channel_id)].append(model_idx)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get hybridized model input residual connection
            hyb_models_input_residual[hyb_model_name] = \
                hyb_models_dict[hyb_model_name]['is_input_residual']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return hyb_models_names, hyb_models, hyb_channels, \
            hyb_models_input_residual
    # -------------------------------------------------------------------------
    def get_hybridized_model(self, hyb_model_name):
        """Get hybridized model from name.
        
        Parameters
        ----------
        hyb_model_name : str
            Hybridized model name.
            
        Returns
        -------
        hyb_model : torch.nn.Module
            Hybridized model.
        """
        # Check hybridized model name
        if hyb_model_name not in self._hyb_models_names:
            raise RuntimeError(f'Unknown hybridized model '
                               f'\'{hyb_model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get hybridized model module index
        model_idx = self._hyb_models_names.index(hyb_model_name)
        # Get hybridized model
        hyb_model = self._hyb_models[model_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return hyb_model
    # -------------------------------------------------------------------------
    def get_hybridized_models(self):
        """Get hybridized models.
        
        Returns
        -------
        hyb_models : torch.nn.ModuleList
            Hybridized models.
        """
        return self._hyb_models
    # -------------------------------------------------------------------------
    def get_hybridized_models_names(self):
        """Get hybridized models names.
        
        Returns
        -------
        hyb_models_names : list[str]
            Hybridized models names.
        """
        return self._hyb_models_names
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
            # Consistent update of hybridized models device
            if hasattr(self, '_hyb_models_dict'):
                # Loop over hybridized models
                for hyb_model_name in self._hyb_models_names:
                    # Get hybridized model
                    hyb_model = self.get_hybridized_model(hyb_model_name)
                    # Update hybridized model device 
                    if hasattr(hyb_model, 'set_device'):
                        hyb_model.set_device(self._device_type)
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
    def get_detached_model_parameters(self, is_normalized_out=False):
        """Get model parameters detached of gradients.
        
        Only collects parameters from hybridized models with explicit learnable
        parameters. Parameters labels are prefixed with hybridized model name.
        
        Parameters
        ----------
        is_normalized_out : bool, default=False
            If True, then model parameters are normalized.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        # Initialize model parameters
        model_parameters = {}
        # Loop over hybridized models
        for hyb_model_name in self._hyb_models_names:
            # Get hybridized model
            hyb_model = self.get_hybridized_model(hyb_model_name)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if hybridized model parameters are collected
            is_collect_params = (hasattr(hyb_model, 'is_explicit_parameters')
                                 and hyb_model.is_explicit_parameters)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip hybridized model parameters
            if not is_collect_params:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get detached hybridized model parameters
            detached_parameters = hyb_model.get_detached_model_parameters(
                is_normalized_out=is_normalized_out)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Collect parameters (prefix with hybridized model name)
            for param, value in detached_parameters.items():
                # Set parameter label
                param_label = f'{hyb_model_name}_{param}'
                # Store parameter
                model_parameters[param_label] = value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters
    # -------------------------------------------------------------------------
    def get_model_parameters_bounds(self):
        """Get model parameters bounds.
        
        Only collects parameters from hybridized models with explicit learnable
        parameters. Parameters labels are prefixed with hybridized model name.

        Returns
        -------
        model_parameters_bounds : dict
            Model learnable parameters bounds. For each parameter (key, str),
            the corresponding bounds are stored as a
            tuple(lower_bound, upper_bound) (item, tuple).
        """
        # Initialize model parameters bounds
        model_parameters_bounds = {}
        # Loop over hybridized models
        for hyb_model_name in self._hyb_models_names:
            # Get hybridized model
            hyb_model = self.get_hybridized_model(hyb_model_name)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if hybridized model parameters are collected
            is_collect_params = (hasattr(hyb_model, 'is_explicit_parameters')
                                 and hyb_model.is_explicit_parameters)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip hybridized model parameters
            if not is_collect_params:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get hybridized model parameters bounds
            parameters_bounds = hyb_model.get_model_parameters_bounds()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Collect parameters bounds (prefix with hybridized model name)
            for param, value in parameters_bounds.items():
                # Set parameter label
                param_label = f'{hyb_model_name}_{param}'
                # Store parameter bounds
                model_parameters_bounds[param_label] = value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters_bounds
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
        # Get sorted hybridization channels ids
        channels_ids = sorted([int(x) for x in self._hyb_channels.keys()])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hybridized models outputs
        list_features_out = []
        # Loop over hybridization channels
        for channel_id in channels_ids:
            # Initialize hybridization channel output features (recursive)
            features_out = features_in
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over hybridization channel hybridized models
            for hyb_model_idx in self._hyb_channels[str(channel_id)]:
                # Get hybridized model name
                hyb_model_name = self._hyb_models_names[hyb_model_idx]
                # Get hybridized model
                hyb_model = self.get_hybridized_model(hyb_model_name)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check hybridized model input residual connection
                if self._hyb_models_input_residual[hyb_model_name]:
                    # Concatenate input residual connection
                    features_out = \
                        torch.cat((features_out, features_in), dim=-1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get hybridized model input features
                if self.check_model_in_normalized(hyb_model):
                    # Normalize hybridized model input features
                    hyb_model_features_in = hyb_model.data_scaler_transform(
                        features_out, 'features_in', mode='normalize')
                else:
                    hyb_model_features_in = features_out
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform forward propagation of hybridized model
                hyb_model_output = hyb_model(hyb_model_features_in)
                # Extract hybridized model output features
                features_out = self.features_out_extractor(hyb_model_output)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get hybridized model output features
                if self.check_model_out_normalized(hyb_model):
                    features_out = hyb_model.data_scaler_transform(
                        features_out, 'features_out', mode='denormalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store hybridization channel output features
            list_features_out.append(features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute hybridization model output features
        features_out = self._hybridization_model(list_features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize hybrid model output features
        if self.is_model_out_normalized:
            features_out = self.data_scaler_transform(
                tensor=features_out, features_type='features_out',
                mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
    # -------------------------------------------------------------------------
    @classmethod
    def check_model_in_normalized(cls, model):
        """Check if generic model expects normalized input features.
        
        A model expects normalized input features if it has an attribute
        'is_model_in_normalized' set to True.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model.
        
        Returns
        -------
        is_model_in_normalized : bool
            If True, then model expects normalized input features (normalized
            input data has been seen during model training).
        """
        # Get model input features normalization
        if hasattr(model, 'is_model_in_normalized'):
            is_model_in_normalized = model.is_model_in_normalized
        else:
            is_model_in_normalized = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_model_in_normalized
    # -------------------------------------------------------------------------
    @classmethod
    def check_model_out_normalized(cls, model):
        """Check if generic model expects normalized output features.
        
        A model expects normalized output features if it has an attribute
        'is_model_out_normalized' set to True.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model.
        
        Returns
        -------
        is_model_out_normalized : bool
            If True, then model expects normalized output features (normalized
            output data has been seen during model training).
        """
        # Get model output features normalization
        if hasattr(model, 'is_model_out_normalized'):
            is_model_out_normalized = model.is_model_out_normalized
        else:
            is_model_out_normalized = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_model_out_normalized
    # -------------------------------------------------------------------------
    @classmethod
    def features_out_extractor(cls, model_output):
        """Extract output features from generic model output.
        
        Parameters
        ----------
        model_output : {torch.Tensor, tuple}
            Model output.
        
        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Extract output features
        if isinstance(model_output, tuple):
            # Assume output features are stored in the first output index
            # of model output
            features_out = model_output[0]
        elif isinstance(model_output, torch.Tensor):
            # Output features correspond directly to model output
            features_out = model_output
        else:
            raise RuntimeError(f'Unexpected model output of type '
                               f'({type(model_output)}). Output features '
                               f'extraction is not implemented.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
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
        model_init_args['hyb_models_dict'] = self._hyb_models_dict
        model_init_args['model_directory'] = self.model_directory
        model_init_args['model_name'] = self.model_name
        model_init_args['is_model_in_normalized'] = \
            self.is_model_in_normalized
        model_init_args['is_model_out_normalized'] = \
            self.is_model_out_normalized
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
    def sync_material_model_parameters(self):
        """Synchronize model parameters with learnable parameters.
        
        Required to update wrapped constitutive model parameters in
        hybridized models of class RecurrentConstitutiveModel.
        """
        # Loop over hybridized models
        for hyb_model_name in self._hyb_models_names:
            # Get hybridized model
            hyb_model = self.get_hybridized_model(hyb_model_name)
            # Synchronize model parameters with learnable parameters
            if (hasattr(hyb_model, 'sync_material_model_parameters')
                and callable(hyb_model.sync_material_model_parameters)):
                hyb_model.sync_material_model_parameters()
    # -------------------------------------------------------------------------
    def check_hyb_models_data_scalers(self):
        """Check hybridized models data scalers."""
        # Loop over hybridized models
        for hyb_model_name in self._hyb_models_names:
            # Get hybridized model
            hyb_model = self.get_hybridized_model(hyb_model_name)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check hybridized model features normalization
            is_model_in_normalized = \
                self.check_model_in_normalized(hyb_model)
            is_model_out_normalized = \
                self.check_model_out_normalized(hyb_model)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Skip checking procedures if normalization is not required
            if not is_model_in_normalized and not is_model_out_normalized:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check hybridized model data scalers
            if (not hasattr(hyb_model, '_data_scalers')
                    or hyb_model._data_scalers is None):
                raise RuntimeError(
                    f'Data scalers are not available to hybridized '
                    f'model \'{hyb_model_name}\'. Either the data_scalers '
                    f'attribute is missing or is set to None.')
            else:
                data_scalers = hyb_model._data_scalers
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check hybridized model input features data scaler
            is_features_in_data_scaler = \
                ('features_in' in data_scalers.keys()
                 and data_scalers['features_in'] is not None)
            # Check hybridized model input features normalization
            if is_model_in_normalized and not is_features_in_data_scaler:
                raise RuntimeError(
                    f'Data scalers to normalize \'features_in\' are not '
                    f'available to hybridized model \'{hyb_model_name}\' '
                    f'with \'is_model_in_normalized=True\'.'
                    f'The corresponding data_scaler[\'features_in\'] is '
                    f'either missing or set to None.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check hybridized model output features data scaler
            is_features_out_data_scaler = \
                ('features_out' in data_scalers.keys()
                 and data_scalers['features_out'] is not None)
            # Check hybridized model output features normalization
            if is_model_out_normalized and not is_features_out_data_scaler:
                raise RuntimeError(
                    f'Data scalers to normalize \'features_out\' are not '
                    f'available to hybridized model \'{hyb_model_name}\' '
                    f'with \'is_model_out_normalized=True\'.'
                    f'The corresponding data_scaler[\'features_in\'] is '
                    f'either missing or set to None.')
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
        load_model_state : {'best', 'last', int, None}, default=None
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
    def fit_data_scalers(self, dataset, is_verbose=False):
        """Fit model data scalers.
        
        Data scalers are set a standard scalers where features are normalized
        by removing the mean and scaling to unit variance.
        
        Calling this method turns on model data normalization.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Time series data set. Each sample is stored as a dictionary where
            each feature (key, str) data is a torch.Tensor(2d) of shape
            (sequence_length, n_features).
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
        # Instantiate data scalers
        scaler_features_in = TorchStandardScaler(
            n_features=self._n_features_in, device_type=self._device_type)
        scaler_features_out = TorchStandardScaler(
            n_features=self._n_features_out, device_type=self._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get scaling parameters and fit data scalers: input features
        mean, std = standard_partial_fit(dataset, features_type='features_in',
                                         n_features=self._n_features_in)
        scaler_features_in.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: output features
        mean, std = standard_partial_fit(dataset, features_type='features_out',
                                         n_features=self._n_features_out)
        scaler_features_out.set_mean_and_std(mean, std)
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
        data_scaler : sklearn.preprocessing.StandardScaler
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
# =============================================================================
def standard_partial_fit(dataset, features_type, n_features, is_verbose=False):
    """Perform batch fitting of standardization data scalers.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    features_type : str
        Features for which data scaler is required:
        
        'features_in'  : Input features

        'features_out' : Output features
    
    n_features : int
        Number of features to standardize.
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
                raise RuntimeError(f'Mismatch between input features tensor '
                                   f'({features_tensor.shape[-1]}) and '
                                   f'model ({n_features}) number of '
                                   f'features for features type: '
                                   f'{features_type}')
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