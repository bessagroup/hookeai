"""Hybrid model.

Classes
-------
HybridModel(torch.nn.Module)
    Hybrid model.
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
from model_architectures.hybrid_base_model.model.hybridization import \
    HybridizationModel
from model_architectures.procedures.model_data_scaling import init_data_scalers
from model_architectures.procedures.model_data_scaling import \
    data_scaler_transform
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
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
                if data_scaler is not None:
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Increment hybridized model index
            model_idx += 1
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
            features_out = data_scaler_transform(self, tensor=features_out,
                                                 features_type='features_out',
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
        model_init_args['hybridization_type'] = self._hybridization_type
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