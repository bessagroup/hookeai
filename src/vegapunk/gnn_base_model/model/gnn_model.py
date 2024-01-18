"""Graph Neural Network based model.

Classes
-------
GNNEPDBaseModel(torch.nn.Module)
    GNN Encoder-Processor-Decoder base model.
TorchStandardScaler
    PyTorch tensor standardization data scaler.
    
Functions
---------
graph_standard_partial_fit
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
import torch_geometric.nn
import torch_geometric.data
import torch_geometric.loader
import tqdm
import sklearn.preprocessing
# Local
from gnn_base_model.model.gnn_epd_model import EncodeProcessDecode
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class GNNEPDBaseModel(torch.nn.Module):
    """GNN Encoder-Processor-Decoder base model.
    
    Attributes
    ----------
    model_directory : str
        Directory where model is stored.
    model_name : str, default='gnn_epd_model'
        Name of model.
    _n_node_in : int
        Number of node input features.
    _n_node_out : int
        Number of node output features.
    _n_edge_in : int
        Number of edge input features.
    _n_edge_out : int
        Number of edge output features.
    _n_global_in : int
        Number of global input features.
    _n_global_out : int
        Number of global output features.
    _n_message_steps : int
        Number of message-passing steps.
    _enc_n_hidden_layers : int
        Encoder: Number of hidden layers of multilayer feed-forward neural
        network update functions.
    _pro_n_hidden_layers : int
        Processor: Number of hidden layers of multilayer feed-forward
        neural network update functions.
    _dec_n_hidden_layers : int
        Decoder: Number of hidden layers of multilayer feed-forward neural
        network update functions.
    _hidden_layer_size : int
        Number of neurons of hidden layers of multilayer feed-forward
        neural network update functions.
    _pro_edge_to_node_aggr : {'add',}, default='add'
        Processor: Edge-to-node aggregation scheme.
    _pro_node_to_global_aggr : {'add',}, default='add'
        Processor: Node-to-global aggregation scheme.
    _enc_node_hidden_activ_type : str, default='identity'
        Encoder: Hidden unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _enc_node_output_activ_type : str, default='identity'
        Encoder: Output unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _enc_edge_hidden_activ_type : str, default='identity'
        Encoder: Hidden unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _enc_edge_output_activ_type : str, default='identity'
        Encoder: Output unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _enc_global_hidden_activ_type : str, default='identity'
        Encoder: Hidden unit activation function type of global update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _enc_global_output_activ_type : str, default='identity'
        Encoder: Output unit activation function type of global update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_node_hidden_activ_type : str, default='identity'
        Processor: Hidden unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_node_output_activ_type : str, default='identity'
        Processor: Output unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_edge_hidden_activ_type : str, default='identity'
        Processor: Hidden unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_edge_output_activ_type : str, default='identity'
        Processor: Output unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_global_hidden_activ_type : str, default='identity'
        Processor: Hidden unit activation function type of global update
        function (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _pro_global_output_activ_type : str, default='identity'
        Processor: Output unit activation function type of global update
        function (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_node_hidden_activ_type : str, default='identity'
        Decoder: Hidden unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_node_output_activ_type : str, default='identity'
        Decoder: Output unit activation function type of node update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_edge_hidden_activ_type : str, default='identity'
        Decoder: Hidden unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_edge_output_activ_type : str, default='identity'
        Decoder: Output unit activation function type of edge update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_global_hidden_activ_type : str, default='identity'
        Decoder: Hidden unit activation function type of global update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _dec_global_output_activ_type : str, default='identity'
        Decoder: Output unit activation function type of global update function
        (multilayer feed-forward neural network). Defaults to identity
        (linear) unit activation function.
    _gnn_epd_model : EncodeProcessDecode
        GNN-based Encoder-Process-Decoder model.
    _device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.
    is_data_normalization : bool, default=False
        If True, then input and output features are normalized for training
        False otherwise. Data scalers need to be fitted with fit_data_scalers()
        and are stored as model attributes.
    _data_scalers : dict
        Data scaler (item, sklearn.preprocessing.StandardScaler) for each
        feature data (key, str).

    Methods
    -------
    init_model_from_file(model_directory)
        Initialize GNN-based model from initialization file.
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self)
        Forward propagation.
    save_model_init_file(self)
        Save model class initialization attributes.
    get_input_features_from_graph(self, graph, is_normalized=False)
        Get input features from graph.
    get_output_features_from_graph(self, graph, is_normalized=False)
        Get output features from graph.
    predict_node_output_features(self, input_graph)
        Predict node output features.
    predict_output_features(self, input_graph, is_normalized=False)
        Predict output features.
    save_model_state(self)
        Save model state to file.
    load_model_state(self)
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
    fit_data_scalers(self, dataset, is_verbose=False)
        Fit model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    get_fitted_data_scaler(self, features_type)
        Get fitted model data scalers.
    load_model_data_scalers_from_file(self)
        Load data scalers from model initialization file.
    check_normalized_return(self)
        Check if model data normalization is available.
    """
    def __init__(self, n_node_in, n_node_out, n_edge_in, n_edge_out,
                 n_global_in, n_global_out, n_message_steps,
                 enc_n_hidden_layers, pro_n_hidden_layers, dec_n_hidden_layers,
                 hidden_layer_size, model_directory,
                 model_name='gnn_epd_model',
                 is_data_normalization=False,
                 pro_edge_to_node_aggr='add', pro_node_to_global_aggr='add',
                 enc_node_hidden_activ_type='identity',
                 enc_node_output_activ_type='identity',
                 enc_edge_hidden_activ_type='identity',
                 enc_edge_output_activ_type='identity',
                 enc_global_hidden_activ_type='identity',
                 enc_global_output_activ_type='identity',
                 pro_node_hidden_activ_type='identity',
                 pro_node_output_activ_type='identity',
                 pro_edge_hidden_activ_type='identity',
                 pro_edge_output_activ_type='identity',
                 pro_global_hidden_activ_type='identity',
                 pro_global_output_activ_type='identity',
                 dec_node_hidden_activ_type='identity',
                 dec_node_output_activ_type='identity',
                 dec_edge_hidden_activ_type='identity',
                 dec_edge_output_activ_type='identity',
                 dec_global_hidden_activ_type='identity',
                 dec_global_output_activ_type='identity',
                 is_save_model_init_file=True,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        n_node_in : int
            Number of node input features.
        n_node_out : int
            Number of node output features.
        n_edge_in : int
            Number of edge input features.
        n_edge_out : int
            Number of edge output features.
        n_global_in : int
            Number of global input features.
        n_global_out : int
            Number of global output features.
        n_message_steps : int
            Number of message-passing steps.
        enc_n_hidden_layers : int
            Encoder: Number of hidden layers of multilayer feed-forward neural
            network update functions.
        pro_n_hidden_layers : int
            Processor: Number of hidden layers of multilayer feed-forward
            neural network update functions.
        dec_n_hidden_layers : int
            Decoder: Number of hidden layers of multilayer feed-forward neural
            network update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        model_directory : str
            Directory where model is stored.
        model_name : str, default='gnn_epd_model'
            Name of model.
        is_data_normalization : bool, default=False
            If True, then input and output features are normalized for
            training, False otherwise. Data scalers need to be fitted with
            fit_data_scalers() and are stored as model attributes.
        pro_edge_to_node_aggr : {'add',}, default='add'
            Processor: Edge-to-node aggregation scheme.
        pro_node_to_global_aggr : {'add',}, default='add'
            Processor: Node-to-global aggregation scheme.
        enc_node_hidden_activ_type : str, default='identity'
            Encoder: Hidden unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        enc_node_output_activ_type : str, default='identity'
            Encoder: Output unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        enc_edge_hidden_activ_type : str, default='identity'
            Encoder: Hidden unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        enc_edge_output_activ_type : str, default='identity'
            Encoder: Output unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        enc_global_hidden_activ_type : str, default='identity'
            Encoder: Hidden unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        enc_global_output_activ_type : str, default='identity'
            Encoder: Output unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_node_hidden_activ_type : str, default='identity'
            Processor: Hidden unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_node_output_activ_type : str, default='identity'
            Processor: Output unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_edge_hidden_activ_type : str, default='identity'
            Processor: Hidden unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_edge_output_activ_type : str, default='identity'
            Processor: Output unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_global_hidden_activ_type : str, default='identity'
            Processor: Hidden unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_global_output_activ_type : str, default='identity'
            Processor: Output unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_node_hidden_activ_type : str, default='identity'
            Decoder: Hidden unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_node_output_activ_type : str, default='identity'
            Decoder: Output unit activation function type of node update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_edge_hidden_activ_type : str, default='identity'
            Decoder: Hidden unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_edge_output_activ_type : str, default='identity'
            Decoder: Output unit activation function type of edge update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_global_hidden_activ_type : str, default='identity'
            Decoder: Hidden unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_global_output_activ_type : str, default='identity'
            Decoder: Output unit activation function type of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
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
        super(GNNEPDBaseModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = n_node_in
        self._n_node_out = n_node_out
        self._n_edge_in = n_edge_in
        self._n_edge_out= n_edge_out
        self._n_global_in = n_global_in
        self._n_global_out = n_global_out
        # Set architecture parameters
        self._n_message_steps = n_message_steps
        self._enc_n_hidden_layers = enc_n_hidden_layers
        self._pro_n_hidden_layers = pro_n_hidden_layers
        self._dec_n_hidden_layers = dec_n_hidden_layers
        self._hidden_layer_size = hidden_layer_size
        self._pro_edge_to_node_aggr = pro_edge_to_node_aggr
        self._pro_node_to_global_aggr = pro_node_to_global_aggr
        self._enc_node_hidden_activ_type = enc_node_hidden_activ_type
        self._enc_node_output_activ_type = enc_node_output_activ_type
        self._enc_edge_hidden_activ_type = enc_edge_hidden_activ_type
        self._enc_edge_output_activ_type = enc_edge_output_activ_type
        self._enc_global_hidden_activ_type = enc_global_hidden_activ_type
        self._enc_global_output_activ_type = enc_global_output_activ_type
        self._pro_node_hidden_activ_type = pro_node_hidden_activ_type
        self._pro_node_output_activ_type = pro_node_output_activ_type
        self._pro_edge_hidden_activ_type = pro_edge_hidden_activ_type
        self._pro_edge_output_activ_type = pro_edge_output_activ_type
        self._pro_global_hidden_activ_type = pro_global_hidden_activ_type
        self._pro_global_output_activ_type = pro_global_output_activ_type
        self._dec_node_hidden_activ_type = dec_node_hidden_activ_type
        self._dec_node_output_activ_type = dec_node_output_activ_type
        self._dec_edge_hidden_activ_type = dec_edge_hidden_activ_type
        self._dec_edge_output_activ_type = dec_edge_output_activ_type
        self._dec_global_hidden_activ_type = dec_global_hidden_activ_type
        self._dec_global_output_activ_type = dec_global_output_activ_type
        # Set model directory and name
        if os.path.isdir(model_directory):
            self.model_directory = str(model_directory)
        else:
            raise RuntimeError('The model directory has not been found.')
        if not isinstance(model_name, str):
            raise RuntimeError('The model name must be a string.')
        else:
            self.model_name = model_name
        # Set normalization flag
        self.is_data_normalization = is_data_normalization
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model
        self._gnn_epd_model = EncodeProcessDecode(
            n_message_steps=n_message_steps,
            n_node_out=n_node_out, n_edge_out=n_edge_out,
            n_global_out=n_global_out,
            enc_n_hidden_layers=enc_n_hidden_layers,
            pro_n_hidden_layers=pro_n_hidden_layers,
            dec_n_hidden_layers=dec_n_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
            pro_edge_to_node_aggr=pro_edge_to_node_aggr,
            pro_node_to_global_aggr=pro_node_to_global_aggr,
            enc_node_hidden_activation=type(self).get_pytorch_activation(
                self._enc_node_hidden_activ_type),
            enc_node_output_activation=type(self).get_pytorch_activation(
                self._enc_node_output_activ_type),
            enc_edge_hidden_activation=type(self).get_pytorch_activation(
                self._enc_edge_hidden_activ_type),
            enc_edge_output_activation=type(self).get_pytorch_activation(
                self._enc_edge_output_activ_type),
            enc_global_hidden_activation=type(self).get_pytorch_activation(
                self._enc_global_hidden_activ_type),
            enc_global_output_activation=type(self).get_pytorch_activation(
                self._enc_global_output_activ_type),
            pro_node_hidden_activation=type(self).get_pytorch_activation(
                self._pro_node_hidden_activ_type),
            pro_node_output_activation=type(self).get_pytorch_activation(
                self._pro_node_output_activ_type),
            pro_edge_hidden_activation=type(self).get_pytorch_activation(
                self._pro_edge_hidden_activ_type),
            pro_edge_output_activation=type(self).get_pytorch_activation(
                self._pro_edge_output_activ_type),
            pro_global_hidden_activation=type(self).get_pytorch_activation(
                self._pro_global_hidden_activ_type),
            pro_global_output_activation=type(self).get_pytorch_activation(
                self._pro_global_output_activ_type),
            dec_node_hidden_activation=type(self).get_pytorch_activation(
                self._dec_node_hidden_activ_type),
            dec_node_output_activation=type(self).get_pytorch_activation(
                self._dec_node_output_activ_type),
            dec_edge_hidden_activation=type(self).get_pytorch_activation(
                self._dec_edge_hidden_activ_type),
            dec_edge_output_activation=type(self).get_pytorch_activation(
                self._dec_edge_output_activ_type),
            dec_global_hidden_activation=type(self).get_pytorch_activation(
                self._dec_global_hidden_activ_type),
            dec_global_output_activation=type(self).get_pytorch_activation(
                self._dec_global_output_activ_type),
            is_node_res_connect=False, is_edge_res_connect=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data scalers
        self._data_scalers = None
        if self.is_data_normalization:
            self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model initialization file
        if is_save_model_init_file:
            self.save_model_init_file()
    # -------------------------------------------------------------------------
    @staticmethod
    def init_model_from_file(model_directory):
        """Initialize model from initialization file.
        
        Initialization file is assumed to be stored in the model directory
        under the name model_init_file.pkl.
        
        Parameters
        ----------
        model_directory : str
            Directory where model is stored.
        """
        # Check model directory
        if not os.path.isdir(model_directory):
            raise RuntimeError('The model directory has not been found:\n\n'
                               + model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The model initialization file has not been '
                               'found:\n\n' + model_init_file_path)
        else:
            with open(model_init_file_path, 'rb') as model_init_file:
                model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization attributes
        model_init_args = model_init_attributes['model_init_args']
        # Update model directory
        model_init_args['model_directory'] = model_directory
        # Initialize model
        model = GNNEPDBaseModel(**model_init_args,
                                is_save_model_init_file=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data scalers
        model_data_scalers = model_init_attributes['model_data_scalers']
        model._data_scalers = model_data_scalers
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
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def forward(self, graph, is_normalized=False, batch_vector=None):
        """Forward propagation.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from graph, False
            otherwise.
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.
            
        Returns
        -------
        node_features_out : {torch.Tensor, None}
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : {torch.Tensor, None}
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict output features
        node_features_out, edge_features_out, global_features_out = \
            self.predict_output_features(
                graph, is_normalized=is_normalized, batch_vector=batch_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
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
        model_init_args['n_node_in'] = self._n_node_in
        model_init_args['n_node_out'] = self._n_node_out
        model_init_args['n_edge_in'] = self._n_edge_in
        model_init_args['n_edge_out'] = self._n_edge_out
        model_init_args['n_global_in'] = self._n_global_in
        model_init_args['n_global_out'] = self._n_global_out
        model_init_args['n_message_steps'] = self._n_message_steps
        model_init_args['dec_n_hidden_layers'] = self._enc_n_hidden_layers
        model_init_args['pro_n_hidden_layers'] = self._pro_n_hidden_layers
        model_init_args['enc_n_hidden_layers'] = self._dec_n_hidden_layers
        model_init_args['pro_edge_to_node_aggr'] = \
            self._pro_edge_to_node_aggr
        model_init_args['pro_node_to_global_aggr'] = \
            self._pro_node_to_global_aggr
        model_init_args['hidden_layer_size'] = self._hidden_layer_size
        model_init_args['enc_node_hidden_activ_type'] = \
            self._enc_node_hidden_activ_type
        model_init_args['enc_node_output_activ_type'] = \
            self._enc_node_output_activ_type
        model_init_args['enc_edge_hidden_activ_type'] = \
            self._enc_edge_hidden_activ_type
        model_init_args['enc_edge_output_activ_type'] = \
            self._enc_edge_output_activ_type
        model_init_args['enc_global_hidden_activ_type'] = \
            self._enc_global_hidden_activ_type
        model_init_args['enc_global_output_activ_type'] = \
            self._enc_global_output_activ_type
        model_init_args['pro_node_hidden_activ_type'] = \
            self._pro_node_hidden_activ_type
        model_init_args['pro_node_output_activ_type'] = \
            self._pro_node_output_activ_type
        model_init_args['pro_edge_hidden_activ_type'] = \
            self._pro_edge_hidden_activ_type
        model_init_args['pro_edge_output_activ_type'] = \
            self._enc_edge_output_activ_type
        model_init_args['pro_global_hidden_activ_type'] = \
            self._pro_global_hidden_activ_type
        model_init_args['pro_global_output_activ_type'] = \
            self._enc_global_output_activ_type
        model_init_args['dec_node_hidden_activ_type'] = \
            self._dec_node_hidden_activ_type
        model_init_args['dec_node_output_activ_type'] = \
            self._dec_node_output_activ_type
        model_init_args['dec_edge_hidden_activ_type'] = \
            self._dec_edge_hidden_activ_type
        model_init_args['dec_edge_output_activ_type'] = \
            self._dec_edge_output_activ_type
        model_init_args['dec_global_hidden_activ_type'] = \
            self._dec_global_hidden_activ_type
        model_init_args['dec_global_output_activ_type'] = \
            self._dec_global_output_activ_type
        model_init_args['model_directory'] = self.model_directory
        model_init_args['model_name'] = self.model_name
        model_init_args['is_data_normalization'] = self.is_data_normalization
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
    def get_input_features_from_graph(self, graph, is_normalized=False):
        """Get input features from graph.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized input features from graph, False otherwise.
        
        Returns
        -------
        node_features_in : {torch.Tensor, None}
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : {torch.Tensor, None}
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_in : {torch.Tensor, None}
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        edges_indexes : {torch.Tensor, None}
            Edges indexes matrix stored as torch.Tensor(2d) with shape
            (2, n_edges), where the i-th global is stored in
            edges_indexes[:, i] as (start_node_index, end_node_index).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check consistency with simulator
        if graph.num_node_features != self._n_node_in:
            raise RuntimeError(f'Input graph ({graph.num_node_features}) and '
                               f'simulator ({self._n_node_in}) number of node '
                               f'features are not consistent.')
        if graph.num_edge_features != self._n_edge_in:
            raise RuntimeError(f'Input graph ({graph.num_edge_features}) and '
                               f'simulator ({self._n_edge_in}) number of edge '
                               f'features are not consistent.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get nodes features from graph
        if 'x' in graph.keys() and isinstance(graph.x, torch.Tensor):
            node_features_in = graph.x.clone()
        else:
            # Preserve total number of nodes
            node_features_in = torch.empty(graph.num_nodes, 0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get edges features from graph
        if 'edge_attr' in graph.keys() and isinstance(graph.edge_attr,
                                                      torch.Tensor):
            edge_features_in = graph.edge_attr.clone()
        else:
            edge_features_in = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get edges features from graph
        if ('global_features_matrix' in graph.keys()
            and isinstance(graph.global_features_matrix, torch.Tensor)):
            global_features_in = graph.global_features_matrix.clone()
        else:
            global_features_in = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get graph edges indexes
        if 'edge_index' in graph.keys() and isinstance(graph.edge_index,
                                                       torch.Tensor):
            edges_indexes = graph.edge_index.clone()
        else:
            edges_indexes = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize input features data
        if is_normalized:
            if node_features_in is not None and node_features_in.numel() > 0:
                node_features_in = self.data_scaler_transform(
                    tensor=node_features_in,
                    features_type='node_features_in',
                    mode='normalize')
            if edge_features_in is not None:
                edge_features_in = self.data_scaler_transform(
                    tensor=edge_features_in,
                    features_type='edge_features_in',
                    mode='normalize')
            if global_features_in is not None:
                global_features_in = self.data_scaler_transform(
                    tensor=global_features_in,
                    features_type='global_features_in',
                    mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return (node_features_in, edge_features_in, global_features_in,
                edges_indexes)
    # -------------------------------------------------------------------------
    def get_output_features_from_graph(self, graph, is_normalized=False):
        """Get output features from graph.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from graph, False
            otherwise.
        
        Returns
        -------
        node_features_out : {torch.Tensor, None}
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : {torch.Tensor, None}
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~               
        # Get features from graph
        if 'y' in graph.keys() and isinstance(graph.y, torch.Tensor):
            node_features_out = graph.y.clone()
        else:
            node_features_out = None
        if ('edge_targets_matrix' in graph.keys()
            and isinstance(graph.edge_targets_matrix, torch.Tensor)):
                edge_features_out = graph.edge_targets_matrix.clone()
        else:
            edge_features_out = None
        if ('global_targets_matrix' in graph.keys()
            and isinstance(graph.global_targets_matrix, torch.Tensor)):
                global_features_out = graph.global_targets_matrix.clone()
        else:
            global_features_out = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check consistency with simulator
        if (node_features_out is not None
                and node_features_out.shape[1] != self._n_node_out):
            raise RuntimeError(f'Input graph ({node_features_out.shape[1]}) '
                               f'and simulator ({self._n_node_out}) number of '
                               f'output node features are not consistent.')
        if (edge_features_out is not None
                and edge_features_out.shape[1] != self._n_edge_out):
            raise RuntimeError(f'Input graph ({edge_features_out.shape[1]}) '
                               f'and simulator ({self._n_edge_out}) number of '
                               f'output edge features are not consistent.')
        if (global_features_out is not None
                and global_features_out.shape[1] != self._n_global_out):
            raise RuntimeError(f'Input graph ({global_features_out.shape[1]}) '
                               f'and simulator ({self._n_global_out}) number '
                               f'of output global features are not '
                               f'consistent.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        # Normalize output features data
        if is_normalized:                
            if node_features_out is not None:
                node_features_out = self.data_scaler_transform(
                    tensor=node_features_out,
                    features_type='node_features_out',
                    mode='normalize')
            if edge_features_out is not None:
                edge_features_out = self.data_scaler_transform(
                    tensor=edge_features_out,
                    features_type='edge_features_out',
                    mode='normalize')
            if global_features_out is not None:
                global_features_out = self.data_scaler_transform(
                    tensor=global_features_out,
                    features_type='global_features_out',
                    mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
    # -------------------------------------------------------------------------
    def predict_node_output_features(self, input_graph, is_normalized=False,
                                     batch_vector=None):
        """Predict node output features.
        
        Parameters
        ----------
        input_graph : torch_geometric.data.Data
            Homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from graph, False
            otherwise.
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.

        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """
        # Check input graph type
        if not isinstance(input_graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph must be instance of '
                               'torch_geometric.data.Data.')
        # Check model data normalization
        if is_normalized:
            self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from input graph
        node_features_in, edge_features_in, global_features_in, \
            edges_indexes = self.get_input_features_from_graph(
                input_graph, is_normalized=self.is_data_normalization)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict node output features
        node_features_out, _, _ = \
            self._gnn_epd_model(edges_indexes=edges_indexes,
                                node_features_in=node_features_in,
                                edge_features_in=edge_features_in,
                                batch_vector=batch_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features data
        if self.is_data_normalization and not is_normalized:
            if node_features_out is not None:
                node_features_out = self.data_scaler_transform(
                    tensor=node_features_out,
                    features_type='node_features_out',
                    mode='denormalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out
    # -------------------------------------------------------------------------
    def predict_output_features(self, input_graph, is_normalized=False,
                                batch_vector=None):
        """Predict output features.
        
        Parameters
        ----------
        input_graph : torch_geometric.data.Data
            Homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from graph, False
            otherwise.
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.

        Returns
        -------
        node_features_out : {torch.Tensor, None}
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : {torch.Tensor, None}
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check input graph type
        if not isinstance(input_graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph must be instance of '
                               'torch_geometric.data.Data.')
        # Check model data normalization
        if is_normalized:
            self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from input graph
        node_features_in, edge_features_in, global_features_in, \
            edges_indexes = self.get_input_features_from_graph(
                input_graph, is_normalized=self.is_data_normalization)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict output features
        node_features_out, edge_features_out, global_features_out = \
            self._gnn_epd_model(edges_indexes=edges_indexes,
                                node_features_in=node_features_in,
                                edge_features_in=edge_features_in,
                                global_features_in=global_features_in,
                                batch_vector=batch_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features data
        if self.is_data_normalization and not is_normalized:
            if node_features_out is not None:
                node_features_out = self.data_scaler_transform(
                    tensor=node_features_out,
                    features_type='node_features_out',
                    mode='denormalize')
            if edge_features_out is not None:
                edge_features_out = self.data_scaler_transform(
                    tensor=edge_features_out,
                    features_type='edge_features_out',
                    mode='denormalize')
            if global_features_out is not None:
                global_features_out = self.data_scaler_transform(
                    tensor=global_features_out,
                    features_type='global_features_out',
                    mode='denormalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
    # -------------------------------------------------------------------------
    @staticmethod
    def get_pytorch_activation(activation_type, **kwargs):
        """Get PyTorch unit activation function.
    
        Parameters
        ----------
        activation_type : {'identity', 'relu', 'tanh'}
            Unit activation function type:
            
            'identity' : Linear (torch.nn.Identity)
            
            'relu'     : Rectified linear unit (torch.nn.Identity)
            
            'tanh'     : Hyperbolic Tangent (torch.nn.Tanh)
            
        **kwargs
            Arguments of torch.nn._Module initializer.
            
        Returns
        -------
        activation_function : torch.nn._Module
            PyTorch unit activation function.
        """
        # Set available unit activation function types
        available = ('identity', 'relu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get unit activation function
        if activation_type == 'identity':
            activation_function = torch.nn.Identity(**kwargs)
        elif activation_type == 'relu':
            activation_function = torch.nn.ReLU(**kwargs)
        elif activation_type == 'tanh':
            activation_function = torch.nn.Tanh(**kwargs)
        else:
            raise RuntimeError(f'Unknown or unavailable PyTorch unit '
                               f'activation function: \'{activation_type}\'.'
                               f'\n\nAvailable: {available}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return activation_function
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
        
        Parameters
        ----------            
        load_model_state : {'best', 'last', int, None}, default=None
            Load available GNN-based model state from the model directory.
            Options:
            
            'best'      : Model state corresponding to best performance
            
            'last'      : Model state corresponding to highest training epoch
            
            int         : Model state corresponding to given training epoch
            
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
        self._data_scalers['node_features_in'] = None
        self._data_scalers['edge_features_in'] = None
        self._data_scalers['global_features_in'] = None
        self._data_scalers['node_features_out'] = None
        self._data_scalers['edge_features_out'] = None
        self._data_scalers['global_features_out'] = None
    # -------------------------------------------------------------------------
    def fit_data_scalers(self, dataset, is_verbose=False):
        """Fit model data scalers.
        
        Data scalars are set a standard scalers where features are normalized
        by removing the mean and scaling to unit variance.
        
        Calling this method turns on model data normalization.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            GNN-based data set. Each sample corresponds to a
            torch_geometric.data.Data object describing a homogeneous graph.
        is_verbose : bool, default=False
            If True, enable verbose output.
        """
        if is_verbose:
            print('\nFitting GNN-based model data scalers'
                  '\n------------------------------------\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data normalization
        self.is_data_normalization = True
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate data scalers
        scaler_node_in = None
        if self._n_node_in > 0:
            scaler_node_in = TorchStandardScaler(
                n_features=self._n_node_in, device_type=self._device_type)
        scaler_edge_in = None
        if self._n_edge_in > 0:
            scaler_edge_in = TorchStandardScaler(
                n_features=self._n_edge_in, device_type=self._device_type)
        scaler_global_in = None
        if self._n_global_in > 0:
            scaler_global_in = TorchStandardScaler(
                n_features=self._n_global_in, device_type=self._device_type)
        scaler_node_out = None
        if self._n_node_out > 0:
            scaler_node_out = TorchStandardScaler(
                n_features=self._n_node_out, device_type=self._device_type)
        scaler_edge_out = None
        if self._n_edge_out > 0:
            scaler_edge_out = TorchStandardScaler(
                n_features=self._n_edge_out, device_type=self._device_type)
        scaler_global_out = None
        if self._n_global_out > 0:
            scaler_global_out = TorchStandardScaler(
                n_features=self._n_global_out, device_type=self._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get scaling parameters and fit data scalers: node input features
        if self._n_node_in > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='node_features_in',
                n_features=self._n_node_in)
            scaler_node_in.set_mean_and_std(mean, std)        
        # Get scaling parameters and fit data scalers: edge input features
        if self._n_edge_in > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='edge_features_in',
                n_features=self._n_edge_in)
            scaler_edge_in.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: global input features
        if self._n_global_in > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='global_features_in',
                n_features=self._n_edge_in)
            scaler_global_in.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: node output features
        if self._n_node_out > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='node_features_out',
                n_features=self._n_node_out)
            scaler_node_out.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: edge output features
        if self._n_edge_out > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='edge_features_out',
                n_features=self._n_edge_out)
            scaler_edge_out.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: global output features
        if self._n_global_out > 0:
            mean, std = graph_standard_partial_fit(
                dataset, features_type='global_features_out',
                n_features=self._n_global_out)
            scaler_global_out.set_mean_and_std(mean, std)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Setting fitted standard scalers...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['node_features_in'] = scaler_node_in
        self._data_scalers['edge_features_in'] = scaler_edge_in
        self._data_scalers['global_features_in'] = scaler_global_in
        self._data_scalers['node_features_out'] = scaler_node_out
        self._data_scalers['edge_features_out'] = scaler_edge_out
        self._data_scalers['global_features_out'] = scaler_global_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update model initialization file with fitted data scalers
        self.save_model_init_file()
    # -------------------------------------------------------------------------
    def get_fitted_data_scaler(self, features_type):
        """Get fitted model data scalers.
        
        Parameters
        ----------
        features_type : str
            Features for which data scaler is required:
            
            'node_features_in'    : Node features input matrix
            
            'edge_features_in'    : Edge features input matrix
            
            'global_features_in'  : Global features input matrix
            
            'node_features_out'   : Node features output matrix

            'edge_features_out'   : Edge features output matrix

            'global_features_out' : Global features output matrix

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
            
            'node_features_in'    : Node features input matrix
            
            'edge_features_in'    : Edge features input matrix
            
            'global_features_in'  : Global features input matrix
            
            'node_features_out'   : Node features output matrix

            'edge_features_out'   : Edge features output matrix

            'global_features_out' : Global features output matrix

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
    set_mean(self, mean)
        Set features standardization mean tensor.
    set_std(self, std)
        Set features standardization standard deviation tensor.    
    fit(self, tensor)
        Fit features standardization mean and standard deviation tensors.
    transform(self, tensor)
        Standardize features tensor.
    inverse_transform(self, tensor)
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
    def fit(self, tensor, is_bessel=False):
        """Fit features standardization mean and standard deviation tensors.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
        is_bessel : bool, default=False
            If True, apply Bessel's correction to compute standard deviation,
            False otherwise.
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set standardization mean and standard deviation
        self._mean = self._check_mean(torch.mean(tensor, dim=0))
        self._std = self._check_std(torch.std(tensor, dim=0,
                                              correction=int(is_bessel)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def transform(self, tensor):
        """Standardize features tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
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
        # Check transformed tensor
        if not torch.equal(torch.tensor(transformed_tensor.size()),
                           torch.tensor(tensor.size())):
            raise RuntimeError('Input and transformed tensors do not have the '
                               'same shape.')
        elif torch.any(torch.isnan(transformed_tensor)):
            raise RuntimeError('One or more NaN elements were detected in '
                               'the transformed tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return transformed_tensor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def inverse_transform(self, tensor):
        """Destandardize features tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Standardized features PyTorch tensor stored as torch.Tensor with
            shape (n_samples, n_features).
            
        Returns
        -------
        transformed_tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of samples
        n_samples = tensor.shape[0]
        # Build mean and standard deviation tensors for standardization
        mean = torch.tile(self._mean, (n_samples, 1)).to(self._device)
        std = torch.tile(self._std, (n_samples, 1)).to(self._device)
        # Destandardize features tensor
        transformed_tensor = torch.mul(tensor, std) + mean
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if not torch.equal(torch.tensor(transformed_tensor.size()),
                           torch.tensor(tensor.size())):
            raise RuntimeError('Input and transformed tensors do not have the '
                               'same shape.')
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
        elif len(tensor.shape) != 2:
            raise RuntimeError('Features tensor is not a torch.Tensor with '
                               'shape (n_samples, n_features).')
        elif tensor.shape[1] != self._n_features:
            raise RuntimeError('Features tensor is not consistent with data'
                               'scaler number of features.')
# =============================================================================
def graph_standard_partial_fit(dataset, features_type, n_features,
                               is_verbose=False):
    """Perform batch fitting of standardization data scalers.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        GNN-based data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    features_type : str
        Features for which data scaler is required:
        
        'node_features_in'    : Node features input matrix
        
        'edge_features_in'    : Edge features input matrix
        
        'global_features_in'  : Global features input matrix
        
        'node_features_out'   : Node features output matrix

        'edge_features_out'   : Edge features output matrix

        'global_features_out' : Global features output matrix
    
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
    data_loader = \
        torch_geometric.loader.dataloader.DataLoader(dataset=dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over graph samples
    for pyg_graph in tqdm.tqdm(data_loader,
                               desc='> Processing data samples: ',
                               disable=not is_verbose):
        # Check sample graph type
        if not isinstance(pyg_graph, torch_geometric.data.Data):
            raise RuntimeError('Graph sample must be instance of '
                               'torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set features mapping
        features_map = {'node_features_in': 'x',
                        'edge_features_in': 'edge_attr',
                        'global_features_in': 'global_features_matrix',
                        'node_features_out': 'y',
                        'edge_features_out': 'edge_targets_matrix',
                        'global_features_out': 'global_targets_matrix'}
        # Check sample graph feature
        if features_map[features_type] not in pyg_graph.keys():
            raise RuntimeError(f'Unknown or unexistent attribute '
                               f'{features_map[features_type]} from graph '
                               f'sample.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features tensor
        if features_type == 'node_features_in':
            features_tensor = pyg_graph.x
        elif features_type == 'edge_features_in':
            features_tensor = pyg_graph.edge_attr
        elif features_type == 'global_features_in':
            features_tensor = pyg_graph.global_features_matrix
        elif features_type == 'node_features_out':
            features_tensor = pyg_graph.y
        elif features_type == 'edge_features_out':
            features_tensor = pyg_graph.edge_targets_matrix
        elif features_type == 'global_features_out':
            features_tensor = pyg_graph.global_targets_matrix
        else:
            raise RuntimeError('Unknown features type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process sample to fit data scaler
        if isinstance(features_tensor, torch.Tensor):
            # Check number of features
            if features_tensor.shape[1] != n_features:
                raise RuntimeError(f'Mismatch between input graph '
                                   f'({features_tensor.shape[1]}) and '
                                   f'model ({n_features}) number of '
                                   f'features for features type: '
                                   f'{features_type}')
            # Process sample
            data_scaler.partial_fit(features_tensor.clone())
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
    elif len(mean) != features_tensor.shape[1]:
        raise RuntimeError('Features standardization mean tensor is not a '
                           'torch.Tensor(1d) with shape (n_features,).')
    # Check features standardization standard deviation tensor
    if not isinstance(std, torch.Tensor):
        raise RuntimeError('Features standardization standard deviation '
                            'tensor is not a torch.Tensor.')
    elif len(std) != features_tensor.shape[1]:
        raise RuntimeError('Features standardization standard deviation '
                           'tensor is not a torch.Tensor(1d) with shape '
                           '(n_features,).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mean, std