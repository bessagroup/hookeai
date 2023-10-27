"""Graph Neural Network based material patch model.

Classes
-------
GNNMaterialPatchModel(torch.nn.Module)
    GNN-based material patch model.
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
from gnn_model.gnn_epd_model import EncodeProcessDecode
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class GNNMaterialPatchModel(torch.nn.Module):
    """GNN-based material patch model.
    
    Attributes
    ----------
    model_directory : str
        Directory where material patch model is stored.
    model_name : str, default='material_patch_model'
        Name of material patch model.
    _n_node_in : int
        Number of node input features.
    _n_node_out : int
        Number of node output features.
    _n_edge_in : int
        Number of edge input features.
    _n_message_steps : int
        Number of message-passing steps.
    _n_hidden_layers : int
        Number of hidden layers of multilayer feed-forward neural network
        update functions.
    _hidden_layer_size : int
        Number of neurons of hidden layers of multilayer feed-forward
        neural network update functions.
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
        Initialize GNN-based material patch model from initialization file.
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self)
        Forward propagation.
    save_model_init_file(self)
        Save material patch model class initialization attributes.
    get_input_features_from_graph(self, graph, is_normalized=False)
        Get input features from material patch graph.
    get_output_features_from_graph(self, graph, is_normalized=False)
        Get output features from material patch graph.
    predict_internal_forces(self, input_graph)
        Predict material patch internal forces.
    save_model_state(self)
        Save material patch model state to file.
    load_model_state(self)
        Load material patch model state from file.
    _check_state_file(self, filename)
        Check if file is model training step state file.
    _check_best_state_file(self, filename)
        Check if file is model training step best state file.
    _remove_posterior_state_files(self, training_step)
        Delete model training step state files posterior to given step.
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
    def __init__(self, n_node_in, n_node_out, n_edge_in, n_message_steps,
                 n_hidden_layers, hidden_layer_size, model_directory,
                 model_name='material_patch_model',
                 is_data_normalization=False, is_save_model_init_file=True,
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
        n_message_steps : int
            Number of message-passing steps.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        model_directory : str
            Directory where material patch model is stored.
        model_name : str, default='material_patch_model'
            Name of material patch model.
        is_data_normalization : bool, default=False
            If True, then input and output features are normalized for
            training, False otherwise. Data scalers need to be fitted with
            fit_data_scalers() and are stored as model attributes.
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
        super(GNNMaterialPatchModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = n_node_in
        self._n_node_out = n_node_out
        self._n_edge_in = n_edge_in
        # Set architecture parameters
        self._n_message_steps = n_message_steps
        self._n_hidden_layers = n_hidden_layers
        self._hidden_layer_size = hidden_layer_size
        # Set material patch model directory and name
        if os.path.isdir(model_directory):
            self.model_directory = model_directory
        else:
            raise RuntimeError('The material patch model directory has not '
                               'been found.')
        if not isinstance(model_name, str):
            raise RuntimeError('The material patch model name must be a '
                               'string.')
        else:
            self.model_name = model_name
        # Set normalization flag
        self.is_data_normalization = is_data_normalization
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize GNN-based Encoder-Process-Decoder model
        self._gnn_epd_model = \
            EncodeProcessDecode(n_message_steps=n_message_steps,
                                n_node_in=n_node_in,
                                n_node_out=n_node_out,
                                n_edge_in=n_edge_in,
                                enc_n_hidden_layers=n_hidden_layers,
                                pro_n_hidden_layers=n_hidden_layers,
                                dec_n_hidden_layers=n_hidden_layers,
                                hidden_layer_size=hidden_layer_size)
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
        """Initialize GNN-based material patch model from initialization file.
        
        Initialization file is assumed to be stored in the material patch model
        directory under the name model_init_file.pkl.
        
        Parameters
        ----------
        model_directory : str
            Directory where material patch model is stored.
        """
        # Check material patch model directory
        if not os.path.isdir(model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The material patch model initialization file '
                               'has not been found:\n\n'
                                + model_init_file_path)
        else:
            with open(model_init_file_path, 'rb') as model_init_file:
                model_init_attributes = pickle.load(model_init_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize GNN-based material patch model
        model_init_args = model_init_attributes['model_init_args']
        model = GNNMaterialPatchModel(**model_init_args,
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
    def forward(self, graph, is_normalized=False):
        """Forward propagation.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Material patch homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from material patch graph,
            False otherwise.
            
        Returns
        -------
        node_internal_forces : torch.Tensor
            Nodes internal forces matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_dim).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict material patch internal forces
        node_internal_forces = self.predict_internal_forces(
            graph, is_normalized=is_normalized)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_internal_forces 
    # -------------------------------------------------------------------------
    def save_model_init_file(self):
        """Save material patch model initialization file.
        
        Initialization file is stored in the material patch model directory
        under the name model_init_file.pkl.
        
        Initialization file contains a dictionary model_init_attributes that
        includes:
        
        'model_init_args' - Model initialization parameters
        
        'model_data_scalers' - Model fitted data scalers
        """
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material patch model initialization attributes
        model_init_attributes = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initialization parameters
        model_init_args = {}
        model_init_args['n_node_in'] = self._n_node_in
        model_init_args['n_node_out'] = self._n_node_out
        model_init_args['n_edge_in'] = self._n_edge_in
        model_init_args['n_message_steps'] = self._n_message_steps
        model_init_args['n_hidden_layers'] = self._n_hidden_layers
        model_init_args['hidden_layer_size'] = self._hidden_layer_size
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
        model_init_file_path = os.path.join(
            self.model_directory, 'model_init_file' + '.pkl')
        # Save model initialization file
        with open(model_init_file_path, 'wb') as init_file:
            pickle.dump(model_init_attributes, init_file)
    # -------------------------------------------------------------------------
    def get_input_features_from_graph(self, graph, is_normalized=False):
        """Get input features from material patch graph.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Material patch homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized input features from material patch graph,
            False otherwise.
        
        Returns
        -------
        node_features_in : torch.Tensor
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : torch.Tensor
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        edges_indexes : torch.Tensor
            Edges indexes matrix stored as torch.Tensor(2d) with shape
            (2, n_edges), where the i-th edge is stored in edges_indexes[:, i]
            as (start_node_index, end_node_index).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check consistency with simulator
        if graph.num_node_features != self._n_node_in:
            raise RuntimeError('Input graph and simulator number of node '
                               'features are not consistent.')
        if graph.num_edge_features != self._n_edge_in:
            raise RuntimeError('Input graph and simulator number of edge '
                               'features are not consistent.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from material patch graph
        if isinstance(graph.x, torch.Tensor):
            node_features_in = graph.x.clone()
        else:
            node_features_in = None
        if isinstance(graph.edge_attr, torch.Tensor):
            edge_features_in = graph.edge_attr.clone()
        else:
            edge_features_in = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch graph edges indexes
        if isinstance(graph.edge_index, torch.Tensor):
            edges_indexes = graph.edge_index.clone()
        else:
            edges_indexes = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize input features data
        if is_normalized:
            if node_features_in is not None:
                node_features_in = self.data_scaler_transform(
                    tensor=node_features_in, features_type='node_features_in',
                    mode='normalize')
            if edge_features_in is not None:
                edge_features_in = self.data_scaler_transform(
                    tensor=edge_features_in, features_type='edge_features_in',
                    mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_in, edge_features_in, edges_indexes
    # -------------------------------------------------------------------------
    def get_output_features_from_graph(self, graph, is_normalized=False):
        """Get output features from material patch graph.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Material patch homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from material patch graph,
            False otherwise.
        
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """
        # Check input graph
        if not isinstance(graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from material patch graph
        if isinstance(graph.y, torch.Tensor):
            node_features_out = graph.y.clone()
        else:
            node_features_out = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        # Normalize output features data
        if is_normalized:                
            if node_features_out is not None:
                node_features_out = self.data_scaler_transform(
                    tensor=node_features_out,
                    features_type='node_features_out',
                    mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out
    # -------------------------------------------------------------------------
    def predict_internal_forces(self, input_graph, is_normalized=False):
        """Predict material patch internal forces.
        
        Parameters
        ----------
        input_graph : torch_geometric.data.Data
            Material patch homogeneous graph.
        is_normalized : bool, default=False
            If True, get normalized output features from material patch graph,
            False otherwise.
            
        Returns
        -------
        node_internal_forces : torch.Tensor
            Nodes internal forces matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_dim).
        """
        # Check input graph type
        if not isinstance(input_graph, torch_geometric.data.Data):
            raise RuntimeError('Material patch input graph must be instance '
                               'of torch_geometric.data.Data.')
        # Check model data normalization
        if is_normalized:
            self.check_normalized_return()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from material patch input graph
        node_features_in, edge_features_in, edges_indexes = \
            self.get_input_features_from_graph(
                input_graph, is_normalized=self.is_data_normalization)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict material patch internal forces
        node_internal_forces = \
            self._gnn_epd_model(node_features_in=node_features_in,
                                edge_features_in=edge_features_in,
                                edges_indexes=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features data
        if self.is_data_normalization and not is_normalized:
            if node_internal_forces is not None:
                node_internal_forces = self.data_scaler_transform(
                    tensor=node_internal_forces,
                    features_type='node_features_out',
                    mode='denormalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_internal_forces
    # -------------------------------------------------------------------------
    def save_model_state(self, training_step=None, is_best_state=False,
                         is_remove_posterior=True):
        """Save material patch model state to file.
        
        Material patch model state file is stored in model_directory under the
        name < model_name >.pt or < model_name >-< training_step >.pt if
        training_step is known.
        
        Material patch model state file corresponding to the best performance
        is stored in model_directory under the name < model_name >-best.pt or
        < model_name >-< training_step >-best.pt if training_step is known.
        
        Parameters
        ----------
        training_step : int, default=None
            Training step corresponding to current material patch model state.
        is_best_state : bool, default=False
            If True, save material patch model state file corresponding to the
            best performance instead of regular state file.
        is_remove_posterior : bool, default=True
            Remove material patch model and optimizer state files corresponding
            to training steps posterior to the saved state file. Effective only
            if saved training step is known.
        """
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        model_state_file = self.model_name
        # Append training step
        if isinstance(training_step, int):
            model_state_file += '-' + str(training_step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch model state corresponding to the best performance
        if is_best_state:
            # Append best performance
            model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove any existent best material patch model state file
            self._remove_best_state_files()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Save material patch model state
        torch.save(self.state_dict(), model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model training step state files posterior to saved training
        # step
        if isinstance(training_step, int) and is_remove_posterior:
            self._remove_posterior_state_files(training_step)  
    # -------------------------------------------------------------------------
    def load_model_state(self, load_model_state=None,
                         is_remove_posterior=True):
        """Load material patch model state from file.
        
        Material patch model state file is stored in model_directory under the
        name < model_name >.pt or < model_name >-< training_step >.pt if
        training_step is known.
        
        Material patch model state file corresponding to the best performance
        is stored in model_directory under the name < model_name >-best.pt or
        < model_name >-< training_step >-best.pt if training_step if known.
        
        Parameters
        ----------            
        load_model_state : {'best', 'last', int, None}, default=None
            Load available GNN-based material patch model state from the model
            directory. Options:
            
            'best'      : Model state corresponding to best performance
            
            'last'      : Model state corresponding to highest training step
            
            int         : Model state corresponding to given training step
            
            None   : Model default state file
        
        is_remove_posterior : bool, default=True
            Remove material patch model state files corresponding to training
            steps posterior to the loaded state file. Effective only if
            loaded training step is known.
            
        Returns
        -------
        training_step : int
            Loaded material patch model state training step. Defaults to None
            if training step is unknown.
        """
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if load_model_state == 'best':
            # Get state files in material patch model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model best state files training steps
            best_state_steps = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in material patch model directory
            for filename in directory_list:
                # Check if file is model training step best state file
                is_best_state_file, best_state_step = \
                    self._check_best_state_file(filename)
                # Store model best state file training step
                if is_best_state_file:
                    best_state_steps.append(best_state_step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set model best state file
            if not best_state_steps:
                raise RuntimeError('Material patch model best state file '
                                   'has not been found in directory:\n\n'
                                   + self.model_directory)
            elif len(best_state_steps) > 1:
                raise RuntimeError('Two or more material patch model best '
                                   'state files have been found in directory:'
                                   '\n\n' + self.model_directory)
            else:
                # Set best state training step
                training_step = best_state_steps[0]
                # Set model best state file
                model_state_file = self.model_name
                if isinstance(training_step, int):
                    model_state_file += '-' + str(training_step)      
                model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model training step state files posterior to loaded
            # training step
            if isinstance(training_step, int) and is_remove_posterior:
                self._remove_posterior_state_files(training_step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif load_model_state == 'last':
            # Get state files in material patch model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model state files training steps
            training_steps = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in material patch model directory
            for filename in directory_list:
                # Check if file is model training step state file
                is_state_file, step = self._check_state_file(filename)
                # Store model state file training step
                if is_state_file:
                    training_steps.append(step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set highest training step model state file
            if training_steps:
                # Set highest training step
                training_step = max(training_steps)
                # Set highest training step model state file
                model_state_file = self.model_name + '-' + str(training_step)
            else:
                raise RuntimeError('Material patch model state files '
                                   'corresponding to training steps have not '
                                   'been found in directory:\n\n'
                                   + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        elif isinstance(load_model_state, int):
            # Get training step
            training_step = load_model_state
            # Set model state filename with training step
            model_state_file = self.model_name + '-' + str(int(training_step))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Delete model training step state files posterior to loaded
            # training step
            if is_remove_posterior:
                self._remove_posterior_state_files(training_step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Set model state filename
            model_state_file = self.model_name
            # Set training step as unknown
            training_step = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material patch model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Check material patch model state file
        if not os.path.isfile(model_path):
            raise RuntimeError('Material patch model state file has not been '
                               'found:\n\n' + model_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load material patch model state
        self.load_state_dict(torch.load(model_path,
                                        map_location=torch.device('cpu')))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return training_step
    # -------------------------------------------------------------------------
    def _check_state_file(self, filename):
        """Check if file is model training step state file.
        
        Material patch model training step state file is stored in
        model_directory under the name < model_name >-< training_step >.pt.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_state_file : bool
            True if model training step state file, False otherwise.
        training_step : {None, int}
            Training step corresponding to model state file if
            is_state_file=True, None otherwise.
        """
        # Check if file is model training step state file
        is_state_file = bool(re.search(r'^' + self.model_name + r'-[0-9]+'
                                       + r'\.pt', filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        training_step = None
        if is_state_file:
            # Get model state training step
            training_step = int(os.path.splitext(filename)[0].split('-')[-1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_state_file, training_step
    # -------------------------------------------------------------------------
    def _check_best_state_file(self, filename):
        """Check if file is model best state file.
        
        Material patch model state file corresponding to the best performance
        is stored in model_directory under the name < model_name >-best.pt. or
        < model_name >-< training_step >-best.pt if the training step is known.
        
        Parameters
        ----------
        filename : str
            File name.
        
        Returns
        -------
        is_best_state_file : bool
            True if model training step state file, False otherwise.
        training_step : {None, int}
            Training step corresponding to model state file if
            is_best_state_file=True and training step is known, None otherwise.
        """
        # Check if file is model training step best state file
        is_best_state_file = bool(re.search(r'^' + self.model_name
                                            + r'-?[0-9]*' + r'-best' + r'\.pt',
                                            filename))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        training_step = None
        if is_best_state_file:
            # Get model state training step
            training_step = int(os.path.splitext(filename)[0].split('-')[-2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_best_state_file, training_step
    # -------------------------------------------------------------------------
    def _remove_posterior_state_files(self, training_step):
        """Delete model training step state files posterior to given step.
        
        Parameters
        ----------
        training_step : int
            Training step.
        """
        # Get files in material patch model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in material patch model directory
        for filename in directory_list:
            # Check if file is model training step state file
            is_state_file, step = self._check_state_file(filename)
            # Delete model training step state file posterior to given training
            # step
            if is_state_file and step > training_step:
                os.remove(os.path.join(self.model_directory, filename))
    # -------------------------------------------------------------------------
    def _remove_best_state_files(self):
        """Delete existent model best state files."""
        # Get files in material patch model directory
        directory_list = os.listdir(self.model_directory)
        # Loop over files in material patch model directory
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
        self._data_scalers['node_features_out'] = None
    # -------------------------------------------------------------------------
    def fit_data_scalers(self, dataset, is_verbose=False):
        """Fit model data scalers.
        
        Data scalars are set a standard scalers where features are normalized
        by removing the mean and scaling to unit variance.
        
        Calling this method turns on model data normalization.
        
        Parameters
        ----------
        dataset : GNNMaterialPatchDataset
            GNN-based material patch data set. Each sample corresponds to a
            torch_geometric.data.Data object describing a homogeneous graph.
        is_verbose : bool, default=False
            If True, enable verbose output.
        """
        if is_verbose:
            print('\nFitting GNN-based material patch model data scalers'
                  '\n---------------------------------------------------\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model data normalization
        self.is_data_normalization = True
        # Initialize data scalers
        self._init_data_scalers()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate data scalers
        scaler_node_in = TorchStandardScaler(n_features=self._n_node_in,
                                             device_type=self._device_type)
        scaler_edge_in = TorchStandardScaler(n_features=self._n_edge_in,
                                             device_type=self._device_type)
        scaler_node_out = TorchStandardScaler(n_features=self._n_node_out,
                                              device_type=self._device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get scaling parameters and fit data scalers: node input features
        mean, std = graph_standard_partial_fit(
            dataset, features_type='node_features_in')
        scaler_node_in.set_mean_and_std(mean, std)        
        # Get scaling parameters and fit data scalers: edge input features
        mean, std = graph_standard_partial_fit(
            dataset, features_type='edge_features_in')
        scaler_edge_in.set_mean_and_std(mean, std)
        # Get scaling parameters and fit data scalers: node output features
        mean, std = graph_standard_partial_fit(
            dataset, features_type='node_features_out')
        scaler_node_out.set_mean_and_std(mean, std)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print('\n> Setting fitted standard scalers...\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set fitted data scalers
        self._data_scalers['node_features_in'] = scaler_node_in
        self._data_scalers['edge_features_in'] = scaler_edge_in
        self._data_scalers['node_features_out'] = scaler_node_out
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
            
            'node_features_in'  : Node features input matrix
            
            'edge_features_in'  : Edge features input matrix
            
            'node_features_out' : Node features output matrix

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
            
            'node_features_in'  : Node features input matrix
            
            'edge_features_in'  : Edge features input matrix
            
            'node_features_out' : Node features output matrix

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
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get model initialization file path from model directory
        model_init_file_path = os.path.join(self.model_directory,
                                            'model_init_file' + '.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model initialization attributes from file
        if not os.path.isfile(model_init_file_path):
            raise RuntimeError('The material patch model initialization file '
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
    def fit(self, tensor, is_bessel=True):
        """Fit features standardization mean and standard deviation tensors.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (n_samples, n_features).
        is_bessel : bool, default=False
            Apply Bessel's correction to compute standard deviation, False
            otherwise.
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set standardization mean and standard deviation
        self._mean = self._check_mean(torch.mean(tensor, dim=0))
        self._std = \
            self._check_std(torch.std(tensor, dim=0, unbiased=is_bessel))
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
        # Standardize features tensor
        transformed_tensor = torch.div(tensor - mean, std)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check transformed tensor
        if not torch.equal(torch.tensor(transformed_tensor.size()),
                           torch.tensor(tensor.size())):
            raise RuntimeError('Input and transformed tensors do not have the '
                               'same shape.')
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
def graph_standard_partial_fit(dataset, features_type, is_verbose=False):
    """Perform batch fitting of standardization data scalers.
    
    Parameters
    ----------
    dataset : GNNMaterialPatchDataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    features_type : str
        Features for which data scaler is required:
        
        'node_features_in'  : Node features input matrix
        
        'edge_features_in'  : Edge features input matrix
        
        'node_features_out' : Node features output matrix
    
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
            raise RuntimeError('Material patch sample graph must be instance '
                               'of torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features tensor
        if features_type == 'node_features_in':
            features_tensor = pyg_graph.x
        elif features_type == 'edge_features_in':
            features_tensor = pyg_graph.edge_attr
        elif features_type == 'node_features_out':
            features_tensor = pyg_graph.y
        else:
            raise RuntimeError('Unknown features type.')   
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Process sample to fit data scaler
        if isinstance(features_tensor, torch.Tensor):
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