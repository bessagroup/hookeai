"""Graph Neural Network based material patch model.

Classes
-------
GNNMaterialPatchModel(torch.nn.Module)
    GNN-based material patch model.
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
    _gnn_epd_model : EncodeProcessDecode
        GNN-based Encoder-Process-Decoder model.
    _device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    forward(self)
        Forward propagation.
    _get_features_from_input_graph(self, input_graph)
        Get features from material patch input graph.
    predict_internal_forces(self, input_graph)
        Predict material patch internal forces.
    save_model_state(self)
        Save material patch model state to file.
    load_model_state(self)
        Load material patch model state from file. 
    """
    def __init__(self, n_node_in, n_node_out, n_edge_in, n_message_steps,
                 n_hidden_layers, hidden_layer_size, model_directory,
                 model_name='material_patch_model', device_type='cpu'):
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
        # Set device
        self._device_type = device_type
        if device_type in ('cpu', 'cuda'):
            self._device = torch.device(device_type)
        else:
            RuntimeError('Invalid device type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model initialization parameters
        self._save_model_init_args()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize GNN-based Encoder-Process-Decoder model
        self._gnn_epd_model = \
            EncodeProcessDecode(n_message_steps=n_message_steps,
                                n_node_in=n_node_in,
                                n_node_out=n_node_out,
                                n_edge_in=n_edge_in,
                                n_hidden_layers=n_hidden_layers,
                                hidden_layer_size=hidden_layer_size)
    # -------------------------------------------------------------------------
    def forward(self):
        """Forward propagation."""
        pass
    # -------------------------------------------------------------------------
    def _save_model_init_args(self):
        """Save material patch model class initialization parameters."""
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initialization parameters
        model_init_args = {}
        model_init_args['n_node_in'] = self._n_node_in
        model_init_args['n_node_out'] = self._n_node_out
        model_init_args['n_edge_in'] = self._n_edge_in
        model_init_args['n_message_steps'] = self._n_message_steps
        model_init_args['n_hidden_layers'] = self._n_hidden_layers
        model_init_args['hidden_layer_size'] = self._hidden_layer_size
        model_init_args['model_name'] = self.model_name
        model_init_args['device_type'] = self._device_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set initialization parameters file path
        model_init_file_path = os.path.join(
            self.model_directory, 'model_init_args' + '.pkl')
        # Save initialization parameters
        with open(model_init_file_path, 'wb') as init_file:
            pickle.dump(model_init_args, init_file)
    # -------------------------------------------------------------------------
    def _get_features_from_input_graph(self, input_graph):
        """Get features from material patch input graph.
        
        Parameters
        ----------
        input_graph : torch_geometric.data.Data
            Material patch homogeneous graph.
        
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
        # Check consistency with simulator
        if input_graph.num_node_features() != self._n_node_in:
            raise RuntimeError('Input graph and simulator number of node '
                               'features are not consistent.')
        if input_graph.num_edge_features() != self._n_edge_in:
            raise RuntimeError('Input graph and simulator number of edge '
                               'features are not consistent.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from material patch input graph
        if isinstance(input_graph.x, torch.Tensor):
            node_features_in = input_graph.x.clone()
        else:
            node_features_in = None
        if isinstance(input_graph.edge_attr, torch.Tensor):
            edge_features_in = input_graph.edge_attr.clone()
        else:
            edge_features_in = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material patch input graph edges indexes
        if isinstance(input_graph.edge_index, torch.Tensor):
            edges_indexes = input_graph.edge_index.clone()
        else:
            edges_indexes = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_in, edge_features_in, edges_indexes
    # -------------------------------------------------------------------------
    def predict_internal_forces(self, input_graph):
        """Predict material patch internal forces.
        
        Parameters
        ----------
        input_graph : torch_geometric.data.Data
            Material patch homogeneous graph.
            
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get features from material patch input graph
        node_features_in, edge_features_in, edges_indexes = \
            self._get_features_from_input_graph(input_graph)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Predict material patch internal forces
        node_internal_forces = \
            self._gnn_epd_model(node_features_in=node_features_in,
                                edge_features_in=edge_features_in,
                                edges_indexes=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_internal_forces
    # -------------------------------------------------------------------------
    def save_model_state(self, training_step=None):
        """Save material patch model state to file.
        
        Material patch model state file is stored in model_directory with
        basename model_name and extension '.pt' by default.
        
        Parameters
        ----------
        training_step : int, default=None
            Training step corresponding to current material patch model state.
            If provided, then state file basename is appended by
            '-training_step'.
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
        # Set material patch model state file path
        model_path = os.path.join(self.model_directory,
                                  model_state_file + '.pt')
        # Save material patch model state
        torch.save(self.state_dict(), model_path)
    # -------------------------------------------------------------------------
    def load_model_state(self, is_latest=False, training_step=None):
        """Load material patch model state from file.
        
        Material patch model state file is loaded from model_directory with
        basename model_name and extension '.pt' by default.
        
        Parameters
        ----------
        is_latest_state : bool, default=False
            If True, load the material patch model state file corresponding to
            the highest training step available. Overriden if training_step is
            provided.
        training_step : int, default=None
            Training step corresponding to loaded material patch model state.
            If provided, then state file basename is appended by
            '-training_step'.
            
        Returns
        -------
        training_step : int
            Loaded material patch model state training step. Defaults to None
            if training step is unknown (e.g., is_latest_state=False and
            training_step=None).
        """
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found:\n\n' + self.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state filename
        if isinstance(training_step, int):
            # Set model state filename with training step
            model_state_file = self.model_name + '-' + str(int(training_step))
        elif is_latest:
            # Get state files in material patch model directory
            directory_list = os.listdir(self.model_directory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize model state files training steps
            training_steps = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over files in material patch model directory
            for filename in directory_list:
                # Check if model state file corresponds to given training step
                is_state_file = \
                    bool(re.search(r'^' + self.model_name + r'-[0-9]+'
                                   + r'\.pt', filename))
                if is_state_file:
                    # Get model state training step
                    training_step = \
                        int(os.path.splitext(filename)[0].split('-')[-1])
                    # Store model state file training step
                    training_steps.append(training_step)
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