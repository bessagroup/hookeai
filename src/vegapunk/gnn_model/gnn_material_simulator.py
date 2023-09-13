"""Graph Neural Network based material patch simulator.

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
    _device : {'cpu', 'cuda'}, default='cpu'
        Type of device on with torch.Tensor is allocated.

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
                 model_name='material_patch_model', device='cpu'):
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
        device : {'cpu', 'cuda'}, default='cpu'
            Type of device on with torch.Tensor is allocated.
        """
        # Initialize from base class
        super(GNNMaterialPatchModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = n_node_in
        self._n_node_out = n_node_out
        self._n_edge_in = n_edge_in
        # Set material patch model directory and name
        if os.path.isdir(model_directory):
            self.model_directory = model_directory
        else:
            raise RuntimeError('The material patch model directory has not '
                               'been found.')
        self.model_name = model_name
        # Set device
        if device in ('cpu', 'cuda'):
            self._device = device
        else:
            RuntimeError('Invalid device type.')
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
    def save_model_state(self):
        """Save material patch model state to file."""
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The simulator model directory has not been '
                               'found.')
        # Set material patch model file path
        model_path = os.path.join(self.model_directory,
                                  self.model_name + '.pt')
        # Save material patch model state
        torch.save(self.state_dict(), model_path)
    # -------------------------------------------------------------------------
    def load_model_state(self):
        """Load material patch model state from file."""
        # Check material patch model directory
        if not os.path.isdir(self.model_directory):
            raise RuntimeError('The material patch model directory has not '
                               'been found.')
        # Set material patch model file path
        model_path = os.path.join(self.model_directory,
                                  self.model_name + '.pt')
        # Save material patch model state
        self.load_state_dict(torch.load(model_path,
                                        map_location=torch.device('cpu')))
        
        