"""Graph Neural Network based material patch model.

Classes
-------
EncodeProcessDecode(torch.nn.Module)
    GNN-based material patch model.
Encoder(GraphIndependentNetwork)
    GNN-based material patch model encoder.
Processor(torch_geometric.nn.MessagePassing)
    GNN-based material patch model processor.
Decoder(torch.nn.Module)
    GNN-based material patch model decoder.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import torch_geometric.nn
# Local
from gnn_model.gnn_architectures import build_fnn, GraphIndependentNetwork,\
    GraphInteractionNetwork
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class EncodeProcessDecode(torch.nn.Module):
    """GNN-based material patch model.
    
    GNN-based model with an Encoder-Processor-Decoder architecture that takes
    a material patch state graph as input and predicts node output features.
    
    Based on: geoelements/gns/graph_network.py
    """
    def __init__(self, n_message_steps, n_node_in, n_node_out, n_edge_in,
                 n_hidden_layers, hidden_layer_size):
        """Constructor.
        
        Parameters
        ----------
        n_message_steps : int
            Number of message-passing steps.
        n_node_in : int
            Number of node input features.
        n_node_out : int
            Number of node output features.
        n_edge_in : int
            Number of edge input features.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        """
        # Initialize from base class
        super(EncodeProcessDecode, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch model encoder
        self._encoder = Encoder(n_node_in=n_node_in,
                                n_node_out=hidden_layer_size,
                                n_edge_in=n_edge_in,
                                n_edge_out=hidden_layer_size,
                                n_hidden_layers=n_hidden_layers,
                                hidden_layer_size=hidden_layer_size)
        # Set GNN-based material patch model processor
        self._processor = Processor(n_message_steps=n_message_steps,
                                    n_node_in=hidden_layer_size,
                                    n_node_out=hidden_layer_size,
                                    n_edge_in=hidden_layer_size,
                                    n_edge_out=hidden_layer_size,
                                    n_hidden_layers=n_hidden_layers,
                                    hidden_layer_size=hidden_layer_size)
        # Set GNN-based material patch model decoder
        self._decoder = Decoder(n_node_in=hidden_layer_size,
                                n_node_out=n_node_out,
                                n_hidden_layers=n_hidden_layers,
                                hidden_layer_size=hidden_layer_size)
    # -------------------------------------------------------------------------
    def forward(self, node_features_in, edge_features_in, edges_indexes):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.tensor
            Nodes features input matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : torch.tensor
            Edges features input matrix stored as a torch.tensor(2d) of shape
            (n_edges, n_features).
        edges_indexes : torch.tensor
            Edges indexes matrix stored as torch.tensor(2d) with shape
            (n_edges, 2), where the i-th edge is stored in edges_indexes[i, :]
            as (start_node_index, end_node_index).
        
        Returns
        -------
        node_features_out : torch.tensor
            Nodes features output matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
        """
        # Perform encoding
        node_features, edge_features = \
            self._encoder(node_features_in=node_features_in,
                          edge_features_in=edge_features_in)
        # Perform processing (message-passing steps)
        node_features, edge_features = \
            self._processor(node_features_in=node_features,
                            edge_features_in=edge_features,
                            edges_indexes=edges_indexes)
        # Perform decoding
        node_features_out = self._decoder(node_features_in=node_features)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out
# =============================================================================
class Encoder(GraphIndependentNetwork):
    """GNN-based material patch model encoder.
    
    Encodes material patch input state graph into latent graph by means of a
    Graph Independent Network. Node, edge and global features update functions
    are implemented as multilayer feed-forward neural networks and are
    independent (no aggregation).
    """
    pass
# =============================================================================
class Processor(torch_geometric.nn.MessagePassing):
    """GNN-based material patch model processor.
    
    Performs a given number of graph message-passing steps to generate a
    sequence of updated latent graphs as the information is propagated through
    the graph neural network. All message-passing steps are performed by means
    of an identical Graph Interaction Network (unshared parameters), where
    node, edge and global features update functions are implemented as
    multilayer feed-forward neural networks.
    
    Residual connections are adopted between the input and output latent
    features of both nodes and edges at each message-passing step.
    
    Based on: geoelements/gns/graph_network.py
    
    Questions:
    (1) I believe that the aggr='max' in the Processor initializer is
        meaningless (not used).
        
    Attributes
    ----------
    _processor : torch.nn.ModuleList
        Sequence of graph neural networks.
        
    Methods
    -------
    forward(self, node_features_in, edge_features_in, edges_indexes)
        Forward propagation.
    """
    def __init__(self, n_message_steps, n_node_in, n_node_out, n_edge_in,
                 n_edge_out, n_hidden_layers, hidden_layer_size):
        """Constructor.
        
        Parameters
        ----------
        n_message_steps : int
            Number of message-passing steps.
        n_node_in : int
            Number of node input features.
        n_node_out : int
            Number of node output features.
        n_edge_in : int
            Number of edge input features.
        n_edge_out : int
            Number of edge output features.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        """
        # Initialize from base class
        super(Processor, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sequence of identical Graph Interaction Networks
        self._processor = torch.nn.ModuleList(
            [GraphInteractionNetwork(n_node_in=n_node_in,
                                     n_node_out=n_node_out,
                                     n_edge_in=n_edge_in,
                                     n_edge_out=n_edge_out,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_layer_size=hidden_layer_size)
             for _ in range(n_message_steps)])
    # -------------------------------------------------------------------------
    def forward(self, node_features_in, edge_features_in, edges_indexes):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.tensor
            Nodes features input matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : torch.tensor
            Edges features input matrix stored as a torch.tensor(2d) of shape
            (n_edges, n_features).
        edges_indexes : torch.tensor
            Edges indexes matrix stored as torch.tensor(2d) with shape
            (n_edges, 2), where the i-th edge is stored in edges_indexes[i, :]
            as (start_node_index, end_node_index).
        
        Returns
        -------
        node_features_out : torch.tensor
            Nodes features output matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : torch.tensor
            Edges features output matrix stored as a torch.tensor(2d) of shape
            (n_edges, n_features).
        """
        # Initialize recurrent update features
        node_features_out = node_features_in.clone()
        edge_features_out = edge_features_in.clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over message-passing steps
        for gnn_model in self._processor:
            # Perform graph neural network message-passing step
            node_features_out, edge_features_out = \
                gnn_model(node_features_in=node_features_out,
                          edge_features_in=edge_features_out,
                          edges_indexes=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out
# =============================================================================
class Decoder(torch.nn.Module):
    """GNN-based material patch model decoder.
    
    Decodes material patch latent graph into output graph node features by
    means of a feed-forward neural network.
    
    Based on: geoelements/gns/graph_network.py
        
    Attributes
    ----------
    _node_fn : torch.nn.Sequential
        Node update function.
        
    Methods
    -------
    forward(self, node_features_in)
        Forward propagation.
    """
    def __init__(self, n_node_in, n_node_out, n_hidden_layers,
                 hidden_layer_size):
        """Constructor.
        
        Parameters
        ----------
        n_node_in : int
            Number of node input features.
        n_node_out : int
            Number of node output features.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        """
        # Initialize from base class
        super(Decoder, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set decoding feed-forward neural network
        self._node_fn = build_fnn(
            input_size=n_node_in, output_size=n_node_out,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=torch.nn.ReLU)
    # -------------------------------------------------------------------------
    def forward(self, node_features_in):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.tensor
            Nodes features input matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
            
        Returns
        -------
        node_features_out : torch.tensor
            Nodes features output matrix stored as a torch.tensor(2d) of shape
            (n_nodes, n_features).
        """
        return self._node_fn(node_features_in=node_features_in)