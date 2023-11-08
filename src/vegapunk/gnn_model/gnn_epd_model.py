"""Graph Neural Network based Encoder-Process-Decoder model.

Classes
-------
EncodeProcessDecode(torch.nn.Module)
    GNN-based Encoder-Process-Decoder model.
Encoder(GraphIndependentNetwork)
    GNN-based encoder.
Processor(torch_geometric.nn.MessagePassing)
    GNN-based processor.
Decoder(torch.nn.Module)
    FNN-based decoder.
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
    """GNN-based Encoder-Process-Decoder model.
    
    Attributes
    ----------
    _n_message_steps : int
        Number of message-passing steps.
    _encoder : Encoder
        GNN-based encoder.
    _processor : Processor
        GNN-based processor.
    _decoder : Decoder
        GNN-based decoder.
    
    Methods
    -------
    forward(self, node_features_in, edge_features_in, edges_indexes)
        Forward propagation.
    """
    def __init__(self, n_message_steps, n_node_in, n_node_out, n_edge_in,
                 enc_n_hidden_layers, pro_n_hidden_layers, dec_n_hidden_layers,
                 hidden_layer_size, pro_aggregation_scheme='add',
                 enc_node_hidden_activation=torch.nn.Identity,
                 enc_node_output_activation=torch.nn.Identity,
                 enc_edge_hidden_activation=torch.nn.Identity,
                 enc_edge_output_activation=torch.nn.Identity,
                 pro_node_hidden_activation=torch.nn.Identity,
                 pro_node_output_activation=torch.nn.Identity,
                 pro_edge_hidden_activation=torch.nn.Identity,
                 pro_edge_output_activation=torch.nn.Identity,
                 dec_node_hidden_activation=torch.nn.Identity,
                 dec_node_output_activation=torch.nn.Identity,
                 is_node_res_connect=False,
                 is_edge_res_connect=False):
        """Constructor.
        
        Parameters
        ----------
        n_message_steps : int
            Number of message-passing steps. Setting number of message-passing
            steps to 0 results in Encoder-Decoder model (Processor is not
            initialized).
        n_node_in : int
            Number of node input features.
        n_node_out : int
            Number of node output features.
        n_edge_in : int
            Number of edge input features.
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
        pro_aggregation_scheme : {'add',}, default='add'
            Processor: Message-passing aggregation scheme.
        enc_node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Encoder: Hidden unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        enc_node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Encoder: Output unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        enc_edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Encoder: Hidden unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        enc_edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Encoder: Output unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        pro_node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Processor: Hidden unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        pro_node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Processor: Output unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        pro_edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Processor: Hidden unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        pro_edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Processor: Output unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Hidden unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Output unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        is_node_res_connect : bool, default=False
            Processor: Add residual connections between nodes input and
            output features if True, False otherwise. Number of input and
            output features must match to process residual connections.
        is_edge_res_connect : bool, default=False
            Processor: Add residual connections in between edges input and
            output features if True, False otherwise. Number of input and
            output features must match to process residual connections.
        """
        # Initialize from base class
        super(EncodeProcessDecode, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store number of message-passing steps
        self._n_message_steps = n_message_steps
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch model encoder
        self._encoder = \
            Encoder(n_hidden_layers=enc_n_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    n_node_in=n_node_in, n_node_out=hidden_layer_size,
                    n_edge_in=n_edge_in, n_edge_out=hidden_layer_size,
                    node_hidden_activation=enc_node_hidden_activation,
                    node_output_activation=enc_node_output_activation,
                    edge_hidden_activation=enc_edge_hidden_activation,
                    edge_output_activation=enc_edge_output_activation)
        # Set GNN-based material patch model processor if positive number of
        # message-passing steps
        if self._n_message_steps > 0:
            self._processor = \
                Processor(n_message_steps=n_message_steps,
                          n_node_in=hidden_layer_size,
                          n_node_out=hidden_layer_size,
                          n_edge_in=hidden_layer_size,
                          n_edge_out=hidden_layer_size,
                          n_hidden_layers=pro_n_hidden_layers,
                          hidden_layer_size=hidden_layer_size,
                          aggregation_scheme=pro_aggregation_scheme,
                          node_hidden_activation=pro_node_hidden_activation,
                          node_output_activation=pro_node_output_activation,
                          edge_hidden_activation=pro_edge_hidden_activation,
                          edge_output_activation=pro_edge_output_activation,
                          is_node_res_connect=is_node_res_connect,
                          is_edge_res_connect=is_edge_res_connect)
        else:
            self._processor = None
        # Set GNN-based material patch model decoder
        self._decoder = \
            Decoder(n_node_in=hidden_layer_size, n_node_out=n_node_out,
                    n_hidden_layers=dec_n_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    node_hidden_activation=dec_node_hidden_activation,
                    node_output_activation=dec_node_output_activation)
    # -------------------------------------------------------------------------
    def forward(self, node_features_in, edge_features_in, edges_indexes):
        """Forward propagation.
        
        Processor is skipped if number of message-passing steps is set to zero.
        
        Parameters
        ----------
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
        
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """
        # Perform encoding
        node_features, edge_features = \
            self._encoder(node_features_in=node_features_in,
                          edge_features_in=edge_features_in)
        # Perform processing (message-passing steps)
        if self._n_message_steps > 0:
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
    """GNN-based encoder.
    
    Encodes input state graph data into latent graph by means of a Graph
    Independent Network. Node, edge and global features update functions are
    implemented as multilayer feed-forward neural networks and are independent
    (no aggregation).
    """
    pass
# =============================================================================
class Processor(torch_geometric.nn.MessagePassing):
    """GNN-based processor.
    
    Performs a given number of graph message-passing steps to generate a
    sequence of updated latent graphs as the information is propagated through
    the graph neural network. All message-passing steps are performed by means
    of an identical Graph Interaction Network (unshared parameters), where
    node, edge and global features update functions are implemented as
    multilayer feed-forward neural networks.
    
    Residual connections are adopted between the input and output latent
    features of both nodes and edges at each message-passing step.
        
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
                 n_edge_out, n_hidden_layers, hidden_layer_size,
                 aggregation_scheme='add',
                 node_hidden_activation=torch.nn.Identity,
                 node_output_activation=torch.nn.Identity,
                 edge_hidden_activation=torch.nn.Identity,
                 edge_output_activation=torch.nn.Identity,
                 is_node_res_connect=False, is_edge_res_connect=False):
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
        aggregation_scheme : {'add',}, default='add'
            Message-passing aggregation scheme.
        node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        is_node_res_connect : bool, default=False
            Add residual connections between nodes input and output features
            if True, False otherwise. Number of input and output features must
            match to process residual connections.
        is_edge_res_connect : bool, default=False
            Add residual connections between edges input and output features
            if True, False otherwise. Number of input and output features must
            match to process residual connections.
        """
        # Initialize from base class
        super(Processor, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of message-passing steps
        if n_message_steps < 1:
            raise RuntimeError('Number of message-passing steps must be at '
                               'least 1.')
        elif n_message_steps > 1 and (n_node_in != n_node_out
                                      or n_edge_in != n_edge_out):
            raise RuntimeError('Number of node/edge input and output features '
                               'must match to process multiple '
                               'message-passing steps in sequence.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node residual connections
        if is_node_res_connect and n_node_in != n_node_out:
            raise RuntimeError('Number of node input and output features '
                               'must match to process residual '
                               'connections.')
        else:
            self._is_node_res_connect = is_node_res_connect
        # Set edge residual connections
        if is_edge_res_connect and n_edge_in != n_edge_out:
            raise RuntimeError('Number of edge input and output features '
                               'must match to process residual '
                               'connections.')
        else:
            self._is_edge_res_connect = is_edge_res_connect
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sequence of identical Graph Interaction Networks
        self._processor = torch.nn.ModuleList(
            [GraphInteractionNetwork(
                n_node_in=n_node_in, n_node_out=n_node_out,
                n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                n_hidden_layers=n_hidden_layers,
                hidden_layer_size=hidden_layer_size,
                aggregation_scheme=aggregation_scheme,
                node_hidden_activation=node_hidden_activation,
                node_output_activation=node_output_activation,
                edge_hidden_activation=edge_hidden_activation,
                edge_output_activation=edge_output_activation)
             for _ in range(n_message_steps)])
    # -------------------------------------------------------------------------
    def forward(self, node_features_in, edge_features_in, edges_indexes):
        """Forward propagation.
        
        Parameters
        ----------
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

        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        """
        # Initialize recurrent update features
        node_features_out = node_features_in.clone()
        edge_features_out = edge_features_in.clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over message-passing steps
        for gnn_model in self._processor:
            # Save features matrix (residual connection)
            if self._is_node_res_connect:
                node_features_res = node_features_out.clone()
            if self._is_edge_res_connect:
                edge_features_res = edge_features_out.clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform graph neural network message-passing step
            node_features_out, edge_features_out = \
                gnn_model(node_features_in=node_features_out,
                          edge_features_in=edge_features_out,
                          edges_indexes=edges_indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add residual connections to features output
            if self._is_node_res_connect:
                node_features_out += node_features_res
            if self._is_edge_res_connect:
                edge_features_out += edge_features_res
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out
# =============================================================================
class Decoder(torch.nn.Module):
    """FNN-based decoder.
    
    Decodes latent graph into output graph node features by means of a
    feed-forward neural network.
        
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
                 hidden_layer_size, node_hidden_activation=torch.nn.Identity,
                 node_output_activation=torch.nn.Identity):
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
        node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        """
        # Initialize from base class
        super(Decoder, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set decoding feed-forward neural network
        self._node_fn = build_fnn(
            input_size=n_node_in, output_size=n_node_out,
            output_activation=node_output_activation,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=node_hidden_activation)
    # -------------------------------------------------------------------------
    def forward(self, node_features_in):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.Tensor
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
            
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """
        return self._node_fn(node_features_in)