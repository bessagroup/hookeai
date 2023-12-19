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
from gnn_base_model.model.gnn_architectures import build_fnn, \
    GraphIndependentNetwork, GraphInteractionNetwork
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
    _n_node_out : int
        Number of node output features.
    _n_edge_out : int
        Number of edge output features.
    _n_global_out : int
        Number of node output features.
    
    Methods
    -------
    forward(self, node_features_in, edge_features_in, edges_indexes, \
            global_features_in)
        Forward propagation.
    """
    def __init__(self, n_message_steps, n_node_out, n_edge_out,
                 n_global_out, enc_n_hidden_layers, pro_n_hidden_layers,
                 dec_n_hidden_layers, hidden_layer_size,
                 n_node_in=0, n_edge_in=0, n_global_in=0,
                 pro_edge_to_node_aggr='add', pro_node_to_global_aggr='add',
                 enc_node_hidden_activation=torch.nn.Identity(),
                 enc_node_output_activation=torch.nn.Identity(),
                 enc_edge_hidden_activation=torch.nn.Identity(),
                 enc_edge_output_activation=torch.nn.Identity(),
                 enc_global_hidden_activation=torch.nn.Identity(),
                 enc_global_output_activation=torch.nn.Identity(),
                 pro_node_hidden_activation=torch.nn.Identity(),
                 pro_node_output_activation=torch.nn.Identity(),
                 pro_edge_hidden_activation=torch.nn.Identity(),
                 pro_edge_output_activation=torch.nn.Identity(),
                 pro_global_hidden_activation=torch.nn.Identity(),
                 pro_global_output_activation=torch.nn.Identity(),
                 dec_node_hidden_activation=torch.nn.Identity(),
                 dec_node_output_activation=torch.nn.Identity(),
                 dec_edge_hidden_activation=torch.nn.Identity(),
                 dec_edge_output_activation=torch.nn.Identity(),
                 dec_global_hidden_activation=torch.nn.Identity(),
                 dec_global_output_activation=torch.nn.Identity(),
                 is_node_res_connect=False, is_edge_res_connect=False,
                 is_global_res_connect=False):
        """Constructor.
        
        Parameters
        ----------
        n_message_steps : int
            Number of message-passing steps. Setting number of message-passing
            steps to 0 results in Encoder-Decoder model (Processor is not
            initialized).
        n_node_out : int
            Number of node output features.
        n_edge_out : int
            Number of edge output features.
        n_global_out : int
            Number of node output features.
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
        n_node_in : int, default=0
            Number of node input features.
        n_edge_in : int, default=0
            Number of edge input features.
        n_global_in : int, default=0
            Number of global input features.
        pro_edge_to_node_aggr : {'add',}, default='add'
            Processor: Edge-to-node aggregation scheme.
        pro_node_to_global_aggr : {'add', 'mean'}, default='add'
            Processor: Node-to-global aggregation scheme.
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
        enc_global_hidden_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Encoder: Hidden unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        enc_global_output_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Encoder: Output unit activation function of global update function
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
        pro_global_hidden_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Processor: Hidden unit activation function of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        pro_global_output_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Processor: Output unit activation function of global update
            function (multilayer feed-forward neural network). Defaults to
            identity (linear) unit activation function.
        dec_node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Hidden unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Output unit activation function of node update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Hidden unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Decoder: Output unit activation function of edge update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_global_hidden_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Decoder: Hidden unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        dec_global_output_activation : torch.nn.Module, \
                default=torch.nn.Identity
            Decoder: Output unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        is_node_res_connect : bool, default=False
            Processor: Add residual connections between nodes input and
            output features if True, False otherwise. Number of input and
            output features must match to process residual connections.
            Automatically set to False if number of node input features is
            zero.
        is_edge_res_connect : bool, default=False
            Processor: Add residual connections in between edges input and
            output features if True, False otherwise. Number of input and
            output features must match to process residual connections.
            Automatically set to False if number of edge input features is
            zero.
        is_global_res_connect : bool, default=False
            Processor: Add residual connections in between global input and
            output features if True, False otherwise. Number of input and
            output features must match to process residual connections.
            Automatically set to False if number of global input features is
            zero.
        """
        # Initialize from base class
        super(EncodeProcessDecode, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of input features
        if int(n_node_in) < 1 and int(n_edge_in) < 1 and int(n_global_in) < 1:
            raise RuntimeError(f'Impossible to setup model without node '
                               f'({int(n_node_in)}), edge ({int(n_edge_in)}) '
                               f'and global ({int(n_global_in)}) input '
                               f'features.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store number of message-passing steps
        self._n_message_steps = int(n_message_steps)
        # Store number of output features
        self._n_node_out = n_node_out
        self._n_edge_out = n_edge_out
        self._n_global_out = n_global_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node update function hidden layer input size
        if int(n_node_in) < 1:
            # Overwrite hidden layer input size when number of node input
            # features is zero
            n_node_hidden_in = 0
            # Turn off node residual connections
            is_node_res_connect = False
        else:
            n_node_hidden_in = hidden_layer_size
        # Set edge update function hidden layer input size
        if int(n_edge_in) < 1:
            # Overwrite hidden layer input size when number of edge input
            # features is zero
            n_edge_hidden_in = 0
            # Turn off edge residual connections
            is_edge_res_connect = False
        else:
            n_edge_hidden_in = hidden_layer_size
        # Set global update function hidden layer input size
        if int(n_global_in) < 1:
            # Overwrite hidden layer input size when number of global input
            # features is zero
            n_global_hidden_in = 0
            # Turn off global residual connections
            is_global_res_connect = False
        else:
            n_global_hidden_in = hidden_layer_size
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model encoder
        self._encoder = \
            Encoder(n_hidden_layers=enc_n_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    n_node_in=n_node_in, n_node_out=hidden_layer_size,
                    n_edge_in=n_edge_in, n_edge_out=hidden_layer_size,
                    n_global_in=n_global_in, n_global_out=hidden_layer_size,
                    node_hidden_activation=enc_node_hidden_activation,
                    node_output_activation=enc_node_output_activation,
                    edge_hidden_activation=enc_edge_hidden_activation,
                    edge_output_activation=enc_edge_output_activation,
                    global_hidden_activation=enc_global_hidden_activation,
                    global_output_activation=enc_global_output_activation,
                    is_norm_layer=True, is_skip_unset_update=True)
        # Set model processor if positive number of message-passing steps
        if self._n_message_steps > 0:
            self._processor = \
                Processor(n_message_steps=n_message_steps,
                          n_node_out=hidden_layer_size,
                          n_edge_out=hidden_layer_size,
                          n_global_out=hidden_layer_size,
                          n_hidden_layers=pro_n_hidden_layers,
                          hidden_layer_size=hidden_layer_size,
                          n_node_in=n_node_hidden_in,
                          n_edge_in=n_edge_hidden_in,
                          n_global_in=n_global_hidden_in,
                          edge_to_node_aggr=pro_edge_to_node_aggr,
                          node_to_global_aggr=pro_node_to_global_aggr,
                          node_hidden_activation=pro_node_hidden_activation,
                          node_output_activation=pro_node_output_activation,
                          edge_hidden_activation=pro_edge_hidden_activation,
                          edge_output_activation=pro_edge_output_activation,
                          global_hidden_activation=\
                              pro_global_hidden_activation,
                          global_output_activation=\
                              pro_global_output_activation,
                          is_node_res_connect=is_node_res_connect,
                          is_edge_res_connect=is_edge_res_connect,
                          is_global_res_connect=is_global_res_connect)
        else:
            self._processor = None
        # Set model decoder
        self._decoder = \
            Decoder(n_hidden_layers=dec_n_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    n_node_in=hidden_layer_size, n_node_out=n_node_out,
                    n_edge_in=hidden_layer_size, n_edge_out=n_edge_out,
                    n_global_in=hidden_layer_size, n_global_out=n_global_out,
                    node_hidden_activation=dec_node_hidden_activation,
                    node_output_activation=dec_node_output_activation,
                    edge_hidden_activation=dec_edge_hidden_activation,
                    edge_output_activation=dec_edge_output_activation,
                    global_hidden_activation=dec_global_hidden_activation,
                    global_output_activation=dec_global_output_activation,
                    is_norm_layer=False, is_skip_unset_update=True)
    # -------------------------------------------------------------------------
    def forward(self, edges_indexes, node_features_in=None,
                edge_features_in=None, global_features_in=None,
                batch_vector=None):
        """Forward propagation.
        
        Processor is skipped if number of message-passing steps is set to zero.
        
        Parameters
        ----------
        edges_indexes : torch.Tensor
            Edges indexes matrix stored as torch.Tensor(2d) with shape
            (2, n_edges), where the i-th edge is stored in edges_indexes[:, i]
            as (start_node_index, end_node_index).
        node_features_in : torch.Tensor, default=None
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : torch.Tensor, default=None
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_in : torch.Tensor, default=None
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features). Ignored if global update function is not setup.
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
        # Check input features matrices
        if (node_features_in is None and edge_features_in is None
            and global_features_in is None):
            raise RuntimeError('Impossible to compute forward propagation of '
                               'model without node (None), edge (None) and '
                               'global (None) input features matrices.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform encoding
        node_features, edge_features, global_features = \
            self._encoder(node_features_in=node_features_in,
                          edge_features_in=edge_features_in,
                          global_features_in=global_features_in,
                          batch_vector=batch_vector)
        # Perform processing (message-passing steps)
        if self._n_message_steps > 0:
            # Compute message-passing step
            node_features, edge_features, global_features = \
                self._processor(edges_indexes=edges_indexes,
                                node_features_in=node_features,
                                edge_features_in=edge_features,
                                global_features_in=global_features,
                                batch_vector=batch_vector)
        # Perform decoding
        node_features_out, edge_features_out, global_features_out = \
            self._decoder(node_features_in=node_features,
                          edge_features_in=edge_features,
                          global_features_in=global_features,
                          batch_vector=batch_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Discard unsolicited output features
        if self._n_node_out < 1:
            node_features_out = None
        if self._n_edge_out < 1:
            edge_features_out = None
        if self._n_global_out < 1:
            global_features_out = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
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
        
    Methods
    -------
    forward(self, edges_indexes, node_features_in=None, edge_features_in=None)
        Forward propagation.
    """
    def __init__(self, n_message_steps, n_node_out, n_edge_out,
                 n_hidden_layers, hidden_layer_size,
                 n_node_in=0, n_edge_in=0, n_global_in=0, n_global_out=0,
                 edge_to_node_aggr='add', node_to_global_aggr='add',
                 node_hidden_activation=torch.nn.Identity(),
                 node_output_activation=torch.nn.Identity(),
                 edge_hidden_activation=torch.nn.Identity(),
                 edge_output_activation=torch.nn.Identity(),
                 global_hidden_activation=torch.nn.Identity(),
                 global_output_activation=torch.nn.Identity(),
                 is_node_res_connect=False, is_edge_res_connect=False,
                 is_global_res_connect=False):
        """Constructor.
        
        Parameters
        ----------
        n_message_steps : int
            Number of message-passing steps.
        n_node_out : int
            Number of node output features.
        n_edge_out : int
            Number of edge output features.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        n_node_in : int, default=0
            Number of node input features.
        n_edge_in : int, default=0
            Number of edge input features.
        n_global_in : int, default=0
            Number of global input features.
        n_global_out : int, default=0
            Number of global output features.
        edge_to_node_aggr : {'add',}, default='add'
            Edge-to-node aggregation scheme.
        node_to_global_aggr : {'add', 'mean'}, default='add'
            Node-to-global aggregation scheme.
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
        global_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        global_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        is_node_res_connect : bool, default=False
            Add residual connections between nodes input and output features
            if True, False otherwise. Number of input and output features must
            match to process residual connections.
        is_edge_res_connect : bool, default=False
            Add residual connections between edges input and output features
            if True, False otherwise. Number of input and output features must
            match to process residual connections.
        is_global_res_connect : bool, default=False
            Add residual connections between global input and output features
            if True, False otherwise. Number of input and output features must
            match to process residual connections.
        """
        # Initialize from base class
        super(Processor, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = int(n_node_in)
        self._n_node_out = int(n_node_out)
        self._n_edge_in = int(n_edge_in)
        self._n_edge_out = int(n_edge_out)
        self._n_global_in = int(n_global_in)
        self._n_global_out = int(n_global_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of input features
        if (self._n_node_in < 1 and self._n_edge_in < 1
            and self._n_global_in < 1):
            raise RuntimeError(f'Impossible to setup model without node '
                               f'({self._n_node_in}), edge '
                               f'({self._n_edge_in}), or global '
                               f'({self._n_global_in}) input features.')
        # Check number of output features
        if self._n_node_out < 1 or self._n_edge_out < 1:
            raise RuntimeError(f'Number of node ({self._n_node_out}) and '
                               f'edge ({self._n_edge_out}) output features '
                               f'must be greater than 0.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of message-passing steps
        if n_message_steps < 1:
            raise RuntimeError('Number of message-passing steps must be at '
                               'least 1.')
        elif (n_message_steps > 1
              and ((n_node_in > 0 and n_node_in != n_node_out)
                    or (n_edge_in > 0 and n_edge_in != n_edge_out)
                    or (n_global_in > 0 and n_global_in != n_global_out))):
            raise RuntimeError('Number of node/edge/global input and output '
                               'features must match to process multiple '
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
        # Set global residual connections
        if is_global_res_connect and n_global_in != n_global_out:
            raise RuntimeError('Number of global input and output features '
                               'must match to process residual connections.')
        else:
            self._is_global_res_connect = is_global_res_connect
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sequence of identical Graph Interaction Networks
        self._processor = torch.nn.ModuleList(
            [GraphInteractionNetwork(
                n_node_out=n_node_out, n_edge_out=n_edge_out,
                n_hidden_layers=n_hidden_layers,
                hidden_layer_size=hidden_layer_size,
                n_node_in=n_node_in, n_edge_in=n_edge_in,
                n_global_in=n_global_in, n_global_out=n_global_out,
                edge_to_node_aggr=edge_to_node_aggr,
                node_to_global_aggr=node_to_global_aggr,
                node_hidden_activation=node_hidden_activation,
                node_output_activation=node_output_activation,
                edge_hidden_activation=edge_hidden_activation,
                edge_output_activation=edge_output_activation,
                global_hidden_activation=global_hidden_activation,
                global_output_activation=global_output_activation)
             for _ in range(n_message_steps)])
    # -------------------------------------------------------------------------
    def forward(self, edges_indexes, node_features_in=None,
                edge_features_in=None, global_features_in=None,
                batch_vector=None):
        """Forward propagation.
        
        Parameters
        ----------
        edges_indexes : torch.Tensor
            Edges indexes matrix stored as torch.Tensor(2d) with shape
            (2, n_edges), where the i-th edge is stored in edges_indexes[:, i]
            as (start_node_index, end_node_index).
        node_features_in : torch.Tensor, default=None
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features). If None, the edge-to-node aggregation is
            only built up to the highest receiver node index according with
            edges_indexes. To preserve total number of nodes in edge-to-node
            aggregation, pass torch.empty(n_nodes, 0) instead of None.
        edge_features_in : torch.Tensor, default=None
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_in : torch.Tensor, default=None
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
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
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check number of nodes and edges features
        if node_features_in is not None and node_features_in.numel() > 0 \
                and node_features_in.shape[1] != self._n_node_in:
            raise RuntimeError(f'Mismatch of number of node features of model '
                               f'({self._n_node_in}) and nodes input features '
                               f'matrix ({node_features_in.shape[1]}).')
        elif edge_features_in is not None \
                and edge_features_in.shape[1] != self._n_edge_in:
            raise RuntimeError(f'Mismatch of number of edge features of model '
                               f'({self._n_edge_in}) and edges input features '
                               f'matrix ({edge_features_in.shape[1]}).')
        elif global_features_in is not None \
                and global_features_in.shape[1] != self._n_global_in:
            raise RuntimeError(f'Mismatch of number of global features of '
                               f'model ({self._n_global_in}) and global input '
                               f'features matrix '
                               f'({global_features_in.shape[1]}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect number of nodes to preserve total number of nodes in
        # edge-to-node aggregation when number of node input features is zero
        n_nodes = None
        if self._n_node_in < 1 and node_features_in is not None:
            n_nodes = node_features_in.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual update node features
        node_features_out = None
        if node_features_in is not None:
            node_features_out = node_features_in.clone()
        # Initialize residual update edge features
        edge_features_out = None
        if edge_features_in is not None:
            edge_features_out = edge_features_in.clone()
        # Initialize residual update global features
        global_features_out = None
        if global_features_in is not None:
            global_features_out = global_features_in.clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over message-passing steps
        for i, gnn_model in enumerate(self._processor):
            # Save features matrix (residual connection)
            if self._is_node_res_connect:
                node_features_res = node_features_out.clone()
            if self._is_edge_res_connect and self._n_edge_in > 0:
                edge_features_res = edge_features_out.clone()
            if self._is_global_res_connect:
                global_features_res = global_features_out.clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform graph neural network message-passing step
            node_features_out, edge_features_out, global_features_out = \
                gnn_model(edges_indexes=edges_indexes,
                          node_features_in=node_features_out,
                          edge_features_in=edge_features_out,
                          global_features_in=global_features_out,
                          batch_vector=batch_vector)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if last message-passing step
            is_last_step = i == len(self._processor) - 1
            # Discard node features output matrix except in the last
            # message-passing step
            if self._n_node_in < 1 and not is_last_step:
                if isinstance(n_nodes, int):
                    node_features_out = torch.empty(n_nodes, 0)
                else:
                    node_features_out = None
            # Discard edge features output matrix except in the last
            # message-passing step
            if self._n_edge_in < 1 and not is_last_step:
                edge_features_out = None
            # Discard global features output matrix except in the last
            # message-passing step
            if self._n_global_in < 1 and not is_last_step:
                global_features_out = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add residual connections to features output
            if self._is_node_res_connect and self._n_node_in > 0:
                node_features_out += node_features_res
            if self._is_edge_res_connect and self._n_edge_in > 0:
                edge_features_out += edge_features_res
            if self._is_global_res_connect and self._n_global_in > 0:
                global_features_out += global_features_res
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
# =============================================================================
class Decoder(GraphIndependentNetwork):
    """GNN-based decoder.
    
    Decodes latent graph into output graph by means of a Graph Independent
    Network. Node, edge and global features update functions are implemented as
    multilayer feed-forward neural networks and are independent
    (no aggregation).
    """
    pass