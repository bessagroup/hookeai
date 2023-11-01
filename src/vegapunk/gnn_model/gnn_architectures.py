"""Graph Neural Networks architectures.

Classes
-------
GraphIndependentNetwork(torch.nn.Module)
    Graph Independent Network.
GraphInteractionNetwork(torch_geometric.nn.MessagePassing)
    Graph Interaction Network.
 
Functions
---------
build_fnn
    Build multilayer feed-forward neural network.
"""
#
#                                                                       Modules
# =============================================================================
from collections import OrderedDict
# Third-party
import torch
import torch_geometric.nn
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def build_fnn(input_size, output_size,
              output_activation=torch.nn.Identity,
              hidden_layer_sizes=[],
              hidden_activation=torch.nn.ReLU):
    """Build multilayer feed-forward neural network.
    
    Parameters
    ----------
    input_size : int
        Number of neurons of input layer.
    output_size : int
        Number of neurons of output layer.
    output_activation : torch.nn.Module, default=torch.nn.Identity
        Output unit activation function. Defaults to identity (linear) unit
        activation function.
    hidden_layer_sizes : list[int], default=[]
        Number of neurons of hidden layers.
    hidden_activation : torch.nn.Module, default=torch.nn.ReLU
        Hidden unit activation function. Defaults to ReLU (rectified linear
        unit function) unit activation function.

    Returns
    -------
    fnn : torch.nn.Sequential
        Multilayer feed-forward neural network.
    """
    # Set number of neurons of each layer
    layer_sizes = []
    layer_sizes.append(input_size)
    layer_sizes += hidden_layer_sizes
    layer_sizes.append(output_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of layers of adaptive weights
    n_layer = len(layer_sizes) - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hidden and output layers unit activation functions
    if not callable(hidden_activation):
        raise RuntimeError('Hidden unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions = [hidden_activation for i in range(n_layer - 1)]
    if not callable(output_activation):
        raise RuntimeError('Output unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions.append(output_activation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create multilayer feed-forward neural network:
    # Initialize neural network
    fnn = torch.nn.Sequential()
    # Loop over neural network layers
    for i in range(n_layer):
        # Set layer linear transformation
        fnn.add_module("Layer-" + str(i),
                       torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1],
                                       bias=True))
        # Set layer unit activation function
        fnn.add_module("Activation-" + str(i), activation_functions[i]())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return fnn
# =============================================================================
class GraphIndependentNetwork(torch.nn.Module):
    """Graph Independent Network.
    
    A Graph Network block with (1) distinct update functions for node, edge and
    global features implemented as multilayer feed-forward neural networks
    and (2) no aggregation functions, i.e., independent node, edges and global
    blocks.
    
    Attributes
    ----------
    _node_fn : torch.nn.Sequential
        Node update function.
    _edge_fn : torch.nn.Sequential
        Edge update function.
        
    Methods
    -------
    forward(self, node_features_in, edge_features_in)
        Forward propagation.
    """
    def __init__(self, n_node_in, n_node_out, n_edge_in, n_edge_out,
                 n_hidden_layers, hidden_layer_size,
                 node_hidden_activation=torch.nn.Identity,
                 node_output_activation=torch.nn.Identity,
                 edge_hidden_activation=torch.nn.Identity,
                 edge_output_activation=torch.nn.Identity):
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
        edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        """
        # Initialize Graph Network block from base class
        super(GraphIndependentNetwork, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node update function as multilayer feed-forward neural network
        # with layer normalization:
        # Build multilayer feed-forward neural network
        fnn = build_fnn(
            input_size=n_node_in,
            output_size=n_node_out,
            output_activation=node_output_activation,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=node_hidden_activation)
        # Build normalization layer (per-feature)
        norm_layer = torch.nn.BatchNorm1d(num_features=n_node_out, affine=True)
        # Set node update function
        self._node_fn = torch.nn.Sequential()
        self._node_fn.add_module('FNN', fnn)
        self._node_fn.add_module('Norm-Layer', norm_layer)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edge update function as multilayer feed-forward neural network
        # with layer normalization:
        # Build multilayer feed-forward neural network
        fnn = build_fnn(
            input_size=n_edge_in,
            output_size=n_edge_out,
            output_activation=edge_output_activation,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=edge_hidden_activation)
        # Build normalization layer (per-feature)
        norm_layer = torch.nn.BatchNorm1d(num_features=n_edge_out, affine=True)
        # Set edge update function
        self._edge_fn = torch.nn.Sequential()
        self._edge_fn.add_module('FNN', fnn)
        self._edge_fn.add_module('Norm-Layer', norm_layer)
    # -------------------------------------------------------------------------
    def forward(self, node_features_in, edge_features_in):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.Tensor
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_in : torch.Tensor
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
            
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        """
        # Check number of nodes and edges
        if node_features_in.shape[0] < 2 or edge_features_in.shape[0] < 2:
            raise RuntimeError('Number of nodes and number of edges must be '
                               'greater than 1 to compute standard deviation '
                               'in corresponding update functions the '
                               'normalization layer.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return self._node_fn(node_features_in), \
            self._edge_fn(edge_features_in)
# =============================================================================
class GraphInteractionNetwork(torch_geometric.nn.MessagePassing):
    """Graph Interaction Network.
    
    A Graph Network block with (1) distinct update functions for node, edge and
    global features implemented as multilayer feed-forward neural networks
    and (2) aggregation functions implemented as elementwise summations. Global
    features are not used to update the edge features.
    
    Attributes
    ----------
    _node_fn : torch.nn.Sequential
        Node update function.
    _edge_fn : torch.nn.Sequential
        Edge update function.
        
    Methods
    -------
    forward(self, node_features_in, edge_features_in, edges_indexes)
        Forward propagation.
    message(self, node_features_in_i, node_features_in_j, edge_features_in)
        Builds messages to node i from each edge (j, i) (edge update).
    update(self, node_features_in_aggr, node_features_in, edge_features_out)
        Update node features.
    """
    def __init__(self, n_node_in, n_node_out, n_edge_in, n_edge_out,
                 n_hidden_layers, hidden_layer_size,
                 aggregation_scheme='add',
                 node_hidden_activation=torch.nn.Identity,
                 node_output_activation=torch.nn.Identity,
                 edge_hidden_activation=torch.nn.Identity,
                 edge_output_activation=torch.nn.Identity):
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
        """
        # Set aggregation scheme
        if aggregation_scheme == 'add':
            aggregation = torch_geometric.nn.aggr.SumAggregation()
        else:
            raise RuntimeError('Unknown aggregation scheme.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set flow direction of message passing
        flow = 'source_to_target'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Graph Network block from base class
        super(GraphInteractionNetwork, self).__init__(aggr=aggregation,
                                                      flow=flow)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node update function as multilayer feed-forward neural network
        # with layer normalization:
        # Build multilayer feed-forward neural network
        fnn = build_fnn(
            input_size=n_node_in+n_edge_out,
            output_size=n_node_out,
            output_activation=node_output_activation,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=node_hidden_activation)
        # Build normalization layer (per-feature)
        norm_layer = torch.nn.BatchNorm1d(num_features=n_node_out, affine=True)
        # Set node update function
        self._node_fn = torch.nn.Sequential()
        self._node_fn.add_module('FNN', fnn)
        self._node_fn.add_module('Norm-Layer', norm_layer)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edge update function as multilayer feed-forward neural network
        # with layer normalization:
        # Build multilayer feed-forward neural network
        fnn = build_fnn(
            input_size=n_edge_in+2*n_node_in,
            output_size=n_edge_out,
            output_activation=edge_output_activation,
            hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
            hidden_activation=edge_hidden_activation)
        # Build normalization layer (per-feature)
        norm_layer = torch.nn.BatchNorm1d(num_features=n_edge_out, affine=True)
        # Set node update function
        self._edge_fn = torch.nn.Sequential()
        self._edge_fn.add_module('FNN', fnn)
        self._edge_fn.add_module('Norm-Layer', norm_layer)
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
        # Check number of nodes and edges
        if node_features_in.shape[0] < 2 or edge_features_in.shape[0] < 2:
            raise RuntimeError('Number of nodes and number of edges must be '
                               'greater than 1 to compute standard deviation '
                               'in corresponding update functions the '
                               'normalization layer.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform graph neural network message-passing step (message,
        # aggregation, update) and get updated node features
        node_features_out = self.propagate(
            edge_index=edges_indexes, node_features_in=node_features_in,
            edge_features_in=edge_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get updated edge features
        edge_features_out = self._edge_features_out
        self._edge_features_out = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out
    # -------------------------------------------------------------------------
    def message(self, node_features_in_i, node_features_in_j,
                edge_features_in):
        """Builds messages to node i from each edge (j, i) (edge update).
        
        Assumes that j is the source node and i is the receiver node (flow
        direction set as 'source_to_target'). For each edge (j, i), the
        update function input features result from concatenation of the edge
        features and the corresponding nodes features.
        
        The source and receiver node input features mappings based on the edges
        indexes matrix are built in the __collect__() method of class
        torch_geometric.nn.MessagePassing.
        
        The edges features output matrix is passed as the input tensor to the
        aggregation operator (class torch.nn.aggr.Aggregation) set in the
        initialization of the torch_geometric.nn.MessagePassing class.
        
        Parameters
        ----------
        node_features_in_i : torch.Tensor
            Source node input features for each edge stored as a
            torch.Tensor(2d) of shape (n_edges, n_features). Mapping is
            performed based on the edges indexes matrix.
        node_features_in_j : torch.Tensor
            Receiver node input features for each edge stored as a
            torch.Tensor(2d) of shape (n_edges, n_features). Mapping is
            performed based on the edges indexes matrix.
        edge_features_in : torch.Tensor
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
            
        Returns
        -------
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        """
        # Concatenate features for each edge
        edge_features_in_cat = torch.cat([node_features_in_i,
                                          node_features_in_j,
                                          edge_features_in], dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update edge features
        edge_features_out = self._edge_fn(edge_features_in_cat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store updated edges features
        self._edge_features_out = edge_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features_out
    # -------------------------------------------------------------------------
    def update(self, node_features_in_aggr, node_features_in):
        """Update node features.
        
        The nodes features input matrix resulting from message passing and
        aggregation is built in the aggregation operator (class
        torch.nn.aggr.Aggregation) set in the initialization of the
        torch_geometric.nn.MessagePassing class.
        
        Parameters
        ----------
        node_features_in_aggr : torch.Tensor
            Nodes features input matrix resulting from message passing and
            aggregation, stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        node_features_in : torch.Tensor
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
              
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """
        # Concatenate features for each node
        node_features_in_cat = torch.cat([node_features_in_aggr,
                                          node_features_in], dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update node features
        node_features_out = self._node_fn(node_features_in_cat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out