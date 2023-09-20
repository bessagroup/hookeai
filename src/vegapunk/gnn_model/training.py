"""Training of Graph Neural Network based material patch model.

Functions
---------
train_model
    Training of GNN-based material patch model.
get_pytorch_loss
    Get PyTorch loss function.
get_pytorch_optimizer
    Get PyTorch optimizer.
get_learning_rate_scheduler
    Get PyTorch optimizer learning rate scheduler.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import torch
# Local
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def train_model(n_train_steps, dataset, model_init_args, learning_rate_init,
                opt_algorithm='adam', lr_scheduler_type=None,
                lr_scheduler_kwargs={}, loss_type='mse', loss_kwargs={},
                batch_size=1, is_sampler_shuffle=False, load_model_state=None,
                save_every=None, device_type='cpu', is_verbose=False):
    """Training of GNN-based material patch model.
    
    Parameters
    ----------
    n_train_steps : int
        Number of training steps.
    dataset : torch_geometric.data.Dataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_init_args : dict
        GNN-based material patch model class initialization parameters (check
        class GNNMaterialPatchModel).
    learning_rate_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    opt_algorithm : {'adam',}, default='adam'
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)
        
    lr_scheduler_type : {'steplr',}, default=None
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)

    lr_scheduler_kwargs : dict, default={}
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict, default={}
        Arguments of torch.nn._Loss initializer.   
    batch_size : int, default=1
        Number of samples loaded per batch.
    is_sampler_shuffle : bool, default=False
        If True, shuffles data set samples at every epoch.
    load_model_state : {'last', int}, default=None
        Load available GNN-based material patch model state from the model
        directory. The option 'last' loads the highest training step available,
        otherwise the specified training step (int) is loaded.
    save_every : int, default=None
        Save GNN-based material patch model every save_every training steps.
        If None, then saves only last training step.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Set device
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\nGNN-based material patch data model training'
              '\n--------------------------------------------\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    model = GNNMaterialPatchModel(
        n_node_in=model_init_args['n_node_in'],
        n_node_out=model_init_args['n_node_out'],
        n_edge_in=model_init_args['n_edge_in'],
        n_message_steps=model_init_args['n_message_steps'],
        n_hidden_layers=model_init_args['n_hidden_layers'],
        hidden_layer_size=model_init_args['hidden_layer_size'],
        model_directory=model_init_args['model_directory'],
        model_name=model_init_args['model_name'],
        device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    # Move model to process ID
    model.to(device=device)
    # Set model in training mode
    model.train()
    # Get model parameters
    model_parameters = model.parameters(recurse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=is_sampler_shuffle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize learning rate
    learning_rate = learning_rate_init
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer    
    if opt_algorithm == 'adam':
        # Initialize optimizer, specifying the model (and submodels) parameters
        # that should be optimized. By default, model parameters gradient flag
        # is set to True, meaning that gradients with respect to the parameters
        # are required (operations on the parameters are recorded for automatic
        # differentiation)
        optimizer = torch.optim.Adam(params=model_parameters, lr=learning_rate)
    else:
        raise RuntimeError('Unknown optimization algorithm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize learning rate scheduler
    is_lr_scheduler = False
    if lr_scheduler_type is not None:
        is_lr_scheduler = True
        lr_scheduler = get_learning_rate_scheduler(
            optimizer=optimizer, scheduler_type=lr_scheduler_type,
            **lr_scheduler_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize loss function
    loss_function = get_pytorch_loss(loss_type, **loss_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training flag
    is_keep_training = True
    # Initialize training step counter
    step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load GNN-based material patch model state
    if load_model_state == 'last' or isinstance(load_model_state, int):
        # Load GNN-based material patch model state
        loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                          load_model_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update training step counter
        step = int(loaded_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training iterations
    while is_keep_training:
        # Loop over batches
        for py_graph_batch in data_loader:
            # Initialize batch loss
            loss = torch.tensor(0)
            # Loop over graph samples
            for pyg_graph in py_graph_batch:
                # Move graph sample to process ID
                pyg_graph.to(device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute node internal forces predictions (forward
                # propagation). During the foward pass, PyTorch creates a
                # computation graph for the tensors that require gradients
                # (gradient flag set to True) to keep track of the operations
                # on these tensors, i.e., the model parameters. In addition,
                # PyTorch additionally stores the corresponding 'gradient
                # functions' (mathematical operator) of the executed operations
                # to the output tensor, stored in the .grad_fn attribute of the
                # corresponding tensors. Tensor.grad_fn is set to None for
                # tensors corresponding to leaf-nodes of the computation graph
                # or for tensors with the gradient flag set to False.
                node_internal_forces = \
                    model.predict_internal_forces(pyg_graph)
                # Get nodel internal forces ground-truth
                node_internal_forces_target = pyg_graph.y
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute and accumulate loss
                loss += loss_function(node_internal_forces,
                                      node_internal_forces_target)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize gradients (set to zero)
            optimizer.zero_grad()
            # Compute gradients with respect to model parameters (backward
            # propagation). PyTorch backpropagates recursively through the
            # computation graph of loss and computes the gradients with respect
            # to the model parameters. On each Tensor, PyTorch computes the
            # local gradients using the previously stored .grad_fn mathematical
            # operators and combines them with the incoming gradients to
            # compute the complete gradient (i.e., building the
            # differentiation chain rule). The backward propagation recursive
            # path stops when a leaf-node is reached (e.g., a model parameter),
            # where .grad_fn is set to None. Gradients are cumulatively stored
            # in the .grad attribute of the corresponding tensors
            loss.backward()
            # Perform optimization step. Gradients are stored in the .grad
            # attribute of model parameters
            optimizer.step()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update optimizer learning rate
            if is_lr_scheduler:
                lr_scheduler.step()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print('> Training step: {:d}/{:d} | Loss: {:.8e}'.format(
                    step, n_train_steps, loss))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save model and optimizer current states
            if step % save_every == 0:
                save_training_state(model=model, optimizer=optimizer,
                                    training_step=step)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check training process completion
            if step >= n_train_steps:
                # Completed prescribed number of training steps
                is_keep_training = False
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Increment training step counter
            step += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model and optimizer final states
    save_training_state(model=model, optimizer=optimizer, training_step=step)
# =============================================================================
def get_pytorch_loss(loss_type, **kwargs):
    """Get PyTorch loss function.
   
    Parameters
    ----------
    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    **kwargs
        Arguments of torch.nn._Loss initializer.
        
    Returns
    -------
    loss_function : torch.nn._Loss
        PyTorch loss function.
    """
    if loss_type == 'mse':
        loss_function = torch.nn.MSELoss(**kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch loss function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_function
# =============================================================================
def get_pytorch_optimizer(algorithm, params, **kwargs):
    """Get PyTorch optimizer.
   
    Parameters
    ----------
    algorithm : {'adam',}
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)
        
    params : list
        List of parameters (torch.Tensors) to optimize or list of dicts
        defining parameter groups.
    **kwargs
        Arguments of torch.optim.Optimizer initializer.
        
    Returns
    -------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    """
    if algorithm == 'adam':
        optimizer = torch.optim.Adam(params, **kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch optimizer.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return optimizer
# =============================================================================
def get_learning_rate_scheduler(optimizer, scheduler_type='steplr', **kwargs):
    """Get PyTorch optimizer learning rate scheduler.
    
    Parameters
    ----------
    scheduler_type : {'steplr',}
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    **kwargs
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        PyTorch optimizer learning rate scheduler.
    """
    if scheduler_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.SetpLR(optimizer, **kwargs)
    else:
        raise RuntimeError('Unknown or unavailable PyTorch optimizer '
                           'learning rate scheduler.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return scheduler
# =============================================================================
def save_training_state(model, optimizer, training_step):
    """Save model and optimizer states at given training step.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    training_step : int
        Training step.
    """
    # Save GNN-based material patch model
    model.save_model_state(training_step=training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file path
    optimizer_path = os.path.join(model.module.model_directory,
                                  model.module.model_name
                                  + '_optim-' + str(training_step) + '.pt')
    # Save optimizer state
    optimizer_state = dict(state=optimizer.state_dict(),
                           training_step=training_step)
    torch.save(optimizer_state, optimizer_path)
# =============================================================================
def load_training_state(model, opt_algorithm, optimizer, load_model_state):
    """Load model and optimizer states from available training data.
    
    Both model and optimizer are updated 'in-place' with loaded state data.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    opt_algorithm : {'adam',}, default='adam'
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)

    optimizer : torch.optim.Optimizer
        PyTorch optimizer.
    load_model_state : {'last', int}, default=None
        Load available GNN-based material patch model state from the model
        directory. The option 'last' loads the highest training step
        available, otherwise the specified training step (int) is loaded.
        
    Returns
    -------
    loaded_step : int
        Training step corresponding to loaded state data.
    """
    # Load GNN-based material patch model state
    if load_model_state == 'last':
        loaded_step = model.load_model_state(is_latest=True)
    else:
        loaded_step = model.load_model_state(
            training_step=int(load_model_state))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file path
    optimizer_path = os.path.join(model.module.model_directory,
                                  model.module.model_name
                                  + '_optim-' + str(loaded_step) + '.pt')
    # Load optimizer state
    if not os.path.isfile(optimizer_path):
        raise RuntimeError('Optimizer state file has not been found:\n\n'
                           + optimizer_path)
    else:
        # Initialize optimizer
        if opt_algorithm == 'adam':
            optimizer = torch.optim.Adam(params=model.parameters(recurse=True))
        else:
            raise RuntimeError('Unknown optimization algorithm')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load optimizer state
        optimizer_state = torch.load(optimizer_path)
        # Set loaded optimizer state
        optimizer.load_state_dict(optimizer_state['state'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loaded_step
        