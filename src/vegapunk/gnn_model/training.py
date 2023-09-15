"""Training of Graph Neural Network based material patch model.

Functions
---------
train_model
    Distributed data-parallel training of GNN-based material patch model.
train
    Train GNN-based material patch model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import os
# Third-party
import torch
# Local
from gnn_model.gnn_material_simulator import GNNMaterialPatchModel
from gnn_model.gnn_patch_dataset import get_gnn_material_patch_data_loader
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def train_model(model_directory, train_args, device='cpu', world_size=None):
    """Distributed data-parallel training of GNN-based material patch model.
    
    Parameters
    ----------
    model_directory : str
        Directory where material patch model is stored.
    train_args : dict
        Arguments passed to model training function.
    device : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    world_size : int, default=None
        Number of processes participating in the distributed training.
        If None, then all processes available for device are allocated. 
    """
    # Create directory where material patch model training files are stored
    if not os.path.exists(model_directory):
        make_directory(model_directory, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spawn processes for distributed data-parallel training
    DistributedTrainingTools.spawn_processes(train_function=train,
                                             train_args=train_args,
                                             device=device,
                                             world_size=world_size)
# =============================================================================
def train(rank, n_train_steps, dataset, model_init_args, opt_algorithm='Adam',
          batch_size=1, is_sampler_shuffle=False, load_model_state=None,
          save_every=None, world_size=1, device='cpu', is_verbose=False):
    """Train GNN-based material patch model.
    
    Parameters
    ----------
    rank : int
        Process ID (between 0 and world_size-1).
    n_train_steps : int
        Number of training steps.
    dataset : torch_geometric.data.Dataset
        GNN-based material patch data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_init_args : dict
        GNN-based material patch model initialization parameters (check
        class GNNMaterialPatchModel).
    opt_algorithm : {'Adam',}, default='Adam'
        Optimization algorithm.
    batch_size : int, default=1
        Number of samples loaded per batch.
    is_shuffle : bool, default=False
        If True, sampler shuffles data set samples at every epoch.
    load_model_state : {'last', int}, default=None
        Load available GNN-based material patch model state from the model
        directory. The option 'last' loads the highest training step available,
        otherwise the specified training step (int) is loaded.
    save_every : int, default=None
        Save GNN-based material patch model every save_every training steps.
        If None, then saves only last training step.
    world_size : int, default=1
        Number of processes participating in the distributed training.
    device : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Questions
    (1) Is it required to update the optimizer learning rate PER-PARAMETER?
    (2) Is collate required in the distributed data loader?
    """
    if is_verbose:
        if rank == 0:
            print('\nGNN-based material patch data model training'
                  '\n--------------------------------------------\n')
        print('\n> Rank ' + str(rank) + '/' + str(world_size)
              + ': Starting training process...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch model
    gnn_material_model = GNNMaterialPatchModel(
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
    # Check rank
    if rank < 0 or rank >= world_size:
        raise RuntimeError('Process ID (rank) must be contained between '
                           '0 and world_size-1. Current world_size: '
                            + str(world_size))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training distribution tools
    dist_training = DistributedTrainingTools(world_size=world_size,
                                             device=device)
    # Initialize training distribution process
    dist_training.initialize(rank=rank)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Wrap model for distributed data-parallel training
    model_ddp = DistributedTrainingTools.wrap_model(gnn_material_model)
    # Set model in training mode
    model_ddp.train()
    # Move model to process ID
    model_ddp.to(device=rank)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get data loader for distributed data-parallel training
    data_loader = DistributedTrainingTools.get_distributed_data_loader(
        dataset, batch_size=batch_size, is_sampler_shuffle=is_sampler_shuffle)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer
    if opt_algorithm == 'Adam':
        # Set learning rate
        learning_rate = None
        # Initialize optimizer, specifying the model (and submodels) parameters
        # that should be optimized. By default, model parameters gradient flag
        # is set to True, meaning that gradients with respect to the parameters
        # are required (operations on the parameters are recorded for automatic
        # differentiation)
        optimizer = torch.optim.Adam(params=model_ddp.parameters(recurse=True),
                                     lr=learning_rate)
    else:
        raise RuntimeError('Unknown optimization algorithm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize training flag
    is_keep_training = True
    # Initialize training step counter
    step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if load_model_state == 'last' or isinstance(load_model_state, int):
        # Load GNN-based material patch model state (in-place update)
        loaded_step = dist_training.load_training_state(
            rank, model_ddp, optimizer, load_model_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update training step counter
        step = int(loaded_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training iterations
    while is_keep_training:
        # Synchronize all processes before starting next training step
        torch.distributed.barrier()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over batch graph samples
        for pyg_graph in data_loader:
            # Move graph sample to process ID
            pyg_graph.to(rank)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute node internal forces predictions
            node_internal_forces = \
                model_ddp.module.predict_internal_forces(pyg_graph)
            # Get nodel internal forces ground-truth
            node_internal_forces_target = pyg_graph.y
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute loss
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(node_internal_forces, node_internal_forces_target)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize gradients (set to zero)
            optimizer.zero_grad()
            # Compute gradients with respect to model parameters (backward
            # propagation). Gradients are cumulatively stored as attributes of
            # the parameters (.grad)
            loss.backward()
            # Perform optimization step
            optimizer.step()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute updated optimizer learning rate
            learning_rate = None
            # Update optimizer learning rate
            optimizer.lr = learning_rate
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose and rank == 0:
                print('> Training step: {:d}/{:d} | Loss: {:.8e}'.format(
                    step, n_train_steps, loss))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save model and optimizer current states
            if rank == 0:
                DistributedTrainingTools.save_training_state(
                    model_ddp=model_ddp, optimizer=optimizer,
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
    if rank == 0:
        DistributedTrainingTools.save_training_state(
            model_ddp=model_ddp, optimizer=optimizer, training_step=step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Clean-up the default distributed process group
    DistributedTrainingTools.terminate()
# =============================================================================
class DistributedTrainingTools():
    """PyTorch Distributed Data-Parallel Training (DDP) tools.
    
    Attributes
    ----------
    _world_size : int
        Number of processes participating in the distributed training.
    _device : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    _backend : {'gloo', 'nccl'}
        Built-in backend for distributed training.
        
    Methods
    -------
    init(rank, world_size, device='cpu')
        Initialize training distribution process.
    initialize(self, rank)
        Initialize training distribution process.
    spawn_processes(train_function, train_args, device='cpu', world_size=None)
        Spawns processes for distributed data-parallel training.
    terminate()
        Terminate training distribution process.
    wrap_model(model)
        Wrap model for distributed data-parallel training.
    get_distributed_data_loader(dataset, batch_size=1, is_sampler_shuffle=True)
        Get data loader for distributed data-parallel training.
    save_training_step(model_ddp, optimizer, training_step)
        Save model and optimizer states at given training step.
    optimizer_to(self, optimizer, rank)
        Move optimizer to process ID.
    """
    def __init__(self, world_size=1, device='cpu'):
        """Constructor.
        
        Parameters
        ----------
        world_size : int, default=1
            Number of processes participating in the distributed training.
        device : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        self._world_size = int(world_size)
        if device in ('cpu', 'gpu'):
            self._device = device
        else:
            raise RuntimeError('Invalid device for torch.Tensor allocation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set built-in backend according with type of PyTorch tensor
        if self._device == 'gpu':
            # Set NCCL backend for distributed GPU training
            self._backend = 'nccl'
        else:
            # Set Gloo backend for distributed CPU training
            self._backend = 'gloo'
    # -------------------------------------------------------------------------  
    def initialize(self, rank):
        """Initialize training distribution process.
        
        Parameters
        ----------
        rank : int
            Process ID (between 0 and world_size-1).
        """
        # Check rank
        if rank < 0 or rank >= self._world_size:
            raise RuntimeError('Process ID (rank) must be contained between '
                               '0 and world_size-1. Current world_size: '
                               + str(self._world_size))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize the distributed training process group
        torch.distributed.init_process_group(backend=self._backend, rank=rank,
                                             world_size=self._world_size)
    # -------------------------------------------------------------------------
    @staticmethod
    def spawn_processes(train_function, train_args, device='cpu',
                        world_size=None):
        """Spawn processes for distributed data-parallel training.
        
        Parameters
        ----------
        train_function : function
            Model training function called as function(rank, args), where
            rank is the process ID and args is a tuple of arguments.
        train_args : dict
            Arguments passed to model training function.
        device : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        world_size : int, default=None
            Number of processes participating in the distributed training.
            If None, then all processes available for device are allocated.
        """
        # Get number of available processes for distributed training
        if world_size is None:
            if device == 'cpu':
                n_process = os.cpu_count()
            elif device == 'gpu':
                n_process = torch.cuda.device_count()
            else:
                RuntimeError('Invalid device type.')
        else:
            n_process = int(world_size)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Spawn processes for distributed data-parallel training
        torch.multiprocessing.spawn(train_function, args=(train_args,),
                                    nprocs=n_process)
    # -------------------------------------------------------------------------
    @staticmethod
    def terminate():
        """Terminate training distribution process."""
        torch.distributed.destroy_process_group()
    # -------------------------------------------------------------------------
    @staticmethod
    def wrap_model(model):
        """Wrap model for distributed data-parallel training.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model whose training is parallelized.
            
        Returns
        -------
        model_ddp : DistributedDataParallel
            Wrapper over model for distributed data-parallel training.
        """
        return torch.nn.parallel.DistributedDataParallel(model)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_distributed_data_loader(dataset, batch_size=1,
                                    is_sampler_shuffle=True):
        """Get data loader for distributed data-parallel training.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Data set.
        batch_size : int, default=1
            Number of samples loaded per batch.
        is_sampler_shuffle : bool, default=True
            If True, sampler shuffles data set samples.
            
        Returns
        -------
        data_loader : torch.utils.data.DataLoader
            Data loader.
        """
        # Check data set
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise RuntimeError('Data set must be torch.utils.data.Dataset.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sampler for distributed data-parallel training. Sampler restricts
        # data loading to a subset of the dataset. In a DistributedDataParallel
        # environment, each process can then have a data loader that loads a
        # subset of the original data set that is exclusive to it
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=is_sampler_shuffle)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=sampler)
    # -------------------------------------------------------------------------
    @staticmethod
    def save_training_state(model_ddp, optimizer, training_step):
        """Save model and optimizer states at given training step.
        
        Parameters
        ----------
        model_ddp : DistributedDataParallel
            Wrapper over model for distributed data-parallel training.
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        training_step : int
            Training step.
        """
        # Save GNN-based material patch model
        model_ddp.module.save_model_state(training_step=training_step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set optimizer state file path
        optimizer_path = os.path.join(model_ddp.module.model_directory,
                                      model_ddp.module.model_name
                                      + '_optim-' + str(training_step) + '.pt')
        # Save optimizer state
        optimizer_state = dict(state=optimizer.state_dict(),
                               training_step=training_step)
        torch.save(optimizer_state, optimizer_path)
    # -------------------------------------------------------------------------
    def load_training_state(self, rank, model_ddp, optimizer,
                            load_model_state):
        """Load model and optimizer states from available training data.
        
        Both model and optimizer are updated 'in-place' with loaded state data.
        
        Parameters
        ----------
        rank : int
            Process ID (between 0 and world_size-1).
        model_ddp : DistributedDataParallel
            Wrapper over model for distributed data-parallel training.
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
            loaded_step = model_ddp.module.load_model_state(is_latest=True)
        else:
            loaded_step = model_ddp.module.load_model_state(
                training_step=int(load_model_state))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set optimizer state file path
        optimizer_path = os.path.join(model_ddp.module.model_directory,
                                      model_ddp.module.model_name
                                      + '_optim-' + str(loaded_step) + '.pt')
        # Load optimizer state
        if not os.path.isfile(optimizer_path):
            raise RuntimeError('Optimizer state file has not been found:\n\n'
                               + optimizer_path)
        else:
            # Initialize optimizer, specifying the model (and submodels)
            # parameters that should be optimized
            optimizer = torch.optim.Adam(
                params=model_ddp.parameters(recurse=True))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load optimizer state
            optimizer_state = torch.load(optimizer_path)
            # Set loaded optimizer state
            optimizer.load_state_dict(optimizer_state['state'])
            # Move optimizer to process ID
            self.optimizer_to(optimizer=optimizer, rank=rank)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return loaded_step
    # -------------------------------------------------------------------------
    def optimizer_to(self, optimizer, rank):
        """Move optimizer to process ID.
        
        Similar to torch.nn.Module.to() but applied to torch.optim.Optimizer.
        
        Required to load a previously stored optimizer state in a distributed
        data-parallel training environment.
        
        Currently this method is not available as a built-in function of
        PyTorch. Workaround is taken from:
        https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        rank : int
            Process ID (between 0 and world_size-1).
        """
        # Check optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError('Optimizer must be torch.optim.Optimizer.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over optimizer parameters
        for param in optimizer.state.values():
            # Move parameter data to process ID
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(rank)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(rank)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(rank)
                        if subparam._grad is not None:
                            subparam._grad.data = \
                                subparam._grad.data.to(rank)
        