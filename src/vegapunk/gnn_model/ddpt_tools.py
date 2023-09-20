"""PyTorch Distributed Data-Parallel Training (DDP) tools.

Classes
-------
DistributedTrainingTools
    PyTorch Distributed Data-Parallel Training (DDP) tools.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class DistributedTrainingTools:
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
    update_learning_rate(self, learning_rate)
        Update optimizer learning rate.
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
    def update_learning_rate(self, learning_rate):
        """Update optimizer learning rate.
        
        Parameters
        ----------
        learning_rate : float
            PyTorch optimizer learning rate.
        
        Returns
        -------
        learning_rate : float
            Updated PyTorch optimizer learning rate.
        """
        return learning_rate*self._world_size
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
        # Set environment variables (NOT SURE HOW TO SET THESE):
        # Set IP address of the machine that will host the process with rank 0
        os.environ['MASTER_ADDR'] = 'localhost'
        # Set free port of the machine that will host the process with rank 0
        os.environ['MASTER_PORT'] = '12355'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        
        Assumes model class has method save_model_state() that saves
        model.state_dict().
        
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
    def load_training_state(self, rank, model_ddp, opt_algorithm, optimizer,
                            load_model_state):
        """Load model and optimizer states from available training data.
        
        Both model and optimizer are updated 'in-place' with loaded state data.
        
        Assumes model class has method load_model_state() that loads model
        state from model.load_state_dict().
        
        Parameters
        ----------
        rank : int
            Process ID (between 0 and world_size-1).
        model_ddp : DistributedDataParallel
            Wrapper over model for distributed data-parallel training.
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
            # Initialize optimizer
            if opt_algorithm == 'adam':
                optimizer = torch.optim.Adam(
                    params=model_ddp.parameters(recurse=True))
            else:
                raise RuntimeError('Unknown optimization algorithm')
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