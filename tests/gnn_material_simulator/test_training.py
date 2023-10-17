"""Test training of Graph Neural Network based material patch model."""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import random
# Third-party
import pytest
import torch
import numpy as np
# Local
from src.vegapunk.gnn_model.gnn_material_simulator import \
    GNNMaterialPatchModel
from src.vegapunk.gnn_model.training import \
    get_pytorch_loss, get_pytorch_optimizer, get_learning_rate_scheduler, \
    save_training_state, load_training_state, save_loss_history, \
    load_loss_history, load_lr_history, seed_worker
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
@pytest.mark.parametrize('torch_seed', [0, 1, 2])
def test_seed_worker(torch_seed):
    """Test workers seed reproducibility in PyTorch data loaders."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dummy worker ID
    worker_id = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set PyTorch initial seed
    torch.manual_seed(torch_seed)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set workers seed
    seed_worker(worker_id)
    # Generate sequence of random numbers (NumPy)
    np_random_1 = np.random.randint(0, 100, 10)
    # Generate sequence of random numbers (random)
    random_1 = [random.randint(0, 100) for _ in range(10)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set workers seed
    seed_worker(worker_id)
    # Generate sequence of random numbers (NumPy)
    np_random_2 = np.random.randint(0, 100, 10)
    # Generate sequence of random numbers (random)
    random_2 = [random.randint(0, 100) for _ in range(10)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check NumPy random number generator reproducibility
    if not np.allclose(np_random_1, np_random_2):
        errors.append('NumPy random number generator was not reproduced.')
    # Check random random number generator reproducibility
    if not np.allclose(random_1, random_2):
        errors.append('Random random number generator was not reproduced.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('total_n_train_steps, loss_type, loss_history,'
                         'training_step, target_load_history, '
                         'lr_scheduler_type, lr_history, target_lr_history',
                         [(4, 'mse', [0.0, 2.0, 4.0, 6.0],
                           None, [0.0, 2.0, 4.0, 6.0], None, None, 4*[None,]),
                          (4, 'mse', [0.0, 2.0, 4.0, 6.0],
                           3, [0.0, 2.0, 4.0, 6.0], 'steplr',
                           [0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]),
                          (10, 'mse', [0.0, 2.0, 4.0, 6.0],
                           2, [0.0, 2.0, 4.0], 'steplr', [0.0, 0.1, 0.2, 0.3],
                           [0.0, 0.1, 0.2]),
                          (10, 'mse', [0.0, 2.0, 4.0, 6.0],
                           2, [0.0, 2.0, 4.0], 'steplr', None, 3*[None,]),
                          ])
def test_save_and_load_loss_history(gnn_material_simulator,
                                    total_n_train_steps, loss_type,
                                    loss_history, training_step,
                                    target_load_history, lr_scheduler_type,
                                    lr_history, target_lr_history):
    """Test saving and loading of training process loss history."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save training process loss history
    save_loss_history(gnn_material_simulator, total_n_train_steps, loss_type,
                      loss_history, lr_scheduler_type=lr_scheduler_type,
                      lr_history=lr_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss history record file path
    loss_record_path = os.path.join(gnn_material_simulator.model_directory,
                                    'loss_history_record' + '.pkl')
    # Check loss history record file
    if not os.path.isfile(loss_record_path):
        errors.append('Loss history record file has not been found.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training process loss history
    loaded_loss_history = load_loss_history(gnn_material_simulator, loss_type,
                                            training_step=training_step)
    # Load training process learning rate history
    loaded_lr_history = load_lr_history(gnn_material_simulator,
                                        training_step=training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check loaded loss history
    if not np.allclose(loaded_loss_history, target_load_history):
        errors.append('Loss history was not properly recovered from file.')

    # Check loaded learning rate history
    if lr_history is not None:
        if not np.allclose(loaded_lr_history, target_lr_history):
            errors.append('Learning rate history was not properly recovered '
                          'from file.')
    else:
        if not loaded_lr_history == (len(loaded_loss_history))*[None,]:
            errors.append('Learning rate history was not properly set in the '
                          'absence of available history.')   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove loss history record file
    loss_record_path = os.path.join(gnn_material_simulator.model_directory,
                                    'loss_history_record' + '.pkl')
    os.remove(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Attempt to load unexistent training process loss and learning rate
    # histories
    if training_step is not None:
        loaded_loss_history = \
            load_loss_history(gnn_material_simulator, loss_type,
                              training_step=training_step)
        if not loaded_loss_history == (training_step + 1)*[None,]:
            errors.append('Loss history was not properly set in the absence '
                          'of loss history record file.')
        loaded_lr_history = load_lr_history(gnn_material_simulator,
                                            training_step=training_step)        
        if not loaded_lr_history == (training_step + 1)*[None,]:
            errors.append('Learning rate history was not properly set in the '
                          'absence of loss history record file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
def test_invalid_load_loss_history(gnn_material_simulator):
    """Test invalid loading of training process loss history."""
    # Set valid parameters to save and load training process loss history
    total_n_train_steps = 4
    loss_type = 'mse'
    loss_history = [0.0, 2.0, 4.0, 6.0]
    training_step = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save training process loss history
    save_loss_history(gnn_material_simulator, total_n_train_steps, loss_type,
                      loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test inconsistent loss history type
    test_loss_type = 'not_mse'
    with pytest.raises(RuntimeError):
        _ = load_loss_history(gnn_material_simulator, test_loss_type,
                              training_step=training_step)
    # Test training step beyong available loss history
    test_training_step = 4
    with pytest.raises(RuntimeError):
        _ = load_loss_history(gnn_material_simulator, loss_type,
                              training_step=test_training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove loss history record file
    loss_record_path = os.path.join(gnn_material_simulator.model_directory,
                                    'loss_history_record' + '.pkl')
    os.remove(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test loading unexistent training process loss history with unknown
    # training step
    with pytest.raises(RuntimeError):
        _ = load_loss_history(gnn_material_simulator, loss_type,
                              training_step=None)
# -----------------------------------------------------------------------------
def test_invalid_load_lr_history(gnn_material_simulator):
    """Test invalid loading of training process learning rate history."""
    # Set valid parameters to save and load training process loss and learning
    # rate histories
    total_n_train_steps = 4
    loss_type = 'mse'
    loss_history = [0.0, 2.0, 4.0, 6.0]
    lr_scheduler_type = 'steplr'
    lr_history = [0.0, 0.1, 0.2, 0.3]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save training process loss history
    save_loss_history(gnn_material_simulator, total_n_train_steps, loss_type,
                      loss_history, lr_scheduler_type=lr_scheduler_type,
                      lr_history=lr_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test training step beyong available learning rate history
    test_training_step = 4
    with pytest.raises(RuntimeError):
        _ = load_lr_history(gnn_material_simulator,
                            training_step=test_training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove loss history record file
    loss_record_path = os.path.join(gnn_material_simulator.model_directory,
                                    'loss_history_record' + '.pkl')
    os.remove(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test loading unexistent training process learning rate history with
    # unknown training step
    with pytest.raises(RuntimeError):
        _ = load_lr_history(gnn_material_simulator, training_step=None)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('opt_algorithm', ['adam',])
def test_save_and_load_model_state(tmp_path, opt_algorithm):
    """Test saving and loading of model and optimizer states."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model initialization parameters
    model_init_args = dict(n_node_in=2, n_node_out=5, n_edge_in=3,
                           n_message_steps=2, n_hidden_layers=2,
                           hidden_layer_size=2, model_directory=str(tmp_path),
                           model_name='material_patch_model',
                           is_data_normalization=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN-based material patch model
    model = GNNMaterialPatchModel(**model_init_args)
    # Initialize optimizer
    if opt_algorithm == 'adam':
        optimizer = torch.optim.Adam(model.parameters(recurse=True))
    # Save model and optimizer states
    save_training_state(model, optimizer)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store model state
    saved_model_state = model.state_dict()
    # Store optimizer state
    saved_optimizer_state = optimizer.state_dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state state file path
    model_state_path = os.path.join(model.model_directory,
                                    model.model_name + '.pt')
    # Check model state file
    if not os.path.isfile(model_state_path):
        errors.append('Model state file has not been found.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimizer state file path
    optimizer_state_path = os.path.join(model.model_directory,
                                        model.model_name + '_optim' + '.pt')
    # Check optimizer state file
    if not os.path.isfile(optimizer_state_path):
        errors.append('Optimizer state file has not been found.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model and optimizer states
    loaded_step = load_training_state(model, opt_algorithm, optimizer)
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Check loaded model state training step
    if loaded_step != 0:
        errors.append('Unknown training step was not properly recovered from '
                      'file.')
    # Check model state parameters
    if str(saved_model_state) != str(model.state_dict()):
        errors.append('Model state was not properly recovered from file.')
    # Check optimizer state parameters
    if str(saved_optimizer_state) != str(optimizer.state_dict()):
        errors.append('Optimizer state was not properly recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model and optimizer saved training step states
    saved_model_states = []
    saved_optimizer_states = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of training steps
    n_train_steps = 5
    # Set random best states during training process
    best_training_states = (1, 3)
    # Loop over training steps
    for step in range(n_train_steps):
        # Build GNN-based material patch model (reinitializing parameters to
        # emulate parameters update)
        model = GNNMaterialPatchModel(**model_init_args)
        # Initialize optimizer (reinitializing parameters to emulate parameters
        # update)
        if opt_algorithm == 'adam':
            optimizer = torch.optim.Adam(model.parameters(recurse=True))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save model and optimizer states at given training step
        save_training_state(model, optimizer, step)
        if step in best_training_states:
            save_training_state(model, optimizer, step, is_best_state=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model state
        saved_model_states.append(model.state_dict())
        # Store optimizer state
        saved_optimizer_states.append(optimizer.state_dict())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state state file path
        model_state_path = os.path.join(model.model_directory, model.model_name
                                        + '-' + str(step) + '.pt')
        # Check model state file
        if not os.path.isfile(model_state_path):
            errors.append('Model state file has not been found.')        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set optimizer state file path
        optimizer_state_path = \
            os.path.join(model.model_directory, model.model_name
                        + '_optim-' + str(step) + '.pt')
        # Check optimizer state file
        if not os.path.isfile(optimizer_state_path):
            errors.append('Optimizer state file has not been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check best performance state files
        if step in best_training_states:
            # Set model state state file path
            model_state_path = \
                os.path.join(model.model_directory,
                             model.model_name + '-' + str(step) + '-best'
                             + '.pt')
            # Check model state file
            if not os.path.isfile(model_state_path):
                errors.append('Model state file has not been found.')
            # Set optimizer state file path
            optimizer_state_path = \
                os.path.join(model.model_directory,
                             model.model_name + '_optim-' + str(step) + '-best'
                             + '.pt')
            # Check optimizer state file
            if not os.path.isfile(optimizer_state_path):
                errors.append('Optimizer state file has not been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build GNN-based material patch model (reinitializing parameters to
        # emulate parameters update)
        model = GNNMaterialPatchModel(**model_init_args)
        # Initialize optimizer
        if opt_algorithm == 'adam':
            optimizer = torch.optim.Adam(model.parameters(recurse=True))
        # Load model and optimizer states
        loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                          load_model_state=step)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        # Check loaded model state training step
        if loaded_step != step:
            errors.append('Training step was not properly recovered from '
                          'file.')
        # Check model state parameters
        if str(saved_model_states[step]) != str(model.state_dict()):
            errors.append('Training step model state was not properly '
                          'recovered from file.')
        # Check optimizer state parameters
        if str(saved_optimizer_states[step]) != str(optimizer.state_dict()):
            errors.append('Training step optimizer state was not properly '
                          'recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model and optimizer states
    loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                      load_model_state='last')
    # Check loaded model state training step
    if loaded_step != 4:
        errors.append('Last training step was not properly recovered from '
                      'file.')
    # Check model state parameters
    if str(saved_model_states[4]) != str(model.state_dict()):
        errors.append('Last training step model state was not properly '
                      'recovered from file.')
    # Check optimizer state parameters
    if str(saved_optimizer_states[4]) != str(optimizer.state_dict()):
        errors.append('Last training step optimizer state was not properly '
                      'recovered from file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model and optimizer states
    loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                      load_model_state='best')
    # Check loaded model state training step
    if loaded_step != 3:
        errors.append('Best state training step was not properly recovered '
                      'from file.')
    # Check model state parameters
    if str(saved_model_states[3]) != str(model.state_dict()):
        errors.append('Best model state was not properly recovered from file.')
    # Check optimizer state parameters
    if str(saved_optimizer_states[3]) != str(optimizer.state_dict()):
        errors.append('Best optimizer state was not properly recovered from '
                      'file.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model and optimizer states
    loaded_step = load_training_state(model, opt_algorithm, optimizer,
                                      load_model_state=2)
    # Check loaded model state training step
    if loaded_step != 2:
        errors.append('Old training step was not properly recovered from '
                      'file.')
    # Check model state parameters
    if str(saved_model_states[2]) != str(model.state_dict()):
        errors.append('Old training step model state was not properly '
                      'recovered from file.')
    # Check optimizer state parameters
    if str(saved_optimizer_states[2]) != str(optimizer.state_dict()):
        errors.append('Old training step optimizer state was not properly '
                      'recovered from file.')    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('opt_algorithm', ['adam',])
def test_invalid_load_model_state(gnn_material_simulator, opt_algorithm):
    """Test invalid loading of model and optimizer states."""
    # Set GNN-based material patch model
    model = gnn_material_simulator
    # Initialize optimizer
    if opt_algorithm == 'adam':
        optimizer = torch.optim.Adam(model.parameters(recurse=True))
    # Set training step
    training_step = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model and optimizer states at given training step
    save_training_state(model, optimizer, training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test detection of unknown optimizer
    with pytest.raises(RuntimeError):
         _ = load_training_state(model, 'unknown_optimizer', optimizer,
                                 load_model_state=training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove optimizer state file path
    optimizer_state_path = os.path.join(model.model_directory,
                                        model.model_name + '_optim-'
                                        + str(training_step) + '.pt')
    os.remove(optimizer_state_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test loading unexistent optimizer state file
    with pytest.raises(RuntimeError):
         _ = load_training_state(model, opt_algorithm, optimizer,
                                 load_model_state=0)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('scheduler_type', ['steplr',])
def test_get_learning_rate_scheduler(pytorch_optimizer_adam, scheduler_type):
    """Test PyTorch optimizer learning rate scheduler getter."""
    # Set scheduler parameters
    if scheduler_type == 'steplr':
        kwargs = {'step_size': 5, 'gamma': 0.1}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get learning rate scheduler
    scheduler = get_learning_rate_scheduler(pytorch_optimizer_adam,
                                            scheduler_type, **kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)    
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('scheduler_type', ['steplr', 'unknown_type'])
def test_get_learning_rate_scheduler_invalid(pytorch_optimizer_adam,
                                             scheduler_type):
    """Test invalid PyTorch optimizer learning rate scheduler getter."""
    # Set empty scheduler parameters
    kwargs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get learning rate scheduler
    with pytest.raises(RuntimeError):
        _ = get_learning_rate_scheduler(pytorch_optimizer_adam, scheduler_type,
                                        **kwargs)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('algorithm', ['adam',])
def test_get_pytorch_optimizer(gnn_material_simulator, algorithm):
    """Test PyTorch optimizer getter."""
    # Set parameters to optimize
    params = gnn_material_simulator.parameters(recurse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get optimizer
    optimizer = get_pytorch_optimizer(algorithm, params)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(optimizer, torch.optim.Optimizer)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('algorithm', ['unknown_algorithm',])
def test_get_pytorch_optimizer_invalid(gnn_material_simulator, algorithm):
    """Test invalid PyTorch optimizer getter."""
    # Set parameters to optimize
    params = gnn_material_simulator.parameters(recurse=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get optimizer
    with pytest.raises(RuntimeError):
        _ = get_pytorch_optimizer(algorithm, params)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('loss_type', ['mse',])
def test_get_pytorch_loss(loss_type):
    """Test PyTorch optimizer getter."""
    # Get loss function
    loss_function = get_pytorch_loss(loss_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert isinstance(loss_function, torch.nn.modules.loss._Loss)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('loss_type', ['unknown_type',])
def test_get_pytorch_loss_invalid(loss_type):
    """Test invalid PyTorch optimizer getter."""
    # Get loss function
    with pytest.raises(RuntimeError):
        _ = get_pytorch_loss(loss_type)