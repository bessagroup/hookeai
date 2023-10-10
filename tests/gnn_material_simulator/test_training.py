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
from src.vegapunk.gnn_model.training import \
    save_loss_history, load_loss_history, seed_worker
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
                         'training_step, target_load_history',
                         [(4, 'mse', [0.0, 2.0, 4.0, 6.0],
                           None, [0.0, 2.0, 4.0, 6.0]),
                          (4, 'mse', [0.0, 2.0, 4.0, 6.0],
                           3, [0.0, 2.0, 4.0, 6.0]),
                          (10, 'mse', [0.0, 2.0, 4.0, 6.0],
                           2, [0.0, 2.0, 4.0]),
                          ])
def test_save_and_load_loss_history(gnn_material_simulator,
                                    total_n_train_steps, loss_type,
                                    loss_history, training_step,
                                    target_load_history):
    """Test saving and loading of training process loss history."""
    # Save training process loss history
    save_loss_history(gnn_material_simulator, total_n_train_steps, loss_type,
                      loss_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training process loss history
    loaded_loss_history = load_loss_history(gnn_material_simulator, loss_type,
                                            training_step=training_step)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert np.allclose(loaded_loss_history, target_load_history)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove loss history record file
    loss_record_path = os.path.join(gnn_material_simulator.model_directory,
                                    'loss_history_record' + '.pkl')
    os.remove(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Attempt to load unexistent training process loss history
    if training_step is not None:
        loaded_loss_history = \
            load_loss_history(gnn_material_simulator, loss_type,
                              training_step=training_step)
        assert loaded_loss_history == (training_step + 1)*[None,]
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