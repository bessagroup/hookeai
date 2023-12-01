"""Test metrics to assess performance of GNN-based material patch model."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.vegapunk.gnn_model.evaluation_metrics import \
    plot_training_loss_history, plot_training_loss_and_lr_history, \
    plot_loss_convergence_test, plot_truth_vs_prediction, \
    plot_kfold_cross_validation
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
@pytest.mark.parametrize('history_sizes, loss_type, is_log_loss, loss_scale',
                         [((10, 20), 'MSE', False, 'linear'),
                          ((5, 5), 'RMSE', True, 'linear'),
                          ((15, 5), None, True, 'log'),
                          ((15, 5), None, False, 'log'),
                          ])
def test_plot_training_loss_history(tmp_path, monkeypatch, history_sizes,
                                    loss_type, is_log_loss, loss_scale):
    """Test plot of model training process loss history."""
    # Set training processes loss histories    
    loss_history = {
        f'dataset_{i}': list(np.random.uniform(low=0.0, high=1.0e4,
                                               size=history_sizes[i]))
        for i in range(len(history_sizes))}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        plot_training_loss_history(loss_history, loss_type=loss_type,
                                   is_log_loss=is_log_loss,
                                   loss_scale=loss_scale,
                                   save_dir=tmp_path, is_save_fig=True,
                                   is_stdout_display=True)
        plot_training_loss_history(loss_history, loss_type=loss_type,
                                   is_log_loss=is_log_loss,
                                   loss_scale=loss_scale,
                                   save_dir=tmp_path, is_save_fig=True,
                                   is_stdout_display=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'model training process loss history.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_training_loss_history(loss_history='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_training_loss_history(loss_history={'dataset_1': 'invalid_type',})
# =============================================================================
@pytest.mark.parametrize('loss_hist_size, lr_hist_size, loss_type, '
                         'is_log_loss, loss_scale, lr_type',
                         [(10, 10, 'MSE', False, 'linear', 'Step'),
                          (5, 5, 'RMSE', True, 'log', 'Step'),
                          (5, 5, None, True, 'log', None),
                          (15, 15, None, False, 'log', 'Step'),
                          ])
def test_plot_training_loss_and_lr_history(tmp_path, monkeypatch,
                                           loss_hist_size, lr_hist_size,
                                           loss_type, is_log_loss, loss_scale,
                                           lr_type):
    """Test plot of model training process loss and learning rate histories."""
    # Set training processes loss and learning rate histories
    loss_history = list(np.random.uniform(low=0.0, high=1.0e4,
                                          size=loss_hist_size))
    lr_history = list(np.random.uniform(low=0.0, high=1.0, size=lr_hist_size))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        plot_training_loss_and_lr_history(
            loss_history, lr_history, loss_type=loss_type,
            is_log_loss=is_log_loss, loss_scale=loss_scale, lr_type=lr_type,
            save_dir=tmp_path, is_save_fig=True, is_stdout_display=True)
        plot_training_loss_and_lr_history(
            loss_history, lr_history, loss_type=loss_type,
            is_log_loss=is_log_loss, loss_scale=loss_scale, lr_type=lr_type,
            save_dir=tmp_path, is_save_fig=True, is_stdout_display=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'model training process loss and learning rate histories.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_training_loss_and_lr_history(loss_history='invalid_type',
                                          lr_history=[])
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_training_loss_and_lr_history(loss_history=[],
                                          lr_history='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data strcture
        plot_training_loss_and_lr_history(loss_history=[1,], lr_history=[1, 2])
# =============================================================================
@pytest.mark.parametrize('testing_size, training_size, loss_type, '
                         'is_log_loss, loss_scale',
                         [((15, 4), (10, 2), 'MSE', False, 'linear'),
                          ((5, 10), None, 'RMSE', True, 'log'),
                          ((10, 2), (10, 4), None, True, 'log'),
                          ((10, 2), (10, 4), None, False, 'linear'),
                          ])
def test_plot_loss_convergence_test(tmp_path, monkeypatch, testing_size,
                                    training_size, loss_type, is_log_loss,
                                    loss_scale):
    """Test plot of loss for different training data set sizes."""
    # Set testing and training processes
    testing_loss = np.random.uniform(low=0.0, high=1.0e4, size=testing_size)
    testing_loss[:, 0] = np.linspace(10, 1000, testing_size[0])
    if training_size is None:
        training_loss = None
    else:
        training_loss = np.random.uniform(low=0.0, high=1.0e4,
                                          size=training_size)
        training_loss[:, 0] = np.linspace(10, 1000, training_size[0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        plot_loss_convergence_test(
            testing_loss, training_loss=training_loss, loss_type=loss_type,
            is_log_loss=is_log_loss, loss_scale=loss_scale, save_dir=tmp_path,
            is_save_fig=True, is_stdout_display=True)
        plot_loss_convergence_test(
            testing_loss, training_loss=training_loss, loss_type=loss_type,
            is_log_loss=is_log_loss, loss_scale=loss_scale, save_dir=tmp_path,
            is_save_fig=True, is_stdout_display=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'testing and training loss for different training data set sizes.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_loss_convergence_test(testing_loss='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data structure
        plot_loss_convergence_test(testing_loss=np.zeros((1, 1, 1)))
    with pytest.raises(RuntimeError):
        # Test detection of invalid data structure
        plot_loss_convergence_test(testing_loss=np.zeros((1, 1)),
                                   training_loss='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data structure
        plot_loss_convergence_test(testing_loss=np.zeros((1, 1)),
                                   training_loss=np.zeros((1, 1, 1)))
# =============================================================================
@pytest.mark.parametrize('n_processes, error_bound, is_normalize_data',
                         [(1, None, False),
                          (2, 0.1, True),
                          ])
def test_plot_truth_vs_prediction(tmp_path, monkeypatch, n_processes,
                                  error_bound, is_normalize_data):
    """Test plot of ground-truth against predictions."""
    # Set prediction processes  
    prediction_sets = {
        f'dataset_{i}': np.random.uniform(low=0.0, high=1.0,
                                          size=(np.random.randint(10, 20), 2))
        for i in range(n_processes)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        plot_truth_vs_prediction(prediction_sets, error_bound=error_bound,
                                 is_normalize_data=is_normalize_data,
                                 save_dir=tmp_path, is_save_fig=True,
                                 is_stdout_display=True)
        plot_truth_vs_prediction(prediction_sets, error_bound=error_bound,
                                 is_normalize_data=is_normalize_data,
                                 save_dir=tmp_path, is_save_fig=True,
                                 is_stdout_display=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'ground-truth against predictions.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_truth_vs_prediction(prediction_sets='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_truth_vs_prediction(
            prediction_sets={'dataset_1': 'invalid_type',})
    with pytest.raises(RuntimeError):
        # Test detection of invalid data structure
        plot_truth_vs_prediction(
            prediction_sets={'dataset_1': np.zeros((1, 3)),})
# =============================================================================
@pytest.mark.parametrize('loss_type, loss_scale',
                         [('MSE', 'linear'),
                          ('RMSE', 'log'),
                          ])
def test_plot_kfold_cross_validation(tmp_path, monkeypatch, loss_type,
                                     loss_scale):
    """Test plot of k-fold cross-validation results."""
    # Set random k-fold cross-validation loss array
    k_fold_loss_array = np.random.uniform(low=0.0, high=1.0e4, size=(4, 2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plot
    monkeypatch.setattr(plt, 'show', lambda: None)
    is_error_raised = False
    try:
        plot_kfold_cross_validation(k_fold_loss_array, loss_type=loss_type,
                                    loss_scale=loss_scale,
                                    save_dir=tmp_path, is_save_fig=True,
                                    is_stdout_display=True)
        plot_kfold_cross_validation(k_fold_loss_array, loss_type=loss_type,
                                    loss_scale=loss_scale,
                                    save_dir=tmp_path, is_save_fig=True,
                                    is_stdout_display=True)
    except:
        is_error_raised = True
    assert not is_error_raised, 'Error while attempting to generate plot of ' \
        'of k-fold cross-validation results.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pytest.raises(RuntimeError):
        # Test detection of invalid data type
        plot_kfold_cross_validation(k_fold_loss_array='invalid_type')
    with pytest.raises(RuntimeError):
        # Test detection of invalid data structure
        plot_kfold_cross_validation(k_fold_loss_array=np.zeros((4, 3)))