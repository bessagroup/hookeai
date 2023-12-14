"""Test prediction plots of Graph Neural Network model."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.vegapunk.gnn_base_model.train.training_plots import \
    plot_training_loss_history, plot_training_loss_and_lr_history, \
    plot_loss_convergence_test, plot_kfold_cross_validation
from src.vegapunk.gnn_base_model.predict.prediction_plots import \
    plot_truth_vs_prediction
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