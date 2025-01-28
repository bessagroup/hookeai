"""Prediction metrics computed from sample prediction file.

Functions
---------
compute_prediction_metrics
    Compute prediction metrics from sample prediction file path.
get_sample_n_features
    Get number of prediction features.
get_feature_and_target
    Get sample prediction and ground-truth for given feature.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
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
def compute_prediction_metrics(sample_prediction_path, prediction_metrics=[]):
    """Compute prediction metrics from sample prediction file path.
    
    Parameters
    ----------
    sample_prediction_path : str
        Sample prediction file path.
    prediction_metrics : list[str]
        Prediction metrics.

    Returns
    -------
    prediction_metrics_results : dict
        Prediction metrics results (item, torch.Tensor) for each prediction
        metric (key, str).
    """
    # Check sample prediction file path
    if not os.path.exists(sample_prediction_path):
        raise RuntimeError(f'The sample prediction file has not been '
                           f'found:\n\n{sample_prediction_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sample prediction results
    with open(sample_prediction_path, 'rb') as sample_prediction_file:
        sample_results = pickle.load(sample_prediction_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of prediction features
    n_features = get_sample_n_features(sample_results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction metrics results
    prediction_metrics_results = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prediction metrics
    for metric_label in prediction_metrics:
        # Initialize metric results
        metric_results = []
        # Mean Squared Error (MSE) or Root Mean Squared Error (RMSE)
        if metric_label in ('mse', 'rmse'):
            # Loop over prediction features
            for feature_idx in range(n_features):
                # Get feature prediction and ground-truth
                feature_out, target = get_feature_data(
                    sample_results, feature_idx, is_ground_truth=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute Mean Squared Error (MSE)
                metric_value = torch.mean((feature_out - target)**2)
                # Compute Root Mean Squared Error (RMSE)
                if metric_label == 'rmse':
                    metric_value = torch.sqrt(metric_value)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store feature prediction metric
                metric_results.append(float(metric_value))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mean Absolute Value (MAV)
        elif metric_label in ('mav', 'mav_gt'):
            # Loop over prediction features
            for feature_idx in range(n_features):
                # Get feature prediction and ground-truth
                feature_out, target = get_feature_data(
                    sample_results, feature_idx, is_ground_truth=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute Mean Absolute Value (MAV)
                if metric_label == 'mav_gt':
                    metric_value = torch.mean(torch.abs(target))
                else:
                    metric_value = torch.mean(torch.abs(feature_out))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store feature prediction metric
                metric_results.append(float(metric_value))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError(f'Unknown prediction metric: '
                               f'\'{metric_label}\'')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store prediction metric results
        prediction_metrics_results[metric_label] = torch.tensor(metric_results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_metrics_results
# =============================================================================
def get_sample_n_features(sample_results):
    """Get number of sample prediction features.
    
    Parameters
    ----------
    sample_results : dict
        Sample prediction results. Features predictions (and ground-truth if
        available) are stored as torch.Tensor(2d) of shape
        (sequence_length, n_features).
        
    Returns
    -------
    n_features : int
        Number of prediction features.
    """
    # Check sample prediction results
    if not isinstance(sample_results, dict):
        raise RuntimeError('Sample prediction results must be stored in a '
                           'dictionary.')
    elif 'features_out' not in sample_results.keys():
        raise RuntimeError('Sample prediction features must be stored as a '
                           'torch.Tensor(2d) under the key \'features_out\'.')
    elif not isinstance(sample_results['features_out'], torch.Tensor):
        raise RuntimeError('Sample prediction features must be stored as a '
                           'torch.Tensor(2d) of shape '
                           '(sequence_length, n_features) under the key '
                           '\'features_out\'.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get sample prediction features
    features_out = sample_results['features_out']
    # Get number of prediction features
    n_features = features_out.shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_features
# =============================================================================
def get_feature_data(sample_results, feature_idx, is_ground_truth=False):
    """Get sample prediction and ground-truth for given feature.
    
    Parameters
    ----------
    sample_results : dict
        Sample prediction results. Features predictions (and ground-truth if
        available) are each stored as torch.Tensor(2d) of shape
        (sequence_length, n_features).
    feature_idx : int
        Feature index.
    is_ground_truth : bool, default=False
        If True, then output sample feature ground-truth.
        
    Returns
    -------
    feature_out : torch.Tensor
        Tensor of feature prediction stored as torch.Tensor(1d) of shape
        (sequence_length,).
    target : {torch.Tensor, None}
        Tensor of feature ground-truth stored as torch.Tensor(1d) of shape
        (sequence_length,). Set to None if is_ground_truth=False.
    """
    # Get sample feature prediction 
    features_out = sample_results['features_out'][:, feature_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize feature ground-truth
    target = None
    # Get sample feature ground-truth
    if is_ground_truth:
        # Check if ground-truth is available
        if ('targets' not in sample_results.keys()
                or sample_results['targets'] is None):
            raise RuntimeError('Ground-truth is not available to compute '
                               'prediction metrics. Ensure that \'targets\' '
                               'are stored in the sample prediction file '
                               'results.')
        # Get sample feature ground-truth
        target = sample_results['targets'][:, feature_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return features_out, target