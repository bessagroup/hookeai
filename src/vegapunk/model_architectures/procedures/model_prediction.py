"""Procedures associated to model prediction.

Functions
---------
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_predictions
    Save model prediction results for given sample.
write_prediction_summary_file
    Write summary data file for model prediction process.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import datetime
# Local
from ioput.iostandard import make_directory, write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def make_predictions_subdir(predict_directory):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.

    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_directory, 'prediction_set_0')
    while os.path.exists(predict_subdir):
        predict_subdir = os.path.join(
            predict_directory,
            'prediction_set_' + str(int(predict_subdir.split('_')[-1]) + 1))
    # Create model predictions subdirectory
    predict_subdir = make_directory(predict_subdir, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
# =============================================================================
def save_sample_predictions(predictions_dir, sample_id, sample_results):
    """Save model prediction results for given sample.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where sample prediction results are stored.
    sample_id : int
        Sample ID. Sample ID is appended to sample prediction results file
        name.
    sample_results : dict
        Sample prediction results.
    """
    # Check prediction results directory
    if not os.path.exists(predictions_dir):
        raise RuntimeError('The prediction results directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set sample prediction results file path
    sample_path = os.path.join(predictions_dir,
                               'prediction_sample_'+ str(sample_id) + '.pkl')
    # Save sample prediction results
    with open(sample_path, 'wb') as sample_file:
        pickle.dump(sample_results, sample_file)
# =============================================================================
def write_prediction_summary_file(
    predict_subdir, device_type, seed, model_directory, load_model_state,
    loss_type, loss_kwargs, is_normalized_loss, dataset_file_path, dataset,
    avg_predict_loss, total_time_sec, avg_time_sample):
    """Write summary data file for model prediction process.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where model is stored.
    load_model_state : {'best', 'last', int, None}
        Load availabl model state from the model directory. Data scalers are
        also loaded from model initialization file.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    dataset_file_path : str
        Data set file path if such file exists. Only used for output purposes.
    dataset : torch.utils.data.Dataset
        Data set.
    avg_predict_loss : float
        Average prediction loss per sample.
    total_time_sec : int
        Total prediction time in seconds.
    avg_time_sample : float
        Average prediction time per sample.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['load_model_state'] = load_model_state
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['is_normalized_loss'] = is_normalized_loss
    summary_data['Prediction data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Prediction data set size'] = len(dataset)
    summary_data['Avg. prediction loss per sample'] = \
        f'{avg_predict_loss:.8e}' if avg_predict_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per sample'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=predict_subdir,
        summary_title='Summary: Model prediction',
        **summary_data)