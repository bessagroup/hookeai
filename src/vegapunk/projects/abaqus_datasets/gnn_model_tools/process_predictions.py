"""Post-process of GNN-based model predictions.

Functions
---------
build_prediction_data_arrays
    Build samples predictions data arrays with predictions and ground-truth.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
# Third-party
import numpy as np
# Local
from gnn_base_model.predict.prediction import load_sample_predictions
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def build_prediction_data_arrays(predictions_dir, prediction_type,
                                 samples_ids='all'):
    """Build samples predictions data arrays with predictions and ground-truth.
    
    Assumes given node output features (set prediction type accordingly).
    
    Parameters
    ----------
    predictions_dir : str
        Directory where samples predictions results files are stored.
    prediction_type : {'coord_comps',}
        Type of prediction data arrays:
        
        'coord_comps' : Node coordinates (one array per dimension)
        
        'disp_comps'  : Node displacements (one array per dimension)

    samples_ids : {'all', list[int]}, default='all'
        Samples IDs whose prediction results are collated in each prediction
        data array.
    
    Returns
    -------
    prediction_data_arrays : list[numpy.ndarray(2d)]
        Prediction data arrays. Each data array collates data from all
        specified samples and is stored as a numpy.ndarray(2d) of shape
        (n_nodes, 2), where data_array[i, 0] stores the i-th node ground-truth
        and data_array[i, 1] stores the i-th node prediction.
    """
    # Check sample predictions directory
    if not os.path.isdir(predictions_dir):
        raise RuntimeError('The samples predictions directory has not been '
                           'found:\n\n' + predictions_dir)
    # Check samples IDs
    if samples_ids != 'all' and not isinstance(samples_ids, list):
        raise RuntimeError('Samples IDs must be specified as "all" or as '
                           'list[int].')
    elif (isinstance(samples_ids, list)
          and not all([isinstance(x, int) for x in samples_ids])):
        raise RuntimeError('Samples IDs must be specified as a list of '
                           'integers.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Get files in samples predictions results directory
    directory_list = os.listdir(predictions_dir)
    # Check directory
    if not directory_list:
        raise RuntimeError('No files have been found in directory where '
                           'samples predictions results files are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get prediction files samples IDs
    prediction_files_ids = []
    for filename in directory_list:
        # Check if file is sample results file
        id = re.search(r'^prediction_sample_([0-9]+).pkl$', filename)
        # Assemble sample ID
        if id is not None:
            prediction_files_ids.append(int(id.groups()[0]))
    # Check prediction files
    if not prediction_files_ids:
        raise RuntimeError('No sample results files have been found in '
                           'directory where samples predictions results files '
                           'are stored.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set all available samples
    if samples_ids == 'all':
        samples_ids = prediction_files_ids
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Set number of prediction data arrays
    if prediction_type == 'coord_comps':   
        # Set number of prediction data arrays
        n_data_arrays = 3
    elif prediction_type == 'disp_comps':   
        # Set number of prediction data arrays
        n_data_arrays = 3
    else:
        raise RuntimeError('Unknown prediction data array type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize prediction data arrays
    prediction_data_arrays = n_data_arrays*[np.empty((0, 2)),]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for sample_id in samples_ids:
        # Check if sample ID prediction results file is available
        if sample_id not in prediction_files_ids:
            raise RuntimeError(f'The prediction results file for sample '
                               f'{sample_id} has not been found in directory: '
                               f'\n\n{predictions_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample predictions file path
        sample_prediction_path = \
            os.path.join(os.path.normpath(predictions_dir),
                         f'prediction_sample_{sample_id}.pkl')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load sample predictions
        sample_results = load_sample_predictions(sample_prediction_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction data arrays
        for i in range(n_data_arrays):
            # Build sample data array
            if prediction_type in ('coord_comps',):
                # Get node coordinates predictions
                coords = sample_results['node_features_out']
                # Get node internal forces ground-truth
                coords_target = sample_results['node_targets']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check availability of ground-truth
                if coords_target is None:
                    raise RuntimeError(f'Node coordinates ground-truth is not '
                                       f'available for sample {sample_id}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build sample data array
                if prediction_type == 'coord_comps':
                    data_array = np.concatenate(
                        (coords_target[:, i].reshape((-1, 1)),
                         coords[:, i].reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build sample data array
            elif prediction_type in ('disp_comps',):
                # Get node displacements predictions
                disps = sample_results['node_features_out']
                # Get node internal forces ground-truth
                disps_target = sample_results['node_targets']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check availability of ground-truth
                if disps_target is None:
                    raise RuntimeError(f'Node displacements ground-truth is '
                                       f'not available for sample '
                                       f'{sample_id}.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build sample data array
                if prediction_type == 'disp_comps':
                    data_array = np.concatenate(
                        (disps_target[:, i].reshape((-1, 1)),
                         disps[:, i].reshape((-1, 1))), axis=1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample prediction data
            prediction_data_arrays[i] = \
                np.append(prediction_data_arrays[i], data_array, axis=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return prediction_data_arrays