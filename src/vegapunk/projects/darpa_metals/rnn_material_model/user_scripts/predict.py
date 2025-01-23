"""DARPA METALS PROJECT: Local prediction of RNN material model.

Functions
---------
perform_model_prediction
    Perform prediction with RNN-based model.
generate_prediction_plots
    Generate plots of model predictions.
set_default_prediction_options
    Set default model prediction options.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[4])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pickle
import re
# Third-party
import torch
import numpy as np
# Local
from rnn_base_model.data.time_dataset import load_dataset, \
    concatenate_dataset_features, sum_dataset_features, \
    add_dataset_feature_init
from rnn_base_model.predict.prediction import predict
from rnn_base_model.predict.prediction_plots import plot_time_series_prediction
from projects.darpa_metals.rnn_material_model.rnn_model_tools. \
    process_predictions import build_prediction_data_arrays, \
        build_time_series_predictions_data
from gnn_base_model.predict.prediction_plots import plot_truth_vs_prediction
from ioput.iostandard import make_directory, find_unique_file_with_regex
from projects.darpa_metals.rnn_material_model.rnn_model_tools.strain_features \
    import add_strain_features
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_model_prediction(predict_directory, dataset_file_path,
                             model_directory,
                             is_remove_sample_prediction=False,
                             device_type='cpu', is_verbose=False):
    """Perform prediction with RNN-based model.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.
    dataset_file_path : str
        Testing data set file path.
    model_directory : str
        Directory where model is stored.
    is_remove_sample_prediction : bool, default=False
        If True, then remove sample prediction files after plots are generated.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.

    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    avg_predict_loss : float
        Average prediction loss per sample. Defaults to None if ground-truth is
        not available for all data set samples.
    """
    # Set default model prediction options
    loss_nature, loss_type, loss_kwargs = set_default_prediction_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model initialization file path from model directory
    model_init_file_path = os.path.join(model_directory,
                                        'model_init_file' + '.pkl')
    # Load model initialization attributes from file
    if not os.path.isfile(model_init_file_path):
        raise RuntimeError('The model initialization file has not been '
                           'found:\n\n' + model_init_file_path)
    else:
        with open(model_init_file_path, 'rb') as model_init_file:
            model_init_attributes = pickle.load(model_init_file)
    # Get model initialization attributes
    model_init_args = model_init_attributes['model_init_args']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize new strain-based feature
    strain_features_labels = None
    # Initialize features concatenation/summing flags
    features_in_build = 'cat'
    features_out_build = 'cat'
    # Set data features for training
    features_option = 'strain_to_stress'
    if features_option == 'strain_to_stress':
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path',)
        features_in_build = 'cat'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('stress_path',)
        features_out_build = 'cat'
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_i1_i2_to_stress':
        # Set new strain-based features labels
        strain_features_labels = ('i1_strain', 'i2_strain')
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path', *strain_features_labels)
        features_in_build = 'cat'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('stress_path',)
        features_out_build = 'cat'
        # Set number of input and output features
        model_init_args['n_features_in'] = 8
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_to_p_strain':
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path',)
        features_in_weights = {'strain_path': 1.0,}
        features_in_build = 'sum'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('strain_path', 'e_strain_mf')
        features_out_weights = {'strain_path': 1.0, 'e_strain_mf': -1.0}
        features_out_build = 'sum'
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 6
    elif features_option == 'strain_i1_i2_to_p_strain':
        # Set new strain-based features labels
        strain_features_labels = ('i1_strain', 'i2_strain')
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path', *strain_features_labels)
        features_in_build = 'cat'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('strain_path', 'e_strain_mf')
        features_out_weights = {'strain_path': 1.0, 'e_strain_mf': -1.0}
        features_out_build = 'sum'
        # Set number of input and output features
        model_init_args['n_features_in'] = 8
        model_init_args['n_features_out'] = 6
    elif features_option == 'stress_acc_p_strain':
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path',)
        features_in_build = 'cat'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('stress_path', 'acc_p_strain')
        features_out_build = 'cat'
        # Set number of input and output features
        model_init_args['n_features_in'] = 6
        model_init_args['n_features_out'] = 7
    elif features_option == 'strain_vf_to_stress':
        # Set input features
        new_label_in = 'features_in'
        features_in_list = ('strain_path', 'vf_path')
        features_in_build = 'cat'
        # Set output features
        new_label_out = 'features_out'
        features_out_list = ('stress_path',)
        features_out_build = 'cat'
        # Set number of input and output features
        model_init_args['n_features_in'] = 7
        model_init_args['n_features_out'] = 6
    else:
        raise RuntimeError('Unknown features option.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hidden state initialization
    hidden_features_in = torch.zeros((model_init_args['n_recurrent_layers'],
                                      model_init_args['hidden_layer_size']))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(dataset_file_path)
    # Compute new strain-based features
    if strain_features_labels is not None:
        # Loop over strain-based features
        for strain_feature_label in strain_features_labels:
            # Add strain-based feature to data set
            dataset = add_strain_features(dataset, strain_feature_label)
    # Set testing data set features labels
    if features_in_build == 'cat':
        dataset = concatenate_dataset_features(
            dataset, new_label_in, features_in_list,
            is_remove_features=False)
    elif features_in_build == 'sum':
        dataset = sum_dataset_features(
            dataset, new_label_in, features_in_list,
            features_weights=features_in_weights, is_remove_features=False)
    if features_out_build == 'cat':
        dataset = concatenate_dataset_features(
            dataset, new_label_out, features_out_list,
            is_remove_features=False)
    elif features_out_build == 'sum':
        dataset = sum_dataset_features(
            dataset, new_label_out, features_out_list,
            features_weights=features_out_weights, is_remove_features=False)
    # Add hidden state initialization to data set
    dataset = add_dataset_feature_init(
        dataset, 'hidden_features_in', hidden_features_in)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss type
    loss_type = 'mse'
    # Set loss parameters
    loss_kwargs = {}
    # Set prediction loss normalization
    is_normalized_loss = False
    # Set prediction batch size
    batch_size = min((512, len(dataset)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state loading
    load_model_state = 'best'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prediction with RNN-based model
    predict_subdir, avg_predict_loss = \
        predict(dataset, model_directory, predict_directory=predict_directory,
                load_model_state=load_model_state, loss_nature=loss_nature,
                loss_type=loss_type, loss_kwargs=loss_kwargs,
                is_normalized_loss=is_normalized_loss, batch_size=batch_size,
                dataset_file_path=dataset_file_path,
                device_type=device_type, seed=None, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of model predictions
    generate_prediction_plots(dataset_file_path, predict_subdir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove sample prediction files
    if is_remove_sample_prediction:
        # Set sample prediction file regex
        sample_regex = re.compile(r'^prediction_sample_\d+\.pkl$')
        # Walk through prediction set directory recursively
        for root, _, files in os.walk(predict_subdir):
            # Loop over prediction set directory files
            for file in files:
                # Remove sample prediction file
                if sample_regex.match(file):
                    # Set sample prediction file path
                    sample_file_path = os.path.join(root, file)
                    # Remove sample prediction file
                    os.remove(sample_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir, avg_predict_loss
# =============================================================================
def generate_prediction_plots(dataset_file_path, predict_subdir):
    """Generate plots of model predictions.
    
    Parameters
    ----------
    dataset_file_path : str
        Testing data set file path.
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(predict_subdir), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load testing data set
    testing_dataset = load_dataset(dataset_file_path)
    # Get testing data set size
    n_sample = len(testing_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get first sample from testing data set
    probe_response_path = testing_dataset[0]
    # Get strain and stress components
    strain_comps_order = probe_response_path['strain_comps_order']
    stress_comps_order = probe_response_path['stress_comps_order']
    # Build strain and stress components predictions labels
    stress_labels = tuple([f'stress_{x}' for x in stress_comps_order])
    #p_strain_labels = tuple([f'p_strain_{x}' for x in strain_comps_order])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set prediction types and corresponding labels
    prediction_types = {}
    prediction_types['stress_comps'] = stress_labels
    #prediction_types['acc_p_strain'] = ('acc_p_strain',)
    #prediction_types['p_strain_comps'] = p_strain_labels
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model predictions against ground-truth
    for prediction_type, prediction_labels in prediction_types.items():
        # Build samples predictions data arrays with predictions and
        # ground-truth
        prediction_data_arrays = build_prediction_data_arrays(
            predict_subdir, prediction_type, prediction_labels,
            samples_ids='all')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over samples predictions data arrays
        for i, data_array in enumerate(prediction_data_arrays):
            # Get prediction plot file name
            filename = prediction_labels[i]
            # Set prediction process
            if prediction_type == 'stress_comps':
                prediction_sets = \
                    {f'Stress {prediction_labels[i].split("_")[-1]}':
                     data_array,}
            elif prediction_type == 'acc_p_strain':
                prediction_sets = \
                    {f'Accumulated plastic strain': data_array,}
            elif prediction_type == 'p_strain_comps':
                prediction_sets = \
                    {f'Plastic strain {prediction_labels[i].split("_")[-1]}':
                     data_array,}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot model predictions against ground-truth
            plot_truth_vs_prediction(prediction_sets, error_bound=0.1,
                                     is_r2_coefficient=True,
                                     is_normalize_data=False,
                                     filename=filename,
                                     save_dir=plot_dir,
                                     is_save_fig=True, is_stdout_display=False,
                                     is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model time series prediction and ground-truth
    for prediction_type, prediction_labels in prediction_types.items():
        # Set samples for which time series data is plotted
        samples_ids = list(np.arange(np.min((5, n_sample)), dtype=int))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        # Build times series predictions data arrays
        prediction_data_dicts = build_time_series_predictions_data(
            dataset_file_path, predict_subdir, prediction_type,
            prediction_labels, samples_ids=samples_ids)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
        # Loop over times series predictions components
        for i, data_dict in enumerate(prediction_data_dicts):
            # Loop over samples (time series paths)
            for sample_id, prediction_array in data_dict.items():
                # Set prediction processes data
                prediction_sets = {}
                prediction_sets['Ground-truth'] = prediction_array[:, [0, 1]]
                prediction_sets['Prediction'] = prediction_array[:, [0, 2]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get prediction plot file name
                filename = prediction_labels[i] + f'_path_sample_{sample_id}'
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set prediction type label
                if prediction_type == 'stress_comps':
                    y_label = 'Stress (MPa)'
                elif prediction_type == 'acc_p_strain':
                    y_label = 'Accumulated plastic strain'
                elif prediction_type == 'p_strain_comps':
                    y_label = 'Plastic strain'
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot model times series predictions against ground-truth
                plot_time_series_prediction(
                    prediction_sets, is_normalize_data=False,
                    x_label='Time', y_label=y_label,
                    filename=filename,
                    save_dir=plot_dir,is_save_fig=True,
                    is_stdout_display=False, is_latex=True)
# =============================================================================
def set_default_prediction_options():
    """Set default model prediction options.
    
    Returns
    -------
    loss_nature : {'features_out',}, default='features_out'
        Loss nature:
        
        'features_out' : Based on output features

    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.   
    """
    loss_nature = 'features_out'
    loss_type = 'mse'
    loss_kwargs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss_nature, loss_type, loss_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set testing type
    testing_type = \
        ('training', 'validation', 'in_distribution', 'out_distribution')[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_project/8_global_random_specimen/von_mises/'
                '1_local_vanilla_GRU/strain_to_stress')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize case study directories
    case_study_dirs = []
    # Set case study directories
    if True:
        # Set training data set sizes
        training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        # Set case study directories
        case_study_dirs += [os.path.join(os.path.normpath(base_dir), f'n{n}')
                            for n in training_sizes]
    else:
        case_study_dirs += [base_dir,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over case study directories
    for case_study_dir in case_study_dirs:
        # Check case study directory
        if not os.path.isdir(case_study_dir):
            raise RuntimeError('The case study directory has not been found:'
                               '\n\n' + case_study_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set testing data set directory
        if testing_type == 'training':
            # Set testing data set directory (training data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '1_training_dataset')
        elif testing_type == 'validation':
            # Set testing data set directory (validation data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '2_validation_dataset')
        elif testing_type == 'in_distribution':
            # Set testing data set directory (in-distribution testing data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '5_testing_id_dataset')
        elif testing_type == 'out_distribution':
            # Set testing data set directory (out-of-distribution testing
            # data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '6_testing_od_dataset')
        else:
            raise RuntimeError('Unknown testing type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get testing data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, dataset_file_path = \
            find_unique_file_with_regex(dataset_directory, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Testing data set file has not been found '
                               f'in data set directory:\n\n'
                               f'{dataset_directory}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = \
            os.path.join(os.path.normpath(case_study_dir), '3_model')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model predictions directory
        prediction_directory = os.path.join(os.path.normpath(case_study_dir),
                                            '7_prediction')
        # Create model predictions directory
        if not os.path.isdir(prediction_directory):
            make_directory(prediction_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create model predictions subdirectory
        prediction_subdir = os.path.join(
            os.path.normpath(prediction_directory), testing_type)
        # Create prediction subdirectory
        if not os.path.isdir(prediction_subdir):
            make_directory(prediction_subdir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform prediction with model
        perform_model_prediction(prediction_subdir, dataset_file_path,
                                 model_directory, device_type=device_type,
                                 is_verbose=True)