"""DARPA METALS PROJECT: Local training of hybrid material model.

Functions
---------
perform_model_standard_training
    Perform standard training of hybrid material model.
generate_standard_training_plots
    Generate plots of standard training of model.
set_default_model_parameters
    Set default model initialization parameters.
set_default_training_options
    Set default model training options.
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
import copy
# Third-party
import torch
import numpy as np
# Local
from time_series_data.time_dataset import load_dataset, \
    concatenate_dataset_features, sum_dataset_features
from rnn_base_model.model.gru_model import GRURNNModel
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
from hybrid_base_model.model.hybridized_layers import BatchedElasticModel, \
    VolDevCompositionModel
from hybrid_base_model.model.hybridized_model import set_hybridized_model
from hybrid_base_model.model.hybrid_model import HybridModel
from hybrid_base_model.train.training import train_model
from gnn_base_model.train.training import \
    read_loss_history_from_file, read_lr_history_from_file
from gnn_base_model.model.model_summary import get_model_summary
from gnn_base_model.train.training_plots import plot_training_loss_history, \
    plot_training_loss_and_lr_history
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from utilities.fit_data_scalers import fit_data_scaler_from_dataset
from projects.darpa_metals.rnn_material_model.rnn_model_tools.strain_features \
    import add_strain_features
from projects.darpa_metals.rnn_material_model.rnn_model_tools.stress_features \
    import add_stress_features
from ioput.iostandard import make_directory, find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def perform_model_standard_training(train_dataset_file_path, model_directory,
                                    val_dataset_file_path=None,
                                    device_type='cpu', is_verbose=False):
    """Perform standard training of hybrid material model.
    
    Parameters
    ----------
    train_dataset_file_path : str
        Training data set file path.
    model_directory : str
        Directory where model is stored.
    val_dataset_file_path : str, default=None
        Validation data set file path.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Set store model initialization flag
    is_store_model_init_only = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default model training options
    opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
            is_early_stopping, early_stopping_kwargs = \
                set_default_training_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model initialization parameters
    model_init_args = {'model_directory': model_directory,
                       'model_name': 'hybrid_material_model',
                       'is_model_in_normalized': False,
                       'is_model_out_normalized': True,
                       'device_type': device_type}
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
    else:
        raise RuntimeError('Unknown features option.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybridized models dictionary
    hyb_models_dict = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridization scheme
    hybridization_scheme = 4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized models dictionary and hybridization model type
    if hybridization_scheme == 0:
        # Description:
        #
        # HybridModel(e) = GRU(e)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'gru_strain_to_stress'
        # Set hybridized model indices
        hyb_indices = (0, 0)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: GRU (strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_gru_model(
            hyb_indices, 'strain_to_stress', scaling_dataset,
            device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        model_init_args['hyb_models_dict'] = hyb_models_dict
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'identity'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif hybridization_scheme == 1:
        # Description:
        #
        # HybridModel(e) = BatchedElastic(e - GRU(e))
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'gru_strain_to_pstrain'
        # Set hybridized model indices
        hyb_indices = (0, 0)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: GRU (strain to plastic strain)
        hyb_models_dict[hyb_model_name] = set_hybridized_gru_model(
            hyb_indices, 'strain_to_pstrain', scaling_dataset,
            device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'batched_elastic'
        # Set hybridized model indices
        hyb_indices = (0, 1)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: Batched elastic (plastic strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_material_layer(
            hyb_indices, 'batched_elastic', device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        model_init_args['hyb_models_dict'] = hyb_models_dict
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'identity'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif hybridization_scheme == 2:
        # Description:
        #
        # HybridModel(e) = GRU(e) > VolDevComposition(s_vol, s_dev)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'gru_strain_to_stressvd'
        # Set hybridized model indices
        hyb_indices = (0, 0)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: GRU (strain to
        #                            stress volumetric/deviatoric components)
        hyb_models_dict[hyb_model_name] = set_hybridized_gru_model(
            hyb_indices, 'strain_to_stressvd', scaling_dataset,
            device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'vd_composition'
        # Set hybridized model indices
        hyb_indices = (0, 1)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: Batched elastic (plastic strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_material_layer(
            hyb_indices, 'vd_composition', device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        model_init_args['hyb_models_dict'] = hyb_models_dict
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'identity'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif hybridization_scheme == 3:
        # Description:
        #
        # HybridModel(e) = GRU(e, e_i1, e_i2) > VolDevComposition(s_vol, s_dev)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'gru_strain_i1_i2_to_stressvd'
        # Set hybridized model indices
        hyb_indices = (0, 0)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: GRU (strain + i1/i2 strain invariants to
        #                            stress volumetric/deviatoric components)
        hyb_models_dict[hyb_model_name] = set_hybridized_gru_model(
            hyb_indices, 'strain_i1_i2_to_stressvd', scaling_dataset,
            device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'vd_composition'
        # Set hybridized model indices
        hyb_indices = (0, 1)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: Batched elastic (plastic strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_material_layer(
            hyb_indices, 'vd_composition', device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        model_init_args['hyb_models_dict'] = hyb_models_dict
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'identity'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif hybridization_scheme == 4:
        # Description:
        #
        # HybridModel(e) = RCM(e) + GRU(e)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'rcm_strain_to_stress'
        # Set hybridized model indices
        hyb_indices = (0, 0)
        # Set hybridized model: RCM (strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_rcm_model(
            hyb_indices, 'drucker_prager', device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model name
        hyb_model_name = 'gru_strain_to_stress'
        # Set hybridized model indices
        hyb_indices = (1, 0)
        # Load scaling data set
        scaling_dataset = load_dataset(train_dataset_file_path)
        # Set hybridized model: GRU (strain to stress)
        hyb_models_dict[hyb_model_name] = set_hybridized_gru_model(
            hyb_indices, 'strain_to_stress', scaling_dataset,
            device_type=device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized models dictionary
        model_init_args['hyb_models_dict'] = hyb_models_dict
        # Set hybridization model type
        model_init_args['hybridization_type'] = 'additive'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown hybridization scheme.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model training options:
    # Set number of epochs
    n_max_epochs = 200
    # Set batch size
    batch_size = 32
    # Set learning rate
    lr_init = 1.0e-03
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute exponential decay (learning rate scheduler)
    lr_end = 1.0e-5
    gamma = (lr_end/lr_init)**(1/n_max_epochs)
    # Set learning rate scheduler
    lr_scheduler_type = 'explr'
    lr_scheduler_kwargs = {'gamma': gamma}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set early stopping
    is_early_stopping = True
    # Set early stopping parameters
    if is_early_stopping:
        # Check validation data set file path
        if val_dataset_file_path is None:
            raise RuntimeError('The validation data set file path must be '
                               'provided to process early stopping criterion.')
        else:
            # Load validation data set
            val_dataset = load_dataset(val_dataset_file_path)
            # Compute new strain-based features
            if strain_features_labels is not None:
                # Loop over strain-based features
                for strain_feature_label in strain_features_labels:
                    # Add strain-based feature to data set
                    val_dataset = add_strain_features(
                        val_dataset, strain_feature_label)
            # Set validation data set features
            if features_in_build == 'cat':
                val_dataset = concatenate_dataset_features(
                    val_dataset, new_label_in, features_in_list,
                    is_remove_features=False)
            if features_out_build == 'cat':
                val_dataset = concatenate_dataset_features(
                    val_dataset, new_label_out, features_out_list,
                    is_remove_features=True)
        # Set early stopping parameters
        early_stopping_kwargs = {'validation_dataset': val_dataset,
                                 'validation_frequency': 1,
                                 'trigger_tolerance': 20,
                                 'improvement_tolerance': 1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load training data set
    train_dataset = load_dataset(train_dataset_file_path)
    # Compute new strain-based features
    if strain_features_labels is not None:
        # Loop over strain-based features
        for strain_feature_label in strain_features_labels:
            # Add strain-based feature to data set
            train_dataset = add_strain_features(
                train_dataset, strain_feature_label)
    # Set training data set features
    if features_in_build == 'cat':
        train_dataset = concatenate_dataset_features(
            train_dataset, new_label_in, features_in_list,
            is_remove_features=False)
    if features_out_build == 'cat':
        train_dataset = concatenate_dataset_features(
            train_dataset, new_label_out, features_out_list,
            is_remove_features=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model initialization file and initial state
    if is_store_model_init_only:
        # Save model initialization file and initial state
        save_model_init(model_init_args, training_dataset=train_dataset)
        # Skip model training
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loss type
    loss_type = 'mse'
    # Set loss parameters
    loss_kwargs = {}
    # Set model state loading
    model_load_state = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Training of hybrid material model
    model, _, _ = train_model(n_max_epochs, train_dataset, model_init_args,
                              lr_init, opt_algorithm=opt_algorithm,
                              lr_scheduler_type=lr_scheduler_type,
                              lr_scheduler_kwargs=lr_scheduler_kwargs,
                              loss_nature=loss_nature, loss_type=loss_type,
                              loss_kwargs=loss_kwargs,
                              batch_size=batch_size,
                              is_sampler_shuffle=is_sampler_shuffle,
                              is_early_stopping=is_early_stopping,
                              early_stopping_kwargs=early_stopping_kwargs,
                              model_load_state=model_load_state,
                              save_every=None,
                              dataset_file_path=train_dataset_file_path,
                              device_type=device_type, seed=None,
                              is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of model training process
    generate_standard_training_plots(model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display summary of PyTorch model
    _ = get_model_summary(model, device_type=device_type,
                          is_verbose=is_verbose)
# =============================================================================
def save_model_init(model_init_args, training_dataset=None):
    """Save model initialization file and initial state to file.
    
    Parameters
    ----------
    model_init_args : dict
        Model class initialization parameters (check HybridModel).
    training_dataset : torch.utils.data.Dataset, default=None
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Initialize hybrid model
    model = HybridModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit model data scalers
    if model.is_model_in_normalized or model.is_model_out_normalized:
        if training_dataset is None:
            raise RuntimeError('Training data set needs to be provided in '
                               'order to fit model data scalers.')
        else:
            model.fit_data_scalers(training_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save model initial state
    model.save_model_init_state()
# =============================================================================
def set_hybridized_gru_model(hyb_indices, features_option, scaling_dataset,
                             device_type='cpu'):
    """Set hybridized model: GRU.
    
    Parameters
    ----------
    hyb_indices : tuple[int]
        Hybridized model hybridization indices stored as (i, j), where i is the
        hybridization channel index and j the position index along the
        hybridization channel.
    features_option : {'strain_to_stress', 'strain_to_pstrain'}
        Hybridized model input and output features.
    scaling_dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where each
        feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features). Data set is used to fit required data
        scalers.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
        
    Returns
    -------
    hyb_model_data : dict
        Hybridized model data.
    """
    # Set hybridized model class
    model_class = GRURNNModel
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model initialization option
    is_model_init_file_path = False
    # Initialize model initialization data
    model_init_args = None
    model_init_file_path = None
    model_state_file_path = None
    # Initialize new strain/stress-based features
    strain_features_in_labels = None
    stress_features_out_labels = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybridized model
    if is_model_init_file_path:
        # Set model initialization file path
        model_init_file_path = ''
    else:
        # Set hidden layer size
        hidden_layer_size = 444
        # Set number of recurrent layers (stacked RNN)
        n_recurrent_layers = 3
        # Set dropout probability
        dropout = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize features concatenation/summing flags
        features_in_build = 'cat'
        features_out_build = 'cat'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set features dependent parameters
        if features_option == 'strain_to_stress':
            # Set input features
            new_label_in = 'features_in'
            features_in_list = ('strain_path',)
            features_in_build = 'cat'
            # Set number of input features
            n_features_in = 6
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set output features
            new_label_out = 'features_out'
            features_out_list = ('stress_path',)
            features_out_build = 'cat'
            # Set number of output features
            n_features_out = 6
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif features_option == 'strain_to_pstrain':
            # Set input features
            new_label_in = 'features_in'
            features_in_list = ('strain_path',)
            features_in_build = 'sum'
            features_in_weights = {'strain_path': 1.0,}
            # Set number of input features
            n_features_in = 6
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set output features
            new_label_out = 'features_out'
            features_out_list = ('strain_path', 'e_strain_mf')
            features_out_build = 'sum'
            features_out_weights = {'strain_path': 1.0, 'e_strain_mf': -1.0}
            # Set number of output features
            n_features_out = 6
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif features_option == 'strain_to_stressvd':
            # Set input features
            new_label_in = 'features_in'
            features_in_list = ('strain_path',)
            features_in_build = 'cat'
            # Set number of input features
            n_features_in = 6
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set new stress-based features labels
            stress_features_out_labels = ('vol_stress', 'dev_stress')
            # Set output features
            new_label_out = 'features_out'
            features_out_list = ('vol_stress', 'dev_stress')
            features_out_build = 'cat'
            # Set number of output features
            n_features_out = 7
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif features_option == 'strain_i1_i2_to_stressvd':
            # Set new strain-based features labels
            strain_features_in_labels = ('i1_strain', 'i2_strain')
            # Set input features
            new_label_in = 'features_in'
            features_in_list = ('strain_path', 'i1_strain', 'i2_strain')
            features_in_build = 'cat'
            # Set number of input features
            n_features_in = 8
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set new stress-based features labels
            stress_features_out_labels = ('vol_stress', 'dev_stress')
            # Set output features
            new_label_out = 'features_out'
            features_out_list = ('vol_stress', 'dev_stress')
            features_out_build = 'cat'
            # Set number of output features
            n_features_out = 7
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown features option.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model input and output features normalization
        is_model_in_normalized = True
        is_model_out_normalized = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GRU model source
        gru_model_source = 'custom'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build model initialization parameters
        model_init_args = {'n_features_in': n_features_in,
                          'n_features_out': n_features_out,
                          'hidden_layer_size': hidden_layer_size,
                          'n_recurrent_layers': n_recurrent_layers,
                          'dropout': dropout,
                          'model_directory': None,
                          'model_name': None,
                          'is_model_in_normalized': is_model_in_normalized,
                          'is_model_out_normalized': is_model_out_normalized,
                          'gru_model_source': gru_model_source,
                          'device_type': device_type}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model state file path
        model_state_file_path = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize features data scalers
    data_scalers = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit input features data scaler
    if is_model_in_normalized:
        # Compute new strain-based input features
        if strain_features_in_labels is not None:
            # Loop over strain-based features
            for strain_feature_label in strain_features_in_labels:
                # Add strain-based feature to data set
                scaling_dataset = add_strain_features(
                    scaling_dataset, strain_feature_label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set input features
        if features_in_build == 'cat':
            fin_dataset = concatenate_dataset_features(
                scaling_dataset, new_label_in,
                features_in_list, is_remove_features=False)
        elif features_in_build == 'sum':
            fin_dataset = sum_dataset_features(
                scaling_dataset, new_label_in,
                features_in_list, features_weights=features_in_weights,
                is_remove_features=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fitted data scaler
        scaler_features_in = fit_data_scaler_from_dataset(
            fin_dataset, 'features_in', n_features_in,
            scaling_type='mean-std')
        # Store fitted data scaler
        data_scalers['features_in'] = scaler_features_in
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit output features data scaler
    if is_model_out_normalized:
        # Compute new stress-based output features
        if stress_features_out_labels is not None:
            # Loop over stress-based features
            for stress_feature_label in stress_features_out_labels:
                # Add stress-based feature to data set
                scaling_dataset = add_stress_features(
                    scaling_dataset, stress_feature_label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if features_out_build == 'cat':
            fout_dataset = concatenate_dataset_features(
                scaling_dataset, new_label_out,
                features_out_list, is_remove_features=False)
        elif features_out_build == 'sum':
            fout_dataset = sum_dataset_features(
                scaling_dataset, new_label_out,
                features_out_list, features_weights=features_out_weights,
                is_remove_features=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get fitted data scaler
        scaler_features_out = fit_data_scaler_from_dataset(
            fout_dataset, 'features_out', n_features_out,
            scaling_type='mean-std')
        # Store fitted data scaler
        data_scalers['features_out'] = scaler_features_out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model data
    hyb_model_data = set_hybridized_model(
        model_class, hyb_indices, model_init_args=model_init_args,
        model_init_file_path=model_init_file_path,
        model_state_file_path=model_state_file_path, data_scalers=data_scalers)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hyb_model_data
# =============================================================================
def set_hybridized_rcm_model(hyb_indices, material_model_name,
                             device_type='cpu'):
    """Set hybridized model: RCM.
    
    Parameters
    ----------
    hyb_indices : tuple[int]
        Hybridized model hybridization indices stored as (i, j), where i is the
        hybridization channel index and j the position index along the
        hybridization channel.
    material_model_name : str
        Hybridized model material constitutive model name.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.

    Returns
    -------
    hyb_model_data : dict
        Hybridized model data.
    """
    # Set hybridized model class
    model_class = RecurrentConstitutiveModel
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model initialization option
    is_model_init_file_path = False
    # Initialize model initializations
    model_init_args = None
    model_init_file_path = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybridized model
    if is_model_init_file_path:
        # Set model initialization file path
        model_init_file_path = None
    else:
        # Set material constitutive model parameters
        if material_model_name == 'elastic':
            # Set base parameters
            material_model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 110e3, 'v': 0.33,
                 'euler_angles': (0.0, 0.0, 0.0)}
            # Set learnable parameters
            learnable_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif material_model_name in ('von_mises', 'von_mises_vmap'):
            # Set constitutive model parameters
            material_model_parameters = \
                {'elastic_symmetry': 'isotropic',
                 'E': 110e3, 'v': 0.33,
                 'euler_angles': (0.0, 0.0, 0.0),
                 'hardening_law': get_hardening_law('nadai_ludwik'),
                 'hardening_parameters': {'s0': np.sqrt(3)*900,
                                          'a': np.sqrt(3)*700,
                                          'b': 0.5,
                                          'ep0': 1e-5}}
            # Set learnable parameters
            learnable_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif material_model_name in ('drucker_prager', 'drucker_prager_vmap'):
            # Set frictional angle
            friction_angle = 0.08671116054780224
            # Set dilatancy angle
            dilatancy_angle = friction_angle
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute angle-related material parameters
            # (matching with Mohr-Coulomb under uniaxial tension and
            # compression)
            # Set yield surface cohesion parameter
            yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
            # Set yield pressure parameter
            yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
            # Set plastic flow pressure parameter
            flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material constitutive model parameters
            material_model_parameters = \
                {'elastic_symmetry': 'isotropic',
                'E': 110e3, 'v': 0.33,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),
                'hardening_parameters': {'s0': 900/yield_cohesion_parameter,
                                         'a': 700/yield_cohesion_parameter,
                                         'b': 0.5,
                                         'ep0': 1e-5},
                'yield_cohesion_parameter': yield_cohesion_parameter,
                'yield_pressure_parameter': yield_pressure_parameter,
                'flow_pressure_parameter': flow_pressure_parameter,
                'friction_angle': friction_angle}
            # Set learnable parameters
            learnable_parameters = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown material constitutive model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain formulation
        strain_formulation = 'infinitesimal'
        # Set problem type
        problem_type = 4
        # Set number of input features
        n_features_in = 6
        # Set number of output features
        n_features_out = 6
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material constitutive state variables (prediction)
        state_features_out = {}
        # Set parameters normalization
        is_normalized_parameters = True
        # Set model input and output features normalization
        is_model_in_normalized = False
        is_model_out_normalized = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build model initialization parameters
        model_init_args = {
            'n_features_in': n_features_in,
            'n_features_out': n_features_out,
            'learnable_parameters': learnable_parameters,
            'strain_formulation': strain_formulation,
            'problem_type': problem_type,
            'material_model_name': material_model_name,
            'material_model_parameters': material_model_parameters,
            'state_features_out': state_features_out,
            'model_directory': None,
            'model_name': None,
            'is_normalized_parameters': is_normalized_parameters,
            'is_model_in_normalized': is_model_in_normalized,
            'is_model_out_normalized': is_model_out_normalized,
            'device_type': device_type}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model data
    hyb_model_data = set_hybridized_model(
        model_class, hyb_indices, model_init_args=model_init_args,
        model_init_file_path=model_init_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hyb_model_data
# =============================================================================
def set_hybridized_material_layer(hyb_indices, layer_type, device_type='cpu'):
    """Set hybridized model: Material layer.
    
    Parameters
    ----------
    hyb_indices : tuple[int]
        Hybridized model hybridization indices stored as (i, j), where i is the
        hybridization channel index and j the position index along the
        hybridization channel.
    layer_type : {'batched_elastic', 'vd_composition'}
        Hybridized model material layer type.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
        
    Returns
    -------
    hyb_model_data : dict
        Hybridized model data.
    """
    # Set hybridized model class and initialization parameters
    if layer_type == 'batched_elastic':
        # Set hybridized model class
        model_class = BatchedElasticModel
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem type
        problem_type = 4
        # Set elastic material properties
        elastic_properties = {'E': 110e3, 'v': 0.33}
        # Set elastic symmetry
        elastic_symmetry = 'isotropic'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model initialization parameters
        model_init_args = {'problem_type': problem_type,
                           'elastic_properties': elastic_properties,
                           'elastic_symmetry': elastic_symmetry,
                           'device_type': device_type}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set input residual connection
        is_input_residual=True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif layer_type == 'vd_composition':
        # Set hybridized model class
        model_class = VolDevCompositionModel
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridized model initialization parameters
        model_init_args = {'device_type': device_type}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set input residual connection
        is_input_residual=False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown material layer type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model data
    hyb_model_data = set_hybridized_model(
        model_class, hyb_indices, model_init_args=model_init_args,
        is_input_residual=is_input_residual)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hyb_model_data
# =============================================================================
def generate_standard_training_plots(model_directory):
    """Generate plots of standard training of model.
    
    Parameters
    ----------
    model_directory : str
        Directory where material patch model is stored.
    """
    # Set loss history record file path
    loss_record_path = os.path.join(model_directory, 'loss_history_record.pkl')
    # Read training process training and validation loss history
    loss_nature, loss_type, training_loss_history, validation_loss_history = \
        read_loss_history_from_file(loss_record_path)
    # Build training process loss history
    loss_histories = {}
    loss_histories['Training'] = training_loss_history
    if validation_loss_history is not None:
        loss_histories['Validation'] = validation_loss_history
    # Read training process learning rate history
    lr_scheduler_type, lr_history_epochs = \
        read_lr_history_from_file(loss_record_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create plot directory
    plot_dir = os.path.join(os.path.normpath(model_directory), 'plots')
    if not os.path.isdir(plot_dir):
        make_directory(plot_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model training process loss history
    plot_training_loss_history(loss_histories, loss_type.upper(),
                               loss_scale='log', save_dir=plot_dir,
                               is_save_fig=True, is_stdout_display=False,
                               is_latex=True)
    # Plot model training process loss and learning rate histories
    plot_training_loss_and_lr_history(training_loss_history,
                                      lr_history_epochs, loss_type=None,
                                      is_log_loss=False, loss_scale='log',
                                      lr_type=lr_scheduler_type,
                                      save_dir=plot_dir, is_save_fig=True,
                                      is_stdout_display=False, is_latex=True)
# =============================================================================
def set_default_training_options():
    """Set default model training options.
    
    Returns
    -------
    opt_algorithm : {'adam',}
        Optimization algorithm:
        
        'adam'  : Adam (torch.optim.Adam)

    lr_init : float
        Initial value optimizer learning rate. Constant learning rate value if
        no learning rate scheduler is specified (lr_scheduler_type=None).
    lr_scheduler_type : {'steplr', 'explr', 'linlr'}
        Type of learning rate scheduler:
        
        'steplr'  : Step-based decay (torch.optim.lr_scheduler.SetpLR)
        
        'explr'   : Exponential decay (torch.optim.lr_scheduler.ExponentialLR)
        
        'linlr'   : Linear decay (torch.optim.lr_scheduler.LinearLR)

    lr_scheduler_kwargs : dict
        Arguments of torch.optim.lr_scheduler.LRScheduler initializer.
    loss_nature : {'features_out',}, default='features_out'
        Loss nature:
        
        'features_out' : Based on output features

    loss_type : {'mse',}
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_sampler_shuffle : bool
        If True, shuffles data set samples at every epoch.
    is_early_stopping : bool
        If True, then training process is halted when early stopping criterion
        is triggered.
    early_stopping_kwargs : dict
        Early stopping criterion parameters (key, str, item, value).
    """
    opt_algorithm = 'adam'
    lr_init = 1.0e-04
    lr_scheduler_type = None
    lr_scheduler_kwargs = None
    loss_nature = 'features_out'
    loss_type = 'mse'
    loss_kwargs = {}
    is_sampler_shuffle = False
    is_early_stopping = True
    early_stopping_kwargs = {'validation_dataset': None,
                             'validation_frequency': 1,
                             'trigger_tolerance': 20,
                             'improvement_tolerance':1e-2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return opt_algorithm, lr_init, lr_scheduler_type, lr_scheduler_kwargs, \
        loss_nature, loss_type, loss_kwargs, is_sampler_shuffle, \
        is_early_stopping, early_stopping_kwargs
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_standard_training = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'darpa_paper_examples/local/hybrid_models/dp_plus_gru/n10')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize case study directories
    case_study_dirs = []
    # Set case study directories
    if False:
        # Set training data set sizes
        training_sizes = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
        # Set case study directories
        case_study_dirs += [os.path.join(os.path.normpath(base_dir), f'n{n}/')
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
        # Set training data set directory
        training_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                            '1_training_dataset')
        # Get training data set file path
        regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
        is_file_found, train_dataset_file_path = \
            find_unique_file_with_regex(training_dataset_dir, regex)
        # Check data set file
        if not is_file_found:
            raise RuntimeError(f'Training data set file has not been found '
                               f'in data set directory:\n\n'
                               f'{training_dataset_dir}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model directory
        model_directory = \
            os.path.join(os.path.normpath(case_study_dir), '3_model')
        # Create model directory
        if is_standard_training:
            # Create model directory (overwrite)
            make_directory(model_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set validation data set directory
        val_dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '2_validation_dataset')
        # Get validation data set file path
        val_dataset_file_path = None
        if os.path.isdir(val_dataset_directory):
            regex = (r'^ss_paths_dataset_n[0-9]+.pkl$',)
            is_file_found, val_dataset_file_path = \
                find_unique_file_with_regex(val_dataset_directory, regex)
            # Check data set file
            if not is_file_found:
                raise RuntimeError(f'Validation data set file has not been '
                                   f'found in data set directory:\n\n'
                                   f'{val_dataset_directory}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device type
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform standard training of model
        if is_standard_training:
            perform_model_standard_training(
                train_dataset_file_path, model_directory,
                val_dataset_file_path=val_dataset_file_path,
                device_type=device_type, is_verbose=True)