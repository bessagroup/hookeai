"""Hydra hyperparameter optimization.

Execute (multi-run mode):

    $ python3 hydra_optimization.py -m

Functions
---------
hydra_wrapper
    Wrapper of Hydra hyperparameter optimization main function.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
# Third-party
import hydra
import torch
# Local
from time_series_data.time_dataset import load_dataset, \
    concatenate_dataset_features, add_dataset_feature_init
from model_architectures.rnn_base_model.train.training import train_model
from model_architectures.rnn_base_model.predict.prediction import predict
from model_architectures.rnn_base_model.optimization.\
    hydra_optimization_template import display_hydra_job_header
from ioput.iostandard import make_directory, write_summary_file
from user_scripts.local_model_update.rnn_material_model.train_model import \
    generate_standard_training_plots
from user_scripts.local_model_update.rnn_material_model.predict import \
    generate_prediction_plots
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def hydra_wrapper(process, dataset_paths, device_type='cpu'):
    """Wrapper of Hydra hyperparameter optimization main function.
    
    Parameters
    ----------
    process : str
        Hyperparameter optimization process.
    dataset_paths : dict
        Hyperparameter optimization process required data sets (key, str)
        file paths (item, str).
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    """
    # Set Hydra main function
    @hydra.main(version_base=None, config_path='.', config_name='hydra_config')
    def hydra_optimize_gru_model(cfg):
        """Hydra hyperparameter optimization of GRU material model.
        
        Parameters
        ----------
        cgf : omegaconf.DictConfig
            Configuration dictionary of YAML based hierarchical configuration
            system.
            
        Returns
        -------
        objective : float
            Objective to minimize.
        """
        # Get Hydra configuration singleton
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display Hydra hyperparameter optimization job header
        sweeper, sweeper_optimizer, job_dir = \
            display_hydra_job_header(hydra_cfg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GRU material model initialization parameters
        model_init_args = {}
        model_init_args['n_features_in'] = cfg.n_features_in
        model_init_args['n_features_out'] = cfg.n_features_out
        model_init_args['hidden_layer_size'] = cfg.hidden_layer_size
        model_init_args['model_directory'] = job_dir
        model_init_args['model_name'] = 'gru_material_model'
        model_init_args['is_model_in_normalized'] = True
        model_init_args['is_model_out_normalized'] = True
        model_init_args['n_recurrent_layers'] = cfg.n_recurrent_layers
        model_init_args['dropout'] = cfg.dropout
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data features for training and prediction
        features_option = 'strain_to_stress'
        if features_option == 'strain_to_stress':
            # Set input features
            new_label_in = 'features_in'
            cat_features_in = ('strain_path',)
            # Set output features
            new_label_out = 'features_out'
            cat_features_out = ('stress_path',)
        elif features_option == 'strain_vf_to_stress':
            # Set input features
            new_label_in = 'features_in'
            cat_features_in = ('strain_path', 'vf_path')
            # Set output features
            new_label_out = 'features_out'
            cat_features_out = ('stress_path',)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hidden state initialization
        hidden_features_in = \
            torch.zeros((cfg.n_recurrent_layers, cfg.hidden_layer_size))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load hyperparameter optimization process training and validation data
        # sets
        if process in ('training', 'training-testing'):
            # Get training data set file path
            train_dataset_file_path = dataset_paths['training']
            # Load training data set
            training_dataset = load_dataset(train_dataset_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set training data set features
            training_dataset = concatenate_dataset_features(
                training_dataset, new_label_in, cat_features_in,
                 is_remove_features=True)
            training_dataset = concatenate_dataset_features(
                training_dataset, new_label_out, cat_features_out,
                is_remove_features=True)
            # Add hidden state initialization to data set
            training_dataset = add_dataset_feature_init(
                training_dataset, 'hidden_features_in', hidden_features_in)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set early stopping
            is_early_stopping = cfg.is_early_stopping
            # Set early stopping parameters
            if is_early_stopping:
                # Get validation data set file path
                val_dataset_file_path = dataset_paths['validation']
                # Load validation data set
                validation_dataset = load_dataset(val_dataset_file_path)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set validation data set features
                validation_dataset = concatenate_dataset_features(
                    validation_dataset, new_label_in, cat_features_in,
                    is_remove_features=True)
                validation_dataset = concatenate_dataset_features(
                    validation_dataset, new_label_out, cat_features_out,
                    is_remove_features=True)
                # Add hidden state initialization to data set
                validation_dataset = add_dataset_feature_init(
                    validation_dataset, 'hidden_features_in',
                    hidden_features_in)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get early stopping parameters
                early_stopping_kwargs = {**cfg.early_stopping_kwargs}
                # Add validation dataset to early stopping parameters
                early_stopping_kwargs['validation_dataset'] = \
                    validation_dataset
        else:
            raise RuntimeError('Unknown hyperparameter optimization process.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load hyperparameter optimization process testing data set
        if process in ('training-testing',):
            # Get testing data set file path
            test_dataset_file_path = dataset_paths['testing']
            # Load testing data set
            testing_dataset = load_dataset(test_dataset_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Change training data set features labels
            testing_dataset = concatenate_dataset_features(
                testing_dataset, new_label_in, cat_features_in,
                is_remove_features=True)
            testing_dataset = concatenate_dataset_features(
                testing_dataset, new_label_out, cat_features_out,
                is_remove_features=True)
            # Add hidden state initialization to data set
            testing_dataset = add_dataset_feature_init(
                testing_dataset, 'hidden_features_in', hidden_features_in) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Training of GRU material model
        if process in ('training', 'training-testing'):
            # Set model training subdirectory
            training_subdir = os.path.join(os.path.normpath(job_dir), 'model')
            # Create model training subdirectory
            training_subdir = make_directory(training_subdir,
                                             is_overwrite=True)
            # Set model directory
            model_init_args['model_directory'] = training_subdir
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training
            model, best_training_loss, _ = train_model(
                cfg.n_max_epochs, training_dataset, model_init_args,
                cfg.lr_init, opt_algorithm=cfg.opt_algorithm,
                lr_scheduler_type=cfg.lr_scheduler_type,
                lr_scheduler_kwargs=cfg.lr_scheduler_kwargs,
                loss_nature=cfg.loss_nature, loss_type=cfg.loss_type,
                loss_kwargs=cfg.loss_kwargs, batch_size=cfg.batch_size,
                is_sampler_shuffle=cfg.is_sampler_shuffle,
                is_early_stopping=cfg.is_early_stopping,
                early_stopping_kwargs=early_stopping_kwargs,
                device_type=device_type, is_verbose=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate plots of model training process
            generate_standard_training_plots(training_subdir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set hyperparameter optimization objective
            if process == 'training':
                objective = best_training_loss
            elif process == 'training-testing':
                # Set model testing subdirectory
                testing_subdir = \
                    os.path.join(os.path.normpath(job_dir), 'testing')
                # Create model testing subdirectory
                testing_subdir = \
                    make_directory(testing_subdir, is_overwrite=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set prediction loss normalization
                is_normalized_loss = False
                # Set prediction batch size
                batch_size = batch_size=len(testing_dataset)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Testing of GRU material model
                predict_subdir, avg_predict_loss_sample = predict(
                    testing_dataset, model.model_directory, model=model,
                    predict_directory=testing_subdir,
                    model_load_state='best', loss_nature=cfg.loss_nature,
                    loss_type=cfg.loss_type, loss_kwargs=cfg.loss_kwargs,
                    is_normalized_loss=is_normalized_loss,
                    batch_size=batch_size, device_type=device_type,
                    is_verbose=False)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate plots of model predictions
                generate_prediction_plots(test_dataset_file_path,
                                          predict_subdir)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set hyperparameter optimization objective
                objective = avg_predict_loss_sample
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set flag to remove sample prediction files
                is_remove_sample_prediction = True
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display parameters
        print('\nParameters:')
        for key, val in cfg.items():
            print(f'  > {key:{max([len(x) for x in cfg.keys()])}} : {val}')
        # Display objective
        print(f'\nFunction evaluation:')
        print(f'  > Objective : {objective:.8e}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set summary data
        summary_data = {}
        summary_data['sweeper'] = sweeper
        summary_data['sweeper_optimizer'] = sweeper_optimizer
        for key, val in cfg.items():
            summary_data[key] = val
        summary_data['objective'] = f'{objective:.8e}'
        # Write summary file
        write_summary_file(\
            summary_directory=job_dir,
            filename='job_summary',
            summary_title='Hydra - Hyperparameter Optimization Job',
            **summary_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return objective
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Call Hydra main function
    hydra_optimize_gru_model()
# =============================================================================
if __name__ == "__main__":
    # Set hyperparameter optimization processes
    processes = ('training', 'training-testing')
    # Select hyperparameter optimization process
    process = processes[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimization process data set paths
    datasets_paths = {}
    if process == 'training':
        datasets_paths['training'] = None
    elif process == 'training-testing':
        datasets_paths['training'] = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '2_local_rnn_training/composite_rve/dataset_01_2025/'
             '2_training_strain_vf_to_stress/1_training_dataset/'
             'ss_paths_dataset_n7333.pkl')
        datasets_paths['validation'] = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '2_local_rnn_training/composite_rve/dataset_01_2025/'
             '2_training_strain_vf_to_stress/2_validation_dataset/'
             'ss_paths_dataset_n916.pkl')
        datasets_paths['testing'] = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '2_local_rnn_training/composite_rve/dataset_01_2025/'
             '2_training_strain_vf_to_stress/5_testing_id_dataset/'
             'ss_paths_dataset_n916.pkl')
    else:
        raise RuntimeError('Unknown hyperparameter optimization process.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute Hydra hyperparameter optimization
    hydra_wrapper(process, datasets_paths, device_type)