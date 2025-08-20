"""DARPA METALS: Contigency plan B.

Functions
---------
generate_material_data
    Generate material response data sets with multiple dependencies.
assemble_material_datasets
    Assemble material response data sets.
preview_material_datasets_sizes
    Preview material response data sets sizes.
perform_material_model_updating
    Perform model updating with uncertainty quantification.
set_lmu_dir
    Set local model update main directory.
set_dependencies_dir
    Set directory for given temperature and composition.
set_dataset_type_dir
    Set directory for given dataset type.
set_strain_path_generator
    Set strain path generator and generation parameters.
set_data_material_model
    Set data material model and parameters.
get_model_parameters
    Get model parameters for given temperature and composition.
polynomial
    Evaluate polynomial function with given coefficients.
plot_model_parameters
    Plot model parameters for given material model.
get_dataset_basename
    Set data set basename.
find_dataset_file
    Find data set file in target directory.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import itertools
import time
import datetime
import re
import random
import warnings
import shutil
# Third-party
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
# Local
from time_series_data.time_dataset import TimeSeriesDatasetInMemory, \
    save_dataset, load_dataset
from data_generation.strain_paths.random_path import RandomStrainPathGenerator
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from ioput.iostandard import make_directory, find_unique_file_with_regex
from ioput.plots import plot_surface_xyz_data, save_figure
from user_scripts.synthetic_data.gen_response_dataset import \
    MaterialResponseDatasetGenerator, generate_dataset_plots
from user_scripts.local_model_update.rnn_material_model. \
    uncertainty_quantification import perform_model_uq, gen_model_uq_plots
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def generate_material_data(lmu_dir, strain_formulation, problem_type,
                           model_name, temperatures=[], compositions=[],
                           n_sample_type={}, is_save_dataset_plots=True,
                           is_verbose=False):
    """Generate material response data sets with multiple dependencies.
    
    Generates strain-stress material response data sets for given set of
    discrete temperatures and compositions, using a specified material
    constitutive model with temperature- and composition-dependent parameters.

    Parameters
    ----------
    lmu_dir : str
        Local model updating main directory.
    strain_formulation : {'infinitesimal', 'finite'}
        Strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    model_name : str
        FETorch material constitutive model name.
    temperatures : list[float], default=[]
        Discrete temperatures.
    compositions : list[float], default=[]
        Discrete compositions.
    n_sample_type : dict, default={}
        Number of samples (item, int) per data set type (str, key) for each
        set of material parameters.
    is_save_dataset_plots : bool, default=False
        If True, generate and save data set plots.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nGenerate strain-stress material response path data'
              '\n--------------------------------------------------'
              f'\n\nMaterial model: {model_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material data directory
    data_dir = os.path.join(lmu_dir, '0_simulation')
    # Create material data directory
    data_dir = make_directory(data_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem type parameters
    n_dim, comp_order_sym, _ = \
        get_problem_type_parameters(problem_type)
    # Set strain components order
    strain_comps_order = comp_order_sym
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path generator
    strain_path_generator, strain_path_gen_kwargs = \
        set_strain_path_generator(strain_formulation, n_dim,
                                  strain_comps_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material response path generator
    material_response_generator = \
        MaterialResponseDatasetGenerator(strain_formulation, problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build iterator over temperatures and compositions
    dependencies_iterator = itertools.product(temperatures, compositions)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over temperature and composition pairs
    for temperature, composition in dependencies_iterator:
        if is_verbose:
            print(f'\n\nTemperature: {temperature} |'
                  f' Composition: {composition}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set dependencies directory
        dependencies_dir = \
            set_dependencies_dir(data_dir, temperature, composition)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data constitutive model
        constitutive_model, state_features = \
            set_data_material_model(strain_formulation, problem_type,
                                    model_name, temperature, composition)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over data set types
        for dataset_type, n_sample in n_sample_type.items():
            if is_verbose:
                print(f'\n  > Data set type: {dataset_type} '
                      f'({n_sample} samples)\n')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set type directory
            dataset_type_dir = \
                set_dataset_type_dir(dependencies_dir, dataset_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set maximum number of sample trials
            max_sample_trials = 10
            # Initialize total number of failed sample trials
            total_n_sample_fail = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize data set samples
            dataset_samples = []
            # Loop over samples
            for i in tqdm.tqdm(range(n_sample),
                               desc=f'  > Generating {dataset_type} data set',
                               disable=not is_verbose):
                # Initialize number of sample trials
                n_sample_trials = 0
                # Initialize stress response path failure flag
                is_stress_path_fail = True
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Attempt to compute material response
                while is_stress_path_fail:
                    # Increment number of sample trials
                    n_sample_trials += 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Generate strain path
                    strain_comps_order, time_hist, strain_path = \
                        strain_path_generator.generate_strain_path(
                            **strain_path_gen_kwargs)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute material response
                    stress_comps_order, stress_path, state_path, \
                        is_stress_path_fail = \
                        material_response_generator.compute_stress_path(
                            strain_comps_order, time_hist, strain_path,
                            constitutive_model, state_features=state_features,
                            is_verbose=is_verbose)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update total number of failed sample trials
                    if is_stress_path_fail:
                        total_n_sample_fail += 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check maximum number of sample trials
                    if n_sample_trials > max_sample_trials:
                        raise RuntimeError(
                            f'The maximum number of trials '
                            f'({max_sample_trials}) to compute a material '
                            f'response path sample was reached without '
                            f'success.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize material response path data
                response_path = {}
                # Assemble strain-stress material response path
                response_path['strain_comps_order'] = strain_comps_order
                response_path['strain_path'] = \
                    torch.tensor(strain_path, dtype=torch.get_default_dtype())
                response_path['stress_comps_order'] = stress_comps_order
                response_path['stress_path'] = \
                    torch.tensor(stress_path, dtype=torch.get_default_dtype())
                # Assemble state variables path
                for state_var in state_path.keys():
                    response_path[state_var] = torch.tensor(
                        state_path[state_var], dtype=torch.get_default_dtype())
                # Assemble time path
                response_path['time_hist'] = torch.tensor(
                    time_hist, dtype=torch.get_default_dtype()).reshape(-1, 1)
                # Assemble temperature and composition history
                response_path['temperature_hist'] = torch.tensor(
                    [temperature,]*len(time_hist),
                    dtype=torch.get_default_dtype()).reshape(-1, 1)
                response_path['composition_hist'] = torch.tensor(
                    [composition,]*len(time_hist),
                    dtype=torch.get_default_dtype()).reshape(-1, 1)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store material response path
                dataset_samples.append(response_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build data set
            dataset = TimeSeriesDatasetInMemory(dataset_samples)
            # Get data set basename
            dataset_basename = get_dataset_basename()
            # Save data set
            save_dataset(dataset, dataset_basename, dataset_type_dir,
                         is_append_n_sample=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate data set plots
            if is_save_dataset_plots:
                # Create plots directory
                plots_dir = make_directory(
                    os.path.join(dataset_type_dir, 'plots'), is_overwrite=True)
                # Generate data set plots
                generate_dataset_plots(strain_formulation, n_dim, dataset,
                                       save_dir=plots_dir, is_save_fig=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total data generation time
    total_time_sec = time.time() - start_time_sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\nFinished generation of material response data sets!\n')
        print(f'Total time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))}')
        print('\n--------------------------------------------------')
# =============================================================================
def assemble_material_datasets(lmu_dir, strain_formulation, problem_type,
                               dataset_types, dataset_basename,
                               size_balanced_reduction=None,
                               is_save_dataset_plots=False, is_verbose=False):
    """Assemble material response data sets.
    
    Parameters
    ----------
    lmu_dir : str
        Local model updating main directory.
    strain_formulation : {'infinitesimal', 'finite'}
        Strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    dataset_types : list[str]
        Data set types to be assembled.
    dataset_basename : str
        Data set basename.
    size_balanced_reduction : dict, default=None
        For each data set type (key, str), set the fraction of samples
        (item, float) that should be kept from each loaded data set file.
        If None or not specified, then all loaded samples are kept for the
        corresponding data set type.
    is_save_dataset_plots : bool, default=False
        If True, generate and save data set plots.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nAssemble material response data sets'
              '\n------------------------------------'
              f'\n\nLocal model updating directory: {lmu_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file regex
    dataset_file_regex = re.compile(dataset_basename + r'_n\d+\.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data set types
    for dataset_type in dataset_types:
        if is_verbose:
            print(f'\n\nData set type: {dataset_type}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set type directory regex
        dataset_type_regex = re.compile(fr'{dataset_type}_dataset')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data set type file paths
        dataset_file_paths = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Walk through directory recursively
        for root, dirs, _ in os.walk(os.path.join(lmu_dir, '0_simulation')):
            # Loop over directories
            for dirname in dirs:
                # Check if directory matches data set type regex
                if dataset_type_regex.match(dirname):
                    # Set data set type directory path
                    dataset_type_dir = os.path.join(root, dirname)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over data set type directory files
                    for filename in os.listdir(dataset_type_dir):
                        # Check if file matches data set file regex
                        if dataset_file_regex.match(filename):
                            # Set data set file path
                            dataset_file_path = os.path.join(
                                dataset_type_dir, filename)
                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Collect data set file path
                            dataset_file_paths.append(dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data set colletion
        datasets = []
        # Loop over data set file paths
        for dataset_file_path in tqdm.tqdm(
                dataset_file_paths,
                desc=f'  > Loading {dataset_type} data sets',
                disable=not is_verbose):
            # Load data set
            dataset = load_dataset(dataset_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Reduce loaded data set size by randomly selecting a fraction of
            # samples from each loaded data set file
            if (isinstance(size_balanced_reduction, dict)
                    and dataset_type in size_balanced_reduction.keys()):
                # Get size balanced reduction fraction
                if isinstance(size_balanced_reduction[dataset_type], float):
                    reduced_fraction = size_balanced_reduction[dataset_type]
                else:
                    reduced_fraction = 1.0
                # Check reduction fraction
                if not (0.0 <= reduced_fraction <= 1.0):
                    warnings.warn(
                        f'Invalid reduction fraction ({reduced_fraction}) for '
                        f'data set type {dataset_type}. Setting to 1.0.',
                        category=UserWarning)
                    # Set admissible reduction fraction
                    reduced_fraction = 1.0
                # Get number of samples to keep
                n_sample = int(len(dataset)*reduced_fraction)
                # Randomly select subset of samples
                dataset = TimeSeriesDatasetInMemory(
                    [dataset[i] for i in random.sample(
                        range(len(dataset)), n_sample)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store data set
            datasets.append(dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n  > Concatenating {dataset_type} data sets')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate data sets
        cat_dataset = torch.utils.data.ConcatDataset(datasets)
        # Get shuffle concatenated data set samples indices
        shuffled_indices = torch.randperm(len(cat_dataset)).tolist()
        # Get concatenated data set with shuffled samples
        cat_dataset = torch.utils.data.Subset(cat_dataset, shuffled_indices) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convert data set
        dataset = TimeSeriesDatasetInMemory.from_dataset(cat_dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set directory basename
        if dataset_type == 'training':
            dataset_dirname = '1_training_dataset'
        elif dataset_type == 'validation':
            dataset_dirname = '2_validation_dataset'
        elif dataset_type == 'testing_id':
            dataset_dirname = '5_testing_id_dataset'
        # Set data set directory
        dataset_dir = set_dataset_type_dir(
            lmu_dir, dataset_type, dir_basename=dataset_dirname)
        # Save data set
        dataset_file_path = save_dataset(dataset, dataset_basename,
                                         dataset_dir, is_append_n_sample=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_verbose:
            print(f'\n  > Total number of samples: {len(dataset)}',
                  f'\n\n  > Data set file path: {dataset_file_path}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate data set plots
        if is_save_dataset_plots:
            # Get problem type parameters
            n_dim, _, _ = get_problem_type_parameters(problem_type)
            # Create plots directory
            plots_dir = make_directory(
                os.path.join(dataset_dir, 'plots'), is_overwrite=True)
            # Generate data set plots
            generate_dataset_plots(strain_formulation, n_dim, dataset,
                                   save_dir=plots_dir, is_save_fig=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total data generation time
    total_time_sec = time.time() - start_time_sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\nFinished assembling material response data sets!\n')
        print(f'Total time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))}')
        print('\n------------------------------------')
# =============================================================================
def preview_material_datasets_sizes(temperatures, compositions,
                                    n_sample_type={}):
    """Preview material response data sets sizes.
    
    Parameters
    ----------
    temperatures : list[float]
        Discrete temperatures.
    compositions : list[float]
        Discrete compositions.
    n_sample_type : dict, default={}
        Number of samples (item, int) per data set type (str, key) for each
        set of material parameters.
    """
    print('\nPreview material response data sets sizes'
          '\n-----------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute number of temperature-composition pairs
    n_temp_comp_pairs = len(temperatures)*len(compositions)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nTemperature-Composition dependencies:'
          f'\n\n  > Number of discrete temperatures: {len(temperatures)}'
          f'\n\n  > Number of discrete compositions: {len(compositions)}'
          f'\n\n  > Number of temperature-composition pairs: '
          f'{n_temp_comp_pairs}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n\nData set sizes:')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data set types
    for dataset_type, n_sample in n_sample_type.items():
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute total number of samples for given data set type
        n_sample_total = n_temp_comp_pairs*n_sample
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f'\n  Data set type: \'{dataset_type}\''
              f'\n\n    > Number of samples per temperature-composition pair: '
              f'{n_sample}'
              f'\n\n    > Total number of samples: {n_sample_total}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n-----------------------------------------\n')
# =============================================================================
def perform_material_model_updating(
        lmu_dir, n_model_sample=1, is_model_training=True,
        is_generate_uq_plots=False, is_verbose=False):
    """Perform model updating with uncertainty quantification.
    
    Model features: rnn_material_model/train_model.py
    
    Model architecture: rnn_material_model/train_model.py
    
    Training parameters: rnn_material_model/train_model.py
    
    Prediction parameters: rnn_material_model/predict_model.py
    
    Parameters
    ----------
    lmu_dir : str
        Local model updating main directory.
    n_model_sample : int
        Number of model samples, each with randomly initialized parameters.
    is_model_training : bool, default=True
        If True, then overwrite the uncertainty quantification directory
        and perform both training and prediction for each model sample.
        If False, then perform prediction for each existing model sample only.
    is_generate_uq_plots : bool, default=False
        If True, then generate model prediction uncertainty quantification
        plots.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    start_time_sec = time.time()
    if is_verbose:
        print('\nModel updating with uncertainty quantification'
              '\n----------------------------------------------'
              f'\n\nLocal model updating directory: {lmu_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model base directory
    model_dir = os.path.join(lmu_dir, '3_model')
    make_directory(model_dir, is_overwrite=True)
    # Set prediction type
    testing_type = 'in_distribution'
    # Set model prediction type base directory
    model_prediction_dir = os.path.join(lmu_dir, '7_prediction')
    make_directory(model_prediction_dir, is_overwrite=True)
    model_prediction_type_dir = os.path.join(model_prediction_dir,
                                             testing_type)
    make_directory(model_prediction_type_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get training data set file path
    training_dataset_file_path = find_dataset_file(
        target_dir=os.path.join(lmu_dir, '1_training_dataset'))
    # Get validation data set file path
    validation_dataset_file_path = find_dataset_file(
        target_dir=os.path.join(lmu_dir, '2_validation_dataset'))
    # Get testing data set file path
    testing_dataset_dir = os.path.join(lmu_dir, '5_testing_id_dataset')
    testing_dataset_file_path = find_dataset_file(
        target_dir=testing_dataset_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set uncertainty quantification directory
    uq_dir = os.path.join(lmu_dir, 'uncertainty_quantification')
    make_directory(uq_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform local model updating with uncertainty quantification
    perform_model_uq(uq_dir, n_model_sample, training_dataset_file_path,
                     model_dir, model_prediction_type_dir,
                     testing_dataset_file_path,
                     is_model_training=is_model_training,
                     val_dataset_file_path=validation_dataset_file_path,
                     device_type=device_type, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate plots of model uncertainty quantification
    if is_generate_uq_plots:
        gen_model_uq_plots(uq_dir, n_model_sample, testing_dataset_dir,
                           testing_type, is_save_fig=True, is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove model base directory
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    # Remove model prediction base directory
    if os.path.isdir(model_prediction_dir):
        shutil.rmtree(model_prediction_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total data generation time
    total_time_sec = time.time() - start_time_sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\nFinished model updating with uncertainty quantification!\n')
        print(f'Total time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))}')
        print('\n------------------------------------')
# =============================================================================
def set_lmu_dir(base_dir):
    """Set local model update main directory.
    
    Parameters
    ----------
    base_dir : str
    
    Returns
    -------
    lmu_dir : str
        Local model updating main directory.
    """
    # Check base directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The base directory has not been found:\n\n'
                           + base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set local model update main directory
    lmu_dir = os.path.join(os.path.normpath(base_dir), 'local_model_update')
    # Create data set directory
    if not os.path.isdir(lmu_dir):
        lmu_dir = make_directory(lmu_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return lmu_dir
# =============================================================================
def set_dependencies_dir(target_dir, temperature, composition):
    """Set directory for given temperature and composition.
    
    Parameters
    ----------
    target_dir : str
        Target directory.
    temperature : float
        Temperature.
    composition : float
        Volume fraction.
        
    Returns
    -------
    dependencies_dir : str
        Directory for given values of dependency parameters.
    """
    # Check target directory
    if not os.path.isdir(base_dir):
        raise RuntimeError('The target directory has not been found:\n\n'
                           + target_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dependencies directory basename
    dirname = f'T{temperature:.2f}_C{composition:.2f}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dependencies directory
    dependencies_dir = os.path.join(os.path.normpath(target_dir), dirname)
    # Create dependencies directory
    dependencies_dir = make_directory(dependencies_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dependencies_dir
# =============================================================================
def set_dataset_type_dir(target_dir, dataset_type, dir_basename=None):
    """Set directory for given dataset type.

    Parameters
    ----------
    target_dir : str
        Target directory.
    dataset_type : str
        Dataset type.
    dir_basename : str, default=None
        Dataset type directory basename. If None, then basename is set to
        '{dataset_type}_dataset' by default.

    Returns
    -------
    dataset_type_dir : str
        Directory for given data set type.
    """
    # Check target directory
    if not os.path.isdir(target_dir):
        raise RuntimeError('The target directory has not been found:\n\n'
                           + target_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dataset type directory basename
    if dir_basename is None:
        dir_basename = f'{dataset_type}_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dataset type directory
    dataset_type_dir = os.path.join(
        os.path.normpath(target_dir), dir_basename)
    # Create dataset type directory
    dataset_type_dir = make_directory(dataset_type_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_type_dir
# =============================================================================
def set_strain_path_generator(strain_formulation, n_dim, strain_comps_order):
    """Set strain path generator and generation parameters.
    
    Parameters
    ----------
    strain_formulation : {'infinitesimal', 'finite'}
        Strain formulation.
    n_dim : int
        Number of spatial dimensions.
    strain_comps_order : tuple
        Strain components order.
        
    Returns
    -------
    strain_path_generator : StrainPathGenerator
        Strain path generator.
    strain_path_gen_kwargs : dict
        Strain path generation parameters.
    """
    # Initialize strain path generator
    strain_path_generator = \
        RandomStrainPathGenerator(strain_formulation, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain components bounds
    strain_bounds = {x: (-0.05, 0.05) for x in strain_comps_order}
    # Set number of control points
    n_control = (4, 7)
    # Set number of time steps
    n_time = 200
    # Set initial and end time
    time_init = 0.0
    time_end = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path generation parameters
    strain_path_gen_kwargs = \
        {'n_control': n_control,
         'strain_bounds': strain_bounds,
         'n_time': n_time,
         'generative_type': 'polynomial',
         'time_init': time_init,
         'time_end': time_end}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_path_generator, strain_path_gen_kwargs
# =============================================================================
def set_data_material_model(strain_formulation, problem_type, model_name,
                            temperature, composition):
    """Set data constitutive model for given temperature and composition.
    
    Parameters
    ----------
    strain_formulation : {'infinitesimal', 'finite'}
        Strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    model_name : str
        FETorch material constitutive model name.
    temperature : float
        Temperature.
    composition : float
        Volume fraction.
        
    Returns
    -------
    constitutive_model : ConstitutiveModel
        FETorch material constitutive model.
    state_features : dict, default={}
        FETorch material constitutive model state variables (key, str) and
        corresponding dimensionality (item, int) for which the path history
        is additionally included in the data set.
    """
    # Set data constitutive model
    if model_name == 'von_mises':
        # Get model parameters for given temperature and composition
        model_parameters = get_model_parameters(model_name, temperature,
                                                composition)
        # Set constitutive model
        constitutive_model = VonMises(strain_formulation, problem_type,
                                      model_parameters)
        # Set constitutive model state features
        state_features = {'acc_p_strain': 1}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError(f'Unknown constitutive model: {model_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return constitutive_model, state_features
# =============================================================================
def get_model_parameters(model_name, temperature, composition):
    """Get model parameters for given temperature and composition.
    
    Parameters
    ----------
    model_name : str
        FETorch material constitutive model name.
    temperature : float
        Temperature.
    composition : float
        Volume fraction.
        
    Returns
    -------
    model_parameters : dict
        Model parameters.
    """
    # Set model parameters
    if model_name == 'von_mises':
        # Set temperature and composition normalization parameters
        t_mean, t_std = 325.0, 180.606
        c_mean, c_std = 0.5, 0.3168
        # Set temperature and composition admissible bounds
        t_bounds = (25, 625)
        c_bounds = (0.0, 1.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check temperature and composition bounds
        if temperature < t_bounds[0] or temperature > t_bounds[1]:
            raise RuntimeError(f'Temperature {temperature} is out of bounds: '
                               + f'{t_bounds[0]} <= T <= {t_bounds[1]}.')
        if composition < c_bounds[0] or composition > c_bounds[1]:
            raise RuntimeError(f'Composition {composition} is out of bounds: '
                               + f'{c_bounds[0]} <= C <= {c_bounds[1]}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize temperature and composition
        t_norm = (temperature - t_mean)/t_std
        c_norm = (composition - c_mean)/c_std
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elastic moduli coefficients
        a1, a2 = 20.186, 135.546
        b1, b2 = 0.936, -8.489
        c1, c2 = 8.843, 52.891
        d1, d2 = 0.405, -3.340
        # Compute elastic moduli (convert to MPa)
        young_modulus = (polynomial([a2, a1], c_norm)
                         + polynomial([b2, b1], c_norm)*t_norm)*(10**3)
        shear_modulus = (polynomial([c2, c1], c_norm)
                         + polynomial([d2, d1], c_norm)*t_norm)*(10**3)
        # Infer Poisson's ratio from elastic isotropy
        poisson = young_modulus/(2.0*shear_modulus) - 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set yield strength coefficients
        a1, b1, c1 = 39.738, -47.582, 25.107
        a2, b2, c2 = -0.091, 0.018, 0.883
        a3, b3, c3, d3, e3 = 20.025, 29.518, -121.669, -74.391, 541.225
        # Compute yield strength (MPa)
        s0 = (polynomial([c1, b1, a1], c_norm)**(
              -polynomial([c2, b2, a2], c_norm)*t_norm)
              + polynomial([e3, d3, c3, b3, a3], c_norm))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hardening law coefficients
        a1, a2 = 2213.1, 7253.7
        b1, b2 = -57.7, -68.7
        # Compute linear hardening slope (MPa)
        s1 = polynomial([a2, a1], c_norm) + polynomial([b2, b1], c_norm)*t_norm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hardening function
        def hard_fun(s0, s1, x):
            return s0 + s1*x
        # Set hardening points
        acc_p_strain = np.linspace(0.0, 2.0, num=1000)
        yield_stress = hard_fun(s0, s1, acc_p_strain)
        hardening_points = torch.tensor(
            np.column_stack((acc_p_strain, yield_stress)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model parameters
        model_parameters = \
            {'elastic_symmetry': 'isotropic',
             'E': young_modulus, 'v': poisson,
             'euler_angles': (0.0, 0.0, 0.0),
             'hardening_law': get_hardening_law('piecewise_linear'),
             'hardening_parameters': {'hardening_points': hardening_points}}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError(f'Unknown constitutive model: {model_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model_parameters
# =============================================================================
def polynomial(coefficients, x):
    """Evaluate polynomial function with given coefficients.
    
    Assumes coefficients are ordered from lowest to highest degree and that
    the polynomial is complete up to the highest degree.
    
    Parameters
    ----------
    coefficients : list
        Polynomial coefficients sorted from lowest to highest degree.
    x : float
        Input value.

    Returns
    -------
    value : float
        Polynomial function value.
    """
    # Compute polynomial value
    value = sum(c*(x**i) for i, c in enumerate(coefficients))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return value
# =============================================================================
def plot_model_parameters(model_name, temperatures=[], compositions=[],
                          save_dir=None, is_save_fig=False,
                          is_stdout_display=False):
    """Plot model parameters for given material model.
    
    Parameters
    ----------
    model_name : str
        FETorch material constitutive model name.
    temperatures : list[float], default=[]
        Discrete temperatures.
    compositions : list[float], default=[]
        Discrete compositions.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Check model name
    if model_name == 'von_mises':
        # Set plotting parameters
        plotting_params = {'E': '$E$ (GPa)',
                           'v': '$\\nu$',
                           's0': '$\\sigma_{y,0}$ (MPa)',
                           's1': '$H$ (MPa)'}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build meshgrid of temperature and composition values
        if len(temperatures) > 0 and len(compositions) > 0:
            # Create meshgrid of temperature and composition values
            t_mesh, c_mesh = np.meshgrid(temperatures, compositions,
                                         indexing='ij')
        else:
            raise RuntimeError('Both temperatures and compositions must have '
                               'at least one value to plot model parameters.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over plotting parameters
        for param, param_label in plotting_params.items():
            # Initialize parameter meshgrid
            p_mesh = np.zeros_like(t_mesh)
            # Loop over temperature and composition values
            for i, temperature in enumerate(temperatures):
                for j, composition in enumerate(compositions):
                    # Get model parameters for given temperature and
                    # composition
                    model_parameters = get_model_parameters(
                        model_name, temperature, composition)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set parameter value
                    if param == 'E':
                        # Get parameter (convert to GPa)
                        p_mesh[i, j] = model_parameters[param]*(10**-3)
                    elif param == 'v':
                        # Get parameter
                        p_mesh[i, j] = model_parameters[param]
                    elif param == 's0':
                        # Get hardening points
                        hard = model_parameters[
                            'hardening_parameters']['hardening_points']
                        # Get initial yield stress from hardening points
                        p_mesh[i, j] = hard[0, 1]
                    elif param == 's1':
                        hard = model_parameters[
                            'hardening_parameters']['hardening_points']
                        # Get linear hardening slope
                        p_mesh[i, j] = ((hard[-1, 1] - hard[0, 1])/
                                        (hard[-1, 0] - hard[0, 0]))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot parameter surface
            figure, _ = plot_surface_xyz_data(
                t_mesh, c_mesh, p_mesh, colormap='viridis',
                x_label='$T$ ($^\circ$C)', y_label='$C$', z_label=param_label,
                view_angles_deg=(40, -145, 0), is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save figure
            if is_save_fig:
                # Set figure name
                filename = f'{model_name}_{param}_temperature_composition'
                # Save figure
                save_figure(figure, filename, height=5.0, format='pdf',
                            save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figures
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close figures
        plt.close('all')
    else:
        raise RuntimeError(f'Unknown constitutive model: {model_name}')
# =============================================================================
def get_dataset_basename(dataset_type=None):
    """Get data set basename.
    
    Parameters
    ----------
    dataset_type : str, default=None
        Data set type appended to the basename.
    
    Returns
    -------
    dataset_basename : str
        Data set basename.
    """
    # Set data set basename
    dataset_basename = 'ss_paths_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append data set type to the basename
    if dataset_type is not None:
        dataset_basename += f'_{dataset_type}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_basename
# =============================================================================
def find_dataset_file(target_dir):
    """Find data set file in target directory.
    
    Parameters
    ----------
    target_dir : str
        Target directory.
        
    Returns
    -------
    dataset_file_path : str
        Data set file path.
    """
    # Get data set basename
    dataset_basename = get_dataset_basename()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file regex
    dataset_file_regex = dataset_basename + r'_n\d+\.pkl'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Search data set file in target directory
    is_file_found, dataset_file_path = \
        find_unique_file_with_regex(target_dir, dataset_file_regex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'No data set file with basename '
                           f'\'{dataset_basename}\' has been found in the '
                           f'target directory:\n\n{target_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_file_path
# =============================================================================
if __name__ == '__main__':
    # Set computational processes
    processes = {'preview_material_datasets_sizes': True,
                 'generate_material_data': False,
                 'assemble_material_datasets': False,
                 'perform_material_model_updating': False,
                 'plot_model_parameters': False}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'colaboration_antonios/contigency_plan_b/j2_dependencies')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation and problem type
    strain_formulation = 'infinitesimal'
    problem_type = 4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model name
    model_name = 'von_mises'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set temperatures and compositions
    temperatures = [25.0, 100.0]
    compositions = [0.0, 0.5]
    
    temperatures = list(np.linspace(25.0, 625.0, num=10))
    compositions = list(np.linspace(0.0, 1.0, num=10))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of samples per data set type for each set of material
    # parameters
    n_sample_type = {'training': 5,
                     'validation': 2,
                     'testing_id': 2}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set whether to generate data set plots
    is_save_dataset_plots = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of model samples (uncertainty quantification)
    n_model_sample = 1
    # Set whether to perform model training
    is_model_training = True
    # Set whether to generate uncertainty quantification plots
    is_generate_uq_plots = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set local model update main directory
    lmu_dir = set_lmu_dir(base_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Preview material response data sets sizes
    if processes['preview_material_datasets_sizes']:
        preview_material_datasets_sizes(temperatures, compositions,
                                        n_sample_type=n_sample_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material response data sets
    if processes['generate_material_data']:
        generate_material_data(lmu_dir, strain_formulation, problem_type,
                               model_name, temperatures=temperatures,
                               compositions=compositions,
                               n_sample_type=n_sample_type,
                               is_save_dataset_plots=False, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble material response data sets
    if processes['assemble_material_datasets']:
        # Set data set basename
        dataset_basename = get_dataset_basename()
        # Set data set types to be assembled
        dataset_types = n_sample_type.keys()
        # Assemble material response data sets
        assemble_material_datasets(lmu_dir, strain_formulation, problem_type,
                                   dataset_types, dataset_basename,
                                   size_balanced_reduction=None,
                                   is_save_dataset_plots=is_save_dataset_plots,
                                   is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform model updating with uncertainty quantification
    if processes['perform_material_model_updating']:
        perform_material_model_updating(
            lmu_dir, n_model_sample=n_model_sample,
            is_model_training=is_model_training,
            is_generate_uq_plots=is_generate_uq_plots,
            is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot model parameters
    if processes['plot_model_parameters']:
        # Set save directory
        save_dir = os.path.join(lmu_dir, '0_model_parameters_plots')
        make_directory(save_dir, is_overwrite=True)
        # Plot model parameters
        plot_model_parameters(model_name, temperatures, compositions,
                              save_dir=save_dir, is_save_fig=True,
                              is_stdout_display=True)