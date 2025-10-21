"""Evaluate GRU material model for different temperatures and compositions.
    
Functions
---------
get_dataset_file_paths
    Get MISSION dataset file paths for given test case.
load_strain_stress_data
    Load MISSION dataset.
extract_strain_path
    Extract strain path for given temperature and composition.
get_gru_material_model
    Get GRU material model from model directory.
build_gru_features_input
    Build GRU features input from strain path data.
forward_gru_material_model
    Forward pass through GRU material model.
get_vm_material_model
    Get VM material model from model directory.
forward_vm_material_model
    Forward pass through VM material model.
forward_material_model
    Forward pass through material model.
get_uniaxial_stress_path
    Get uniaxial stress path for given temperature and composition.
plot_strain_stress_paths
    Plot strain-stress paths for given temperature and composition.
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
import pickle
import tqdm
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from data_generation.strain_paths.interface import StrainPathGenerator
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from model_architectures.rnn_base_model.model.gru_model import GRURNNModel
from model_architectures.rc_base_model.model.recurrent_model import \
    RecurrentConstitutiveModel
from model_architectures.procedures.model_state_files import load_model_state
from model_architectures.procedures.model_data_scaling import \
    data_scaler_transform
from ioput.plots import plot_xy_data, save_figure
from ioput.iostandard import make_directory
from user_scripts.complete_pipelines.local_model_update import \
    get_model_parameters
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def get_dataset_file_paths(test_case):
    """Get MISSION dataset file paths for given test case.
    
    Parameters
    ----------
    test_case : int
        MISSION's lab test case index.
    
    Returns
    -------
    dataset_file_path : str
        MISSION dataset file path.
    temperatures : list[int]
        List of temperatures (°C) in the dataset.
    compositions : list[float]
        List of weight fractions of Ti4822 in the dataset.
    """
    if test_case == 1:
        dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                             'colaboration_antonios/contigency_plan_b/'
                             'j2_parameters_2025_10_15/extracted data/'
                             'extracted data/data_fractions.npz')
        temperatures = [25, 325, 625]
        compositions = [0, 0.25, 0.5, 0.75, 1.0]
    elif test_case == 2:
        dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                             'colaboration_antonios/contigency_plan_b/'
                             'j2_parameters_2025_10_15/extracted data/'
                             'extracted data/data_temperatures.npz')
        temperatures = [25, 125, 225, 325, 425, 525, 625]
        compositions = [0, 0.5, 1.0]
    else:
        raise ValueError(f'Invalid test case: {test_case}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_file_path, temperatures, compositions
# =============================================================================
def load_strain_stress_data(dataset_file_path):
    """Load MISSION dataset.
    
    Parameters
    ----------
    dataset_file_path : str
        MISSION dataset file path.

    Returns
    -------
    dataset : dict
        MISSION dataset.
    """
    dataset = np.load(dataset_file_path, allow_pickle=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def extract_strain_path(dataset, temperature, composition, device='cpu'):
    """Extract strain path for given temperature and composition.
    
    Parameters
    ----------
    dataset : dict
        MISSION dataset.
    temperature : int
        Temperature (°C).
    composition : float
        Weight fraction of Ti4822 (0 to 1).
    device : str, default='cpu'
        Torch device.
        
    Returns
    -------
    strain_path : torch.Tensor(2d)
        Strain path history stored as torch.Tensor(2d) of shape
        (n_time, n_strain_comp).
    """
    # Extract strain path
    strain_path = \
        dataset[f'strain_frac_{int(composition*100)}_T_{temperature}']
    # Convert to torch tensor
    strain_path = torch.tensor(strain_path, dtype=torch.get_default_dtype())
    # Move to device
    strain_path = strain_path.to(device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_path
# =============================================================================
def get_gru_material_model(model_directory, model_load_state='best',
                           device_type='cpu'):
    """Get GRU material model from model directory.
    
    Parameters
    ----------
    model_directory : str
        Model directory.
    model_load_state : str, default='best'
        Model load state identifier.
    device_type : str, default='cpu'
        Torch device type.
        
    Returns
    -------
    model : GRURNNModel
        GRU material model.
    """
    # Initialize recurrent neural network model
    model = GRURNNModel.init_model_from_file(model_directory=model_directory)
    # Set model device
    model.set_device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model state
    _ = load_model_state(model, model_load_state=model_load_state,
                         is_remove_posterior=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device
    device = torch.device(device_type)
    # Move model to device
    model.to(device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model
# =============================================================================
def build_gru_features_in(strain_path, temperature, composition):
    """Build GRU material model input features tensor.
    
    Parameters
    ----------
    strain_path : torch.Tensor(2d)
        Strain path history stored as torch.Tensor(2d) of shape
        (n_time, n_strain_comp).
    temperature : int
        Temperature (°C).
    composition : float
        Weight fraction of Ti4822 (0 to 1).

    Returns
    -------
    features_in : torch.Tensor(2d)
        Model input features tensor of shape (n_time, n_features).
    """
    # Get device
    device = strain_path.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of time steps
    n_time = strain_path.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set temperature and composition path
    temperature_path = torch.tensor([temperature,]*n_time,
                                    dtype=strain_path.dtype,
                                    device=device).reshape(-1, 1)
    composition_path = torch.tensor([composition,]*n_time,
                                    dtype=strain_path.dtype,
                                    device=device).reshape(-1, 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Concatenate model input features
    features_in = \
        torch.cat((strain_path, temperature_path, composition_path), dim=1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return features_in
# =============================================================================
def forward_gru_material_model(model, features_in, requires_grad=False):
    """Forward GRU material model.
    
    Parameters
    ----------
    model : GRURNNModel
        GRU material model.
    features_in : torch.Tensor(2d)
        Model input features tensor of shape (n_time, n_features).
    requires_grad : bool, default=False
        Whether to track gradients during forward propagation.
    """
    # Get model input and output features normalization
    is_model_in_normalized = model.is_model_in_normalized
    is_model_out_normalized = model.is_model_out_normalized
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Disable gradients for model parameters (frozen model)
    for param in model.parameters():
        param.requires_grad_(False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    with torch.set_grad_enabled(requires_grad):
        # Normalize input features
        if is_model_in_normalized:
            features_in = data_scaler_transform(model, features_in,
                                                features_type='features_in',
                                                mode='normalize')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set initial hidden state features
        hidden_features_in = None
        # Forward propagation
        features_out, _ = model(features_in, hidden_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denormalize output features
        if is_model_out_normalized:
            features_out = data_scaler_transform(model, features_out,
                                                 features_type='features_out',
                                                 mode='denormalize')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return features_out
# =============================================================================
def get_vm_material_model(strain_formulation, problem_type, temperature,
                          composition, device_type='cpu'):
    """Get von Mises material model.
    
    Parameters
    ----------
    strain_formulation : str
        Strain formulation.
    problem_type : str
        Problem type.
    temperature : float
        Temperature (°C).
    composition : float
        Weight fraction of Ti4822 (0 to 1).
    device_type : str, default='cpu'
        Torch device type.
        
    Returns
    -------
    model : RecurrentConstitutiveModel
        Von Mises material model.
    """
    # Set material model name
    material_model_name = 'von_mises'
    # Get von Mises model parameters for given temperature and composition
    material_model_parameters = \
        get_model_parameters(material_model_name, temperature, composition)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of input and output features
    n_features_in = 6
    n_features_out = 6
    # Set learnable parameters
    learnable_parameters = {}
    # Set model directory
    model_directory = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize von Mises material model
    model = RecurrentConstitutiveModel(n_features_in, n_features_out,
                                       learnable_parameters,
                                       strain_formulation, problem_type,
                                       material_model_name,
                                       material_model_parameters,
                                       model_directory,
                                       is_save_model_init_file=False,
                                       device_type=device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model
# =============================================================================
def forward_vm_material_model(model, strain_path):
    """Forward von Mises material model.
    
    Parameters
    ----------
    model : RecurrentConstitutiveModel
        Von Mises material model.
    strain_path : torch.Tensor(2d)
        Strain path history stored as torch.Tensor(2d) of shape
        (n_time, n_strain_comp).

    Returns
    -------
    stress_path : torch.Tensor(2d)
        Stress path history stored as torch.Tensor(2d) of shape
        (n_time, n_stress_comp).
    """
    # Forward propagation
    stress_path = model(strain_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_path
# =============================================================================
def forward_material_model(model_name, model, strain_path, temperature,
                           composition, requires_grad=False):
    """Forward material model.
    
    Parameters
    ----------
    model_name : str
        Material model name.
    model : {VonMises, GRURNNModel}
        Material model.
    strain_path : torch.Tensor(2d)
        Strain path history stored as torch.Tensor(2d) of shape
        (n_time, n_strain_comp).
    temperature : float
        Temperature (°C).
    composition : float
        Weight fraction of Ti4822 (0 to 1).
    requires_grad : bool, default=False
        Whether to track gradients during forward propagation.

    Returns
    -------
    stress_path : torch.Tensor(2d)
        Stress path history stored as torch.Tensor(2d) of shape
        (n_time, n_stress_comp).
    """
    # Forward material model and extract stress path
    if model_name == 'von_mises':
        stress_path = forward_vm_material_model(model, strain_path)
    elif model_name == 'gru':
        # Build input features
        features_in = build_gru_features_in(strain_path, temperature,
                                            composition)
        # Forward model
        features_out = forward_gru_material_model(model, features_in,
                                                  requires_grad=requires_grad)
        # Extract stress path
        stress_path = features_out[:, :6]
    else:
        raise ValueError(f'Invalid material model name: {model_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check stress path
    if not isinstance(stress_path, torch.Tensor):
        raise TypeError('Stress path is not a torch.Tensor')
    elif stress_path.shape != strain_path.shape:
        raise ValueError('Stress path shape does not match strain path shape.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_path
# =============================================================================
def get_uniaxial_stress_path(strain_33_path, model_name, material_model,
                             temperature, composition, tolerance=1e-8,
                             max_iter=30, device="cpu"):
    """Compute strain path leading to uniaxial stress along 33 direction.

    Parameters
    ----------
    strain_33_path : torch.Tensor(1d)
        Prescribed strain_33 history (n_time,).
    model_name : str
        Material model name.
    material_model : {VonMises, GRURNNModel}
        Material model.
    temperature : float
        Temperature (°C).
    composition : float
        Weight fraction of Ti4822 (0 to 1).
    tolerance : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations per step.
    device : str
        Torch device.

    Returns
    -------
    strain_path : torch.Tensor, shape (n_time, 6)
        Strain path leading to uniaxial stress along 33.
    """
    # Set strain type
    strain_dtype = torch.get_default_dtype()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert strain component loading path to torch tensor
    strain_33_path = strain_33_path.to(dtype=strain_dtype, device=device)
    # Get number of time steps
    n_time = strain_33_path.numel()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain path
    strain_path = torch.zeros((n_time, 6), dtype=strain_dtype, device=device)
    # Initialize uniaxial stress path
    uniaxial_stress_path = torch.zeros_like(strain_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set uniaxial stress residual function
    def residual(x, strain_33, time_idx, strain_path, model_name,
                 material_model, temperature, composition, device):
        # Initialize strain tensor
        strain = torch.zeros((1, 6), dtype=strain_dtype, device=device)
        # Build strain tensor by assembling iterative strain components and
        # prescribed loading component
        strain[0, 0] = x[0]
        strain[0, 1] = x[1]
        strain[0, 2] = strain_33
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble iterative strain tensor to strain path
        strain_path = torch.cat([strain_path[:time_idx, :], strain], dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward material model to compute stress path
        stress_path = forward_material_model(
            model_name, material_model, strain_path, temperature, composition,
            requires_grad=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get residual
        residual = stress_path[-1, [0, 1]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return residual
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over time steps
    for i in tqdm.tqdm(range(1, n_time),
                       desc='Processing time steps: '):
        # Get prescribed loading strain component
        strain_33 = strain_33_path[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize iterative strain components (previous time step)
        x = torch.zeros(2, dtype=strain_dtype, device=device).clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enable gradient tracking of iterative strain components
        x = x.requires_grad_(True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Newton iterations
        for _ in range(max_iter):
            # Compute uniaxial stress residual
            res = residual(x, strain_33, i, strain_path, model_name,
                           material_model, temperature, composition, device)
            # Check convergence
            if torch.linalg.norm(res)/torch.linalg.norm(strain_33) < tolerance:
                break
            # Compute Jacobian
            jacobian = torch.autograd.functional.jacobian(
                lambda xx: residual(xx, strain_33, i, strain_path, model_name,
                                    material_model, temperature, composition,
                                    device), x)
            # Compute iterative update
            dx = torch.linalg.solve(jacobian, -res)
            # Update iterative solution (break gradient tracking)
            x = (x + dx).detach().requires_grad_(True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store converged strain components
        strain_path[i, 0:2] = x
        strain_path[i, 2] = strain_33
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward material model to compute stress path
    uniaxial_stress_path = forward_material_model(model_name, material_model,
                                                  strain_path, temperature,
                                                  composition,
                                                  requires_grad=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detach strain and stress paths from computation graph
    strain_path = strain_path.detach()
    uniaxial_stress_path = uniaxial_stress_path.detach()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_path, uniaxial_stress_path
# =============================================================================
def plot_strain_stress_paths(strain_comp_paths, stress_comp_paths, data_labels,
                             title=None):
    """Plot stress vs strain component paths.
    
    Parameters
    ----------
    strain_comp_paths : list[torch.Tensor(1d)]
        List of strain component paths stored as torch.Tensor(1d).
    stress_comp_paths : list[torch.Tensor(1d)]
        List of stress component paths stored as torch.Tensor(1d).
    data_labels : list[str]
        List of data labels.
    title : str, default=None
        Plot title.
        
    Returns
    -------
    figure : matplotlib.figure.Figure
        Matplotlib figure.
    axes : matplotlib.axes.Axes
        Matplotlib axes.
    """
    # Get number of time steps
    n_time = strain_comp_paths[0].shape[0]
    # Get number of paths
    n_path = len(strain_comp_paths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data array
    data_array = np.zeros((n_time, 2*n_path))
    for i in range(n_path):
        data_array[:, 2*i] = strain_comp_paths[i].flatten()
        data_array[:, 2*i+1] = stress_comp_paths[i].flatten()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits
    x_lims = (0, None)
    y_lims = (None, None)
    # Set axes labels
    x_label = 'Strain'
    y_label = 'Stress (N/mm$^2$)'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot data
    figure, axes = plot_xy_data(data_array, data_labels=data_labels,
                                title=title, x_lims=x_lims, y_lims=y_lims,
                                x_label=x_label, y_label=y_label,
                                is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return figure, axes
# =============================================================================
def evaluate_gru_material_model():
    """Evaluate GRU material model under set of temperatures and compositions.
    
    Notes
    -----
    Update plot_xy_data() default cycler:

    # Extract colors and linestyles
    colors = [d['color'] for d in cycler_color]
    linestyles = [d['linestyle'] for d in cycler_linestyle[:2]]
    # Build colors and linestyles required for cycler
    color_list = [c for c in colors for _ in linestyles]
    linestyle_list = [ls for _ in colors for ls in linestyles]
    # Set default cycler
    default_cycler = (cycler.cycler('color', color_list)
                      + cycler.cycler('linestyle', linestyle_list))
    """
    # Set device
    device_type = ('cpu', 'cuda')[0]
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'colaboration_antonios/contigency_plan_b/'
                 'j2_parameters_2025_10_15/debug/plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation and problem type
    strain_formulation = 'infinitesimal'
    problem_type = 4
    # Get problem type parameters
    _, comp_order_sym, _ = \
        get_problem_type_parameters(problem_type)
    # Set strain and stress components indices
    strain_idx = comp_order_sym.index('33')
    stress_idx = comp_order_sym.index('33')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model name
    model_name = 'gru'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set MISSION test case
    test_case = 1
    # Get MISSION dataset file path temperatures and compositions
    _, temperatures, compositions = \
        get_dataset_file_paths(test_case) 
    # Set uniaxial stress dataset file path
    dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                         'colaboration_antonios/contigency_plan_b/'
                         'j2_parameters_2025_10_15/debug/'
                         'uniaxial_stress_datasets/'
                         f'uniaxial_stress_dataset_test_case_{test_case}.pkl')
    # Load uniaxial stress dataset
    with open(dataset_file_path, 'rb') as file:
        uniaxial_stress_dataset = pickle.load(file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nComputing material model uniaxial stress paths...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process test case
    if test_case == 1:
        # Loop over temperatures
        for temperature in temperatures:
            # Initialize strain and stress component paths
            strain_comp_paths = []
            stress_comp_paths = []
            # Initialize data labels
            data_labels = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over compositions
            for composition in compositions:
                print(f'\nProcessing: T={temperature}°C | '
                      f'C={int(composition*100)}% ...')
                # Set temperature-composition key
                temp_comp_key = f'T_{temperature}_C_{int(composition*100)}'
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Load strain path from uniaxial stress dataset
                strain_path = torch.tensor(
                    uniaxial_stress_dataset[temp_comp_key]['strain_path'],
                    dtype=torch.get_default_dtype(), device=device)
                ref_stress_path = torch.tensor(
                    uniaxial_stress_dataset[temp_comp_key]['stress_path'],
                    dtype=torch.get_default_dtype(), device=device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set material model
                if model_name == 'von_mises':
                    model = get_vm_material_model(strain_formulation,
                                                  problem_type, temperature,
                                                  composition,
                                                  device_type=device_type)
                elif model_name == 'gru':
                    # Set model directory
                    model_directory = (
                        '/home/bernardoferreira/Documents/brown/projects/'
                        'colaboration_antonios/contigency_plan_b/'
                        'j2_parameters_2025_10_15/debug/model_2/3_model')
                    # Get GRU material model
                    model = get_gru_material_model(model_directory,
                                                   model_load_state='best',
                                                   device_type=device_type)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Forward material model to compute stress path
                stress_path = forward_material_model(
                    model_name, model, strain_path, temperature, composition,
                    requires_grad=False)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store strain and stress component paths
                strain_comp_paths.append(strain_path[:, strain_idx].numpy())
                stress_comp_paths.append(stress_path[:, stress_idx].numpy())
                # Store data label
                data_labels.append(f'{int(composition*100)}\%')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store reference strain and stress component path
                strain_comp_paths.append(strain_path[:, strain_idx].numpy())
                stress_comp_paths.append(
                    ref_stress_path[:, stress_idx].numpy())
                # Store reference data label
                data_labels.append(f'{int(composition*100)}\% (J2)')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set plot title
            plot_title = f'T = {temperature} °C'
            # Plot stress vs strain component paths
            figure, _ = plot_strain_stress_paths(
                strain_comp_paths=strain_comp_paths,
                stress_comp_paths=stress_comp_paths,
                data_labels=data_labels,
                title=plot_title)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set figure file path
            fig_file_path = pathlib.Path(plots_dir) / \
                f'stress_vs_strain_T_{int(temperature)}.pdf'
            # Save figure
            save_figure(figure, fig_file_path)
            # Close figure
            plt.close(figure)
# =============================================================================
def build_uniaxial_stress_dataset():
    """Build uniaxial stress dataset using von Mises material model."""
    # Set device
    device_type = ('cpu', 'cuda')[0]
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set uniaxial stress dataset directory
    dataset_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                   'colaboration_antonios/contigency_plan_b/'
                   'j2_parameters_2025_10_15/debug/uniaxial_stress_datasets_')
    # Check dataset directory
    if not pathlib.Path(dataset_dir).exists():
        raise RuntimeError('Uniaxial stress dataset directory does not exist: '
                           f'{dataset_dir}')
    # Set plots directory
    plots_dir = pathlib.Path(dataset_dir) / 'plots'
    # Create plots directory
    plots_dir = make_directory(plots_dir, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation and problem type
    strain_formulation = 'infinitesimal'
    problem_type = 4
    # Get problem type parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
    # Set strain and stress components order
    strain_comps_order = comp_order_sym
    stress_comps_order = comp_order_sym
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model name
    model_name = 'von_mises'
    # Set number of time steps
    n_time = 200
    # Set total loading strain
    total_strain = 0.05
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set MISSION test case
    test_case = 1
    # Get MISSION dataset file path temperatures and compositions
    _, temperatures, compositions = get_dataset_file_paths(test_case)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize dataset samples
    dataset_dict = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain component path along loading dimension (33)
    strain_33_path = torch.linspace(
        0.0, total_strain, steps=n_time,
        dtype=torch.get_default_dtype(), device=device)
    # Set discrete time history
    time_hist = np.linspace(0.0, 1.0, num=n_time)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nBuilding uniaxial stress dataset...')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over temperatures
    for temperature in temperatures:
        # Loop over compositions
        for composition in compositions:
            print(f'\nProcessing: T={temperature}°C | '
                  f'C={int(composition*100)}% ...')
            # Set temperature-composition key
            temp_comp_key = f'T_{temperature}_C_{int(composition*100)}'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material model
            if model_name == 'von_mises':
                model = get_vm_material_model(strain_formulation,
                                              problem_type, temperature,
                                              composition,
                                              device_type=device_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute strain and stress paths corresponding to uniaxial
            # stress state along loading dimension (33)
            strain_path, stress_path = get_uniaxial_stress_path(
                strain_33_path, model_name, model, temperature,
                composition, device=device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store strain and stress paths
            dataset_dict[temp_comp_key] = {
                'strain_path': strain_path.cpu().numpy(),
                'stress_path': stress_path.cpu().numpy()
            }
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot strain path
            StrainPathGenerator.plot_strain_path(
                strain_formulation, n_dim,
                strain_comps_order, time_hist,
                dataset_dict[temp_comp_key]['strain_path'],
                is_plot_strain_path=True,
                strain_label='Strain',
                strain_units='',
                filename=f'strain_path_{temp_comp_key}',
                save_dir=plots_dir,
                is_save_fig=True,
                is_latex=True)
            # Plot stress path
            StrainPathGenerator.plot_strain_path(
                strain_formulation, n_dim,
                stress_comps_order, time_hist,
                dataset_dict[temp_comp_key]['stress_path'],
                is_plot_strain_path=True,
                strain_label='Stress',
                strain_units=' (MPa)',
                filename=f'stress_path_{temp_comp_key}',
                save_dir=plots_dir,
                is_save_fig=True,
                is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set dataset file path
    dataset_file_path = pathlib.Path(dataset_dir) / \
        f'uniaxial_stress_dataset_test_case_{test_case}.pkl'
    # Save dataset
    pickle.dump(dataset_dict, open(dataset_file_path, 'wb'))      
# =============================================================================
if __name__ == "__main__":
    # Set process
    process = ('build_uniaxial_stress_dataset',
               'evaluate_gru_material_model')[1]
    # Execute process
    if process == 'build_uniaxial_stress_dataset':
        build_uniaxial_stress_dataset()
    elif process == 'evaluate_gru_material_model':
        evaluate_gru_material_model()
    else:
        raise ValueError(f'Invalid process: {process}')