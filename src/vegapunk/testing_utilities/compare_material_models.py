# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from projects.darpa_metals.rnn_material_model.user_scripts. \
    gen_response_dataset import MaterialResponseDatasetGenerator
from projects.darpa_metals.rnn_material_model.strain_paths.interface import \
    StrainPathGenerator
from time_series_data.time_dataset import load_dataset
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.vmap.von_mises import VonMisesVMAP
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from simulators.fetorch.material.models.vmap.drucker_prager import \
    DruckerPragerVMAP
from simulators.fetorch.material.models.standard.lou import LouZhangYoon
from simulators.fetorch.material.models.vmap.lou import LouZhangYoonVMAP
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Compare constitutive models stress prediction
# =============================================================================
def compare_material_model_response(strain_formulation, problem_type,
                                    strain_comps_order, time_hist,
                                    strain_path, constitutive_models={},
                                    ref_stress_path=None,
                                    save_dir=None, is_save_fig=False,
                                    is_stdout_display=False, is_latex=False):
    """Compare constitutive models stress predictions for given strain path.
    
    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    strain_comps_order : tuple[str]
        Strain components order.
    time_hist : numpy.ndarray(1d)
        Discrete time history.
    strain_path : numpy.ndarray(2d)
        Strain path history stored as numpy.ndarray(2d) of shape
        (sequence_length, n_strain_comps).
    constitutive_models : dict
        FETorch material constitutive models (key, str, item,
        ConstitutiveModel).
    ref_stress_path : numpy.ndarray(2d), default=None
        Stress path history stored as numpy.ndarray(2d) of shape
        (sequence_length, n_stress_comps).
    save_dir : str, default=None
        Directory where figure is saved.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not
        available, then this option is silently set to False and all input
        strings are processed to remove $(...)$ enclosure.
    """
    # Initialize strain-stress material response path data set generator
    dataset_generator = \
        MaterialResponseDatasetGenerator(strain_formulation, problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize constitutive models predictions
    models_predictions = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material constitutive models
    for model_name, model in constitutive_models.items():
        # Compute constitutive model prediction
        stress_comps_order, stress_path, state_path, \
            is_stress_path_fail = dataset_generator.compute_stress_path(
                strain_comps_order, time_hist, strain_path, model)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build prediction data
        model_prediction = {'stress_comps_order': stress_comps_order,
                            'stress_path': stress_path,
                            'state_path': state_path,
                            'time_hist': time_hist,
                            'is_stress_path_fail': is_stress_path_fail}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check state update failure
        if is_stress_path_fail:
            print(f'Model: {model_name} - State update failure!')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model prediction data
        models_predictions[model_name] = model_prediction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add reference stress path
    if ref_stress_path is not None:
        # Build reference data
        model_prediction = {'stress_comps_order': stress_comps_order,
                            'stress_path': ref_stress_path,
                            'time_hist': time_hist}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store reference data
        models_predictions['reference'] = model_prediction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem type parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(problem_type)
    # Get number of constitutive models
    n_model = len(models_predictions.keys())
    # Get number of discrete times
    n_time = len(time_hist)
    # Loop stress components
    for comp_label in comp_order_sym:
        # Initialize stress data array
        stress_data_xy = np.zeros((n_time, 2*n_model))
        # Initialize data labels
        data_labels = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material constitutive models
        for i, model_name in enumerate(models_predictions.keys()):
            # Get model prediction data
            model_prediction = models_predictions[model_name]
            # Get model stress path prediction
            stress_path = model_prediction['stress_path']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get model stress component index
            k = model_prediction['stress_comps_order'].index(comp_label)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble model stress data
            stress_data_xy[:, 2*i] = time_hist.reshape(-1)
            stress_data_xy[:, 2*i+1] = stress_path[:, k]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble model label
            data_labels.append(model_name)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot stress path
        figure, _ = plot_xy_data(data_xy=stress_data_xy,
                                 data_labels=data_labels,
                                 x_lims=(time_hist[0], time_hist[-1]),
                                 x_label='Time',
                                 y_label=f'Stress {comp_label}',
                                 is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set figure name
        figure_name = f'model_comparison_stress_{comp_label}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure
        if is_save_fig:
            save_figure(figure, figure_name, format='pdf',
                        save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot strain-stress material response path data in principal stress space
    if n_dim == 3:
        # Initialize models stress paths data
        time_hists = []
        stress_paths = []
        # Collect models strain-stress paths data
        for model_name in models_predictions.keys():
            # Get model prediction data
            model_prediction = models_predictions[model_name]
            # Collect discrete time history
            time_hists.append(model_prediction['time_hist'])
            # Collect stress history
            stress_paths.append(model_prediction['stress_path'])
        # Plot stress paths data
        MaterialResponseDatasetGenerator.plot_stress_space_metrics(
            strain_formulation, stress_comps_order,
            stress_paths, time_hists,
            is_plot_principal_stress_path=True,
            is_plot_pi_stress_path_pairs=True,
            stress_units=' (MPa)',
            filename='stress_path',
            save_dir=save_dir,
            is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display, is_latex=True)
# =============================================================================
if __name__ == "__main__":
    # Set reference stress path flag
    is_reference_stress_path = True
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/11_global_learning_lou/'
                 'random_specimen_rc_von_mises_vmap/loading_3/test_paths/'
                 'plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path source
    strain_path_source = 'dataset'
    # Get strain path data
    if strain_path_source == 'dataset':
        # Set data set file path
        dataset_file_path = \
            ('/home/bernardoferreira/Documents/brown/projects/darpa_project/'
             '11_global_learning_lou/random_specimen_rc_von_mises_vmap/'
             'loading_3/3_discover_rc_lou/material_model_finder/'
             '0_simulation/local_response_dataset/ss_paths_dataset_n936.pkl')
        # Set sample index:
        # Loading 2
        sample_idx = 44 # Element 6 - Failed local path index
        sample_idx = 275 # Element 35 - Failed local path index
        sample_idx = 650 # Element 82 - Failed local path index
        sample_idx = 834 # Element 105 - Failed local path index
        # Loading 3
        sample_idx = 2 # Element 1 - Failed local path index (GP #3)
        sample_idx = 12 # Element 2 - Failed local path index (GP #1, #3, #4, #5)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data set
        dataset = load_dataset(dataset_file_path)
        # Extract sample material response path
        response_path = dataset[sample_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect material response path data
        strain_comps_order = response_path['strain_comps_order']
        strain_path = response_path['strain_path']
        stress_comps_order = response_path['stress_comps_order']
        stress_path = response_path['stress_path']
        time_hist = response_path['time_hist']
    else:
        raise RuntimeError('Unknown strain path source.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot strain paths data
    StrainPathGenerator.plot_strain_path(
        strain_formulation, n_dim,
        strain_comps_order, time_hist, strain_path,
        is_plot_strain_path=True,
        is_plot_strain_comp_hist=False,
        is_plot_strain_norm=True,
        is_plot_strain_norm_hist=False,
        is_plot_inc_strain_norm=True,
        is_plot_inc_strain_norm_hist=False,
        is_plot_strain_path_pairs=False,
        is_plot_strain_pairs_hist=False,
        is_plot_strain_pairs_marginals=False,
        is_plot_strain_comp_box=False,
        strain_label='Strain',
        strain_units='',
        filename='strain_path',
        save_dir=plots_dir,
        is_save_fig=True,
        is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize constitutive models
    constitutive_models = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'von_mises_1'
    # Set constitutive model parameters
    model_parameters = {
        'elastic_symmetry': 'isotropic',
        'E': 110e3, 'v': 0.33,
        'euler_angles': (0.0, 0.0, 0.0),
        'hardening_law': get_hardening_law('nadai_ludwik'),
        'hardening_parameters': {'s0': 900,
                                 'a': 700,
                                 'b': 0.5,
                                 'ep0': 1e-5}}
    # Initialize constitutive model
    constitutive_model = VonMises(strain_formulation, problem_type,
                                  model_parameters)
    # Store constitutive model
    #constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'von_mises_2'
    # Set constitutive model parameters
    model_parameters = {
        'elastic_symmetry': 'isotropic',
        'E': 110e3, 'v': 0.33,
        'euler_angles': (0.0, 0.0, 0.0),
        'hardening_law': get_hardening_law('nadai_ludwik'),
        'hardening_parameters': {'s0': 900,
                                 'a': 700,
                                 'b': 0.5,
                                 'ep0': 1e-5}}
    # Initialize constitutive model
    constitutive_model = VonMises(strain_formulation, problem_type,
                                  model_parameters)
    # Store constitutive model
    #constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'drucker_prager_gt'
    # Set frictional angle
    friction_angle = np.deg2rad(5)
    # Set dilatancy angle
    dilatancy_angle = friction_angle
    # Compute angle-related material parameters
    # (matching with Mohr-Coulomb under uniaxial tension and compression)
    # Set yield surface cohesion parameter
    yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
    # Set yield pressure parameter
    yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
    # Set plastic flow pressure parameter
    flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
    # Set constitutive model parameters
    # (matching Von Mises yield surface for null pressure)
    model_parameters = {
        'elastic_symmetry': 'isotropic',
        'E': 110e3, 'v': 0.33,
        'euler_angles': (0.0, 0.0, 0.0),
        'hardening_law': get_hardening_law('nadai_ludwik'),
        'hardening_parameters':
            {'s0': 900/(np.sqrt(3)*yield_cohesion_parameter),
             'a': 700/(np.sqrt(3)*yield_cohesion_parameter),
             'b': 0.5,
             'ep0': 1e-5},
        'yield_cohesion_parameter': yield_cohesion_parameter,
        'yield_pressure_parameter': yield_pressure_parameter,
        'flow_pressure_parameter': flow_pressure_parameter,
        'friction_angle': friction_angle}
    # Initialize constitutive model
    constitutive_model = DruckerPrager(strain_formulation, problem_type,
                                       model_parameters)
    # Store constitutive model
    #constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'drucker_prager_found'
    # Set frictional angle
    friction_angle = 0.08712586238835966
    # Set dilatancy angle
    dilatancy_angle = friction_angle
    # Compute angle-related material parameters
    # (matching with Mohr-Coulomb under uniaxial tension and compression)
    # Set yield surface cohesion parameter
    yield_cohesion_parameter = (2.0/np.sqrt(3))*np.cos(friction_angle)
    # Set yield pressure parameter
    yield_pressure_parameter = (3.0/np.sqrt(3))*np.sin(friction_angle)
    # Set plastic flow pressure parameter
    flow_pressure_parameter = (3.0/np.sqrt(3))*np.sin(dilatancy_angle)
    # Set constitutive model parameters
    # (matching Von Mises yield surface for null pressure)
    model_parameters = {
        'elastic_symmetry': 'isotropic',
        'E': 96371.53685092926, 'v': 0.33,
        'euler_angles': (0.0, 0.0, 0.0),
        'hardening_law': get_hardening_law('nadai_ludwik'),
        'hardening_parameters':
            {'s0': 385.3058099746704,
             'a': 448.69024008512497,
             'b': 0.5,
             'ep0': 1e-5},
        'yield_cohesion_parameter': yield_cohesion_parameter,
        'yield_pressure_parameter': yield_pressure_parameter,
        'flow_pressure_parameter': flow_pressure_parameter,
        'friction_angle': friction_angle}
    # Initialize constitutive model
    constitutive_model = DruckerPrager(strain_formulation, problem_type,
                                       model_parameters)
    # Store constitutive model
    #constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'lou_zhang_yoon_gt'
    # Set constitutive model parameters
    model_parameters = \
        {'elastic_symmetry': 'isotropic',
         'E': 110e3, 'v': 0.33,
         'euler_angles': (0.0, 0.0, 0.0),
         'hardening_law': get_hardening_law('nadai_ludwik'),
         'hardening_parameters': {'s0': 900,
                                  'a': 700,
                                  'b': 0.5,
                                  'ep0': 1e-5},
        'a_hardening_law': get_hardening_law('linear'),
        'a_hardening_parameters': {'s0': math.sqrt(3),
                                   'a': 0},
        'b_hardening_law': get_hardening_law('linear'),
        'b_hardening_parameters': {'s0': 0.03072711378335953,
                                   'a': 0},
        'c_hardening_law': get_hardening_law('linear'),
        'c_hardening_parameters': {'s0': -2.148874580860138,
                                   'a': 0},
        'd_hardening_law': get_hardening_law('linear'),
        'd_hardening_parameters': {'s0': 0.11660067737102509,
                                   'a': 0},
        'is_associative_hardening': True}
        
        
        
    model_parameters = \
        {'elastic_symmetry': 'isotropic',
         'E': 110e3, 'v': 0.33,
         'euler_angles': (0.0, 0.0, 0.0),
         'hardening_law': get_hardening_law('nadai_ludwik'),
         'hardening_parameters': {'s0': 900,
                                  'a': 700,
                                  'b': 0.5,
                                  'ep0': 1e-5},
        'a_hardening_law': get_hardening_law('linear'),
        'a_hardening_parameters': {'s0': 2.0,
                                   'a': 0},
        'b_hardening_law': get_hardening_law('linear'),
        'b_hardening_parameters': {'s0': 0.05,
                                   'a': 0},
        'c_hardening_law': get_hardening_law('linear'),
        'c_hardening_parameters': {'s0': 0.5,
                                   'a': 0},
        'd_hardening_law': get_hardening_law('linear'),
        'd_hardening_parameters': {'s0': 0.5,
                                   'a': 0},
        'is_associative_hardening': True}

    # Initialize constitutive model
    constitutive_model = LouZhangYoon(strain_formulation, problem_type,
                                      model_parameters)
    # Store constitutive model
    constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set constitutive model label
    model_label = 'lou_zhang_yoon_vmap'
    # Set constitutive model parameters
    model_parameters2 = \
        {'elastic_symmetry': 'isotropic',
         'E': 110e3, 'v': 0.33,
         'euler_angles': (0.0, 0.0, 0.0),
         'hardening_law': get_hardening_law('nadai_ludwik'),
         'hardening_parameters': {'s0': 900,
                                  'a': 700,
                                  'b': 0.5,
                                  'ep0': 1e-5},
        'a_hardening_law': get_hardening_law('linear'),
        'a_hardening_parameters': {'s0': 2.416022926568985,
                                   'a': 0},
        'b_hardening_law': get_hardening_law('linear'),
        'b_hardening_parameters': {'s0': 0.03072711378335953,
                                   'a': 0},
        'c_hardening_law': get_hardening_law('linear'),
        'c_hardening_parameters': {'s0': -2.148874580860138,
                                   'a': 0},
        'd_hardening_law': get_hardening_law('linear'),
        'd_hardening_parameters': {'s0': 0.11660067737102509,
                                   'a': 0},
        'is_associative_hardening': True}

    #2.416022926568985
    #0.03072711378335953
    #-2.148874580860138
    #0.11660067737102509

    
        
    # Initialize constitutive model
    constitutive_model = LouZhangYoonVMAP(strain_formulation, problem_type,
                                          model_parameters)
    # Store constitutive model
    constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reference stress path
    if is_reference_stress_path:
        ref_stress_path = stress_path
    else:
        ref_stress_path = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare constitutive models stress predictions
    compare_material_model_response(strain_formulation, problem_type,
                                    strain_comps_order, time_hist, strain_path,
                                    constitutive_models=constitutive_models,
                                    ref_stress_path=ref_stress_path,
                                    save_dir=plots_dir, is_save_fig=True,
                                    is_stdout_display=False, is_latex=True)
    


