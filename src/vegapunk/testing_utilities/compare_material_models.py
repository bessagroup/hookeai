# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from projects.darpa_metals.rnn_material_model.user_scripts. \
    gen_response_dataset import MaterialResponseDatasetGenerator
from time_series_data.time_dataset import load_dataset
from simulators.fetorch.material.models.standard.von_mises import VonMises
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
                            'is_stress_path_fail': is_stress_path_fail}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store model prediction data
        models_predictions[model_name] = model_prediction
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem type parameters
    _, comp_order_sym, _ = get_problem_type_parameters(problem_type)
    # Get number of constitutive models
    n_model = len(constitutive_models.keys())
    # Get number of discrete times
    n_time = len(time_hist)
    # Loop over material constitutive models
    for comp_label in comp_order_sym:
        # Initialize stress data array
        stress_data_xy = np.zeros((n_time, 2*n_model))
        # Initialize data labels
        data_labels = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material constitutive models
        for i, model_name in enumerate(constitutive_models.keys()):
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
# =============================================================================
if __name__ == "__main__":
    # Set plots directory
    plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'darpa_project/9_local_lou_rc_training/testing')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain path source
    strain_path_source = 'dataset'
    # Get strain path data
    if strain_path_source == 'dataset':
        # Set data set file path
        dataset_file_path = ('/home/bernardoferreira/Documents/brown/'
                             'projects/darpa_project/9_local_lou_rc_training/'
                             '0_standard_training/n20/1_training_dataset/'
                             'ss_paths_dataset_n20.pkl')
        # Set sample index
        sample_idx = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data set
        dataset = load_dataset(dataset_file_path)
        # Extract sample material response path
        response_path = dataset[sample_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect strain path
        strain_comps_order = response_path['strain_comps_order']
        strain_path = response_path['strain_path']
        time_hist = response_path['time_hist']
    else:
        raise RuntimeError('Unknown strain path source.')
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
        'E': 90e3, 'v': 0.33,
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
    model_label = 'lou_zhang_yoon'
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
        'a_hardening_parameters': {'s0': 1.0,
                                   'a': 0},
        'b_hardening_law': get_hardening_law('linear'),
        'b_hardening_parameters': {'s0': 0.05,
                                   'a': 0},
        'c_hardening_law': get_hardening_law('linear'),
        'c_hardening_parameters': {'s0': 1.5,
                                   'a': 0},
        'd_hardening_law': get_hardening_law('linear'),
        'd_hardening_parameters': {'s0': 0.75,
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
        'a_hardening_parameters': {'s0': 1.0,
                                   'a': 0},
        'b_hardening_law': get_hardening_law('linear'),
        'b_hardening_parameters': {'s0': 0.05,
                                   'a': 0},
        'c_hardening_law': get_hardening_law('linear'),
        'c_hardening_parameters': {'s0': 1.5,
                                   'a': 0},
        'd_hardening_law': get_hardening_law('linear'),
        'd_hardening_parameters': {'s0': 0.75,
                                   'a': 0},
        'is_associative_hardening': True}
    # Initialize constitutive model
    constitutive_model = LouZhangYoonVMAP(strain_formulation, problem_type,
                                          model_parameters)
    # Store constitutive model
    constitutive_models[model_label] = constitutive_model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare constitutive models stress predictions
    compare_material_model_response(strain_formulation, problem_type,
                                    strain_comps_order, time_hist, strain_path,
                                    constitutive_models=constitutive_models,
                                    save_dir=plots_dir, is_save_fig=True,
                                    is_stdout_display=False, is_latex=True)
    


