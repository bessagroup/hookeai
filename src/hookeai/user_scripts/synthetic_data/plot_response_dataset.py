"""Generate plots of strain-stress material response data set."""
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
# Local
from time_series_data.time_dataset import load_dataset, save_dataset, \
    TimeSeriesDatasetInMemory
from ioput.iostandard import make_directory
from user_scripts.synthetic_data.gen_response_dataset import \
    generate_dataset_plots, MaterialResponseDatasetGenerator
from simulators.fetorch.material.models.standard.hardening import \
    get_hardening_law
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    # Set computation processes
    process = ('plot_response_dataset',
               'recompute_dataset_response_paths')[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file path
    dataset_file_path = (
        '/home/username/Documents/brown/projects/'
        'colaboration_antonios/dtp_validation/4_dtp1_exp_rowan_data/'
        '0_DTP1U_Case_2/1_adimu_forward_hexa8_1GP_synthetic_parameters/'
        '2_discover_rc_von_mises_adimu_force_displacement_test/'
        'material_model_finder/3_model/local_response_dataset/'
        'augmented_response_dataset/ss_paths_dataset_n1483.pkl')
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if process in ('plot_response_dataset',
                   'recompute_dataset_response_paths'):
        if process == 'plot_response_dataset':
            # Load data set
            dataset = load_dataset(dataset_file_path)
            # Set plots directory
            plots_dir = \
                os.path.join(os.path.dirname(dataset_file_path), 'plots')
        elif process == 'recompute_dataset_response_paths':
            # Initialize material response path data set generator
            dataset_generator = MaterialResponseDatasetGenerator(
                strain_formulation, problem_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set constitutive model name
            model_name = 'von_mises'
            # Set constitutive model parameters
            model_parameters = {
                'elastic_symmetry': 'isotropic',
                'E': 118640, 'v': 0.334,
                'euler_angles': (0.0, 0.0, 0.0),
                'hardening_law': get_hardening_law('nadai_ludwik'),
                'hardening_parameters':
                    {'s0': 823.14484,
                     'a': 522.23411,
                     'b': 0.375146,
                     'ep0': 1e-5}}
            # Set constitutive state variables to be additionally included in
            # the data set
            state_features = {'acc_p_strain': 1,}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate strain-stress material response data set from strain
            # data set
            dataset = \
                dataset_generator.gen_response_dataset_from_strain_dataset(
                    dataset_file_path, model_name=model_name,
                    model_parameters=model_parameters,
                    state_features=state_features, is_verbose=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set storage directory
            dataset_save_dir = os.path.join(os.path.dirname(dataset_file_path),
                                            'augmented_response_dataset')
            # Create data set storage directory
            dataset_save_dir = make_directory(dataset_save_dir,
                                              is_overwrite=True)
            # Set data set basename
            dataset_basename = 'ss_paths_dataset'
            # Save data set
            dataset_file_path = save_dataset(dataset, dataset_basename,
                                             dataset_save_dir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set data set plots directory
            plots_dir = os.path.join(dataset_save_dir, 'plots')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Select specific paths from the data set
        is_plot_subset_only = False
        if is_plot_subset_only:
            # Set paths indexes to be plotted
            n_gauss = 1
            paths_indexes = \
                [(element_id - 1)*n_gauss for element_id in range(1, 17)]
            # Extract paths from data set
            dataset_samples = [dataset[k] for k in paths_indexes]
            # Create new data set with selected paths only
            dataset = TimeSeriesDatasetInMemory(dataset_samples)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create plots directory
        plots_dir = make_directory(plots_dir, is_overwrite=True)
        # Generate data set plots
        generate_dataset_plots(strain_formulation, n_dim, dataset,
                               save_dir=plots_dir, is_save_fig=True,
                               is_stdout_display=False)