# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import random
# Third-party
import numpy as np
# Local
from time_series_data.time_dataset import load_dataset
from rnn_base_model.predict.prediction_plots import plot_time_series_prediction
from gnn_base_model.predict.prediction import load_sample_predictions
from ioput.iostandard import make_directory
from testing_utilities.output_prediction_metrics import \
    compute_mean_prediction_metrics
# =============================================================================
# Summary: Compare models stress path prediction
# =============================================================================
def plot_models_stress_prediction(response_path, models_prediction_results,
                                  is_plot_ground_truth=True,
                                  filename='prediction_comparison',
                                  save_dir=None, is_save_fig=False,
                                  is_stdout_display=False):
    """Plot models stress path prediction for given material response path.
    
    Parameters
    ----------
    response_path : dict
        Material response path.
    models_prediction_results : dict
        Material response prediction results (item, dict) for each prediction
        model (key, str).
    is_plot_ground_truth : bool, default=True
        If True, then plot ground-truth data extracted from material response
        sample file.
    filename : str, default='prediction_comparison'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    """
    # Initialize predictions data
    prediction_sets = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get discrete time history
    time_hist = response_path['time_hist']
    # Get stress components
    stress_comps_order = response_path['stress_comps_order']
    # Get ground-truth stress path
    stress_path = response_path['stress_path'] 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over stress components
    for i, stress_comp in enumerate(stress_comps_order):
        # Assemble ground-truth stress data
        if is_plot_ground_truth:
            # Build ground-truth stress data array
            stress_data_xy = np.stack(
                (time_hist.reshape(-1), stress_path[:, i]), axis=1)
            # Assemble ground-truth stress data
            prediction_sets['Ground-truth'] = stress_data_xy
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prediction models
        for model_label, sample_results in models_prediction_results.items():
            # Get model stress path prediction
            model_stress_path = sample_results['features_out']
            # Build model stress data array
            stress_data_xy = np.stack(
                (time_hist.reshape(-1), model_stress_path[:, i]), axis=1)
            # Assemble model stress data
            prediction_sets[model_label] = stress_data_xy
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label='Time'
        y_label = 'Stress (MPa)'
        # Set plot file name
        filename_comp = filename + f'_stress_{stress_comp}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot model times series predictions against ground-truth
        plot_time_series_prediction(
            prediction_sets, is_reference_data=True,
            x_label=x_label, y_label=y_label,
            filename=filename_comp,
            save_dir=save_dir, is_save_fig=is_save_fig,
            is_stdout_display=is_stdout_display, is_latex=True)
# =============================================================================
if __name__ == '__main__':
    # Set samples indexes
    sample_ids = random.sample(range(0, 512), 20)
    # Set testing data set file path
    dataset_file_path = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'darpa_paper_examples/global/random_material_patch_von_mises/'
         'deformation_bounds_0d1/global/gru_model/'
         'lr_exp_0d001_0d00001_32000_epochs/material_model_performance/'
         '6_testing_od_dataset/ss_paths_dataset_n512.pkl')
    # Set models prediction directories
    models_prediction_dirs = {}
    models_prediction_dirs['GRU'] = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'darpa_paper_examples/global/random_material_patch_von_mises/'
         'deformation_bounds_0d1/global/'
         'gru_model/lr_exp_0d001_0d00001_32000_epochs/'
         'material_model_performance/7_prediction/out_distribution/'
         'prediction_set_1')
    models_prediction_dirs['Hybrid'] = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'darpa_paper_examples/global/random_material_patch_von_mises/'
         'deformation_bounds_0d1/global/hybrid_model/vm_candidate_1/'
         'lr_exp_0d001_0d00001_32000_epochs/material_model_performance/'
         '7_prediction/out_distribution/prediction_set_1')
    # Set plots directory
    plots_dir = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'darpa_paper_examples/global/random_material_patch_von_mises/'
         'deformation_bounds_0d1/global/local_paths_comparison_plots')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples 
    for sample_id in sample_ids:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sample plots subdirectory
        sample_plots_dir = os.path.join(os.path.normpath(plots_dir),
                                        f'sample_{sample_id}')
        # Create sample plots subdirectory
        make_directory(sample_plots_dir, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load testing data set
        dataset = load_dataset(dataset_file_path)
        # Extract sample material response path
        response_path = dataset[sample_id]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize models prediction results
        models_prediction_results = {}
        # Loop over models
        for model_label, prediction_dir in models_prediction_dirs.items():
            # Set sample prediction file path
            sample_prediction_path = \
                os.path.join(os.path.normpath(prediction_dir),
                             f'prediction_sample_{sample_id}.pkl')
            # Load material response prediction
            sample_results = load_sample_predictions(sample_prediction_path)
            # Store model material response prediction
            models_prediction_results[model_label] = sample_results
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample mean prediction metric
            mean_metrics_results = compute_mean_prediction_metrics(
                [sample_prediction_path,], ['nrmse',])
            # Display
            print(f'Sample {sample_id:4d} | {model_label:10s} | NRMSE = '
                  f'{mean_metrics_results["nrmse"]}')
        print()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot models stress path prediction
        plot_models_stress_prediction(response_path, models_prediction_results,
                                      is_plot_ground_truth=True,
                                      save_dir=sample_plots_dir,
                                      is_save_fig=True,
                                      is_stdout_display=False)