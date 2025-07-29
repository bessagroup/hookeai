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
from user_scripts.synthetic_data.gen_response_dataset import \
    generate_dataset_plots
from time_series_data.time_dataset import load_dataset
from ioput.iostandard import make_directory
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
    # Set data set file path
    dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                         'colaboration_antonios/dtp_validation/0_rowan_data/'
                         'Ti-6242 HIP2 DTP for Brown v4/DTP2/'
                         'ss_paths_dataset_n1386.pkl')
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set number of spatial dimensions
    n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set plots directory
    plots_dir = os.path.join(os.path.dirname(dataset_file_path), 'plots')
    # Create plots directory
    plots_dir = make_directory(plots_dir, is_overwrite=True)
    # Generate data set plots
    generate_dataset_plots(strain_formulation, n_dim, dataset,
                           save_dir=plots_dir, is_save_fig=True,
                           is_stdout_display=False)