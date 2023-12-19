"""User script: Generate GNN-based data sets."""
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
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset, split_dataset
from projects.shell_knock_down.gnn_model_tools.gen_graphs_files import \
    generate_dataset_samples_files
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
def generate_dataset(case_study_name, dataset_file_path, dataset_directory,
                     is_verbose=False):
    """Generate data sets.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    dataset_csv_file_path : str
        Data set csv file path.
    dataset_directory : str
        Directory where the data set is stored (all ata set samples files).
        All existent files are overridden when saving sample data files.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset : GNNGraphDataset
        Graph Neural Network graph data set.
    dataset_file_path : str
        Graph Neural Network graph data set file path.
    """
    # Set default files and directories storage options
    sample_file_basename, is_save_sample_plot = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data set samples files
    dataset_directory, dataset_samples_files = generate_dataset_samples_files(
        dataset_directory, dataset_csv_file_path,
        sample_file_basename=sample_file_basename,
        is_save_sample_plot=is_save_sample_plot, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based data set
    dataset = GNNGraphDataset(dataset_directory, dataset_samples_files,
                              dataset_basename='graph_dataset',
                              is_store_dataset=False)
    # Save GNN-based data set to file
    dataset_file_path = dataset.save_dataset(is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset, dataset_file_path
# =============================================================================
def set_default_saving_options():
    """Set default files and directories storage options.
    
    Returns
    -------
    sample_file_basename : str
        Basename of data set sample file. The basename is appended with sample
        index.
    is_save_sample_plot : bool
        Save plot of each sample graph in the same directory where the data set
        is stored.
    """
    sample_file_basename = 'shell_graph'
    is_save_sample_plot = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_file_basename, is_save_sample_plot
# =============================================================================
if __name__ == "__main__":
    # Set training/testing data set flag
    is_testing_dataset = False
    # Set computation processes
    is_generate_dataset = True
    is_split_dataset = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file path (shells .csv file)
    dataset_csv_file_path = \
        ('/home/bernardoferreira/Documents/brown/projects/'
         'shell_knock_down/datasets_files/shells_full.csv')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'shell_knock_down/case_studies/')
    # Set case study directory
    case_study_name = 'full'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set simulation directory
    if is_testing_dataset:
        # Set testing data set directory
        testing_dataset_dir = os.path.join(os.path.normpath(case_study_dir),
                                           '4_testing_dataset')
        # Check testing data set directory
        if not os.path.isdir(testing_dataset_dir):
            raise RuntimeError('The case study testing data set directory has '
                               'not been found:\n\n' + testing_dataset_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set directory
    if is_testing_dataset:
        # Set data set directory (testing data set)
        dataset_directory = testing_dataset_dir
    else:
        # Set data set directory (training data set)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '1_training_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data set
    if is_generate_dataset or is_testing_dataset:
        # Create data set directory
        if not is_testing_dataset:
            make_directory(dataset_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate data set
        dataset, _ = generate_dataset(case_study_name, dataset_csv_file_path,
                                      dataset_directory, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split data set
    if is_split_dataset and not is_testing_dataset:
        # Get training data set file path
        regex = r'^graph_dataset_n[0-9]+.pkl$'
        is_file_found, dataset_file_path = \
            find_unique_file_with_regex(dataset_directory, regex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check training data set file
        if not is_file_found:
            raise RuntimeError('Training data set file has not been found in '
                               'directory:\n\n' + dataset_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load training data set
        dataset = GNNGraphDataset.load_dataset(dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set split sizes
        split_sizes = {'training': 0.8, 'testing': 0.2}
        # Split data set
        dataset_split = \
            split_dataset(dataset, split_sizes, is_save_subsets=True,
                          subsets_basename=dataset.get_dataset_basename(),
                          subsets_directory=dataset.get_dataset_directory())

