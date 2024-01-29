"""User script: Generate GNN-based data sets."""
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
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset, split_dataset
from projects.abaqus_datasets.gnn_model_tools.gen_graphs_files import \
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
def generate_dataset(case_study_name, input_files_paths, data_files_paths,
                     dataset_directory, is_verbose=False, bottle_id=None):
    """Generate data sets.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    input_files_paths : list[str]
        Data set ABAQUS input data files paths ('.inp' files).
    data_files_paths : list[str]
        Data set ABAQUS data files paths ('.parquet' > '.csv' files).
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
    # Set GNN-based data set basename
    if bottle_id:
        dataset_basename = f'graph_dataset_bottle_{str(bottle_id)}'
    else:
        dataset_basename = 'graph_dataset'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default files and directories storage options
    sample_file_basename, is_save_sample_plot = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data set samples files
    dataset_directory, dataset_sample_files = generate_dataset_samples_files(
        dataset_directory, input_files_paths, data_files_paths,
        is_save_sample_plot=True, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based data set
    dataset = GNNGraphDataset(dataset_directory, dataset_sample_files,
                              dataset_basename=dataset_basename,
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
    sample_file_basename = 'sample_graph'
    is_save_sample_plot = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_file_basename, is_save_sample_plot
# =============================================================================
if __name__ == "__main__":
    # Set training/testing data set flag
    is_testing_dataset = False
    # Set computation processes
    is_generate_dataset = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set directory of ABAQUS input data files
    input_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                 'abaqus_datasets/datasets/ABAQUS_M5_buckling/'
                 'bottle_inp/')
    # Set directory of ABAQUS data files (coordinates, displacements)
    data_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'abaqus_datasets/datasets/ABAQUS_M5_buckling/'
                'bottle_disp_data/')
    # Set data files IDs
    bottle_id = None
    if is_testing_dataset:
        #bottle_id = 101
        data_files_ids = [x for x in range(bottle_id, bottle_id + 1)]
    else:
        bottle_id_init = 1
        bottle_id_end = 100
        data_files_ids = [x for x in range(bottle_id_init, bottle_id_end + 1)]
    # Initialize sets of ABAQUS data files
    input_files_paths = []
    data_files_paths = []
    # Loop over samples
    for id in data_files_ids:
        # Append ABAQUS input data file
        input_files_paths.append(os.path.join(input_dir, f'{id}.inp'))
        # Append ABAQUS data file
        data_files_paths.append(os.path.join(data_dir, f'{id}_DISP.csv'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'abaqus_datasets/case_studies/M5_buckling')
    # Set case study directory
    case_study_name = 'incremental_model'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                        + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set directory
    if is_testing_dataset:
        # Set data set directory (testing data set)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                        '4_testing_dataset')
    else:
        # Set data set directory (training data set)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                        '1_training_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data set
    if is_generate_dataset:
        # Create data set directory
        make_directory(dataset_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate data set
        dataset, _ = generate_dataset(case_study_name, input_files_paths,
                                    data_files_paths, dataset_directory,
                                    is_verbose=True, bottle_id=bottle_id)

