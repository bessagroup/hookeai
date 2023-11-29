"""User script: Generate GNN-based material patch data sets."""
#
#                                                                       Modules
# =============================================================================
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
# Local
from material_patch.patch_dataset import read_simulation_dataset_from_file
from gnn_model.gnn_patch_dataset import generate_dataset_samples_files, \
    GNNMaterialPatchDataset, split_dataset
from ioput.iostandard import make_directory, find_unique_file_with_regex
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def generate_dataset(case_study_name, sim_dataset_file_path, dataset_directory,
                     is_verbose=False):
    """Generate GNN-based material patch data sets.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    sim_dataset_file_path : str
        Material patches finite element simulations data set file path.        
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files). All existent files are overridden when saving
        sample data files.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    dataset : GNNMaterialPatchDataset
        GNN-based material patch data set.
    dataset_file_path : str
        GNN-based material patch data set file path.
    """
    # Set default files and directories storage options
    sample_file_basename, is_save_plot_patch = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    # Load material patch simulation data set
    dataset_simulation_data = \
        read_simulation_dataset_from_file(sim_dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set node and edge features
    if case_study_name == 'cs_0_2d_elastic_complete_basis':
        # Set node features
        node_features = ('coord_hist', 'disp_hist')
        # Set edge features
        edge_features = ('edge_vector', 'edge_vector_norm', 'relative_disp',
                         'relative_disp_norm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate GNN-based material patch data set samples files
    dataset_directory, dataset_samples_files = generate_dataset_samples_files(
        dataset_directory, node_features, edge_features,
        dataset_simulation_data, sample_file_basename=sample_file_basename,
        is_save_plot_patch=is_save_plot_patch, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch data set
    dataset = GNNMaterialPatchDataset(dataset_directory, dataset_samples_files,
                                      is_store_dataset=False)
    # Save GNN-based material patch data set to file
    dataset_file_path = dataset.save_dataset(is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset, dataset_file_path
# =============================================================================
def set_default_saving_options():
    """Set default files and directories storage options.
    
    Returns
    -------
    sample_file_basename : str, default='material_patch_graph'
        Basename of GNN-based material patch data set sample file. The basename
        is appended with sample index.
    is_save_plot_patch : bool, default=False
        Save plot of each material patch design sample in the same directory
        where the GNN-based material patch data set is stored.
    """
    sample_file_basename = 'material_patch_graph'
    is_save_plot_patch = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_file_basename, is_save_plot_patch
# =============================================================================
if __name__ == "__main__":
    # Set training/testing data set flag
    is_testing_dataset = False
    # Set computation processes
    is_generate_dataset = True
    is_split_dataset = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/home/bernardoferreira/Documents/brown/projects/'
                'gnn_material_patch/case_studies/')
    # Set case study directory
    case_study_name = 'cs_0_2d_elastic_complete_basis'
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set simulation directory (testing data set)
        simulation_directory = os.path.join(
            os.path.normpath(testing_dataset_dir), 'simulation')
    else:
        # Set simulation directory (training data set)
        simulation_directory = os.path.join(os.path.normpath(case_study_dir),
                                            '0_simulation')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material patch simulation data set file path
    regex = r'^material_patch_fem_dataset_n[0-9]+.pkl$'
    is_file_found, sim_dataset_file_path = \
        find_unique_file_with_regex(simulation_directory, regex)
    # Check data set file
    if not is_file_found:
        raise RuntimeError(f'Simulation data set file has not been found  '
                            f'in data set directory:\n\n'
                            f'{simulation_directory}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set directory
    if is_testing_dataset:
        # Set data set directory (testing data set)
        dataset_directory = testing_dataset_dir
    else:
        # Set data set directory (training data set)
        dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                         '1_training_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate GNN-based material patch data set
    if is_generate_dataset or is_testing_dataset:
        # Create data set directory
        if not is_testing_dataset:
            make_directory(dataset_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate GNN-based material patch data set
        dataset, _ = generate_dataset(case_study_name, sim_dataset_file_path,
                                      dataset_directory, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split data set
    if is_split_dataset and not is_testing_dataset:
        # Get GNN-based material patch training data set file path
        regex = r'^material_patch_graph_dataset_n[0-9]+.pkl$'
        is_file_found, dataset_file_path = \
            find_unique_file_with_regex(dataset_directory, regex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load GNN-based material patch training data set
        dataset = GNNMaterialPatchDataset.load_dataset(dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data set split sizes
        split_sizes = {'training': 0.6, 'testing': 0.4}
        # Split data set
        dataset_split = \
            split_dataset(dataset, split_sizes, is_save_subsets=True,
                          subsets_basename=dataset.get_dataset_basename(),
                          subsets_directory=dataset.get_dataset_directory())

