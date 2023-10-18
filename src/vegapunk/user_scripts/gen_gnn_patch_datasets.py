"""User script: Generate GNN-based material patch data sets."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import os
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
# Local
from material_patch.patch_dataset import read_simulation_dataset_from_file
from gnn_model.gnn_patch_dataset import generate_dataset_samples_files, \
    GNNMaterialPatchDataset, split_dataset
from ioput.iostandard import make_directory
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
    """
    # Set default files and directories storage options
    sample_file_basename, is_save_plot_patch = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if case_study_name == '2d_elastic':
        # Load material patch simulation data set
        dataset_simulation_data = \
            read_simulation_dataset_from_file(sim_dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown case study.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate GNN-based material patch data set samples files
    dataset_directory, dataset_samples_files = generate_dataset_samples_files(
        dataset_directory, dataset_simulation_data,
        sample_file_basename=sample_file_basename,
        is_save_plot_patch=is_save_plot_patch, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch data set
    dataset = GNNMaterialPatchDataset(dataset_directory, dataset_samples_files,
                                      is_store_dataset=False)
    # Save GNN-based material patch data set to file
    dataset.save_dataset(is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
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
    # Set case study name
    case_study_name = '2d_elastic'
    # Set case study directory
    case_study_base_dirs = {
        '2d_elastic': f'/home/bernardoferreira/Documents/temp',}
    case_study_dir = \
        os.path.join(os.path.normpath(case_study_base_dirs[case_study_name]),
                     f'cs_{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material patch simulation data set size
    n_sample = 5
    # Set material patch simulation data set file path
    sim_dataset_file_path = os.path.join(
        os.path.normpath(case_study_dir),
        f'0_simulation/material_patch_fem_dataset_n{n_sample}.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set directory
    dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                     '1_dataset')
    # Create data set directory
    make_directory(dataset_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate GNN-based material patch data set
    dataset = generate_dataset(case_study_name, sim_dataset_file_path,
                               dataset_directory, is_verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Randomly split data set into training, validation and testing
    split_sizes = {'training': 0.6, 'validation': 0.2, 'testing': 0.2}
    # Split data set
    dataset_split = \
        split_dataset(dataset, split_sizes, is_save_subsets=True,
                      subsets_basename=dataset.get_dataset_basename(),
                      subsets_directory=dataset.get_dataset_directory())

