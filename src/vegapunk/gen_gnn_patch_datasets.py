"""User script: Generate GNN-based material patch data sets."""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import os
# Third-party
import numpy as np
# Local
from material_patch.patch_dataset import read_simulation_dataset_from_file
from gnn_model.gnn_patch_dataset import generate_dataset_samples_files, \
    get_dataset_sample_files_from_dir, GNNMaterialPatchDataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def generate_dataset(sim_dataset_type, simulation_directory, dataset_directory,
                     is_verbose=False):
    """Generate GNN-based material patch data sets.
    
    Parameters
    ----------
    sim_dataset_type : str
        Material patch simulation data set type.
    simulation_directory : str
        Directory where files associated with the generation of the material
        patch finite element simulations dataset are written. All existent
        files are overridden when saving new data files.
    dataset_directory : str
        Directory where the GNN-based material patch data set is stored (all
        data set samples files). All existent files are overridden when saving
        sample data files.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Set default files and directories storage options
    sample_file_basename, is_save_plot_patch = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if sim_dataset_type == '2d_elastic':
        # Set material patch simulation data set file path
        sim_dataset_file_path = os.path.join(simulation_directory,
            'material_patch_fem_dataset.pkl')
        # Load material patch simulation data set
        dataset_simulation_data = \
            read_simulation_dataset_from_file(sim_dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Unknown material patch simulation data set type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate GNN-based material patch data set samples files
    dataset_directory, dataset_samples_files = generate_dataset_samples_files(
        dataset_directory, dataset_simulation_data,
        sample_file_basename=sample_file_basename,
        is_save_plot_patch=is_save_plot_patch, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize GNN-based material patch data set
    dataset = GNNMaterialPatchDataset(dataset_directory, dataset_samples_files)
    # Save GNN-based material patch data set to file
    dataset.save_dataset()
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
    # Available material patch simulation data set types:
    #
    # ['2d_elastic'] - 2D material patch, homogeneous, elastic
    #
    available_sim_dataset_types = {}
    available_sim_dataset_types['2d_elastic'] = \
        '/home/bernardoferreira/Documents/temp'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set simulation data set type
    sim_dataset_type = '2d_elastic'
    simulation_directory = available_sim_dataset_types['2d_elastic']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set directory
    dataset_directory = simulation_directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate material patch simulation data set
    generate_dataset(sim_dataset_type, simulation_directory, dataset_directory,
                     is_verbose=True)

