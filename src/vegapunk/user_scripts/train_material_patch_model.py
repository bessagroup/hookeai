"""User script: Train Graph Neural Network based material patch model."""
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
# Third-party
import numpy as np
# Local
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
def perform_model_standard_training(case_study_name, dataset_file_path,
                                    model_directory, is_verbose=False):
    """Perform standard training of GNN-based material patch model.
    
    Parameters
    ----------
    case_study_name : str
        Case study.
    dataset_file_path : str
        GNN-based material patch data set file path.        
    model_directory : str
        Directory where material patch model is stored.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    pass
# =============================================================================
if __name__ == "__main__":
    # Set case study name
    case_study_name = '2d_elastic'
    # Set case study base directory
    case_study_dir = \
        f'/home/bernardoferreira/Documents/temp/cs_{case_study_name}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch data set size
    n_sample = 5
    # Set GNN-based material patch data set file path
    dataset_file_path = os.path.join(
        os.path.normpath(case_study_dir),
        f'1_dataset/material_patch_graph_dataset_n{n_sample}.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set GNN-based material patch model directory
    model_directory = os.path.join(os.path.normpath(case_study_dir),
                                   '2_model')
    # Create model directory
    if not os.path.isdir(model_directory):
        make_directory(model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform standard training of GNN-based material patch model
    perform_model_standard_training(case_study_name, dataset_file_path,
                                    model_directory, is_verbose=True)