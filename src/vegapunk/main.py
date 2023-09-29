"""Test script: GNN-based finite element material patch."""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    # Set vegapunk directory
    vegapunk_dir = '/home/bernardoferreira/Documents/repositories/' \
        'gen_material_patch_data/src/vegapunk/'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate datasets    
    os.system('python3 '
              + os.path.join(vegapunk_dir,
                             'user_scripts/gen_patch_simulation_datasets.py'))
    os.system('python3 '
              + os.path.join(vegapunk_dir,
                             'user_scripts/gen_gnn_patch_datasets.py'))






