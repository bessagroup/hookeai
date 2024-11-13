"""Build stress-based features for time series data set.

Functions
---------
add_stress_features(dataset, stress_feature_label)
    Add new stress-based feature history in time series data set.
compute_stressfeature(stress_comps_array, stress_feature_label, n_dim,
                      stress_comps_order, device=None)
    Compute stress-based feature.
build_stress_from_comps(n_dim, stress_comps_order, stress_comps_array,
                        device=None)
    Build stress tensor from given components.
"""
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
import itertools
# Third-party
import torch
# Local
from rnn_base_model.data.time_dataset import TimeSeriesDataset, load_dataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def add_stress_features(dataset, stress_feature_label):
    """Add new stress-based feature history in time series data set.
    
    The computation of a new stress-based feature requires that 'stress_path'
    is available as an existing feature of the time series data set.
    
    The new stress-based feature history is stored as a torch.Tensor(2d) of
    shape (sequence_length, n_features), where n_features is the corresponding
    dimensionality.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    stress_feature_label : {'vol_stress', 'dev_stress'}
        Stress-based feature:
        
        'vol_stress' : Volumetric (or hydrostatic) stress.
        
        'dev_stress' : Deviatoric component of stress tensor.
    
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    """
    # Probe first data set sample
    sample = dataset[0]
    # Check if required data is available in data set
    if 'stress_path' not in sample.keys():
        raise RuntimeError(f'The feature \'stress_path\' must be available '
                           f'in the data set in order to compute a new '
                           f'stress-based feature.')
    elif 'stress_comps_order' not in sample.keys():
        raise RuntimeError(f'The data \'stress_comps_order\' must be '
                           f'available in the data set in order to compute a '
                           f'new stress-based feature.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get stress components order
    stress_comps_order = sample['stress_comps_order']
    # Infer number of spatial dimensions from stress components
    if len(stress_comps_order) in (3, 4):
        n_dim = 2
    else:
        n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set available stress-based features
    available_stress_features = ('vol_stress', 'dev_stress')
    # Check if available stress-based feature
    if stress_feature_label not in available_stress_features:
        raise RuntimeError(
            f'Unavailable stress-based feature \'{stress_feature_label}\'.\n\n'
            f'Available stress-based features: {available_stress_features}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract sample stress path
        stress_path = sample['stress_path']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized stress-based feature history computation (batch along
        # time)
        vmap_compute_stress_feature = \
            torch.vmap(compute_stress_feature,
                       in_dims=(0, None, None, None), out_dims=(0,))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get stress-based feature history
        stress_feature_path = vmap_compute_stress_feature(
            stress_path, stress_feature_label, n_dim, stress_comps_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set stress-based feature history
        sample[stress_feature_label] = stress_feature_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data set sample
        if isinstance(dataset, TimeSeriesDataset):
            dataset.update_dataset_sample(i, sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def compute_stress_feature(stress_comps_array, stress_feature_label, n_dim,
                           stress_comps_order, device=None):
    """Compute stress-based feature.
    
    Parameters
    ----------
    stress_comps_array : torch.Tensor(1d)
        Strain components array.
    stress_feature_label : {'vol_stress', 'dev_stress'}
        Stress-based feature:
        
        'vol_stress' : Volumetric (or hydrostatic) stress.
        
        'dev_stress' : Deviatoric component of stress tensor.
 
    n_dim : int
        Number of spatial dimensions.
    stress_comps_order : tuple[str]
        Strain components order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
        
    Returns
    -------
    stress_feature : torch.Tensor(1d)
        Stress-based feature.
    """
    # Build stress tensor
    stress = build_stress_from_comps(n_dim, stress_comps_order,
                                     stress_comps_array)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stress-based feature
    if stress_feature_label == 'vol_stress':
        # Check number of spatial dimensions
        if n_dim != 3:
            raise RuntimeError('Volumetric (or hydrostatic) stress can only '
                               'be computed from three-dimensional stress '
                               'tensor.')
        # Compute volumetric (or hydrostatic) stress
        stress_feature = torch.trace(stress)
    elif stress_feature_label == 'dev_stress':
        # Check number of spatial dimensions
        if n_dim != 3:
            raise RuntimeError('Deviatoric stress component can only be '
                               'computed from three-dimensional stress '
                               'tensor.')
        # Compute volumetric (or hydrostatic) stress
        vol_stress = torch.trace(stress)
        # Compute deviatoric stress
        dev_stress = stress - vol_stress*torch.eye(n_dim)
        # Store deviatoric stress components
        stress_feature = store_stress_comps(stress_comps_order, dev_stress)
    else:
        raise RuntimeError('Unknown stress-based feature.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce stress-based feature 1d tensor
    stress_feature = stress_feature.view(-1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_feature
# =============================================================================
def build_stress_from_comps(n_dim, stress_comps_order, stress_comps_array,
                            device=None):
    """Build stress tensor from given components.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    stress_comps_order : tuple[str]
        Stress components order.
    stress_comps_array : torch.Tensor(1d)
        Stress components array.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
    
    Returns
    -------
    stress : torch.Tensor(2d)
        Stress tensor.
    """
    # Get device from input tensor
    if device is None:
        device = stress_comps_array.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set row major components order       
    row_major_order = tuple(f'{i + 1}{j + 1}' for i, j
                            in itertools.product(range(n_dim), repeat=2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = [stress_comps_order.index(x) if x in stress_comps_order
                 else stress_comps_order.index(x[::-1])
                 for x in row_major_order]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build stress tensor
    stress = stress_comps_array[index_map].view(n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress
# =============================================================================
def store_stress_comps(stress_comps_order, stress, device=None):
    """Store stress tensor components in array.
    
    Parameters
    ----------
    stress_comps_order : tuple[str]
        Strain components order.
    stress : torch.Tensor(2d)
        Stress tensor.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
    
    Returns
    -------
    stress_comps_array : torch.Tensor(1d)
        Stress components array.
    """
    # Get device from input tensor
    if device is None:
        device = stress.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = tuple(
        [int(x[i]) - 1 for x in stress_comps_order] for i in range(2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build tensor components array
    stress_comps_array = stress[index_map]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return stress_comps_array
# =============================================================================
if __name__ == '__main__':
    # Set data set file path
    dataset_file_path = ('/home/bernardoferreira/Desktop/test/'
                         '1_training_dataset/ss_paths_dataset_n10.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over stress-based features
    for stress_feature_label in ('vol_stress', 'dev_stress'):
        # Add stress-based feature to data set
        dataset = add_stress_features(dataset, stress_feature_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe first data set sample
    sample = dataset[0]
    # Get stress path
    output = sample['stress_path']
    output_labels = ['stress_path']
    # Loop over stress-based features
    for stress_feature_label in ('vol_stress', 'dev_stress'):
        output = torch.cat((output, sample[stress_feature_label]), dim=1)
        output_labels.append(stress_feature_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output first ten time steps
    torch.set_printoptions(linewidth=1000)
    print(output_labels)
    print(output[0:min(10, output.shape[0]), :])