"""Build strain-based features for time series data set.

Functions
---------
add_strain_features(dataset, strain_feature_label)
    Add new strain-based feature history in time series data set.
compute_strain_feature(strain_comps_array, strain_feature_label, n_dim,
                       strain_comps_order, device=None)
    Compute strain-based feature.
build_strain_from_comps(n_dim, strain_comps_order, strain_comps_array,
                        device=None)
    Build strain tensor from given components.
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
def add_strain_features(dataset, strain_feature_label):
    """Add new strain-based feature history in time series data set.
    
    The computation of a new strain-based feature requires that 'strain_path'
    is available as an existing feature of the time series data set.
    
    The new strain-based feature history is stored as a torch.Tensor(2d) of
    shape (sequence_length, n_features), where n_features is the corresponding
    dimensionality.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Time series data set. Each sample is stored as a dictionary where
        each feature (key, str) data is a torch.Tensor(2d) of shape
        (sequence_length, n_features).
    strain_feature_label : {'i1_strain', 'i2_strain'}
        Strain-based feature:
        
        'i1_strain' : First (principal) invariant of strain tensor.
        
        'i2_strain' : Second (principal) invariant of strain tensor.
    
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
    if 'strain_path' not in sample.keys():
        raise RuntimeError(f'The feature \'strain_path\' must be available '
                           f'in the data set in order to compute a new '
                           f'strain-based feature.')
    elif 'strain_comps_order' not in sample.keys():
        raise RuntimeError(f'The data \'strain_comps_order\' must be '
                           f'available in the data set in order to compute a '
                           f'new strain-based feature.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get strain components order
    strain_comps_order = sample['strain_comps_order']
    # Infer number of spatial dimensions from strain components
    if len(strain_comps_order) in (3, 4):
        n_dim = 2
    else:
        n_dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set available strain-based features
    available_strain_features = ('i1_strain', 'i2_strain')
    # Check if available strain-based feature
    if strain_feature_label not in available_strain_features:
        raise RuntimeError(
            f'Unavailable strain-based feature \'{strain_feature_label}\'.\n\n'
            f'Available strain-based features: {available_strain_features}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract sample strain path
        strain_path = sample['strain_path']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized strain-based feature history computation (batch along
        # time)
        vmap_compute_strain_feature = \
            torch.vmap(compute_strain_feature,
                       in_dims=(0, None, None, None), out_dims=(0,))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain-based feature history
        strain_feature_path = vmap_compute_strain_feature(
            strain_path, strain_feature_label, n_dim, strain_comps_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain-based feature history
        sample[strain_feature_label] = strain_feature_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update data set sample
        if isinstance(dataset, TimeSeriesDataset):
            dataset.update_dataset_sample(i, sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset
# =============================================================================
def compute_strain_feature(strain_comps_array, strain_feature_label, n_dim,
                           strain_comps_order, device=None):
    """Compute strain-based feature.
    
    Parameters
    ----------
    strain_comps_array : torch.Tensor(1d)
        Strain components array.
    strain_feature_label : {'i1_strain', 'i2_strain'}
        Strain-based feature:
        
        'i1_strain' : First (principal) invariant of strain tensor.
        
        'i2_strain' : Second (principal) invariant of strain tensor.
 
    n_dim : int
        Number of spatial dimensions.
    strain_comps_order : tuple[str]
        Strain components order.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
        
    Returns
    -------
    strain_feature : torch.Tensor(1d)
        Strain-based feature.
    """
    # Build strain tensor
    strain = build_strain_from_comps(n_dim, strain_comps_order,
                                     strain_comps_array)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute strain-based feature
    if strain_feature_label == 'i1_strain':
        # Check number of spatial dimensions
        if n_dim != 3:
            raise RuntimeError('First (principal) invariant can only be '
                               'computed from three-dimensional strain '
                               'tensor.')
        # Compute first (principal) invariant
        strain_feature = torch.trace(strain)
    elif strain_feature_label == 'i2_strain':
        # Check number of spatial dimensions
        if n_dim != 3:
            raise RuntimeError('Second (principal) invariant can only be '
                               'computed from three-dimensional strain '
                               'tensor.')
        # Compute second (principal) invariant
        strain_feature = 0.5*(torch.trace(strain)**2
                              - torch.trace(torch.matmul(strain, strain)))
    else:
        raise RuntimeError('Unknown strain-based feature.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Enforce strain-based feature 1d tensor
    strain_feature = strain_feature.view(-1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain_feature
# =============================================================================
def build_strain_from_comps(n_dim, strain_comps_order, strain_comps_array,
                            device=None):
    """Build strain tensor from given components.
    
    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    strain_comps_order : tuple[str]
        Strain components order.
    strain_comps_array : torch.Tensor(1d)
        Strain components array.
    device : torch.device, default=None
        Device on which torch.Tensor is allocated.
    
    Returns
    -------
    strain : torch.Tensor(2d)
        Strain tensor.
    """
    # Get device from input tensor
    if device is None:
        device = strain_comps_array.device
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set row major components order       
    row_major_order = tuple(f'{i + 1}{j + 1}' for i, j
                            in itertools.product(range(n_dim), repeat=2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build indexing mapping
    index_map = [strain_comps_order.index(x) if x in strain_comps_order
                 else strain_comps_order.index(x[::-1])
                 for x in row_major_order]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build strain tensor
    strain = strain_comps_array[index_map].view(n_dim, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return strain
# =============================================================================
if __name__ == '__main__':
    # Set data set file path
    dataset_file_path = ('/home/bernardoferreira/Desktop/test/'
                         '1_training_dataset/ss_paths_dataset_n10.pkl')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data set
    dataset = load_dataset(dataset_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over strain-based features
    for strain_feature_label in ('i1', 'i2'):
        # Add strain-based feature to data set
        dataset = add_strain_features(dataset, strain_feature_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe first data set sample
    sample = dataset[0]
    # Get strain path
    output = sample['strain_path']
    output_labels = ['strain_path']
    # Loop over strain-based features
    for strain_feature_label in ('i1', 'i2'):
        output = torch.cat((output, sample[strain_feature_label]), dim=1)
        output_labels.append(strain_feature_label)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output first ten time steps
    torch.set_printoptions(linewidth=1000)
    print(output_labels)
    print(output[0:min(10, output.shape[0]), :])