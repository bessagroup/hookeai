# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Local
from time_series_data.time_dataset import load_dataset
from utilities.fit_data_scalers import fit_data_scaler_from_dataset
from utilities.data_scalers import TorchStandardScaler, TorchMinMaxScaler
# =============================================================================
# Summary: Compute data scalers parameters from data set
# =============================================================================
# Set data set file path
dataset_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                     'darpa_paper_examples/global/'
                     'random_material_patch_von_mises/local/datasets/n10/'
                     '1_training_dataset/ss_paths_dataset_n10.pkl')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data set
dataset = load_dataset(dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize data scalers
data_scalers = {}
# Set data scaling type
scaling_type = 'mean-std'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set features type and number of features
features_type = 'strain_path'
n_features = 6
# Get scaling parameters and fit data scalers
data_scaler = fit_data_scaler_from_dataset(
    dataset, features_type, n_features, scaling_type=scaling_type)
# Store data scaler
data_scalers[features_type] = data_scaler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set features type and number of features
features_type = 'stress_path'
n_features = 6
# Get scaling parameters and fit data scalers
data_scaler = fit_data_scaler_from_dataset(
    dataset, features_type, n_features, scaling_type=scaling_type)
# Store data scaler
data_scalers[features_type] = data_scaler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\nCompute data scalers parameters'
      '\n------------------------------')
print(f'\nData set file: {dataset_file_path}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over model data scalers
for features_type, data_scaler in data_scalers.items():
    # Output data scaler normalization parameters
    if isinstance(data_scaler, TorchStandardScaler):
        # Set scaling type
        scaling_type = 'mean-std'
        # Get features standardization mean tensor
        param_1 = data_scaler._mean
        # Get features standardization standard deviation tensor
        param_2 = data_scaler._std
    elif isinstance(data_scaler, TorchMinMaxScaler):
        # Set scaling type
        scaling_type = 'min-max'
        # Get features normalization minimum tensor
        param_1 = data_scaler._minimum
        # Get features normalization maximum tensor
        param_2 = data_scaler._maximum
    else:
        raise RuntimeError('Unknown type of data scaling.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set parameters names
    if scaling_type == 'mean-std':
        param_1_label = 'MEAN'
        param_2_label = 'STD'
    elif scaling_type == 'min-max':
        param_1_label = 'MIN'
        param_2_label = 'MAX'
    # Set formatted data
    param_1_frmt = ' '.join([f'{x:15.8e}' for x in param_1.numpy()])
    param_2_frmt = ' '.join([f'{x:15.8e}' for x in param_2.numpy()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'\nFeatures: {features_type}')
    print(f'  > {param_1_label:<{4}}: {param_1_frmt}')
    print(f'  > {param_2_label:<{4}}: {param_2_frmt}')
    print(f'')