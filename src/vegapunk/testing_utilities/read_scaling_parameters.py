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
import pickle
# Local
from gnn_base_model.model.gnn_model import TorchStandardScaler
from utilities.data_scalers import TorchMinMaxScaler
# =============================================================================
# Summary: Extract data scalers parameters from model initialization file.
# =============================================================================
# Set model initialization file path
model_init_file_path = ('/home/bernardoferreira/Documents/brown/projects/'
                        'darpa_project/2_local_rnn_training/composite_rve/'
                        'dataset_07_2024/2_training_strain_vf_to_stress/'
                        '3_model/model_init_file.pkl')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model initialization attributes from file
if not os.path.isfile(model_init_file_path):
    raise RuntimeError('The model initialization file has not been found:\n\n'
                       + model_init_file_path)
else:
    with open(model_init_file_path, 'rb') as model_init_file:
        model_init_attributes = pickle.load(model_init_file)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get model data scalers
if 'model_data_scalers' not in model_init_attributes.keys():
    raise RuntimeError('Model data scalers are not available from model '
                       'initialization file.')
else:
    model_data_scalers = model_init_attributes['model_data_scalers']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('\nExtract data scalers parameters'
      '\n-------------------------------')
print(f'\nModel initialization file: {model_init_file_path}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over model data scalers
for features_type, data_scaler in model_data_scalers.items():
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