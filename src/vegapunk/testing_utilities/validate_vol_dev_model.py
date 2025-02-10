# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import torch
# Local
from time_series_data.time_dataset import get_time_series_data_loader, \
    load_dataset, concatenate_dataset_features
from hybrid_base_model.model.hybridized_layers import VolDevCompositionModel
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from projects.darpa_metals.rnn_material_model.rnn_model_tools.stress_features \
    import add_stress_features
from utilities.loss_functions import get_pytorch_loss
# =============================================================================
# Summary: Validate volumetric/decomposition model for batched time series data
# =============================================================================
# Set training data set file path
train_dataset_file_path = \
    ('/home/bernardoferreira/Desktop/test/1_training_dataset/'
     'ss_paths_dataset_n10.pkl')
# Load training data set
train_dataset = load_dataset(train_dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get problem type parameters
_, comp_order_sym, _ = get_problem_type_parameters(4) 
# Set stress components order
stress_comps_order = comp_order_sym
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set device type
device_type = 'cpu'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over stress-based features (volumetric and deviatoric stress components)
for stress_feature_label in ('vol_stress', 'dev_stress'):
    # Add stress-based feature to data set
    train_dataset = add_stress_features(train_dataset, stress_feature_label)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set input features
new_label_in = 'features_in'
cat_features_in = ('vol_stress', 'dev_stress')
# Set training data set features
train_dataset = concatenate_dataset_features(train_dataset, new_label_in,
                                             cat_features_in,
                                             is_remove_features=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set data loader
data_loader = get_time_series_data_loader(dataset=train_dataset,
                                          batch_size=len(train_dataset))
# Get data batch
batch = [x for x in data_loader][0]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get batched data
stress = batch['stress_path']
vol_stress = batch['vol_stress']
dev_stress = batch['dev_stress']
features_in = batch['features_in']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check stress volumetric/deviatoric decomposition (data set features)
validate_stress = dev_stress
for i, comp in enumerate(stress_comps_order):
    if comp[0] == comp[1]:
        validate_stress[:, :, i] += vol_stress[:, :, 0]
if not torch.allclose(stress, validate_stress):
    raise RuntimeError('Stress volumetric/deviatoric decomposition was not '
                       'satisfied.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check results: Mean Squared Error (MSE)
# Initialize loss function
loss_function = get_pytorch_loss('mse')
# Compute loss
mre = loss_function(validate_stress, stress)
# Display results
print(f'\nmse = {mre:11.4e}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize volumetric/deviatoric composition model
voldev_model = VolDevCompositionModel(device_type=device_type)
# Compute volumetric/deviatoric composition model output
validate_stress = voldev_model(features_in)
# Check stress volumetric/deviatoric decomposition (model)
if not torch.allclose(stress, validate_stress):
    raise RuntimeError('Stress volumetric/deviatoric decomposition was not '
                       'satisfied.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check results: Mean Squared Error (MSE)
# Initialize loss function
loss_function = get_pytorch_loss('mse')
# Compute loss
mse = loss_function(validate_stress, stress)
# Display results
print(f'\nmse = {mse:11.4e}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()