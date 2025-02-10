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
    load_dataset, concatenate_dataset_features, sum_dataset_features
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
from hybrid_base_model.model.hybridized_layers import BatchedElasticModel
from utilities.loss_functions import get_pytorch_loss
# =============================================================================
# Summary: Validate elastic constitutive model for batched time series data
# =============================================================================
# Expecting training data set file generated with von Mises and where the
# outputs include both stress and e_strain_mf. The later can be obtained by
# setting state_features = {'e_strain_mf': len(strain_comps_order)} in the
# material response data set generation process.
#
# Example:
#
## Set constitutive model parameters:
#if model_name == 'von_mises':
#    # Set constitutive model parameters
#    model_parameters = {'elastic_symmetry': 'isotropic',
#                        'E': 110e3, 'v': 0.33,
#                        'euler_angles': (0.0, 0.0, 0.0),
#                        'hardening_law': get_hardening_law('nadai_ludwik'),
#                        'hardening_parameters':
#                            {'s0': 900,
#                             'a': 700,
#                             'b': 0.5,
#                             'ep0': 1e-5}}
#    # Set constitutive state variables to be additionally included in the
#    # data set
#    state_features = {'e_strain_mf': len(strain_comps_order)}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set training data set file path
train_dataset_file_path = \
    ('/home/bernardoferreira/Desktop/test/1_training_dataset/'
     'ss_paths_dataset_n10.pkl')
# Load training data set
train_dataset = load_dataset(train_dataset_file_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set device type
device_type = 'cpu'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set model name
model_name = 'material_rc_model'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set number of input features
n_features_in = 6
# Set number of output features
n_features_out = 6
# Set learnable parameters
learnable_parameters = {}
# Set strain formulation
strain_formulation = 'infinitesimal'
# Set problem type
problem_type = 4
# Set material constitutive model name
material_model_name = 'elastic'
# Set material constitutive model parameters
material_model_parameters = {'elastic_symmetry': 'isotropic',
                             'E': 110e3, 'v': 0.33,
                             'euler_angles': (0.0, 0.0, 0.0)}
# Set material constitutive state variables (prediction)
state_features_out = {}
# Set parameters normalization
is_normalized_parameters = True
# Set model input and output features normalization
is_model_in_normalized = False
is_model_out_normalized = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build model initialization parameters
model_init_args = {'n_features_in': n_features_in,
                   'n_features_out': n_features_out,
                   'learnable_parameters': learnable_parameters,
                   'strain_formulation': strain_formulation,
                   'problem_type': problem_type,
                   'material_model_name': material_model_name,
                   'material_model_parameters': material_model_parameters,
                   'state_features_out': state_features_out,
                   'model_directory': None,
                   'model_name': None,
                   'is_normalized_parameters': is_normalized_parameters,
                   'is_model_in_normalized': is_model_in_normalized,
                   'is_model_out_normalized': is_model_out_normalized,
                   'device_type': device_type}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set new feature (plastic strain)
new_label = 'p_strain_mf'
sum_features_labels = ('strain_path', 'e_strain_mf')
features_weights = {'strain_path': 1.0, 'e_strain_mf': -1.0}
# Set training data set new feature
train_dataset = sum_dataset_features(train_dataset, new_label,
                                     sum_features_labels, features_weights,
                                     is_remove_features=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set new feature (concatenated strain and plastic strain)
new_label = 'cat_strain'
cat_features_in = ('p_strain_mf', 'strain_path')
# Set training data set new feature
train_dataset = concatenate_dataset_features(train_dataset, new_label,
                                             cat_features_in, 
                                             is_remove_features=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set input features
new_label_in = 'features_in'
cat_features_in = ('e_strain_mf',)
# Set training data set features
train_dataset = concatenate_dataset_features(train_dataset, new_label_in,
                                             cat_features_in,
                                             is_remove_features=False)
# Set output features
new_label_out = 'features_out'
cat_features_out = ('stress_path',)
# Set training data set features
train_dataset = concatenate_dataset_features(train_dataset, new_label_out,
                                             cat_features_out,
                                             is_remove_features=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set data loader
data_loader = get_time_series_data_loader(dataset=train_dataset,
                                          batch_size=len(train_dataset))
# Get data batch
batch = [x for x in data_loader][0]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get batched data
strain = batch['strain_path']
e_strain_in = batch['e_strain_mf']
p_strain_in = batch['p_strain_mf']
cat_strain_in = batch['cat_strain']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check infinitesimal strain elastoplastic decomposition
if not torch.allclose(strain, e_strain_in + p_strain_in):
    raise RuntimeError('Infinitesimal strains additive elastoplastic '
                       'decomposition was not satisfied.')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize elastic recurrent constitutive model
rc_elastic_model = RecurrentConstitutiveModel(**model_init_args,
                                              is_save_model_init_file=False)
# Compute elastic recurrent constitutive model prediction
rc_elastic_features_out = rc_elastic_model(e_strain_in)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize batched elastic constitutive model
batch_elastic_model = BatchedElasticModel(
    problem_type, elastic_properties=material_model_parameters,
    elastic_symmetry=material_model_parameters['elastic_symmetry'],
    device_type=device_type)
# Compute batched elastic constitutive model prediction
batch_elastic_features_out = batch_elastic_model(cat_strain_in)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check results: Mean Squared Error (MSE)
# Initialize loss function
loss_function = get_pytorch_loss('mse')
# Compute loss
mse = loss_function(rc_elastic_features_out, batch_elastic_features_out)
# Display results
print(f'\nmse = {mse:11.4e}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check results: Element by element maximum tolerance
# Set absolute tolerance
atol = 1e-4
# Set relative tolerance
rtol = 1e-3
# Check element by element tolerances
is_all_close = \
    torch.allclose(rc_elastic_features_out, batch_elastic_features_out,
                   rtol=rtol, atol=atol)
# Display results
print(f'\nall_close(rtol={rtol}, atol={atol}) = {is_all_close}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()