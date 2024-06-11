import torch

def set_parameter_in_dict(param_dict, split_name, value):
    """Store parameter in torch.nn.ParameterDict.
    
    Provided parameter dictionary is updated in-place. Nested dictionaries are
    created to handle parameters from nested modules or submodules.
    
    Parameters
    ----------
    param_dict : torch.nn.ParameterDict()
        Parameter dictionary.
    split_name : list[str]
        Parameter name splitted by dot delimiter (nested structure).
    value : torch.Tensor(0d)
        Parameter value.
    """
    # Get initial parameter key
    key = split_name[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store parameter if leaf module, otherwise create and process nested
    # dictionary
    if len(split_name) == 1:
        # Store parameter
        param_dict[key] = value
    else:
        # Create nested dictionary
        if key not in param_dict:
            param_dict[key] = torch.nn.ParameterDict()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store parameter in nested dictionary (recursive)
        set_parameter_in_dict(param_dict[key], split_name[1:], value)
# -----------------------------------------------------------------------------
def get_model_parameter_dict(model):
    """Store torch Module parameters in torch.nn.ParameterDict.
    
    Nested modules or submodules are stored as nested dictionaries.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
        
    Returns
    -------
    param_dict : torch.nn.ParameterDict()
        Parameter dictionary.
    """
    # Initialize parameter dictionary
    param_dict = torch.nn.ParameterDict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over model parameters    
    for name, value in model.named_parameters():
        # Split parameter name (nested structure)
        split_name = name.split('.')
        # Store parameter
        set_parameter_in_dict(param_dict, split_name, value)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return param_dict
# =============================================================================
# Linear model (emulating RNN)
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self._model = torch.nn.Linear(2, 1)
        self._model_parameters = get_model_parameter_dict(self)
# -----------------------------------------------------------------------------
# Create linear model
model1 = Model1()
print('\n\nMODEL 1:')
# Output parameters: named_parameters()
print('\nnamed_parameters():')
for param, value in model1.named_parameters():
    print(param, value)
# Output parameters: self._model_parameters
print('\n_model_parameters.items():')
for param, value in model1._model_parameters.items():
    print(param, value)
# =============================================================================
# Model (emulating RCM)
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        m2p1 = torch.nn.Parameter(torch.tensor(0.0))
        m2p2 = torch.nn.Parameter(torch.tensor(1.0))
        m2_parameters = torch.nn.ParameterDict({})
        m2_parameters['p1'] = m2p1
        m2_parameters['p2'] = m2p2
        self._model_parameters = m2_parameters
# -----------------------------------------------------------------------------
# Create model
model2 = Model2()
print('\n\nMODEL 2:')
# Output parameters: named_parameters()
print('\nnamed_parameters():')
for param, value in model2.named_parameters():
    print(param, value)
# Output parameters: self._model_parameters
print('\n_model_parameters.items():')
for param, value in model2._model_parameters.items():
    print(param, value)
# =============================================================================
class ModelFinder(torch.nn.Module):
    def __init__(self, model1, model2):
        super(ModelFinder, self).__init__()
        self._model_parameters = torch.nn.ParameterDict({})
        self._model_parameters['m1'] = model1._model_parameters
        self._model_parameters['m2'] = model2._model_parameters
        self._elements_material = {}
        self._elements_material['1'] = model2
        self._elements_material['2'] = model2
# -----------------------------------------------------------------------------
# Create model
model_finder = ModelFinder(model1, model2)
print('\n\nMODEL FINDER:')
# Output parameters: named_parameters()
print('\nnamed_parameters():')
for param, value in model_finder.named_parameters():
    print(param, value)
# Output parameters: self._model_parameters
print('\n_model_parameters.items():')
for param, value in model_finder._model_parameters.items():
    print(param, value)
# =============================================================================
# Update model parameter
model_finder._model_parameters['m2']['p2'] = \
    torch.nn.Parameter(torch.tensor(3.0))
# Output parameters: named_parameters()
print('\nnamed_parameters():')
for param, value in model_finder.named_parameters():
    print(param, value)
# Output parameters: Check if both elements are sharing the model
print('\n\nCheck elements shared model:')
for param, value in model_finder._elements_material['1'].named_parameters():
    print(param, value)
for param, value in model_finder._elements_material['2'].named_parameters():
    print(param, value)
