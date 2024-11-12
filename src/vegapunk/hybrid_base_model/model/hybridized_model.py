"""Setup hybridized model.
    
Functions
---------
set_hybridized_model
    Set hybridized model data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def set_hybridized_model(model_class, hyb_indices, model_init_args=None,
                         model_init_file_path=None,
                         model_state_file_path=None, is_input_residual=False,
                         data_scalers=None):
    """Set hybridized model data.
    
    Parameters
    ----------
    model_class : type
        Hybridized model class.
    hyb_indices : tuple[int]
        Hybridized model hybridization indices stored as (i, j), where i is the
        hybridization channel index and j the position index along the
        hybridization channel.
    model_init_args : dict, default=None
        Hybridized model class initialization parameters.
    model_init_file_path : str, default=None
        Hybridized model initialization file path. Ignored if
        model_init_args is provided.
    model_state_file_path : str, default=None
        Hybridized model state file path. If provided, then model state is
        initialized from state file, otherwise model state stems from model
        (default) initialization.
    is_input_residual : bool, default=False
        If True, then input residual connection is assigned to hybridized
        model, False otherwise.
    data_scalers : dict, default=None
        Data scaler (item, TorchStandardScaler) for each feature data
        (key, str). Only extracts data scalers for 'features_in' and
        'features_out' if available. Overrides data scalers provided in model
        initialization file path.

    Returns
    -------
    hyb_model_data : dict
        Hybridized model data.
    """
    # Initialize hybridized model
    if isinstance(model_init_args, dict):
        # Initialize hybridized model from initialization parameters
        if hasattr(model_class, 'save_model_init_file'):
            hyb_model = model_class(**model_init_args,
                                    is_save_model_init_file=False)
        else:
            hyb_model = model_class(**model_init_args)
    elif isinstance(model_init_file_path, str):
        # Initialize hybridized model from initialization file
        if hasattr(model_class, 'init_model_from_file'):
            hyb_model = model_class.init_model_from_file(
                 model_init_file_path=model_init_file_path)
        else:
            raise RuntimeError(f'Hybridized model \'{model_class}\' cannot '
                               f'be initialized from initialization file '
                               f'because method \'init_model_from_file\' '
                               f'is not available.')
    else:
        raise RuntimeError('Neither valid \'model_init_args\' nor '
                           '\'model_init_file_path\' have been provided '
                           'to initialize hybridized model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model state
    if model_state_file_path is not None:
        # Check hybridized model state file path
        if not os.path.isfile(model_state_file_path):
            raise RuntimeError(f'Hybridized model \'{model_class}\' state '
                               f'file path has not been found:\n\n'
                               f'{model_state_file_path}')
        # Load hybridized model state
        hyb_model.load_state_dict(torch.load(model_state_file_path,
                                             map_location=torch.device('cpu')))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hybridized model data scalers
    if data_scalers is not None:
        if hasattr(model_class, 'set_data_scalers'):
            # Check data scalers
            if isinstance(data_scalers, dict):
                # Check data scaler for input features
                if 'features_in' in data_scalers.keys():
                    scaler_features_in = data_scalers['features_in']
                else:
                    scaler_features_in = None
                # Check data scaler for output features
                if 'features_out' in data_scalers.keys():
                    scaler_features_out = data_scalers['features_out']
                else:
                    scaler_features_out = None
                # Set hybridized model data scalers
                hyb_model.set_data_scalers(scaler_features_in,
                                           scaler_features_out)
            else:
                raise RuntimeError('Hybridized model data scalers must be '
                                'provided as dictionary.') 
        else:
            raise RuntimeError(f'Cannot set hybridized model '
                               f'\'{model_class}\' data scalers because '
                               f'method \'set_data_scalers\' is not '
                               f'available.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check hybridization indices
    if not isinstance(hyb_indices, tuple):
        raise RuntimeError('Hybridized model hybridization indices must be '
                           'provided as tuple[int] as (i, j).')
    elif (len(hyb_indices) != 2
            or not all(isinstance(x, int) for x in hyb_indices)):
        raise RuntimeError('Hybridized model hybridization indices must be '
                           'provided as tuple[int] as (i, j).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input residual connection
    if not isinstance(is_input_residual, bool):
        raise RuntimeError('Hybridized model input residual connection must '
                           'be specified as boolean.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize hybridized model data
    hyb_model_data = {}
    # Set hybridized model data
    hyb_model_data['hyb_model'] = hyb_model
    hyb_model_data['hyb_indices'] = hyb_indices
    hyb_model_data['is_input_residual'] = is_input_residual
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hyb_model_data