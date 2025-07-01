"""Procedures to save and load model state files.

Functions
---------
save_model_state
    Save model state to file.
load_model_state
    Load model state from file.
check_state_file
    Check if file is model training epoch state file.
check_best_state_file
    Check if file is model best state file.
remove_posterior_state_files
    Delete model training epoch state files posterior to given epoch.
remove_best_state_files
    Delete existent model best state files.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import re
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
def save_model_state(model, state_type='default', epoch=None,
                     is_remove_posterior=True):
    """Save model state to file.
    
    Model default state file is stored in model_directory under the name
    < model_name >.pt.
    
    Model initial state file is stored in model_directory under the name
    < model_name >-init.pt.

    Model state file corresponding to given training epoch is stored
    in model_directory under the name < model_name >.pt or
    < model_name >-< epoch >.pt if epoch is known.

    Model state file corresponding to the best performance is stored in
    model_directory under the name < model_name >-best.pt or
    < model_name >-< epoch >-best.pt if epoch is known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    state_type : {'default', 'init', 'epoch', 'best'}, default='default'
        Saved model state file type.
        Options:
        
        'default' : Model default state
        
        'init'    : Model initial state
    
        'epoch'   : Model state of given training epoch
        
        'best'    : Model state of best performance
        
    epoch : int, default=None
        Training epoch corresponding to current model state.
    is_remove_posterior : bool, default=True
        Remove model state files corresponding to training epochs posterior to
        the saved state file. Effective only if saved training epoch is known.
    """
    # Check model
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError('Model is not a torch.nn.Module.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model directory
    if not hasattr(model, 'model_directory'):
        raise RuntimeError('The model directory is not available.')
    elif not os.path.isdir(model.model_directory):
        raise RuntimeError('The model directory has not been found:\n\n'
                           + model.model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model state filename
    model_state_file = model.model_name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state type
    if state_type == 'init':
        # Set model state filename
        model_state_file += '-init'
    else:
        # Append epoch
        if isinstance(epoch, int):
            model_state_file += '-' + str(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set particular model states
        if state_type == 'best':
            # Set model state corresponding to the best performance
            model_state_file += '-' + 'best'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Remove any existent best model state file
            remove_best_state_files(model)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state file path
    model_path = os.path.join(model.model_directory,
                              model_state_file + '.pt')
    # Save model state
    torch.save(model.state_dict(), model_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Delete model epoch state files posterior to saved epoch
    if isinstance(epoch, int) and is_remove_posterior:
        remove_posterior_state_files(model, epoch)
# =============================================================================
def load_model_state(model, load_model_state=None, is_remove_posterior=True):
    """Load model state from file.
    
    Model state file is stored in model_directory under the name
    < model_name >.pt or < model_name >-< epoch >.pt if epoch is known.
    
    Model state file corresponding to the best performance is stored in
    model_directory under the name < model_name >-best.pt or
    < model_name >-< epoch >-best.pt if epoch if known.
    
    Model initial state file is stored in model directory under the name
    < model_name >-init.pt
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    load_model_state : {'default', 'init', int, 'best', 'last'},
                       default='default'
        Load available model state from the model directory.
        Options:
        
        'default'   : Model default state file
        
        'init'      : Model initial state
        
        int         : Model state of given training epoch
        
        'best'      : Model state of best performance
        
        'last'      : Model state of latest training epoch
    
    is_remove_posterior : bool, default=True
        Remove model state files corresponding to training epochs posterior
        to the loaded state file. Effective only if loaded training epoch
        is known.
        
    Returns
    -------
    epoch : int
        Loaded model state training epoch. Defaults to None if training
        epoch is unknown.
    """
    # Check model
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError('Model is not a torch.nn.Module.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model directory
    if not hasattr(model, 'model_directory'):
        raise RuntimeError('The model directory is not available.')
    elif not os.path.isdir(model.model_directory):
        raise RuntimeError('The model directory has not been found:\n\n'
                           + model.model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model state filename
    model_state_file = model.model_name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state file
    if load_model_state == 'init':
        # Set model initial state file
        model_state_file += '-init'
        # Set epoch
        epoch = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model epoch state files posterior to loaded epoch
        if is_remove_posterior:
            remove_posterior_state_files(model, epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif isinstance(load_model_state, int):
        # Get epoch
        epoch = load_model_state
        # Set model state filename with epoch
        model_state_file += '-' + str(int(epoch))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model epoch state files posterior to loaded epoch
        if is_remove_posterior:
            remove_posterior_state_files(model, epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif load_model_state == 'best':
        # Get state files in model directory
        directory_list = os.listdir(model.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model best state files epochs
        best_state_epochs = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model epoch best state file
            is_best_state_file, best_state_epoch = \
                check_best_state_file(model, filename)
            # Store model best state file training epoch
            if is_best_state_file:
                best_state_epochs.append(best_state_epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model best state file
        if not best_state_epochs:
            raise RuntimeError('Model best state file has not been found '
                               'in directory:\n\n' + model.model_directory)
        elif len(best_state_epochs) > 1:
            raise RuntimeError('Two or more model best state files have '
                               'been found in directory:'
                               '\n\n' + model.model_directory)
        else:
            # Set best state epoch
            epoch = best_state_epochs[0]
            # Set model best state file
            if isinstance(epoch, int):
                model_state_file += '-' + str(epoch)
            model_state_file += '-' + 'best'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delete model epoch state files posterior to loaded epoch
        if isinstance(epoch, int) and is_remove_posterior:
            remove_posterior_state_files(model, epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif load_model_state == 'last':
        # Get state files in model directory
        directory_list = os.listdir(model.model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize model state files training epochs
        epochs = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over files in model directory
        for filename in directory_list:
            # Check if file is model epoch state file
            is_state_file, epoch = check_state_file(model, filename)
            # Store model state file training epoch
            if is_state_file:
                epochs.append(epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set highest epoch model state file
        if epochs:
            # Set highest epoch
            epoch = max(epochs)
            # Set highest epoch model state file
            model_state_file += '-' + str(epoch)
        else:
            raise RuntimeError('Model state files corresponding to epochs '
                               'have not been found in directory:\n\n'
                               + model.model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Set epoch as unknown
        epoch = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state file path
    model_path = os.path.join(model.model_directory,
                              model_state_file + '.pt')
    # Check model state file
    if not os.path.isfile(model_path):
        raise RuntimeError('Model state file has not been found:\n\n'
                           + model_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model state
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device('cpu')))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return epoch
# =============================================================================
def check_state_file(model, filename):
    """Check if file is model training epoch state file.
    
    Model training epoch state file is stored in model_directory under the
    name < model_name >-< epoch >.pt.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    filename : str
        File name.
    
    Returns
    -------
    is_state_file : bool
        True if model training epoch state file, False otherwise.
    epoch : {None, int}
        Training epoch corresponding to model state file if
        is_state_file=True, None otherwise.
    """
    # Check if file is model epoch state file
    is_state_file = bool(re.search(r'^' + model.model_name + r'-[0-9]+'
                                   + r'\.pt', filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    epoch = None
    if is_state_file:
        # Get model state epoch
        epoch = int(os.path.splitext(filename)[0].split('-')[-1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_state_file, epoch
# =============================================================================
def check_best_state_file(model, filename):
    """Check if file is model best state file.
    
    Model state file corresponding to the best performance is stored in
    model_directory under the name < model_name >-best.pt. or
    < model_name >-< epoch >-best.pt if the training epoch is known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    filename : str
        File name.
    
    Returns
    -------
    is_best_state_file : bool
        True if model training epoch state file, False otherwise.
    epoch : {None, int}
        Training epoch corresponding to model state file if
        is_best_state_file=True and training epoch is known, None
        otherwise.
    """
    # Check if file is model epoch best state file
    is_best_state_file = bool(re.search(r'^' + model.model_name
                                        + r'-?[0-9]*' + r'-best' + r'\.pt',
                                        filename))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    epoch = None
    if is_best_state_file:
        # Get model state epoch
        epoch = int(os.path.splitext(filename)[0].split('-')[-2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return is_best_state_file, epoch
# =============================================================================
def remove_posterior_state_files(model, epoch):
    """Delete model training epoch state files posterior to given epoch.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    epoch : int
        Training epoch.
    """
    # Get files in model directory
    directory_list = os.listdir(model.model_directory)
    # Loop over files in model directory
    for filename in directory_list:
        # Check if file is model epoch state file
        is_state_file, file_epoch = model._check_state_file(filename)
        # Delete model epoch state file posterior to given epoch
        if is_state_file and file_epoch > epoch:
            os.remove(os.path.join(model.model_directory, filename))
# =============================================================================
def remove_best_state_files(model):
    """Delete existent model best state files.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    """
    # Get files in model directory
    directory_list = os.listdir(model.model_directory)
    # Loop over files in model directory
    for filename in directory_list:
        # Check if file is model best state file
        is_best_state_file, _ = check_best_state_file(model, filename)
        # Delete state file
        if is_best_state_file:
            os.remove(os.path.join(model.model_directory, filename))