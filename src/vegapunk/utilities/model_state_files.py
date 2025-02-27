"""Procedures to save and load model state files.

Functions
---------

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
def save_model_state(model, state_type='default', epoch=None,
                     is_best_state=False,
                        is_best_fit_param=False, is_remove_posterior=True):
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

    Model state file corresponding to the best fitting parameters is stored
    in model_directory under the name < model_name >-best-param.pt or
    < model_name >-< epoch >-best-fit.pt if epoch is known.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    state_type : {'default', 'init', 'epoch', 'best', 'best-fit'}, \
                 default = 'default'
        Saved model state file type.
        Options:
        
        'default' : Model default state
        
        'init'    : Model initial state
    
        'epoch'   : Model state of given training epoch
        
        'best'    : Model state of best performance
        
        'best-fit : Model state of best fitting parameters
        
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
    if not hasattr(model, 'model_directory') or os.path.isdir(self.model_directory):
        pass
    
    
    # Check model directory
    if not os.path.isdir(self.model_directory):
        raise RuntimeError('The model directory has not been found:\n\n'
                            + self.model_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state filename
    model_state_file = self.model_name
    # Append epoch
    if isinstance(epoch, int):
        model_state_file += '-' + str(epoch)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set particular model states
    if is_best_state:
        # Set model state corresponding to the best performance
        model_state_file += '-' + 'best'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove any existent best model state file
        self._remove_best_state_files()
    elif is_best_fit_param:
        # Set model state corresponding to the best fitting parameters
        model_state_file += '-' + 'best-fit-param'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove any existent best fitting parameters model state file
        self._remove_best_state_files()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set model state file path
    model_path = os.path.join(self.model_directory,
                                model_state_file + '.pt')
    # Save model state
    torch.save(self.state_dict(), model_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Delete model epoch state files posterior to saved epoch
    if isinstance(epoch, int) and is_remove_posterior:
        self._remove_posterior_state_files(epoch)