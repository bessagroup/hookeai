"""Hybridization model.

Classes
-------
HybridizationModel(torch.nn.Module)
    Hybridization model.
HMIdentity(torch.nn.Module)
    Hybridization model: Identity.
HMAdditive(torch.nn.Module)
    Hybridization model: Additive.
"""
#
#                                                                       Modules
# =============================================================================
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
class HybridizationModel(torch.nn.Module):
    """Hybridization model.
    
    Attributes
    ----------
    hybridization_type : str
        Hybridization model type.
    hybridization_model : torch.nn.Module
        Hybridization model.
    
    Methods
    -------
    forward(self, list_features_in)
        Forward propagation.

    """
    def __init__(self, hybridization_type='identity'):
        """Constructor.
        
        Parameters
        ----------
        hybridization_type : str, default='identity'
            Hybridization model type.
        """
        # Initialize from base class
        super(HybridizationModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set hybridization model type
        self._hybridization_type = hybridization_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hybridization model
        if hybridization_type == 'identity':
            self.hybridization_model = HMIdentity()
        elif hybridization_type == 'additive':
            self.hybridization_model = HMAdditive()
        else:
            raise RuntimeError('Unknown hybridization type.')
    # -------------------------------------------------------------------------
    def forward(self, list_features_in):
        """Forward propagation.
        
        Parameters
        ----------
        list_features_in : list[torch.Tensor]
            List of similar shaped tensors of input features stored as
            torch.Tensor(2d) of shape (sequence_length, n_features_in) for
            unbatched input or torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        """
        # Forward propagation
        features_out = self.hybridization_model(list_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
# =============================================================================
class HMIdentity(torch.nn.Module):
    """Hybridization model: Identity.
    
    Let first input tensor pass unchanged, discard all remainder input tensors.

    Methods
    -------
    forward(self, list_features_in)
        Forward propagation.
    """
    def __init__(self):
        """Constructor."""
        # Initialize from base class
        super(HMIdentity, self).__init__()
    # -------------------------------------------------------------------------
    def forward(self, list_features_in):
        """Forward propagation.

        Parameters
        ----------
        list_features_in : list[torch.Tensor]
            List of similar shaped tensors of input features stored as
            torch.Tensor(2d) of shape (sequence_length, n_features_in) for
            unbatched input or torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        """
        # Get first input tensor
        features_out = list_features_in[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
# =============================================================================
class HMAdditive(torch.nn.Module):
    """Hybridization model: Additive.
    
    Add input tensors.

    Methods
    -------
    forward(self, list_features_in)
        Forward propagation.
    """
    def __init__(self):
        """Constructor."""
        # Initialize from base class
        super(HMAdditive, self).__init__()
    # -------------------------------------------------------------------------
    def forward(self, list_features_in):
        """Forward propagation.

        Parameters
        ----------
        list_features_in : list[torch.Tensor]
            List of similar shaped tensors of input features stored as
            torch.Tensor(2d) of shape (sequence_length, n_features_in) for
            unbatched input or torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.
        """
        # Add input tensors
        features_out = sum(list_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out