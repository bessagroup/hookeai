"""FETorch: Material constitutive model interface.

This module includes the interface to implement a material constitutive model.

Classes
-------
ConstitutiveModel
    Material constitutive model interface.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
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
class ConstitutiveModel(ABC):
    """Material constitutive model interface.

    Attributes
    ----------
    _name : str
        Material constitutive model name.
    _strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
        Material constitutive model strain formulation: infinitesimal strain
        formulation ('infinitesimal'), finite strain formulation ('finite') or
        finite strain formulation through kinematic extension
        ('finite-kinext').
    _model_parameters : dict
        Material constitutive model parameters.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    get_required_model_parameters()
        Get required material constitutive model parameters.
    state_init(self)
        Get initialized material constitutive model state variables.
    state_update(self, inc_strain, state_variables_old)
        Perform material constitutive model state update.
    get_name(self)
        Get material constitutive model name.
    get_strain_type(self)
        Get material constitutive model strain formulation.
    get_model_parameters(self)
        Get material constitutive model parameters.
    """
    @abstractmethod
    def __init__(self, strain_formulation, problem_type, model_parameters,
                 device_type='cpu'):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        model_parameters : dict
            Material constitutive model parameters.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_required_model_parameters():
        """Get required material constitutive model parameters.

        Returns
        -------
        model_parameters_names : tuple[str]
            Material constitutive model parameters names (str).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def state_init(self):
        """Get initialized material constitutive model state variables.

        Returns
        -------
        state_variables_init : dict
            Initialized material constitutive model state variables.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def state_update(self, inc_strain, state_variables_old):
        """Perform material constitutive model state update.

        Parameters
        ----------
        inc_strain : torch.Tensor(2d)
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged material constitutive model state variables.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : torch.Tensor(2d)
            Material constitutive model consistent tangent modulus stored in
            matricial form.
        """
        pass
    # -------------------------------------------------------------------------
    def get_name(self):
        """Get material constitutive model name.

        Returns
        -------
        name : str
            Material constitutive model name.
        """
        return self._name
    # -------------------------------------------------------------------------
    def get_strain_type(self):
        """Get material constitutive model strain formulation.

        Returns
        -------
        strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
            Material constitutive model strain formulation: infinitesimal
            strain formulation ('infinitesimal'), finite strain formulation
            ('finite') or finite strain formulation through kinematic extension
            ('finite-kinext').
        """
        return self._strain_type
    # -------------------------------------------------------------------------
    def get_model_parameters(self):
        """Get material constitutive model parameters.

        Returns
        -------
        model_parameters : dict
            Material constitutive model parameters.
        """
        return copy.deepcopy(self._model_parameters)
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Returns
        -------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device