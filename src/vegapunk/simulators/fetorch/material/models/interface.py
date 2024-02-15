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
    def __init__(self, strain_formulation, problem_type, model_parameters):
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