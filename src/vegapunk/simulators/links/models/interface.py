"""Links constitutive model: Interface.

Links (Large Strain Implicit Nonlinear Analysis of Solids Linking Scales) is
a finite element code developed by the CM2S research group at the Faculty of
Engineering, University of Porto.

Classes
-------
LinksConstitutiveModel(ABC)
    Links constitutive model interface.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class LinksConstitutiveModel(ABC):
    """Links constitutive model interface.
    
    Attributes
    ----------
    _name : str
        Constitutive model name.
        
    Methods
    -------
    _get_required_material_descriptors()
        *abstract*: Get required constitutive model descriptors for Links input
        file.
    _format_material_descriptors(self)
        *abstract*: Format constitutive model descriptors to Links input file.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    @abstractmethod
    def _get_required_material_descriptors():
        """Get required constitutive model descriptors for Links input file.
        
        Returns
        -------
        material_descriptors : tuple[str]
            Required constitutive model descriptors for Links input file.
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @abstractmethod
    def _format_material_descriptors(self):
        """Format constitutive model descriptors to Links input file.
        
        Returns
        -------
        descriptors_input_lines : list[str]
            Constitutive model descriptors formatted to Links input data file.
        """
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_name(self):
        """Get constitutive model name.
        
        Returns
        -------
        name : str
            Constitutive model name.
        """
        return self._name