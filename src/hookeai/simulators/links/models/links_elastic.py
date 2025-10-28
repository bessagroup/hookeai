"""Links constitutive model: ELASTIC.

Links (Large Strain Implicit Nonlinear Analysis of Solids Linking Scales) is
a finite element code developed by the CM2S research group at the Faculty of
Engineering, University of Porto.

Classes
-------
LinksElastic(LinksConstitutiveModel)
    Elastic constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Local
from simulators.links.models.interface import LinksConstitutiveModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================
class LinksElastic(LinksConstitutiveModel):
    """Elastic constitutive model.
    
    Linear elastic constitutive model (infinitesimal strains) or Hencky
    hyperelastic constitutive model (finite strains).
    
    Attributes
    ----------
    _name : str
        Constitutive model name.
        
    Methods
    -------
    _get_required_material_descriptors()
        Get required constitutive model descriptors for Links input file.
    _format_material_descriptors(self)
        Format constitutive model descriptors to Links input file.
    """
    def __init__(self):
        """Constructor."""
        self._name = 'ELASTIC'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def _get_required_material_descriptors():
        """Get required constitutive model descriptors for Links input file.
        
        Returns
        -------
        material_descriptors : tuple[str]
            Required constitutive model descriptors for Links input file.
        """
        # Set required constitutive model descriptors
        material_descriptors = ('name', 'density', 'young', 'poisson')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return material_descriptors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _format_material_descriptors(self, mat_phase, descriptors):
        """Format constitutive model descriptors to Links input file.
        
        Parameters
        ----------
        mat_phase : int
            Material phase identifier.
        descriptors : str
            Constitutive model descriptors.
        
        Returns
        -------
        descriptors_input_lines : list[str]
            Constitutive model descriptors formatted to Links input data file.
        """
        # Check descriptors
        required_descriptors = type(self)._get_required_material_descriptors()
        if not set(required_descriptors).issubset(set(descriptors.keys())):
            raise RuntimeError('Missing required constitutive model '
                               'descriptors.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get descriptors
        name = descriptors['name']
        density = descriptors['density']
        young = descriptors['young']
        poisson = descriptors['poisson']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set spacing
        spacing = (len(str(mat_phase)) + 1)*' '
        # Set formatted constitutive model descriptors
        descriptors_input_lines = \
            [str(mat_phase) + ' ' + name + '\n'] + \
            [spacing + '{:<16.8e}'.format(density) + '\n'] + \
            [spacing + '{:<16.8e}'.format(young) +
             '{:<16.8e}'.format(poisson) + '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return descriptors_input_lines