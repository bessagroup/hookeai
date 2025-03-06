"""Links constitutive model: DRUCKER_PRAGER.

Links (Large Strain Implicit Nonlinear Analysis of Solids Linking Scales) is
a finite element code developed by the CM2S research group at the Faculty of
Engineering, University of Porto.

Classes
-------
LinksDruckerPrager(LinksConstitutiveModel)
    Drucker-Prager constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
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
class LinksDruckerPrager(LinksConstitutiveModel):
    """Drucker-Prager constitutive model.
    
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
        self._name = 'DRUCKER_PRAGER'
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
        material_descriptors = ('name', 'density', 'young', 'poisson',
                                'friction_angle_deg', 'dilatancy_angle_deg',
                                'n_hard_point', 'hardening_law',
                                'hardening_parameters')
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
        friction_angle_deg = descriptors['friction_angle_deg']
        dilatancy_angle_deg = descriptors['dilatancy_angle_deg']
        n_hard_point = descriptors['n_hard_point']
        hardening_law = descriptors['hardening_law']
        hardening_parameters = descriptors['hardening_parameters']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set spacing
        spacing = (len(str(mat_phase)) + 1)*' '
        # Initialize formatted constitutive model descriptors
        descriptors_input_lines = \
            [str(mat_phase) + ' ' + name + '\n'] + \
            [spacing + '{:<16.8e}'.format(density) + '\n'] + \
            [spacing + '{:<16.8e}'.format(young) +
             '{:<16.8e}'.format(poisson) + \
             '{:<16.8e}'.format(friction_angle_deg) + \
             '{:<16.8e}'.format(dilatancy_angle_deg) + \
             '3' + '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set formatted constitutive model descriptors (number of hardening
        # points)
        descriptors_input_lines += \
            [spacing + '{:<4d}'.format(n_hard_point) + '\n']
        # Set accumulated plastic strain hardening points
        acc_p_strains = np.linspace(0.0, 2.0, n_hard_point)
        # Loop over hardening points
        for acc_p_strain in acc_p_strains:
            # Compute yield stress
            yield_stress, _ = \
                hardening_law(hardening_parameters, acc_p_strain)
            # Set formatted constitutive model descriptors (hardening law)
            descriptors_input_lines += \
                [spacing + '{:<16.8e}'.format(acc_p_strain)
                 + '{:<16.8e}'.format(yield_stress) + '\n'] 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return descriptors_input_lines