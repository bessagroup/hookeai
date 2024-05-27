"""FETorch: Structure material state.

Classes
-------
StructureMaterialState
    FETorch structure material state.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class StructureMaterialState:
    """FETorch structure material state.
    
    Attributes
    ----------
    _strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    _problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _elements_material : dict
        FETorch material constitutive model (item, ConstitutiveModel) of each
        finite element mesh element (str[int]). Elements are labeled from
        1 to n_elem.
    _elements_state : dict
        For each finite element mesh element (key, str[int]), stores a
        dictionary with the material constitutive model state variables
        (item, dict) for each Gauss integration point (key, str[int]).
    _elements_state_old : dict
        For each finite element mesh element (key, str[int]), stores a
        dictionary with the last converged material constitutive model state
        variables (item, dict) for each Gauss integration point
        (key, str[int]).
    _elements_is_state : dict
        For each finite element mesh element (key, str[int]), stores a bool
        that defines if the material constitutive model state variables are
        available. When False, element state variables are set to None.

    Methods
    -------
    init_elements_model(self, model_name, model_parameters, element_ids, \
                        elements_type)
        Initialize constitutive model assigned to given set of elements.
    get_strain_formulation(self)
        Get problem strain formulation.
    get_problem_type(self)
        Get problem type.
    get_elements_material(self)
        Get elements material constitutive models.
    update_element_state(self, element_id, state_variables, time='current')
        Update element material constitutive state variables.
    get_element_state(self, element_id, element_state, time='current')
        Get element material constitutive state variables.
    update_converged_elements_state(self)
        Update elements last converged material state variables.
    """
    def __init__(self, strain_formulation, problem_type, n_elem):
        """Constructor.
        
        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        n_elem : int
            Number of elements of finite element mesh.
        """
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements material model
        self._elements_material = {str(i): None for i in range(1, n_elem + 1)}
        # Initialize elements material constitutive state variables
        self._elements_state = None
        self._elements_state_old = None
        self._elements_is_state = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
    # -------------------------------------------------------------------------
    def init_elements_model(self, model_name, model_parameters, element_ids,
                            elements_type):
        """Initialize constitutive model assigned to given set of elements.
        
        While the constitutive model object is shared between all the elements
        in the provided set, the corresponding state variables are initialized
        independently for each element (and corresponding Gauss integration
        points) in the provided set.
        
        Parameters
        ----------
        model_name : str
            Material constitutive model name.
        model_parameters : dict
            Material constitutive model parameters.
        element_ids : tuple[int]
            Set of element labels. Elements labels must be within the range of
            1 to n_elem (included).
        elements_type : dict
            FETorch element type (item, ElementType) of each finite element
            mesh element (str[int]). Elements labels must be within the range
            of 1 to n_elem (included).
        """
        # Initialize constitutive model
        if model_name == 'elastic':
            constitutive_model = Elastic(self._strain_formulation,
                                         self._problem_type,
                                         model_parameters)
        elif model_name == 'von_mises':
            constitutive_model = VonMises(self._strain_formulation,
                                          self._problem_type,
                                          model_parameters)
        elif model_name == 'drucker_prager':
            constitutive_model = DruckerPrager(self._strain_formulation,
                                               self._problem_type,
                                               model_parameters)
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model state variables availability
        if model_name in ('elastic', 'von_mises', 'drucker_prager'):
            is_state_available = True
        else:
            is_state_available = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements Gauss integration points state variables
        self._elements_state = {}
        self._elements_state_old = {}
        self._elements_is_state = {}
        # Loop over elements
        for element_id in element_ids:
            # Assign constitutive model
            self._elements_material[str(element_id)] = constitutive_model
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model state variables
            if is_state_available:
                # Initialize constitutive model state variables
                state_variables = constitutive_model.state_init()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get element type
                element_type = elements_type[str(element_id)]
                # Get element number of Gauss quadrature integration points
                n_gauss = element_type.get_n_gauss()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize element Gauss integration points state variables
                self._elements_state[str(element_id)] = \
                    {str(i): copy.deepcopy(state_variables)
                    for i in range(1, n_gauss + 1)}
                self._elements_state_old[str(element_id)] = \
                    copy.deepcopy(self._elements_state[str(element_id)])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set element state variables availability
                self._elements_is_state[str(element_id)] = True
            else:
                # Set undefined state variables
                self._elements_state[str(element_id)] = None
                self._elements_state_old[str(element_id)] = None
                self._elements_is_state[str(element_id)] = False
    # -------------------------------------------------------------------------
    def get_strain_formulation(self):
        """Get problem strain formulation.
        
        Returns
        -------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        """
        return self._strain_formulation
    # -------------------------------------------------------------------------
    def get_problem_type(self):
        """Get problem type.
        
        Returns
        -------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        """
        return self._problem_type
    # -------------------------------------------------------------------------
    def get_elements_material(self):
        """Get elements material constitutive models.
        
        Returns
        -------
        elements_material : dict
            FETorch material constitutive model (item, ConstitutiveModel) of
            each finite element mesh element (str[int]). Elements are labeled
            from 1 to n_elem.
        """
        return self._elements_material
    # -------------------------------------------------------------------------
    def update_element_state(self, element_id, element_state, time='current'):
        """Update element material constitutive state variables.
        
        Parameters
        ----------
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).
        element_state : dict
            Material constitutive model state variables (item, dict) for each
            Gauss integration point (key, str[int]).
        time : {'last', 'current'}, default='current'
            Time where update of element state variables is performed: last
            converged state variables ('last') or current state variables
            ('current').
        """
        # Update element material constitutive state variables
        if time == 'last':
            self._elements_state_old[str(element_id)] = \
                copy.deepcopy(element_state)
        elif time == 'current':
            self._elements_state[str(element_id)] = \
                copy.deepcopy(element_state)
        else:
            raise RuntimeError('Unknown time option.')
    # -------------------------------------------------------------------------
    def get_element_state(self, element_id, time='current'):
        """Get element material constitutive state variables.
        
        Parameters
        ----------
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).
        time : {'last', 'current'}, default='current'
            Time where update of element state variables is performed: last
            converged state variables ('last') or current state variables
            ('current').

        Returns
        -------
        element_state : dict
            Material constitutive model state variables (item, dict) for each
            Gauss integration point (key, str[int]).
        """
        # Update element material constitutive state variables
        if time == 'last':
            element_state = self._elements_state_old[str(element_id)]
        elif time == 'current':
            element_state = self._elements_state[str(element_id)]
        else:
            raise RuntimeError('Unknown time option.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return copy.deepcopy(element_state)
    # -------------------------------------------------------------------------
    def update_converged_elements_state(self):
        """Update elements last converged material state variables."""
        self._elements_state_old = copy.deepcopy(self._elements_state)