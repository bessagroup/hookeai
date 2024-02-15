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
from simulators.fetorch.material.models.standard import elastic
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

    Methods
    -------
    init_elements_model(self, model_name, model_parameters, element_ids, \
                        elements_type)
        Initialize constitutive model assigned to given set of elements.
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
            constitutive_model = elastic(
                self._strain_formulation, self._problem_type, model_parameters)
        elif model_name == 'data_driven_model':
            constitutive_model = None
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for element_id in element_ids:
            # Assign constitutive model
            self._elements_material[str(element_id)] = constitutive_model
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model state variables
            state_variables = constitutive_model.state_init()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element type
            element_type = elements_type[str(element_id)]
            # Get element number of Gauss quadrature integration points
            n_gauss = element_type.get_n_gauss()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize element Gauss integration points state variables
            self._elements_state[str(element_id)] = \
                {str(i): copy.deepcopy(state_variables)
                 for i in range(1, n_gauss + 1)}
            self._elements_state_old[str(element_id)] = \
                copy.deepcopy(self._elements_state[str(element_id)])