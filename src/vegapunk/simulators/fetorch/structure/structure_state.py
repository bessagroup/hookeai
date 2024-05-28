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
import re
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
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
    _elements_is_recurrent_material : dict
        For each finite element mesh element (key, str[int]), stores a bool
        that defines if the material constitutive model has a recurrent
        structure (processes full deformation path when called).

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
    get_element_state_availability(self, element_id)
        Get element constitutive model state variables availability.
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
        # Initialize elements material model recurrency
        self._elements_is_recurrent_material = None
        # Initialize elements material constitutive state variables
        self._elements_state = None
        self._elements_state_old = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
    # -------------------------------------------------------------------------
    def init_elements_model(self, model_name, model_parameters, element_ids,
                            elements_type, model_kwargs={}):
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
        model_kwargs : dict, default={}
            Other parameters required to initialize constitutive model.
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
        elif bool(re.search(r'^rc_.*$', model_name)):
            # Get number of input and output features
            n_features_in = model_kwargs['n_features_in']
            n_features_out = model_kwargs['n_features_out']
            # Get learnable parameters
            learnable_parameters = model_kwargs['learnable_parameters']
            # Set strain formulation
            strain_formulation = self._strain_formulation
            # Set problem type
            problem_type = self._problem_type
            # Set material constitutive model name
            material_model_name = model_kwargs['material_model_name']
            # Set material constitutive model parameters
            material_model_parameters = model_parameters
            # Get material constitutive state variables (prediction)
            state_features_out = model_kwargs['state_features_out']
            # Get model directory
            model_directory = model_kwargs['model_directory']
            # Get parameters normalization
            is_normalized_parameters = model_kwargs['is_normalized_parameters']
            # Get data normalization
            is_data_normalization = model_kwargs['is_data_normalization']
            # Get device type
            device_type = model_kwargs['device_type']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build model initialization parameters
            model_init_args = {
                'n_features_in': n_features_in,
                'n_features_out': n_features_out,
                'learnable_parameters': learnable_parameters,
                'strain_formulation': strain_formulation,
                'problem_type': problem_type,
                'material_model_name': material_model_name,
                'material_model_parameters': material_model_parameters,
                'state_features_out': state_features_out,
                'model_directory': model_directory,
                'model_name': model_name,
                'is_normalized_parameters': is_normalized_parameters,
                'is_data_normalization': is_data_normalization,
                'device_type': device_type}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize recurrent constitutive model
            constitutive_model = RecurrentConstitutiveModel(**model_init_args) 
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive model recurrency structure
        if model_name in ('elastic', 'von_mises', 'drucker_prager'):
            is_recurrent_model = False
        else:
            is_recurrent_model = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements Gauss integration points state variables
        self._elements_state = {}
        self._elements_state_old = {}
        # Initialize elements model recurrency structure
        self._elements_is_recurrent_material = {}
        # Loop over elements
        for element_id in element_ids:
            # Assign constitutive model
            self._elements_material[str(element_id)] = constitutive_model
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model state variables
            if is_recurrent_model:
                # Set element material model recurrency
                self._elements_is_recurrent_material[str(element_id)] = True
                # Initialize constitutive model state variables
                self._elements_state[str(element_id)] = None
                self._elements_state_old[str(element_id)] = None
            else:
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
                # Set element material model recurrency
                self._elements_is_recurrent_material[str(element_id)] = False
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
    # -------------------------------------------------------------------------
    def get_element_model_recurrency(self, element_id):
        """Get element constitutive model recurrent structure.
        
        Parameters
        ----------
        element_id : int
            Element label. Elements labels must be within the range of
            1 to n_elem (included).

        Returns
        -------
        is_recurrent_model : bool
            True if the material constitutive model has a recurrent structure
            (processes full deformation path when called), False otherwise.
        """
        # Check element constitutive model recurrent structure
        is_recurrent_model = \
            self._elements_is_recurrent_material[str(element_id)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_recurrent_model
        