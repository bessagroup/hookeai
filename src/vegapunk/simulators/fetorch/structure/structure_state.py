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
# Third-party
import torch
# Local
from simulators.fetorch.math.matrixops import get_problem_type_parameters
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.material.models.standard.von_mises import VonMises
from simulators.fetorch.material.models.standard.von_mises_mixed import \
    VonMisesMixed
from simulators.fetorch.material.models.standard.drucker_prager import \
    DruckerPrager
from rc_base_model.model.recurrent_model import RecurrentConstitutiveModel
from rnn_base_model.model.gru_model import GRURNNModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class StructureMaterialState(torch.nn.Module):
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
    _n_mat_model : int
        Number of FETorch material constitutive models.
    _material_models : dict
        FETorch material constitutive models (key, str[int], item,
        ConstitutiveModel). Models are labeled from 1 to n_mat_model.
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
    get_material_models(self)
        Get material constitutive models.
    get_elements_material(self)
        Get elements material constitutive models.
    get_n_element_material_type(self)
        Get the number of material model types of finite element mesh.
    update_element_state(self, element_id, state_variables, time='current', \
                         is_copy=True)
        Update element material constitutive state variables.
    get_element_state(self, element_id, element_state, time='current', \
                      is_copy=True)
        Get element material constitutive state variables.
    update_converged_elements_state(self, is_copy=True)
        Update elements last converged material state variables.
    get_element_model_recurrency(self, element_id)
        Get element constitutive model recurrent structure.
    update_material_models_device(self, device_type)
        Update material models Torch device.
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
        # Initialize from base class
        super(StructureMaterialState, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem strain formulation and type
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize number of material models
        self._n_mat_model = 0
        # Initialize material models
        self._material_models = torch.nn.ModuleDict()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements material model
        self._elements_material = {str(i): None for i in range(1, n_elem + 1)}
        # Initialize elements material model recurrency
        self._elements_is_recurrent_material = {}
        # Initialize elements material constitutive state variables
        self._elements_state = {}
        self._elements_state_old = {}
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
        
        Constitutive models are automatically labeled from 1 to n_mat_model,
        following the assignment order. A unique constitutive model is created
        whenever this method is called for a given set of elements.
        
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
        # Initialize constitutive model and set related parameters
        if model_name == 'elastic':
            # Initialize constitutive model
            constitutive_model = Elastic(self._strain_formulation,
                                         self._problem_type,
                                         model_parameters)
            # Set recurrency structure
            is_recurrent_model = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'von_mises':
            # Initialize constitutive model
            constitutive_model = VonMises(self._strain_formulation,
                                          self._problem_type,
                                          model_parameters)
            # Set recurrency structure
            is_recurrent_model = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'von_mises_mixed':
            # Initialize constitutive model
            constitutive_model = VonMisesMixed(self._strain_formulation,
                                               self._problem_type,
                                               model_parameters)
            # Set recurrency structure
            is_recurrent_model = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'drucker_prager':
            # Initialize constitutive model
            constitutive_model = DruckerPrager(self._strain_formulation,
                                               self._problem_type,
                                               model_parameters)
            # Set recurrency structure
            is_recurrent_model = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            # Get automatic synchronization of material model parameters
            is_auto_sync_parameters = model_kwargs['is_auto_sync_parameters']
            # Get state update failure checking
            is_check_su_fail = model_kwargs['is_check_su_fail']
            # Get model directory
            model_directory = model_kwargs['model_directory']
            # Get parameters normalization
            is_normalized_parameters = model_kwargs['is_normalized_parameters']
            # Get model input and output features normalization
            is_model_in_normalized = model_kwargs['is_model_in_normalized']
            is_model_out_normalized = model_kwargs['is_model_out_normalized']
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
                'is_auto_sync_parameters': is_auto_sync_parameters,
                'is_check_su_fail': is_check_su_fail,
                'model_directory': model_directory,
                'model_name': model_name,
                'is_normalized_parameters': is_normalized_parameters,
                'is_model_in_normalized': is_model_in_normalized,
                'is_model_out_normalized': is_model_out_normalized,
                'device_type': device_type}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = RecurrentConstitutiveModel(**model_init_args)
            # Set recurrency structure
            is_recurrent_model = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_name == 'gru_material_model':
            # Get number of input and output features
            n_features_in = model_kwargs['n_features_in']
            n_features_out = model_kwargs['n_features_out']
            # Get hidden layer size
            hidden_layer_size = model_kwargs['hidden_layer_size']
            # Get number of recurrent layers
            n_recurrent_layers = model_kwargs['n_recurrent_layers']
            # Get dropout probability
            dropout = model_kwargs['dropout']
            # Get model directory
            model_directory = model_kwargs['model_directory']
            # Get model input and output features normalization
            is_model_in_normalized = model_kwargs['is_model_in_normalized']
            is_model_out_normalized = model_kwargs['is_model_out_normalized']
            # Set GRU model source
            gru_model_source = model_kwargs['gru_model_source']
            # Get device type
            device_type = model_kwargs['device_type']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build model initialization parameters
            model_init_args = {
                'n_features_in': n_features_in,
                'n_features_out': n_features_out,
                'hidden_layer_size': hidden_layer_size,
                'n_recurrent_layers': n_recurrent_layers,
                'dropout': dropout,
                'model_directory': model_directory,
                'model_name': model_name,
                'is_model_in_normalized': is_model_in_normalized,
                'is_model_out_normalized': is_model_out_normalized,
                'gru_model_source': gru_model_source,
                'device_type': device_type}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model
            constitutive_model = GRURNNModel(**model_init_args)
            # Set recurrency structure
            is_recurrent_model = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError(f'Unknown material constitutive model '
                               f'\'{model_name}\'.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update number of material constitutive models
        self._n_mat_model += 1
        # Set new material constitutive model label
        model_id = str(self._n_mat_model)
        # Store constitutive model
        self._material_models[model_id] = constitutive_model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    def get_material_models(self):
        """Get material constitutive models.
        
        Returns
        -------
        material_models : dict
            FETorch material constitutive models (key, str[int], item,
            ConstitutiveModel). Models are labeled from 1 to n_mat_model.
        """
        return self._material_models
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
    def get_n_element_material_type(self):
        """Get the number of material model types of finite element mesh.
        
        Returns
        -------
        n_element_material_type : int
            Number of types of constitutive material models of finite element
            mesh.
        """
        return len({type(x) for x in self._elements_material.values()})
    # -------------------------------------------------------------------------
    def update_element_state(self, element_id, element_state, time='current',
                             is_copy=True):
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
        is_copy : bool, default=True
            If True, then update is performed by copying the state variables.
            If False, then update is performed by direct assignment (without
            copy).
        """
        # Copy element material constitutive state variables
        if is_copy:
            element_state = copy.deepcopy(element_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update element material constitutive state variables
        if time == 'last':
           self._elements_state_old[str(element_id)] = element_state
        elif time == 'current':
            self._elements_state[str(element_id)] = element_state
        else:
            raise RuntimeError('Unknown time option.')
    # -------------------------------------------------------------------------
    def get_element_state(self, element_id, time='current', is_copy=True):
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
        is_copy : bool, default=True
            If True, then return copy of the element state variables. If False,
            then return the element state variables by direct assignment
            (without copy).

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
        # Copy element material constitutive state variables
        if is_copy:
            element_state = copy.deepcopy(element_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_state
    # -------------------------------------------------------------------------
    def update_converged_elements_state(self, is_copy=True):
        """Update elements last converged material state variables.
        
        Parameters
        ----------
        is_copy : bool, default=True
            If True, then update is performed by copying the state variables.
            If False, then update is performed by direct assignment (without
            copy).
        """
        if is_copy:
            self._elements_state_old = copy.deepcopy(self._elements_state)
        else:
            self._elements_state_old = self._elements_state
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
    # -------------------------------------------------------------------------
    def update_material_models_device(self, device_type):
        """Update material models Torch device.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        """
        # Check device type
        if device_type not in ('cpu', 'cuda'):
            raise RuntimeError('Invalid device type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material models
        for _, model in self._material_models.items():
            # Check model device
            if hasattr(model, '_device_type'):
                # Update model device
                model.set_device(device_type)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check model embedded material model
            if hasattr(model, '_constitutive_model'):
                # Update embedded material model device
                model._constitutive_model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return