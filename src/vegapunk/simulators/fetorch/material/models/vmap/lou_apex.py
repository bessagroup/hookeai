"""Lou-Zhang-Yoon model with general differentiable yield function.

This module includes the implementation of the Lou-Zhang-Yoon model with
general differentiable yield function and isotropic hardening.

The apex singularity is handled by means of a purely volumetric return-mapping
along the hydrostatic axis.

This implementation is made compatible with the use of PyTorch vectorizing
maps that, at the current moment, do not support auto differentiable
data-dependent control flows based on if statements or similar constructs
(e.g., torch.cond()). Workarounds based on torch.where() were successfully
implemented, but these lead to complex or inefficient coding, mainly because
they are constrained by elementwise operations (require pre-computations of
true and false paths or repeated true/false function calls for each element).

When torch.cond() is available, the state_update() method can be simplified as
follows:

1. The condition in torch.cond() does not need to be a Tensor with the same
   shape as the true/false output tensors
   
2. Avoid flow vector pre-computations - only perform the needed step
   computation based on torch.cond()

3. Avoid elastic and plastic steps pre-computations - only perform the needed
   step computation based on torch.cond() condition

4. The is_elastic_step flag is no longer required in _plastic_step()

5. Avoid elastic and plastic consistent tangent moduli pre-computations - only
   compute the required tangent based on torch.cond() condition


Classes
-------
LouZhangYoonVMAP
    Lou-Zhang-Yoon model with general differentiable yield function.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
# Local
from simulators.fetorch.material.models.interface import ConstitutiveModel
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf, vget_state_3Dmf_from_2Dmf, \
    vget_state_2Dmf_from_3Dmf
from simulators.fetorch.math.tensorops import get_id_operators, dyad22_1, \
    ddot42_1, ddot24_1, ddot22_1, ddot44_1, fo_dinv_sym
from utilities.type_conversion import convert_dict_to_tensor, \
    convert_tensor_to_float64, convert_dict_to_float64, \
    convert_dict_to_float32
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class LouZhangYoonApexVMAP(ConstitutiveModel):
    """Lou-Zhang-Yoon model with general differentiable yield function.

    Compatible with vectorized mapping.

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
        Material constitutive model strain formulation: infinitesimal strain
        formulation ('infinitesimal'), finite strain formulation ('finite') or
        finite strain formulation through kinematic extension
        ('finite-kinext').
    _model_parameters : dict
        Material constitutive model parameters.
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
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
    get_stress_invariants(cls, stress)
        Compute invariants of stress and deviatoric stress.
    get_effective_stress(cls, stress, yield_a, yield_b, yield_c, yield_d)
        Compute effective stress.
    get_flow_vector(cls, stress, yield_a, yield_b, yield_c, yield_d)
        Compute flow vector.
    _elastic_step(cls, e_trial_strain_mf, trial_stress_mf, acc_p_strain_old)
        Perform elastic step.
    _plastic_step(cls, is_elastic_step, e_trial_strain_mf, \
                  e_consistent_tangent, acc_p_strain_old, E, G, \
                  hardening_law, hardening_parameters, a_hardening_law, \
                  a_hardening_parameters, b_hardening_law, \
                  b_hardening_parameters, c_hardening_law, \
                  c_hardening_parameters, d_hardening_law, \
                  d_hardening_parameters, is_associative_hardening, \
                  su_conv_tol, su_max_n_iterations)
        Perform plastic step.
    get_residual(cls, e_strain, e_trial_strain, acc_p_strain, \
                 acc_p_strain_old, inc_p_mult, effective_stress, \
                 yield_stress, init_yield_stress, flow_vector, \
                 norm_flow_vector, is_associative_hardening=False)
        Compute state update residuals.
    _nr_iteration(cls, residual, n_dim, comp_order_sym, stress, inc_p_mult, \
                  flow_vector, norm_flow_vector, init_yield_stress, \
                  hard_slope, yield_a, a_hard_slope, yield_b, b_hard_slope, \
                  yield_c, c_hard_slope, yield_d, d_hard_slope, \
                  e_consistent_tangent, is_associative_hardening)
        Newton-Raphson iteration (return-mapping).
    """
    def __init__(self, strain_formulation, problem_type, model_parameters,
                 is_su_float64=True, device_type='cpu'):
        """Constitutive model constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        model_parameters : dict
            Material constitutive model parameters.
        is_su_float64 : bool, default=True
            If True, then state update is locally computed in floating-point
            double precision. If False, then default floating-point precision
            is assumed.
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Set material constitutive model name
        self._name = 'lou_zhang_yoon'
        # Set constitutive model strain formulation
        self._strain_type = 'infinitesimal'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set initialization parameters
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._model_parameters = model_parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update floating-point precision
        self._is_su_float64 = is_su_float64
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, self._comp_order_nsym = \
            get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get elastic symmetry
        elastic_symmetry = model_parameters['elastic_symmetry']
        # Compute technical constants of elasticity
        if elastic_symmetry == 'isotropic':
            # Compute technical constants of elasticity
            technical_constants = Elastic.get_technical_from_elastic_moduli(
                elastic_symmetry, model_parameters)
            # Assemble technical constants of elasticity
            self._model_parameters.update(technical_constants)
        else:
            raise RuntimeError('The Lou-Zhang-Yoon constitutive model is '
                               'currently only available for the elastic '
                               'isotropic case.')
    # -------------------------------------------------------------------------
    @staticmethod
    def get_required_model_parameters():
        """Get required material constitutive model parameters.
        
        Model parameters:
        
        - 'elastic_symmetry' : Elastic symmetry (str, {'isotropic',
          'transverse_isotropic', 'orthotropic', 'monoclinic', 'triclinic'})
        - 'elastic_moduli' : Elastic moduli (dict, {'Eijkl': float})
        - 'euler_angles' : Euler angles (degrees) sorted according with Bunge
           convention (tuple[float])
        - 'hardening_law' : Isotropic hardening law (function)
        - 'hardening_parameters' : Isotropic hardening law parameters (dict)
        - 'a_hardening_law': Yield parameter hardening law (function)
        - 'a_hardening_parameters': Yield parameter hardening parameters (dict)
        - 'b_hardening_law': Yield parameter hardening law (function)
        - 'b_hardening_parameters': Yield parameter hardening parameters (dict)
        - 'c_hardening_law': Yield parameter hardening law (function)
        - 'c_hardening_parameters': Yield parameter hardening parameters (dict)
        - 'd_hardening_law': Yield parameter hardening law (function)
        - 'd_hardening_parameters': Yield parameter hardening parameters (dict)
        - 'is_associative_hardening': Assume associative hardening rule (bool)

        Notes:
        
        - Associative hardening rule is only admissible if the yield parameters
          a, b, c and d are constant, i.e., do not depend on the accumulated
          plastic strain through the corresponding hardening laws

        Returns
        -------
        model_parameters_names : tuple[str]
            Material constitutive model parameters names (str).
        """
        # Set material properties names
        model_parameters_names = ('elastic_symmetry', 'elastic_moduli',
                                  'euler_angles',
                                  'hardening_law', 'hardening_parameters',
                                  'a_hardening_law', 'a_hardening_parameters',
                                  'b_hardening_law', 'b_hardening_parameters',
                                  'c_hardening_law', 'c_hardening_parameters',
                                  'd_hardening_law', 'd_hardening_parameters',
                                  'is_associative_hardening')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters_names
    # -------------------------------------------------------------------------
    def state_init(self):
        """Get initialized material constitutive model state variables.

        Constitutive model state variables:

        * ``e_strain_mf``

            * *Infinitesimal strains*: Elastic infinitesimal strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon^{e}}`

        * ``acc_p_strain``

            * Accumulated plastic strain.

            * *Symbol*: :math:`\\bar{\\varepsilon}^{p}`

        * ``strain_mf``

            * *Infinitesimal strains*: Infinitesimal strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon}`

        * ``stress_mf``

            * *Infinitesimal strains*: Cauchy stress tensor (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\sigma}`

        * ``is_plastic``

            * Plastic step flag.

        * ``is_su_fail``

            * State update failure flag.

        ----

        Returns
        -------
        state_variables_init : dict
            Initialized material constitutive model state variables.
        """
        # Initialize constitutive model state variables
        state_variables_init = dict()
        # Initialize strain tensors
        state_variables_init['e_strain_mf'] = vget_tensor_mf(
            torch.zeros((self._n_dim, self._n_dim), device=self._device),
                        self._n_dim, self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            state_variables_init['e_strain_mf'].clone()
        # Initialize Cauchy stress tensor
        state_variables_init['stress_mf'] = vget_tensor_mf(
            torch.zeros((self._n_dim, self._n_dim), device=self._device),
                        self._n_dim, self._comp_order_sym)
        # Initialize internal variables
        state_variables_init['acc_p_strain'] = \
            torch.tensor(0.0, device=self._device)
        # Initialize state flags
        state_variables_init['is_plast'] = \
            torch.tensor(False, device=self._device)
        state_variables_init['is_su_fail'] = \
            torch.tensor(False, device=self._device)
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = \
                torch.tensor(0.0, device=self._device)
            state_variables_init['stress_33'] = \
                torch.tensor(0.0, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables_init
    # -------------------------------------------------------------------------
    def state_update(self, inc_strain, state_variables_old):
        """Perform material constitutive model state update.

        Parameters
        ----------
        inc_strain : torch.Tensor(2d)
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged constitutive model material state variables.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : torch.Tensor(2d)
            Material constitutive model consistent tangent modulus stored in
            matricial form.
        """
        # Get model parameters
        model_parameters = self._model_parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize floating-point precision conversion flag
        is_precision_conversion = False
        # Handle state update floating-point precision
        if torch.get_default_dtype() == torch.float32 and self._is_su_float64:
            # Set floating-point precision conversion flag
            is_precision_conversion = True
            # Set default floating-point precision
            torch.set_default_dtype(torch.float64)
            # Perform floating-point precision conversion
            model_parameters = convert_dict_to_float64(model_parameters,
                                                       is_inplace=False)
            inc_strain = convert_tensor_to_float64(inc_strain)
            state_variables_old = convert_dict_to_float64(state_variables_old,
                                                          is_inplace=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update convergence tolerance
        su_conv_tol = 1e-6
        # Set state update maximum number of iterations
        su_max_n_iterations = 10
        # Set minimum threshold to handle values close or equal to zero
        small = 1e-8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental strain tensor matricial form
        inc_strain_mf = vget_tensor_mf(inc_strain, self._n_dim,
                                       self._comp_order_sym,
                                       device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material properties
        E = model_parameters['E']
        v = model_parameters['v']
        # Get material isotropic strain hardening law
        hardening_law = model_parameters['hardening_law']
        hardening_parameters = model_parameters['hardening_parameters']
        # Get yield parameters hardening laws
        a_hardening_law = model_parameters['a_hardening_law']
        a_hardening_parameters = model_parameters['a_hardening_parameters']
        b_hardening_law = model_parameters['b_hardening_law']
        b_hardening_parameters = model_parameters['b_hardening_parameters']
        c_hardening_law = model_parameters['c_hardening_law']
        c_hardening_parameters = model_parameters['c_hardening_parameters']
        d_hardening_law = model_parameters['d_hardening_law']
        d_hardening_parameters = model_parameters['d_hardening_parameters']
        # Get hardening rule associativity
        is_associative_hardening = model_parameters['is_associative_hardening']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute bulk and shear modulus
        K = E/(3.0*(1.0 - 2.0*v))
        G = E/(2.0*(1.0 + v))
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute initial yield parameters
        yield_a_init, _ = a_hardening_law(
            a_hardening_parameters,
            acc_p_strain=torch.tensor(0.0, device=self._device))
        yield_b_init, _ = b_hardening_law(
            b_hardening_parameters,
            acc_p_strain=torch.tensor(0.0, device=self._device))
        # Get Drucker-Prager pressure and cohesion equivalent parameters
        etay = 3.0*yield_a_init*yield_b_init
        xi = (2.0*math.sqrt(3)/3.0)*torch.sqrt(1.0 - (1.0/3.0)*etay**2)
        # Compute additional material parameter
        alpha = xi/etay
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get last increment converged state variables
        e_strain_old_mf = state_variables_old['e_strain_mf']
        p_strain_old_mf = state_variables_old['strain_mf'] - e_strain_old_mf
        acc_p_strain_old = state_variables_old['acc_p_strain']
        if self._problem_type == 1:
            e_strain_33_old = state_variables_old['e_strain_33']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state update failure flag
        is_su_fail = torch.tensor(False, device=self._device)
        # Initialize plastic step flag
        is_plast = torch.tensor(False, device=self._device)
        #
        #                                                    2D > 3D conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, perform the state
        # update and consistent tangent computation as in the 3D case,
        # considering the appropriate out-of-plain strain and stress components
        if self._problem_type == 4:
            n_dim = self._n_dim
            comp_order_sym = self._comp_order_sym
        else:
            # Set 3D problem parameters
            n_dim, comp_order_sym, _ = get_problem_type_parameters(4)
            # Build strain tensors (matricial form) by including the
            # appropriate out-of-plain components
            inc_strain_mf = vget_state_3Dmf_from_2Dmf(
                inc_strain_mf, comp_33=0.0, device=self._device)
            e_strain_old_mf = vget_state_3Dmf_from_2Dmf(
                e_strain_old_mf, e_strain_33_old, device=self._device)
        # Get number of components
        n_comps = len(comp_order_sym)
        #
        #                                                          State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        _, _, _, fosym, fodiagtrace, _, _ = \
            get_id_operators(n_dim, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic trial strain
        e_trial_strain_mf = e_strain_old_mf + inc_strain_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic consistent tangent modulus according to problem type
        # and store it in matricial form
        if self._problem_type in [1, 4]:
            e_consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
        e_consistent_tangent_mf = vget_tensor_mf(e_consistent_tangent,
                                                 n_dim, comp_order_sym,
                                                 device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial stress
        trial_stress_mf = torch.matmul(e_consistent_tangent_mf,
                                       e_trial_strain_mf)
        trial_stress = vget_tensor_from_mf(trial_stress_mf, n_dim,
                                           comp_order_sym,
                                           device=self._device)
        # Compute trial accumulated plastic strain
        acc_p_trial_strain = acc_p_strain_old
        # Compute trial yield stress
        yield_stress, _ = hardening_law(hardening_parameters,
                                        acc_p_trial_strain)
        # Compute current yield parameters
        yield_a, _ = a_hardening_law(a_hardening_parameters,
                                     acc_p_trial_strain)
        yield_b, _ = b_hardening_law(b_hardening_parameters,
                                     acc_p_trial_strain)
        yield_c, _ = c_hardening_law(c_hardening_parameters,
                                     acc_p_trial_strain)
        yield_d, _ = d_hardening_law(d_hardening_parameters,
                                     acc_p_trial_strain)
        # Compute trial effective stress
        effective_trial_stress = self.get_effective_stress(
            trial_stress, yield_a, yield_b, yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield function
        yield_function = effective_trial_stress - yield_stress
        # Set admissible yield function condition
        yield_function_cond = (yield_function/yield_stress) \
            *torch.ones(2*n_comps + 5, device=self._device) \
                <= su_conv_tol
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the trial stress state lies inside the yield function,
        # then the state update is purely elastic and coincident with the
        # elastic trial state. Otherwise, the state update is elastoplastic
        # and the return-mapping system of nonlinear equations must be solved
        # in order to update the state variables
        #
        # Perform elastic step
        elastic_step_output = self._elastic_step(
            e_trial_strain_mf, trial_stress_mf, acc_p_strain_old)
        # Perform plastic step
        is_elastic_step = (yield_function/yield_stress) <= su_conv_tol
        plastic_step_output = self._plastic_step(
            is_elastic_step, e_trial_strain_mf, trial_stress,
            e_consistent_tangent, acc_p_strain_old, E, K, alpha, hardening_law,
            hardening_parameters, a_hardening_law, a_hardening_parameters,
            b_hardening_law, b_hardening_parameters, c_hardening_law,
            c_hardening_parameters, d_hardening_law, d_hardening_parameters,
            is_associative_hardening, su_conv_tol, su_max_n_iterations, small)
        # Pick elastic or plastic step according with yielding condition
        step_output = torch.where(yield_function_cond,
                                  elastic_step_output,
                                  plastic_step_output)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Unpack state updated variables
        e_strain_mf = step_output[:n_comps]
        stress_mf = step_output[n_comps:2*n_comps]
        acc_p_strain = step_output[2*n_comps]
        inc_p_mult = step_output[2*n_comps + 1]
        is_su_fail = torch.logical_not(
            step_output[2*n_comps + 2].to(torch.bool))
        is_plast = step_output[2*n_comps + 3].to(torch.bool)
        is_apex_return = step_output[2*n_comps + 4].to(torch.bool)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get the out-of-plane strain and stress components
        if self._problem_type == 1:
            e_strain_33 = e_strain_mf[comp_order_sym.index('33')]
            stress_33 = stress_mf[comp_order_sym.index('33')]
        #
        #                                                    3D > 2D Conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the 2D
        # strain and stress tensors (matricial form) once the state update has
        # been performed
        if self._problem_type == 1:
            # Builds 2D strain and stress tensors (matricial form) from the
            # associated 3D counterparts
            e_trial_strain_mf = vget_state_2Dmf_from_3Dmf(
                e_trial_strain_mf, device=self._device)
            e_strain_mf = vget_state_2Dmf_from_3Dmf(
                e_strain_mf, device=self._device)
            stress_mf = vget_state_2Dmf_from_3Dmf(
                stress_mf, device=self._device)
        #
        #                                                Update state variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state variables dictionary
        state_variables = self.state_init()
        # Store updated state variables
        state_variables['e_strain_mf'] = e_strain_mf
        state_variables['acc_p_strain'] = acc_p_strain
        state_variables['strain_mf'] = e_trial_strain_mf + p_strain_old_mf
        state_variables['stress_mf'] = stress_mf
        state_variables['is_su_fail'] = is_su_fail
        state_variables['is_plast'] = is_plast
        if self._problem_type == 1:
            state_variables['e_strain_33'] = e_strain_33
            state_variables['stress_33'] = stress_33
        #
        #                                            Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step condition
        is_plast_cond = is_plast.expand(e_consistent_tangent.shape)
        # Set return-mapping to apex condition
        is_apex_return_cond = is_apex_return.expand(e_consistent_tangent.shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the state update was purely elastic, then the consistent tangent
        # modulus is the elastic consistent tangent modulus. Otherwise, compute
        # the elastoplastic consistent tangent modulus
        consistent_tangent_mf = None
        #
        #                                                    3D > 2D Conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the 2D
        # consistent tangent modulus (matricial form) from the 3D counterpart
        if self._problem_type == 1 and consistent_tangent_mf is not None:
            consistent_tangent_mf = vget_state_2Dmf_from_3Dmf(
                consistent_tangent_mf, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Restore floating-point precision
        if is_precision_conversion:
            # Reset default floating-point precision
            torch.set_default_dtype(torch.float32)
            # Perform floating-point precision conversion
            state_variables = convert_dict_to_float32(state_variables,
                                                      is_inplace=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables, consistent_tangent_mf
    # -------------------------------------------------------------------------
    @classmethod
    def get_stress_invariants(cls, stress):
        """Compute invariants of stress and deviatoric stress.
        
        Parameters
        ----------
        stress : torch.Tensor(2d)
            Stress.
            
        Returns
        -------
        i1 : torch.Tensor(0d)
            First (principal) invariant of stress tensor.
        i2 : torch.Tensor(0d)
            Second (principal) invariant of stress tensor.
        i3 : torch.Tensor(0d)
            Third (principal) invariant of stress tensor.
        j1 : torch.Tensor(0d)
            First invariant of deviatoric stress tensor.
        j2 : torch.Tensor(0d)
            Second invariant of deviatoric stress tensor.
        j3 : torch.Tensor(0d)
            Third invariant of deviatoric stress tensor.
        """
        # Compute first (principal) invariant of stress tensor.
        i1 = torch.trace(stress)
        # Compute second (principal) invariant of stress tensor.
        i2 = 0.5*(torch.trace(stress)**2
                  - torch.trace(torch.matmul(stress, stress)))
        # Compute third (principal) invariant of stress tensor.
        i3 = torch.det(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute first invariant of deviatoric stress tensor
        j1 = i1
        # Compute second invariant of deviatoric stress tensor
        j2 = (1/3)*(i1**2) - i2
        # Compute third invariant of deviatoric stress tensor
        j3 = (2/27)*(i1**3) - (1/3)*i1*i2 + i3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return i1, i2, i3, j1, j2, j3
    # -------------------------------------------------------------------------
    @classmethod
    def get_effective_stress(cls, stress, yield_a, yield_b, yield_c, yield_d):
        """Compute effective stress.
        
        Parameters
        ----------
        stress : torch.Tensor(2d)
            Stress.
        yield_a : torch.Tensor(0d)
            Yield parameter.
        yield_b : torch.Tensor(0d)
            Yield parameter.
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
            
        Returns
        -------
        effective_stress : torch.Tensor(2d)
            Effective stress.
        """
        # Compute stress invariants
        i1, _, _, _, j2, j3 = cls.get_stress_invariants(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliary terms
        aux_1 = yield_b*i1
        aux_2 = j2**3 - yield_c*(j3**2)
        aux_3 = yield_d*j3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute effective stress
        effective_stress = yield_a*(aux_1 + (aux_2**(1/2) - aux_3)**(1/3))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return effective_stress
    # -------------------------------------------------------------------------
    @classmethod
    def get_flow_vector(cls, stress, yield_a, yield_b, yield_c, yield_d):
        """Compute flow vector.
        
        Parameters
        ----------
        stress : torch.Tensor(2d)
            Stress.
        yield_a : torch.Tensor(0d)
            Yield parameter.
        yield_b : torch.Tensor(0d)
            Yield parameter.
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
            
        Returns
        -------
        flow_vector : torch.Tensor(2d)
            Flow vector.
        """
        # Set number of spatial dimensions
        n_dim = 3
        # Get device from stress
        device = stress.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, _, _, _, fodevprojsym = \
            get_id_operators(n_dim, device=device)
        # Compute deviatoric stress tensor
        dev_stress = ddot42_1(fodevprojsym, stress)
        # Compute inverse of deviatoric stress tensor
        dev_stress_inv = torch.inverse(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants
        _, _, _, _, j2, j3 = cls.get_stress_invariants(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get first stress invariant derivative w.r.t. stress
        di1_dstress = soid
        # Get second stress invariant derivative w.r.t. stress
        dj2_dstress = dev_stress
        # Get third stress invariant derivative w.r.t. stress
        dj3_dstress = \
            torch.det(dev_stress)*ddot24_1(dev_stress_inv, fodevprojsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliary terms
        aux_1 = j2**3 - yield_c*(j3**2)
        aux_2 = 3*(j2**2)*dj2_dstress - yield_c*2*j3*dj3_dstress
        aux_3 = yield_d*j3
        aux_4 = yield_d*dj3_dstress
        term_1 = yield_a*yield_b*di1_dstress
        term_2 = yield_a*(1/3)*((aux_1**(1/2) - aux_3)**(-2/3))
        term_3 = (1/2)*(aux_1**(-1/2))*aux_2 - aux_4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute flow vector
        flow_vector = term_1 + term_2*term_3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return flow_vector
    # -------------------------------------------------------------------------
    @classmethod
    def _elastic_step(cls, e_trial_strain_mf, trial_stress_mf,
                      acc_p_strain_old):
        """Perform elastic step.
        
        Parameters
        ----------
        e_trial_strain_mf : torch.Tensor(1d)
            Elastic trial strain (matricial form).
        trial_stress_mf : torch.Tensor(1d)
            Trial stress (matricial form).
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        
        Returns
        -------
        elastic_step_output : torch.Tensor(1d)
            Elastic step concatenated output data.
        """
        # Get device from elastic trial strain
        device = e_trial_strain_mf.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic strain
        e_strain_mf = e_trial_strain_mf
        # Update stress
        stress_mf = trial_stress_mf
        # Update accumulated plastic strain
        acc_p_strain = acc_p_strain_old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step flag
        is_plast = torch.tensor([False], device=device)
        # Set incremental plastic multiplier initial iterative guess
        inc_p_mult = torch.tensor(0.0, device=device)
        # Set state update convergence flag
        is_converged = torch.tensor([True], device=device)
        # Set return-mapping to apex flag
        is_apex_return = torch.tensor([False], device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated elastic step output
        elastic_step_output = \
            torch.cat([e_strain_mf, stress_mf, acc_p_strain.view(-1),
                       inc_p_mult.view(-1),
                       is_converged.view(-1),
                       is_plast.view(-1),
                       is_apex_return.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def _plastic_step(cls, is_elastic_step, e_trial_strain_mf, trial_stress,
                      e_consistent_tangent, acc_p_strain_old, E, K, alpha,
                      hardening_law, hardening_parameters, a_hardening_law,
                      a_hardening_parameters, b_hardening_law,
                      b_hardening_parameters, c_hardening_law,
                      c_hardening_parameters, d_hardening_law,
                      d_hardening_parameters, is_associative_hardening,
                      su_conv_tol, su_max_n_iterations, small):
        """Perform plastic step.
        
        Parameters
        ----------
        is_elastic_step : torch.Tensor(0d)
            If True, then avoid return mapping computations and compute elastic
            response. This flag avoids non-admissible values stemming from
            invalid return-mapping problem and consequent runtime errors when
            computing gradients with autograd.
        e_trial_strain_mf : torch.Tensor(1d)
            Elastic trial strain (matricial form).
        trial_stress : torch.Tensor(2d)
            Trial stress.
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        E : torch.Tensor(0d)
            Young modulus.
        K : torch.Tensor(0d)
            Bulk modulus.
        alpha : torch.Tensor(0d)
            Ratio between yield surface cohesion parameter and yield surface
            pressure parameter.
        hardening_law : function
            Hardening law.
        hardening_parameters : dict
            Hardening law parameters.
        a_hardening_law : function
            Yield parameter hardening law.
        a_hardening_parameters : function
            Yield parameter hardening law parameters.
        b_hardening_law : function
            Yield parameter hardening law.
        b_hardening_parameters : function
            Yield parameter hardening law parameters.
        c_hardening_law : function
            Yield parameter hardening law.
        c_hardening_parameters : function
            Yield parameter hardening law parameters.
        d_hardening_law : function
            Yield parameter hardening law.
        d_hardening_parameters : function
            Yield parameter hardening law parameters.
        is_associative_hardening : bool
            If True, then adopt associative hardening rule.
        su_conv_tol : float
            State update convergence tolerance.
        su_max_n_iterations : int
            State update maximum number of iterations.
        small : float
            Minimum threshold to handle values close or equal to zero.

        Returns
        -------
        plastic_step_output : torch.Tensor(1d)
            Plastic step concatenated output data.
        """
        # Set 3D problem parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(4)
        # Get device from elastic trial strain
        device = e_trial_strain_mf.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step flag
        is_plast = torch.tensor([True], device=device)
        # Get elastic trial strain tensor
        e_trial_strain = vget_tensor_from_mf(e_trial_strain_mf, n_dim,
                                             comp_order_sym,
                                             is_kelvin_notation=True,
                                             device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial accumulated plastic strain
        acc_p_trial_strain = acc_p_strain_old
        # Compute trial yield stress
        yield_stress, _ = hardening_law(hardening_parameters,
                                        acc_p_trial_strain)
        # Compute current yield parameters
        yield_a, _ = a_hardening_law(a_hardening_parameters,
                                     acc_p_trial_strain)
        yield_b, _ = b_hardening_law(b_hardening_parameters,
                                     acc_p_trial_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial pressure
        trial_pressure = (1.0/3.0)*torch.trace(trial_stress)
        # Compute current apex pressure
        pressure_apex = (1.0/(3.0*yield_a*yield_b))*yield_stress
        # Set return-mapping type
        is_apex_return = trial_pressure > pressure_apex
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the trial pressure is greater than the Lou-Zhang-Yoon apex
        # pressure, then solve return-mapping system of nonlinear equations to
        # the cone apex. Otherwise, solve return-mapping system of nonlinear
        # equations to the cone surface
        #
        # Perform plastic step (return-mapping to cone surface)
        plastic_step_cone_output = cls._plastic_step_cone(
            is_elastic_step, e_trial_strain_mf, e_trial_strain,
            e_consistent_tangent, acc_p_strain_old, E, hardening_law,
            hardening_parameters, a_hardening_law, a_hardening_parameters,
            b_hardening_law, b_hardening_parameters, c_hardening_law,
            c_hardening_parameters, d_hardening_law, d_hardening_parameters,
            is_associative_hardening, su_conv_tol, su_max_n_iterations, small)
        # Perform plastic step (return-mapping to cone apex)
        plastic_step_apex_output = cls._plastic_step_apex(
            is_elastic_step, e_trial_strain_mf, trial_pressure,
            acc_p_strain_old, K, alpha, hardening_law, hardening_parameters,
            a_hardening_law, a_hardening_parameters, b_hardening_law,
            b_hardening_parameters, su_conv_tol, su_max_n_iterations, small)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set return-mapping type condition
        cone_surface_cond = torch.logical_not(is_apex_return).expand(
            plastic_step_cone_output.shape)
        # Pick surface or apex plastic step according with condition
        plastic_step_output = torch.where(cone_surface_cond,
                                          plastic_step_cone_output,
                                          plastic_step_apex_output)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated plastic step output
        plastic_step_output = \
            torch.cat([plastic_step_output,
                       is_plast.view(-1),
                       is_apex_return.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return plastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def _plastic_step_cone(cls, is_elastic_step, e_trial_strain_mf,
                           e_trial_strain, e_consistent_tangent,
                           acc_p_strain_old, E, hardening_law,
                           hardening_parameters, a_hardening_law,
                           a_hardening_parameters, b_hardening_law,
                           b_hardening_parameters, c_hardening_law,
                           c_hardening_parameters, d_hardening_law,
                           d_hardening_parameters, is_associative_hardening,
                           su_conv_tol, su_max_n_iterations, small):
        """Perform plastic step (return-mapping to cone surface).

        Parameters
        ----------
        is_elastic_step : torch.Tensor(0d)
            If True, then avoid return mapping computations and compute elastic
            response. This flag avoids non-admissible values stemming from
            invalid return-mapping problem and consequent runtime errors when
            computing gradients with autograd.
        e_trial_strain_mf : torch.Tensor(1d)
            Elastic trial strain (matricial form).
        e_trial_strain : torch.Tensor(2d)
            Elastic trial strain.
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        E : torch.Tensor(0d)
            Young modulus.
        hardening_law : function
            Hardening law.
        hardening_parameters : dict
            Hardening law parameters.
        a_hardening_law : function
            Yield parameter hardening law.
        a_hardening_parameters : function
            Yield parameter hardening law parameters.
        b_hardening_law : function
            Yield parameter hardening law.
        b_hardening_parameters : function
            Yield parameter hardening law parameters.
        c_hardening_law : function
            Yield parameter hardening law.
        c_hardening_parameters : function
            Yield parameter hardening law parameters.
        d_hardening_law : function
            Yield parameter hardening law.
        d_hardening_parameters : function
            Yield parameter hardening law parameters.
        is_associative_hardening : bool
            If True, then adopt associative hardening rule.
        su_conv_tol : float
            State update convergence tolerance.
        su_max_n_iterations : int
            State update maximum number of iterations.
        small : float
            Minimum threshold to handle values close or equal to zero.

        Returns
        -------
        plastic_step_output : torch.Tensor(1d)
            Plastic step concatenated output data.
        """
        # Set 3D problem parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(4)
        # Get device from elastic trial strain
        device = e_trial_strain_mf.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store elastic consistent tangent modulus in matricial form
        e_consistent_tangent_mf = vget_tensor_mf(e_consistent_tangent,
                                                 n_dim, comp_order_sym,
                                                 device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute initial yield stress
        init_yield_stress, _ = hardening_law(
            hardening_parameters,
            acc_p_strain=torch.tensor(0.0, device=device))
        # Set unknowns initial iterative guess
        e_strain = e_trial_strain
        acc_p_strain = acc_p_strain_old
        inc_p_mult = torch.tensor(0.0, device=device)
        # Set null iterative solution vector
        d_iter_null = torch.zeros(len(comp_order_sym)+2, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize norm of iterative solution vector (convergence check)
        diter_norm = torch.tensor(0.0, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Newton-Raphson iterative loop
        for nr_iter in range(su_max_n_iterations):
            # Compute current stress
            stress = ddot42_1(e_consistent_tangent, e_strain)
            # Compute current yield stress and hardening modulus
            yield_stress, hard_slope = \
                hardening_law(hardening_parameters, acc_p_strain)
            # Compute current yield parameters and hardening moduli
            yield_a, a_hard_slope = \
                a_hardening_law(a_hardening_parameters, acc_p_strain)
            yield_b, b_hard_slope = \
                b_hardening_law(b_hardening_parameters, acc_p_strain)
            yield_c, c_hard_slope = \
                c_hardening_law(c_hardening_parameters, acc_p_strain)
            yield_d, d_hard_slope = \
                d_hardening_law(d_hardening_parameters, acc_p_strain)
            # Compute effective stress
            effective_stress = cls.get_effective_stress(
                stress, yield_a, yield_b, yield_c, yield_d)
            # Compute current flow vector and norm
            flow_vector = cls.get_flow_vector(
                stress, yield_a, yield_b, yield_c, yield_d)
            norm_flow_vector = torch.linalg.norm(flow_vector)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute residuals
            residual_1, residual_2, residual_3 = cls.get_residual(
                e_strain, e_trial_strain, acc_p_strain, acc_p_strain_old,
                inc_p_mult, effective_stress, yield_stress, init_yield_stress,
                flow_vector, norm_flow_vector,
                is_associative_hardening=is_associative_hardening)
            # Build residuals matrices
            r1 = vget_tensor_mf(residual_1, n_dim, comp_order_sym,
                                is_kelvin_notation=True, device=device)
            r2 = residual_2.reshape(-1)
            r3 = residual_3.reshape(-1)
            # Build residual vector
            residual = torch.cat((r1, r2, r3), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute residuals convergence norm
            conv_norm_res_1 = torch.where(
                torch.linalg.norm(e_trial_strain) < small,
                torch.linalg.norm(residual_1),
                (torch.linalg.norm(residual_1)
                 /torch.linalg.norm(e_trial_strain)))
            conv_norm_res_2 = torch.where(
                torch.abs(acc_p_strain_old) < small,
                torch.abs(residual_2),
                torch.abs(residual_2/acc_p_strain_old))
            conv_norm_res_3 = torch.abs(residual_3)*(init_yield_stress/E)
            # Compute residual vector convergence norm
            conv_norm_residual = torch.mean(
                torch.stack((conv_norm_res_1, conv_norm_res_2,
                             conv_norm_res_3)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute converge condition
            conv_cond = torch.all(
                torch.stack((conv_norm_residual < su_conv_tol,
                             diter_norm < su_conv_tol,
                             torch.tensor(nr_iter > 0, dtype=torch.bool,
                                          device=device))))
            # Check Newton-Raphson iterative procedure convergence
            is_converged = torch.where(is_elastic_step,
                                       is_elastic_step,
                                       conv_cond)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute iterative solution
            d_iter = torch.where(
                is_converged*torch.ones(len(comp_order_sym) + 2,
                                        dtype=bool, device=device),
                d_iter_null,
                cls._nr_iteration(
                    residual, n_dim, comp_order_sym, stress, inc_p_mult,
                    flow_vector, norm_flow_vector, init_yield_stress,
                    hard_slope, yield_a, a_hard_slope, yield_b, b_hard_slope,
                    yield_c, c_hard_slope, yield_d, d_hard_slope,
                    e_consistent_tangent, is_associative_hardening))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute norm of iterative solution vector (convergence check)
            diter_norm = torch.linalg.norm(d_iter)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Extract iterative solution
            e_strain_iter = vget_tensor_from_mf(
                d_iter[:len(comp_order_sym)], n_dim, comp_order_sym,
                is_kelvin_notation=True, device=device)
            acc_p_strain_iter = d_iter[len(comp_order_sym)]
            inc_p_mult_iter = d_iter[len(comp_order_sym) + 1]
            # Update iterative unknowns
            e_strain = e_strain + e_strain_iter
            acc_p_strain = acc_p_strain + acc_p_strain_iter
            inc_p_mult = inc_p_mult + inc_p_mult_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic strain
        e_strain_mf = vget_tensor_mf(e_strain, n_dim, comp_order_sym,
                                     is_kelvin_notation=True, device=device)
        # Update stress
        stress_mf = torch.matmul(e_consistent_tangent_mf, e_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Expand plastic step convergence condition
        is_converged_cond = is_converged.expand(e_strain_mf.shape)
        # Set elastic strain to NaN if state update fails
        e_strain_mf = torch.where(is_converged_cond,
                                  e_strain_mf,
                                  torch.full(e_strain_mf.shape, torch.nan,
                                             device=device))
        # Set stress to NaN if state update fails
        stress_mf = torch.where(is_converged_cond,
                                stress_mf,
                                torch.full(stress_mf.shape, torch.nan,
                                           device=device))
        # Set accumulated plastic strain to NaN if state update fails
        acc_p_strain = torch.where(is_converged,
                                   acc_p_strain,
                                   torch.tensor(torch.nan, device=device))
        # Set incremental plastic multiplier to NaN if state update fails
        inc_p_mult = torch.where(is_converged,
                                 inc_p_mult,
                                 torch.tensor(torch.nan, device=device))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated plastic step output
        plastic_step_output = \
            torch.cat([e_strain_mf, stress_mf, acc_p_strain.view(-1),
                       inc_p_mult.view(-1),
                       is_converged.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return plastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def get_residual(cls, e_strain, e_trial_strain, acc_p_strain,
                     acc_p_strain_old, inc_p_mult, effective_stress,
                     yield_stress, init_yield_stress, flow_vector,
                     norm_flow_vector, is_associative_hardening=False):
        """Compute state update residuals.
        
        Parameters
        ----------
        e_strain : torch.Tensor(2d)
            Elastic strain.
        e_trial_strain : torch.Tensor(2d)
            Elastic trial strain.
        acc_p_strain : torch.Tensor(0d)
            Accumulated plastic strain.
        acc_p_strain_old : torch.Tensor(0d)
            Last converged accumulated plastic strain.
        inc_p_mult : torch.Tensor(0d)
            Incremental plastic multiplier.
        effective_stress : torch.Tensor(0d)
            Effective stress.
        yield_stress : torch.Tensor(0d)
            Yield stress.
        init_yield_stress : torch.Tensor(0d)
            Initial yield stress.
        flow_vector : torch.Tensor(2d)
            Flow vector.
        norm_flow_vector : torch.Tensor(0d)
            Flow vector norm.
        is_associative_hardening : bool, default=False
            If True, then adopt associative hardening rule.

        Returns
        -------
        residual_1 : torch.Tensor(2d)
            First residual.
        residual_2 : torch.Tensor(2d)
            Second residual.
        residual_3 : torch.Tensor(2d)
            Third residual.
        """
        # Compute first residual
        residual_1 = e_strain - e_trial_strain + inc_p_mult*flow_vector
        # Compute second residual
        if is_associative_hardening:
            residual_2 = acc_p_strain - acc_p_strain_old - inc_p_mult
        else:
            residual_2 = (acc_p_strain - acc_p_strain_old
                          - inc_p_mult*(math.sqrt(2/3))*norm_flow_vector)
        # Compute third residual
        residual_3 = (effective_stress - yield_stress)/init_yield_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return residual_1, residual_2, residual_3
    # -------------------------------------------------------------------------
    @classmethod
    def _nr_iteration(cls, residual, n_dim, comp_order_sym, stress, inc_p_mult,
                      flow_vector, norm_flow_vector, init_yield_stress,
                      hard_slope, yield_a, a_hard_slope, yield_b, b_hard_slope,
                      yield_c, c_hard_slope, yield_d, d_hard_slope,
                      e_consistent_tangent, is_associative_hardening):
        """Newton-Raphson iteration (return-mapping).
        
        Parameters
        ----------
        residual : torch.Tensor(1d)
            Residual.
        n_dim : int
            Problem number of spatial dimensions.
        comp_order_sym : list
            Strain/Stress components symmetric order.
        stress : torch.Tensor(2d)
            Stress.
        inc_p_mult : torch.Tensor(0d)
            Incremental plastic multiplier.
        flow_vector : torch.Tensor(2d)
            Flow vector.
        norm_flow_vector : torch.Tensor(0d)
            Flow vector norm.
        init_yield_stress : torch.Tensor(0d)
            Initial yield stress.
        hard_slope : torch.Tensor(0d)
            Hardening modulus.
        yield_a : torch.Tensor(0d)
            Yield parameter.
        a_hard_slope : torch.Tensor(0d)
            Yield parameter hardening modulus.
        yield_b : torch.Tensor(0d)
            Yield parameter.
        b_hard_slope : torch.Tensor(0d)
            Yield parameter hardening modulus.
        yield_c : torch.Tensor(0d)
            Yield parameter.
        c_hard_slope : torch.Tensor(0d)
            Yield parameter hardening modulus.
        yield_d : torch.Tensor(0d)
            Yield parameter.
        d_hard_slope : torch.Tensor(0d)
            Yield parameter hardening modulus.
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        is_associative_hardening : bool
            If True, then adopt associative hardening rule.
            
        Return
        ------
        d_iter : torch.Tensor(1d)
            Iterative solution vector.
        """
        # Set number of spatial dimensions
        n_dim = 3
        # Get device from elastic trial strain
        device = residual.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, fosym, _, _, fodevprojsym = \
            get_id_operators(n_dim, device=device)
        # Compute deviatoric stress tensor
        dev_stress = ddot42_1(fodevprojsym, stress)
        # Compute inverse of deviatoric stress tensor
        dev_stress_inv = torch.inverse(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants
        i1, _, _, _, j2, j3 = cls.get_stress_invariants(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get first stress invariant derivative w.r.t. stress
        di1_dstress = soid
        # Get second deviatoric stress invariant derivative w.r.t. stress
        dj2_dstress = dev_stress
        # Get third deviatoric stress invariant derivative w.r.t. stress
        dj3_dstress = \
            torch.det(dev_stress)*ddot24_1(dev_stress_inv, fodevprojsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of inverse of deviatoric stress tensor w.r.t
        # stress
        ddevstressinv_dstress = \
            ddot44_1(fo_dinv_sym(dev_stress_inv), fodevprojsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get second stress invariant second-order derivative w.r.t. stress
        d2j2_dstress2 = fodevprojsym
        # Get third stress invariant second-order derivative w.r.t. stress
        d2j3_dstress2 = (
            dyad22_1(ddot24_1(dev_stress_inv, fodevprojsym), dj3_dstress)
            + torch.det(dev_stress)*(ddevstressinv_dstress
            - (1/3)*dyad22_1(soid, ddot24_1(soid, ddevstressinv_dstress))))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliar terms a, b and c
        auxa = yield_b*i1
        auxb = j2**3 - yield_c*(j3**2)
        auxc = yield_d*j3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliar term b derivatives
        dauxb_dstress = 3*(j2**2)*dj2_dstress - yield_c*2*j3*dj3_dstress
        dauxb_daccpstr = -c_hard_slope*(j3**2)
        d2auxb_dstress2 = (3*(2*j2*dyad22_1(dj2_dstress, dj2_dstress)
                              + (j2**2)*d2j2_dstress2)
                           - yield_c*2*(dyad22_1(dj3_dstress, dj3_dstress)
                                        + j3*d2j3_dstress2))
        d2auxb_daccpstrdstress = -c_hard_slope*2*j3*dj3_dstress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliar term
        aux_1 = yield_a*(1/3)*((auxb**(1/2) - auxc)**(-2/3))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of flow vector w.r.t. stress
        dflow_dstress = \
            (aux_1*((1/2)*(-(1/2)*(auxb**(-3/2))*dyad22_1(dauxb_dstress,
                                                          dauxb_dstress)
                           + (auxb**(-1/2))*d2auxb_dstress2)
                    - yield_d*d2j3_dstress2)
             + yield_a*(1/3)*dyad22_1(
                (1/2)*(auxb**(-1/2))*dauxb_dstress - yield_d*dj3_dstress,
                (-2/3)*((auxb**(1/2) - auxc)**(-5/3))*(
                (1/2)*(auxb**(-1/2))*dauxb_dstress - yield_d*dj3_dstress)))
        # Compute derivative of flow vector w.r.t. elastic strain
        dflow_destrain = ddot44_1(dflow_dstress, e_consistent_tangent)
        # Compute derivative of flow vector w.r.t. accumulated plastic strain
        dflow_daccpstr = \
            ((a_hard_slope*yield_b + yield_a*b_hard_slope)*di1_dstress) \
             + (aux_1*((1/2)*(
                 (-1/2)*(auxb**(-3/2))*dauxb_daccpstr*dauxb_dstress
                 + (auxb**(-1/2))*d2auxb_daccpstrdstress)
                 - d_hard_slope*dj3_dstress)) \
             + yield_a*(1/3)*((1/2)*(auxb**(-1/2))*dauxb_dstress
                              - yield_d*dj3_dstress)*(
            -(2/3)*((auxb**(1/2) - d_hard_slope*j3)**(-5/3))*(
                (1/2)*(auxb**(-1/2)))*dauxb_daccpstr - d_hard_slope*j3)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of effective stress w.r.t. stress
        deff_dstress = \
            (yield_a*yield_b*di1_dstress
             + aux_1*((1/2)*(auxb**(-1/2))*dauxb_dstress
                      - yield_d*dj3_dstress))
        # Compute derivative of effective stress w.r.t. elastic strain
        deff_destrain = ddot24_1(deff_dstress, e_consistent_tangent)
        # Compute derivative of effective stress w.r.t. accumulated plastic
        # strain
        deff_daccpstr = \
            (a_hard_slope*(auxa + ((auxb**(1/2)) - auxc)**(1/3))
             + yield_a*b_hard_slope*i1
             + aux_1*((1/2)*(auxb**(-1/2))*dauxb_daccpstr - d_hard_slope*j3))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of first residual w.r.t. to elastic strain
        dr1_destrain = fosym + inc_p_mult*dflow_destrain
        # Compute derivative of first residual w.r.t. to accumulated plastic
        # strain
        dr1_daccpstr = inc_p_mult*dflow_daccpstr
        # Compute derivative of first residual w.r.t. to incremental plastic
        # multiplier
        dr1_dincpm = flow_vector
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivatives of second residual
        if is_associative_hardening:
            # Compute derivative of second residual w.r.t. to elastic strain
            dr2_destrain = torch.zeros_like(flow_vector, device=device)
            # Compute derivative of second residual w.r.t. to accumulated
            # plastic strain
            dr2_daccpstr = torch.tensor(1.0, device=device)
            # Compute derivative of second residual w.r.t. to incremental
            # plastic multiplier
            dr2_dincpm = torch.tensor(-1.0, device=device)
        else:
            # Compute derivative of second residual w.r.t. to elastic strain
            dr2_destrain = \
                -inc_p_mult*math.sqrt(2/3)*(1/norm_flow_vector)*ddot24_1(
                    flow_vector, dflow_destrain)
            # Compute derivative of second residual w.r.t. to accumulated
            # plastic strain
            dr2_daccpstr = \
                1.0 - inc_p_mult*math.sqrt(2/3)*(1/norm_flow_vector)*ddot22_1(
                    flow_vector, dflow_daccpstr)
            # Compute derivative of second residual w.r.t. to incremental
            # plastic multiplier
            dr2_dincpm = -math.sqrt(2/3)*norm_flow_vector
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of third residual w.r.t. to elastic strain
        dr3_destrain = (1/init_yield_stress)*deff_destrain
        # Compute derivative of third residual w.r.t. to accumulated plastic
        # strain
        dr3_daccpstr = (1/init_yield_stress)*(deff_daccpstr - hard_slope)
        # Compute derivative of third residual w.r.t. to incremental plastic
        # multiplier
        dr3_dincpm = torch.tensor(0.0, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build first residual derivatives matrices
        j11 = vget_tensor_mf(dr1_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True, device=device)
        j12 = vget_tensor_mf(dr1_daccpstr, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=device).reshape(-1, 1)
        j13 = vget_tensor_mf(dr1_dincpm, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=device).reshape(-1, 1)
        # Build second residual derivatives matrices
        j21 = vget_tensor_mf(dr2_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=device).reshape(1, -1)
        j22 = dr2_daccpstr.reshape(1, 1)
        j23 = dr2_dincpm.reshape(1, 1)
        # Build third residual derivatives matrices
        j31 = vget_tensor_mf(dr3_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=device).reshape(1, -1)
        j32 = dr3_daccpstr.reshape(1, 1)
        j33 = dr3_dincpm.reshape(1, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build Jacobian matrix
        jacobian = torch.cat((torch.cat((j11, j12, j13), dim=1),
                              torch.cat((j21, j22, j23), dim=1),
                              torch.cat((j31, j32, j33), dim=1)), dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve return-mapping linearized equation
        d_iter = torch.linalg.solve(jacobian, -residual)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return d_iter
    # -------------------------------------------------------------------------
    @classmethod
    def _plastic_step_apex(cls, is_elastic_step, e_trial_strain_mf,
                           trial_pressure, acc_p_strain_old, K, alpha,
                           hardening_law, hardening_parameters,
                           a_hardening_law, a_hardening_parameters,
                           b_hardening_law, b_hardening_parameters,
                           su_conv_tol, su_max_n_iterations, small):
        """Perform plastic step (return-mapping to cone apex).

        Parameters
        ----------
        is_elastic_step : torch.Tensor(0d)
            If True, then avoid return mapping computations and compute elastic
            response. This flag avoids non-admissible values stemming from
            invalid return-mapping problem and consequent runtime errors when
            computing gradients with autograd.
        e_trial_strain_mf : torch.Tensor(1d)
            Elastic trial strain (matricial form).
        trial_pressure : torch.Tensor(0d)
            Trial pressure.
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        K : torch.Tensor(0d)
            Bulk modulus.
        alpha : torch.Tensor(0d)
            Yield parameter equivalent to Drucker-Prager ratio between yield
            surface cohesion parameter and yield surface pressure parameter.
        hardening_law : function
            Hardening law.
        hardening_parameters : dict
            Hardening law parameters.
        a_hardening_law : function
            Yield parameter hardening law.
        a_hardening_parameters : function
            Yield parameter hardening law parameters.
        b_hardening_law : function
            Yield parameter hardening law.
        b_hardening_parameters : function
            Yield parameter hardening law parameters.
        su_conv_tol : float
            State update convergence tolerance.
        su_max_n_iterations : int
            State update maximum number of iterations.
        small : float
            Minimum threshold to handle values close or equal to zero.

        Returns
        -------
        plastic_step_output : torch.Tensor(1d)
            Plastic step concatenated output data.
        """
        # Set 3D problem parameters
        n_dim, comp_order_sym, _ = get_problem_type_parameters(4)
        # Get device from elastic trial strain
        device = e_trial_strain_mf.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required second-order tensor
        soid, _, _, _, _, _, _ = get_id_operators(n_dim, device=device)
        soid_mf = vget_tensor_mf(soid, n_dim, comp_order_sym, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set incremental plastic multiplier
        inc_p_mult = torch.tensor(0.0, device=device)
        # Set incremental plastic volumetric strain initial iterative guess
        inc_vol_p_strain = torch.tensor(0.0, device=device)
        # Compute initial (iterative) yield stress and hardening modulus
        yield_stress, hard_slope = hardening_law(
            hardening_parameters,
            acc_p_strain_old + alpha*inc_vol_p_strain)
        # Compute initial (iterative) yield parameters
        yield_a, _ = a_hardening_law(
            a_hardening_parameters,
            acc_p_strain_old + alpha*inc_vol_p_strain)
        yield_b, _ = b_hardening_law(
            b_hardening_parameters,
            acc_p_strain_old + alpha*inc_vol_p_strain)
        # Compute initial (iterative) material parameter
        beta = 1.0/(3.0*yield_a*yield_b)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize norm of iterative solution vector (convergence
        # check)
        diter_norm = torch.tensor(0.0, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Newton-Raphson iterative loop
        for nr_iter in range(su_max_n_iterations):
            # Compute current yield stress and hardening modulus
            yield_stress, hard_slope = hardening_law(
                hardening_parameters,
                acc_p_strain_old + alpha*inc_vol_p_strain)
            # Compute current yield parameters
            yield_a, _ = a_hardening_law(
                a_hardening_parameters,
                acc_p_strain_old + alpha*inc_vol_p_strain)
            yield_b, _ = b_hardening_law(
                b_hardening_parameters,
                acc_p_strain_old + alpha*inc_vol_p_strain)
            # Compute current material parameter
            beta = 1.0/(3.0*yield_a*yield_b)
            # Compute return-mapping residual (apex)
            residual = yield_stress*beta \
                - (trial_pressure - K*inc_vol_p_strain)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute residual convergence norm
            conv_norm_residual = \
                torch.where(torch.abs(yield_stress) < small,
                            torch.abs(residual),
                            torch.abs(residual)/torch.abs(yield_stress))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute converge condition
            conv_cond = torch.all(
                torch.stack((conv_norm_residual < su_conv_tol,
                             diter_norm < su_conv_tol,
                             torch.tensor(nr_iter > 0, dtype=torch.bool,
                                          device=device))))
            # Check Newton-Raphson iterative procedure convergence 
            is_converged = \
                torch.where(is_elastic_step,
                            is_elastic_step,
                            conv_cond)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute iterative incremental plastic volumetric strain
            inc_vol_p_strain = torch.where(is_converged,
                                           inc_vol_p_strain,
                                           cls._nr_iteration_apex(
                                               inc_vol_p_strain, residual,
                                               alpha, beta, K, hard_slope))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute pressure
        pressure = trial_pressure - K*inc_vol_p_strain
        # Update stress
        stress_mf = pressure*soid_mf
        # Update elastic strain
        e_strain_mf = (1.0/(3.0*K))*pressure*soid_mf
        # Update accumulated plastic strain
        acc_p_strain = acc_p_strain_old + alpha*inc_vol_p_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Expand plastic step convergence condition
        is_converged_cond = is_converged.expand(e_strain_mf.shape)
        # Set elastic strain to NaN if state update fails
        e_strain_mf = torch.where(is_converged_cond,
                                  e_strain_mf,
                                  torch.full(e_strain_mf.shape, torch.nan,
                                             device=device))
        # Set stress to NaN if state update fails
        stress_mf = torch.where(is_converged_cond,
                                stress_mf,
                                torch.full(stress_mf.shape, torch.nan,
                                           device=device))
        # Set accumulated plastic strain to NaN if state update fails
        acc_p_strain = torch.where(is_converged,
                                   acc_p_strain,
                                   torch.tensor(torch.nan, device=device))
        # Set incremental plastic multiplier to NaN if state update fails
        inc_p_mult = torch.where(is_converged,
                                 inc_p_mult,
                                 torch.tensor(torch.nan, device=device))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated plastic step output
        plastic_step_output = \
            torch.cat([e_strain_mf, stress_mf, acc_p_strain.view(-1),
                       inc_p_mult.view(-1), is_converged.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return plastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def _nr_iteration_apex(cls, inc_vol_p_strain, residual, alpha, beta, K,
                           hard_slope):
        """Newton-Raphson iteration (return-mapping to cone apex).
        
        Parameters
        ----------
        inc_vol_p_strain : torch.Tensor(0d)
            Incremental plastic volumetric strain.
        residual : torch.Tensor(0d)
            Residual.
        alpha : torch.Tensor(0d)
            Yield parameter equivalent to Drucker-Prager ratio between yield
            surface cohesion parameter and yield surface pressure parameter.
        beta : torch.Tensor(0d)
            Ratio between yield surface cohesion parameter and plastic flow
            rule pressure parameter.
        K : torch.Tensor(0d)
            Bulk modulus.
        hard_slope : torch.Tensor(0d)
            Hardening modulus.
            
        Return
        ------
        inc_vol_p_strain : torch.Tensor(0d)
            Incremental plastic volumetric strain.
        """
        # Compute return-mapping Jacobian (scalar)
        jacobian = alpha*beta*hard_slope + K
        # Solve return-mapping linearized equation
        d_iter = -residual/jacobian
        # Update incremental plastic multiplier
        inc_vol_p_strain = inc_vol_p_strain + d_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return inc_vol_p_strain