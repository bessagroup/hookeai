"""Von Mises elasto-plastic constitutive model with mixed hardening.

This module includes the implementation of the von Mises constitutive model
with isotropic and kinematic strain hardening.

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
VonMisesMixedVMAP
    Von Mises constitutive model with isotropic and kinematic hardening.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import torch
# Local
from simulators.fetorch.material.models.interface import ConstitutiveModel
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf, vget_state_3Dmf_from_2Dmf, \
    vget_state_2Dmf_from_3Dmf
from simulators.fetorch.math.tensorops import get_id_operators, dyad22_1
from utilities.type_conversion import convert_dict_to_tensor, \
    convert_tensor_to_float64, convert_dict_to_float64, \
    convert_dict_to_float32, convert_tensor_to_float32
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class VonMisesMixedVMAP(ConstitutiveModel):
    """Von Mises constitutive model with isotropic and kinematic hardening.

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
    _is_su_float64 : bool
        If True, then state update is locally computed in floating-point
        double precision. If False, then default floating-point precision
        is assumed.
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
    _elastic_step(cls, e_trial_strain_mf, trial_stress_mf, acc_p_strain_old, \
                  back_stress_old_mf)
        Perform elastic step.
    _plastic_step(cls, is_elastic_step, e_trial_strain_mf, \
                  relative_eq_trial_stress, e_consistent_tangent_mf, \
                  flow_vector_mf, acc_p_strain_old, back_stress_old_mf, \
                  G, hardening_law, hardening_parameters, \
                  kinematic_hardening_law, kinematic_hardening_parameters, \
                  su_conv_tol, su_max_n_iterations):
        Perform plastic step.
    _nr_iteration(cls, inc_p_mult, residual, G, H, kin_hard_slope)
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
        self._name = 'von_mises_mixed'
        # Set constitutive model strain formulation
        self._strain_type = 'infinitesimal'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set initialization parameters
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._model_parameters = convert_dict_to_tensor(model_parameters,
                                                        is_inplace=True)
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
            raise RuntimeError('The von Mises constitutive model is currently '
                               'only available for the elastic isotropic '
                               'case.')
    # -------------------------------------------------------------------------
    @staticmethod
    def get_required_model_parameters():
        """Get required material constitutive model parameters.
        
        Model parameters:
        
        - 'elastic_symmetry' : Elastic symmetry (str, {'isotropic',
          'transverse_isotropic', 'orthotropic', 'monoclinic', 'triclinic'});
        - 'elastic_moduli' : Elastic moduli (dict, {'Eijkl': float});
        - 'euler_angles' : Euler angles (degrees) sorted according with Bunge
           convention (tuple[float]).
        - 'hardening_law' : Isotropic hardening law (function)
        - 'hardening_parameters' : Isotropic hardening law parameters (dict)
        - 'kinematic_hardening_law' : Kinematic hardening law (function)
        - 'kinematic_hardening_parameters' : Kinematic hardening law
                                             parameters (dict)

        Returns
        -------
        model_parameters_names : tuple[str]
            Material constitutive model parameters names (str).
        """
        # Set material properties names
        model_parameters_names = ('elastic_symmetry', 'elastic_moduli',
                                  'euler_angles', 'hardening_law',
                                  'hardening_parameters',
                                  'kinematic_hardening_law',
                                  'kinematic_hardening_parameters')
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

        * ``back_stress_mf``

            * *Infinitesimal strains*: Back-stress tensor (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\beta}`

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
        state_variables_init['back_stress_mf'] = vget_tensor_mf(
            torch.zeros((self._n_dim, self._n_dim), device=self._device),
                        self._n_dim, self._comp_order_sym)
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
            state_variables_init['back_stress_33'] = \
                torch.tensor(0.0, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
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
        su_max_n_iterations = 20
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
        # Get material kinematic strain hardening law
        kinematic_hardening_law = \
            model_parameters['kinematic_hardening_law']
        kinematic_hardening_parameters = \
            model_parameters['kinematic_hardening_parameters']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute shear modulus
        G = E/(2.0*(1.0 + v))
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get last increment converged state variables
        e_strain_old_mf = state_variables_old['e_strain_mf']
        p_strain_old_mf = state_variables_old['strain_mf'] - e_strain_old_mf
        acc_p_strain_old = state_variables_old['acc_p_strain']
        back_stress_old_mf = state_variables_old['back_stress_mf']
        if self._problem_type == 1:
            e_strain_33_old = state_variables_old['e_strain_33']
            back_stress_33_old = state_variables_old['back_stress_33']
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
            # Build back-stress tensor (matricial form) by including the
            # appropriate out-of-plain component
            back_stress_old_mf = vget_state_3Dmf_from_2Dmf(
                back_stress_old_mf, back_stress_33_old, device=self._device)
        # Get number of components
        n_comps = len(comp_order_sym)
        #
        #                                                          State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        _, _, _, fosym, fodiagtrace, _, fodevprojsym = \
            get_id_operators(n_dim, device=self._device)
        fodevprojsym_mf = vget_tensor_mf(fodevprojsym, n_dim, comp_order_sym,
                                         device=self._device)
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
        # Compute deviatoric trial stress
        dev_trial_stress_mf = torch.matmul(fodevprojsym_mf, trial_stress_mf)
        # Compute trial relative stress
        relative_stress_mf = dev_trial_stress_mf - back_stress_old_mf
        # Compute relative equivalent trial stress
        relative_eq_trial_stress = \
            math.sqrt(3.0/2.0)*torch.norm(relative_stress_mf)
        # Compute trial accumulated plastic strain
        acc_p_trial_strain = acc_p_strain_old
        # Compute trial yield stress
        yield_stress, _ = \
            hardening_law(hardening_parameters, acc_p_trial_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial relative stress norm divison factor
        norm_div_factor = torch.where(
            torch.norm(relative_stress_mf) > small,
            1.0/torch.norm(relative_stress_mf + small),
            torch.zeros(1, device=self._device))
        # Compute flow vector
        flow_vector_mf = math.sqrt(3.0/2.0)*norm_div_factor*relative_stress_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield function
        yield_function = relative_eq_trial_stress - yield_stress        
        # Set admissible yield function condition
        yield_function_cond = (yield_function/yield_stress) \
            *torch.ones(3*n_comps + 4, device=self._device) \
                <= su_conv_tol
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the trial stress state lies inside the von Mises yield function,
        # then the state update is purely elastic and coincident with the
        # elastic trial state. Otherwise, the state update is elastoplastic
        # and the return-mapping system of nonlinear equations must be solved
        # in order to update the state variables
        #
        # Perform elastic step
        elastic_step_output = self._elastic_step(
            e_trial_strain_mf, trial_stress_mf, acc_p_strain_old,
            back_stress_old_mf)
        # Perform plastic step
        is_elastic_step = (yield_function/yield_stress) <= su_conv_tol
        plastic_step_output = self._plastic_step(
            is_elastic_step, e_trial_strain_mf, relative_eq_trial_stress,
            e_consistent_tangent_mf, flow_vector_mf, acc_p_strain_old,
            back_stress_old_mf, G, hardening_law, hardening_parameters,
            kinematic_hardening_law, kinematic_hardening_parameters,
            su_conv_tol, su_max_n_iterations)
        # Pick elastic or plastic step according with yielding condition
        step_output = torch.where(yield_function_cond,
                                  elastic_step_output,
                                  plastic_step_output)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Unpack state updated variables
        e_strain_mf = step_output[:n_comps]
        stress_mf = step_output[n_comps:2*n_comps]
        acc_p_strain = step_output[2*n_comps]
        back_stress_mf = step_output[2*n_comps + 1:3*n_comps + 1]
        inc_p_mult = step_output[3*n_comps + 1]
        is_plast = step_output[3*n_comps + 2].to(torch.bool)
        is_su_fail = \
            torch.logical_not(step_output[3*n_comps + 3].to(torch.bool))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get the out-of-plane strain and stress components
        if self._problem_type == 1:
            e_strain_33 = e_strain_mf[comp_order_sym.index('33')]
            stress_33 = stress_mf[comp_order_sym.index('33')]
            back_stress_33 = back_stress_mf[comp_order_sym.index('33')]
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
            back_stress_mf = vget_state_2Dmf_from_3Dmf(
                back_stress_mf, device=self._device)
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
        state_variables['back_stress_mf'] = back_stress_mf
        state_variables['is_su_fail'] = is_su_fail
        state_variables['is_plast'] = is_plast
        if self._problem_type == 1:
            state_variables['e_strain_33'] = e_strain_33
            state_variables['stress_33'] = stress_33
            state_variables['back_stress_33'] = back_stress_33
        #
        #                                            Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step condition
        is_plast_cond = is_plast.expand(e_consistent_tangent.shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the state update was purely elastic, then the consistent tangent
        # modulus is the elastic consistent tangent modulus. Otherwise, compute
        # the elastoplastic consistent tangent modulus
        #
        # Compute plastic consistent tangent modulus
        _, H = hardening_law(hardening_parameters, acc_p_strain)
        factor_1 = ((inc_p_mult*6.0*G**2)/relative_eq_trial_stress)
        factor_2 = (6.0*G**2)*((inc_p_mult/relative_eq_trial_stress)
                                - (1.0/(3.0*G + H)))
        unit_flow_vector = math.sqrt(2.0/3.0)*vget_tensor_from_mf(
            flow_vector_mf, n_dim, comp_order_sym, device=self._device)
        p_consistent_tangent = e_consistent_tangent \
            - factor_1*fodevprojsym + factor_2*dyad22_1(
                unit_flow_vector, unit_flow_vector)
        # Pick consistent tangent modulus according with plastic step condition
        consistent_tangent = torch.where(is_plast_cond,
                                         p_consistent_tangent,
                                         e_consistent_tangent)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build consistent tangent modulus matricial form
        consistent_tangent_mf = vget_tensor_mf(consistent_tangent, n_dim,
                                               comp_order_sym,
                                               device=self._device)
        #
        #                                                    3D > 2D Conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the 2D
        # consistent tangent modulus (matricial form) from the 3D counterpart
        if self._problem_type == 1:
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
            consistent_tangent_mf = \
                convert_tensor_to_float32(consistent_tangent_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables, consistent_tangent_mf
    # -------------------------------------------------------------------------
    @classmethod
    def _elastic_step(cls, e_trial_strain_mf, trial_stress_mf,
                      acc_p_strain_old, back_stress_old_mf):
        """Perform elastic step.
        
        Parameters
        ----------
        e_trial_strain_mf : torch.Tensor(1d)
            Elastic trial strain (matricial form).
        trial_stress_mf : torch.Tensor(1d)
            Trial stress (matricial form).
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        back_stress_old_mf : torch.Tensor(1d)
            Last converged back-stress (matricial form).
        
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
        # Update back-stress tensor
        back_stress_mf = back_stress_old_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step flag
        is_plast = torch.tensor([False], device=device)
        # Set incremental plastic multiplier initial iterative guess
        inc_p_mult = torch.tensor(0.0, device=device)
        # Set state update convergence flag
        is_converged = torch.tensor([True], device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated elastic step output
        elastic_step_output = \
            torch.cat([e_strain_mf, stress_mf, acc_p_strain.view(-1),
                       back_stress_mf.view(-1),
                       inc_p_mult.view(-1),
                       is_plast.view(-1),
                       is_converged.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return elastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def _plastic_step(cls, is_elastic_step, e_trial_strain_mf,
                      relative_eq_trial_stress, e_consistent_tangent_mf,
                      flow_vector_mf, acc_p_strain_old, back_stress_old_mf,
                      G, hardening_law, hardening_parameters,
                      kinematic_hardening_law, kinematic_hardening_parameters,
                      su_conv_tol, su_max_n_iterations):
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
        relative_eq_trial_stress : torch.Tensor(1d)
            Relative equivalent trial stress.
        e_consistent_tangent_mf : torch.Tensor(2d)
            Elastic consistent tangent modulus (matricial form).
        flow_vector_mf : torch.Tensor(1d)
            Flow vector.
        acc_p_strain_old : torch.Tensor(0d)
            Last convergence accumulated plastic strain.
        back_stress_old_mf : torch.Tensor(1d)
            Last converged back-stress (matricial form).
        G : torch.Tensor(0d)
            Shear modulus.
        hardening_law : function
            Hardening law.
        hardening_parameters : dict
            Hardening law parameters.
        kinematic_hardening_law : function
            Kinematic hardening law.
        kinematic_hardening_parameters : dict
            Kinematic hardening law parameters.
        su_conv_tol : float
            State update convergence tolerance.
        su_max_n_iterations : int
            State update maximum number of iterations.

        Returns
        -------
        plastic_step_output : torch.Tensor(1d)
            Plastic step concatenated output data.
        """
        # Get device from elastic trial strain
        device = e_trial_strain_mf.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plastic step flag
        is_plast = torch.tensor([True], device=device)
        # Set incremental plastic multiplier initial iterative guess
        inc_p_mult = torch.tensor(0.0, device=device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute last converged kinematic hardening stress
        kin_hard_stress_old, _ = kinematic_hardening_law(
            kinematic_hardening_parameters, acc_p_strain_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Newton-Raphson iterative loop
        for nr_iter in range(su_max_n_iterations + 1):
            # Compute current yield stress and hardening modulus
            yield_stress, H = hardening_law(hardening_parameters,
                                            acc_p_strain_old + inc_p_mult)
            # Compute current kinematic hardening stress and modulus
            kin_hard_stress, kin_hard_slope = \
                kinematic_hardening_law(kinematic_hardening_parameters,
                                        acc_p_strain_old + inc_p_mult)
            # Compute return-mapping residual (scalar)
            residual = relative_eq_trial_stress - 3.0*G*inc_p_mult \
                - kin_hard_stress + kin_hard_stress_old - yield_stress
            # Compute converge condition
            error = abs(residual/yield_stress)
            conv_cond = torch.all(
                torch.stack((error < su_conv_tol,
                             torch.tensor(nr_iter > 0, dtype=torch.bool,
                                          device=device))))
            # Check Newton-Raphson iterative procedure convergence
            is_converged = torch.where(is_elastic_step,
                                       is_elastic_step,
                                       conv_cond)
            # Compute iterative incremental plastic multiplier
            inc_p_mult = torch.where(is_converged,
                                     inc_p_mult,
                                     cls._nr_iteration(inc_p_mult, residual,
                                                       G, H, kin_hard_slope))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set incremental plastic multiplier to NaN if state update fails
        inc_p_mult = torch.where(is_converged,
                                 inc_p_mult,
                                 torch.tensor(torch.nan))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic strain
        e_strain_mf = e_trial_strain_mf - inc_p_mult*flow_vector_mf
        # Update stress
        stress_mf = torch.matmul(e_consistent_tangent_mf, e_strain_mf)
        # Update accumulated plastic strain
        acc_p_strain = acc_p_strain_old + inc_p_mult
        # Update back-stress
        back_stress_mf = back_stress_old_mf \
            + (kin_hard_stress - kin_hard_stress_old)*flow_vector_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build concatenated plastic step output
        plastic_step_output = \
            torch.cat([e_strain_mf, stress_mf, acc_p_strain.view(-1),
                       back_stress_mf.view(-1),
                       inc_p_mult.view(-1),
                       is_plast.view(-1),
                       is_converged.view(-1)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return plastic_step_output
    # -------------------------------------------------------------------------
    @classmethod
    def _nr_iteration(cls, inc_p_mult, residual, G, H, kin_hard_slope):
        """Newton-Raphson iteration (return-mapping).
        
        Parameters
        ----------
        inc_p_mult : torch.Tensor(0d)
            Incremental plastic multiplier.
        residual : torch.Tensor(0d)
            Residual.
        G : torch.Tensor(0d)
            Shear modulus.
        H : torch.Tensor(0d)
            Hardening modulus.
        kin_hard_slope : torch.Tensor(0d)
            Kinematic hardening modulus.
            
        Return
        ------
        inc_p_mult : torch.Tensor(0d)
            Incremental plastic multiplier.
        """
        # Compute return-mapping Jacobian (scalar)
        jacobian = -3.0*G - kin_hard_slope - H
        # Solve return-mapping linearized equation
        d_iter = -residual/jacobian
        # Update incremental plastic multiplier
        inc_p_mult = inc_p_mult + d_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return inc_p_mult