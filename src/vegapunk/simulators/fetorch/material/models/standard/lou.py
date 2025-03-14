"""Lou-Zhang-Yoon model with general differentiable yield function.

This module includes the implementation of the Lou-Zhang-Yoon model with
general differentiable yield function and isotropic hardening.

The apex singularity is handled by means of a purely volumetric return-mapping
along the hydrostatic axis.

It also includes several tools to check the yield surface convexity.

Classes
-------
LouZhangYoon
    Lou-Zhang-Yoon model with general differentiable yield function.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from ioput.plots import plot_xy_data, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class LouZhangYoon(ConstitutiveModel):
    """Lou-Zhang-Yoon model with general differentiable yield function.

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
    get_stress_invariants(self, stress)
        Compute invariants of stress and deviatoric stress.
    get_stress_invariants_and_derivatives(self, n_dim, stress)
        Compute stress invariants and derivatives w.r.t. stress.
    get_effective_stress(self, stress, yield_a, yield_b, yield_c, yield_d)
        Compute effective stress.
    get_residual_and_jacobian(self, n_dim, comp_order_sym, e_strain, \
                              e_trial_strain, acc_p_strain, \
                              acc_p_strain_old, inc_p_mult, \
                              e_consistent_tangent, init_yield_stress, \
                              hardening_law, hardening_parameters, \
                              a_hardening_law, a_hardening_parameters, \
                              b_hardening_law, b_hardening_parameters, \
                              c_hardening_law, c_hardening_parameters, \
                              d_hardening_law, d_hardening_parameters, \
                              is_associative_hardening=False)
        Compute state update residuals and Jacobian matrix.
    get_numerical_jacobian(self, n_dim, comp_order_sym, e_strain, \
                           e_trial_strain, acc_p_strain, acc_p_strain_old, \
                           inc_p_mult, e_consistent_tangent, \
                           init_yield_stress, \
                           hardening_law, hardening_parameters,
                           a_hardening_law, a_hardening_parameters,
                           b_hardening_law, b_hardening_parameters,
                           c_hardening_law, c_hardening_parameters,
                           d_hardening_law, d_hardening_parameters,
                           is_associative_hardening=False, is_verbose=False)
        Compute state update Jacobian matrix with finite differences.
    compute_num_derivatives(self, n_dim, comp_order_sym, e_strain,
                            e_consistent_tangent)
        Compute numerical derivatives by finite differences.
    convexity_return_mapping(cls, yield_c, yield_d)
        Perform convexity return-mapping.
    compute_convex_boundary(cls, n_theta=360)
        Compute convexity domain boundary.
    directional_convex_boundary(cls, theta, r_lower=0.0, r_upper=4.0, \
                                search_tol=1e-6)
        Compute convexity domain boundary along given angular direction.
    check_yield_surface_convexity(cls, yield_c, yield_d)
        Check yield surface convexity.
    plot_convexity_boundary(cls, convex_boundary, parameters_paths=None, \
                            is_plot_legend=False, save_dir=None, \
                            is_save_fig=False, is_stdout_display=False, \
                            is_latex=False)
        Plot convexity domain boundary.
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
        # Set verbose flag
        is_verbose = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Set apex return-mapping switch tolerance
        apex_switch_tol = 0.045
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
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
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
        #
        #                                                          State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, fosym, fodiagtrace, _, _ = \
            get_id_operators(n_dim, device=self._device)
        soid_mf = vget_tensor_mf(soid, n_dim, comp_order_sym,
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
        # If the trial stress state lies inside the yield function,
        # then the state update is purely elastic and coincident with the
        # elastic trial state. Otherwise, the state update is elastoplastic
        # and the return-mapping system of nonlinear equations must be solved
        # in order to update the state variables
        if yield_function/yield_stress <= su_conv_tol:
            # Update elastic strain
            e_strain_mf = e_trial_strain_mf
            # Update stress
            stress_mf = trial_stress_mf
            # Update accumulated plastic strain
            acc_p_strain = acc_p_strain_old
        else:
            # Set plastic step flag
            is_plast = torch.tensor(True, device=self._device)
            # Get elastic trial strain tensor
            e_trial_strain = vget_tensor_from_mf(e_trial_strain_mf, n_dim,
                                                 comp_order_sym,
                                                 is_kelvin_notation=True,
                                                 device=self._device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute initial yield parameters
            yield_a_init, _ = a_hardening_law(
                a_hardening_parameters,
                acc_p_strain=torch.tensor(0.0, device=self._device))
            yield_b_init, _ = b_hardening_law(
                b_hardening_parameters,
                acc_p_strain=torch.tensor(0.0, device=self._device))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute trial pressure
            trial_pressure = (1.0/3.0)*torch.trace(trial_stress)
            # Compute current apex pressure
            safe_yield_b = torch.max(
                torch.abs(yield_b), torch.tensor(1e-6, device=self._device))
            pressure_apex = (1.0/(3.0*yield_a*safe_yield_b))*yield_stress
            # Set return-mapping type
            is_apex_return = \
                trial_pressure > (1.0 - apex_switch_tol)*pressure_apex
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute return-mapping to surface or apex
            if is_apex_return:
                # Get Drucker-Prager pressure and cohesion equivalent
                # parameters
                etay = 3.0*yield_a_init*yield_b_init
                xi = (2.0*math.sqrt(3)/3.0)*torch.sqrt(1.0 - (1.0/3.0)*etay**2)
                # Compute yield parameter equivalent to Drucker-Prager ratio
                # between yield surface cohesion parameter and yield surface
                # pressure parameter
                alpha = xi/etay
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set incremental plastic volumetric strain initial iterative
                # guess
                inc_vol_p_strain = torch.tensor(0.0, device=self._device)
                # Compute initial (iterative) yield stress and hardening
                # modulus
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
                # Initialize Newton-Raphson iteration counter
                nr_iter = 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize convergence norm of iterative solution vector
                conv_diter_norm = torch.tensor(0.0, device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print(f'\n\nPlastic Increment - Newton-Raphson')
                    print('----------------------------------')
                    print('nr_iter   conv_norm_res   conv_diter_norm')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Start Newton-Raphson iterative loop
                while True:
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
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
                    # Compute residual convergence norm
                    if torch.abs(yield_stress) < small:
                        conv_norm_res = torch.abs(residual)
                    else:
                        conv_norm_res = \
                            torch.abs(residual)/torch.abs(yield_stress)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Display iterative residuals
                    if is_verbose:
                        print(f'{nr_iter:7d}   {conv_norm_res:^15.4e}   '
                              f'{conv_diter_norm:^12.4e}')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check Newton-Raphson iterative procedure convergence 
                    is_converged = (conv_norm_res < su_conv_tol
                                    and conv_diter_norm < su_conv_tol
                                    and nr_iter > 0)
                    # Control Newton-Raphson iteration loop flow
                    if is_converged:
                        # Display convergence status
                        if is_verbose:
                            print(f'{"Solution converged!":^74s}')
                        # Leave Newton-Raphson iterative loop (converged
                        # solution)
                        break
                    elif nr_iter == su_max_n_iterations:
                        # Display convergence status
                        if is_verbose:
                            print(f'{"Solution convergence failure!":^74s}')
                        # Update state update failure flag
                        is_su_fail = torch.tensor(True, device=self._device)
                        # Leave Newton-Raphson iterative loop (failed solution)
                        break
                    else:
                        # Increment iteration counter
                        nr_iter = nr_iter + 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute return-mapping Jacobian (scalar)
                    jacobian = alpha*beta*hard_slope + K
                    # Solve return-mapping linearized equation
                    d_iter = -residual/jacobian
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute convergence norm of iterative solution vector
                    conv_diter = d_iter.detach().clone()
                    if torch.abs(acc_p_strain_old) > small:
                        conv_diter = conv_diter/acc_p_strain_old
                    conv_diter_norm = torch.linalg.norm(conv_diter)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update incremental plastic volumetric strain
                    inc_vol_p_strain = inc_vol_p_strain + d_iter   
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute pressure
                pressure = trial_pressure - K*inc_vol_p_strain
                # Update stress
                stress_mf = pressure*soid_mf
                # Update elastic strain
                e_strain = (1.0/(3.0*K))*pressure*soid
                # Update accumulated plastic strain
                acc_p_strain = acc_p_strain_old + alpha*inc_vol_p_strain
            else:
                # Compute initial yield stress
                init_yield_stress, _ = hardening_law(
                    hardening_parameters,
                    acc_p_strain=torch.tensor(0.0, device=self._device))
                # Set unknowns initial iterative guess
                e_strain = e_trial_strain
                acc_p_strain = acc_p_strain_old
                inc_p_mult = torch.tensor(0.0, device=self._device)
                # Initialize Newton-Raphson iteration counter
                nr_iter = 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize convergence norm of iterative solution vector
                conv_diter_norm = torch.tensor(0.0, device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if is_verbose:
                    print(f'\n\nPlastic Increment - Newton-Raphson')
                    print('----------------------------------')
                    print('nr_iter   conv_norm_res_1   conv_norm_res_2   '
                          'conv_norm_res_3   conv_diter_norm')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Start Newton-Raphson iterative loop
                while True:
                    # Compute return-mapping residuals and Jacobian
                    residual_1, residual_2, residual_3, jacobian = \
                        self.get_residual_and_jacobian(
                            n_dim, comp_order_sym, e_strain, e_trial_strain,
                            acc_p_strain, acc_p_strain_old, inc_p_mult,
                            e_consistent_tangent, init_yield_stress,
                            hardening_law, hardening_parameters,
                            a_hardening_law, a_hardening_parameters,
                            b_hardening_law, b_hardening_parameters,
                            c_hardening_law, c_hardening_parameters,
                            d_hardening_law, d_hardening_parameters,
                            is_associative_hardening=is_associative_hardening)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Build residuals matrices
                    r1 = vget_tensor_mf(residual_1, n_dim, comp_order_sym,
                                        is_kelvin_notation=True,
                                        device=self._device)
                    r2 = residual_2.reshape(-1)
                    r3 = residual_3.reshape(-1)
                    # Build residual vector
                    residual = torch.cat((r1, r2, r3), dim=0)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute residuals convergence norm
                    if torch.linalg.norm(e_trial_strain) < small:
                        conv_norm_res_1 = torch.linalg.norm(residual_1)
                    else:
                        conv_norm_res_1 = (torch.linalg.norm(residual_1)/
                                           torch.linalg.norm(e_trial_strain))
                    if torch.abs(acc_p_strain_old) < small:
                        conv_norm_res_2 = torch.abs(residual_2)
                    else:
                        conv_norm_res_2 = \
                            torch.abs(residual_2/acc_p_strain_old)
                    conv_norm_res_3 = torch.abs(residual_3)
                    # Compute residual vector convergence norm
                    conv_norm_residual = torch.mean(
                        torch.tensor((conv_norm_res_1, conv_norm_res_2,
                                      conv_norm_res_3)))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Display iterative residuals
                    if is_verbose:
                        print(f'{nr_iter:7d}   {conv_norm_res_1:^15.4e}   '
                              f'{conv_norm_res_2:^15.4e}   '
                              f'{conv_norm_res_3:^15.4e}   '
                              f'{conv_diter_norm:^15.4e}')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check Newton-Raphson iterative procedure convergence
                    is_converged = (conv_norm_residual < su_conv_tol
                                    and conv_diter_norm < su_conv_tol
                                    and nr_iter > 0)
                    # Control Newton-Raphson iteration loop flow
                    if is_converged:
                        # Display convergence status
                        if is_verbose:
                            print(f'{"Solution converged!":^74s}')
                        # Leave Newton-Raphson iterative loop (converged
                        # solution)
                        break
                    elif nr_iter == su_max_n_iterations:
                        # Display convergence status
                        if is_verbose:
                            print(f'{"Solution convergence failure!":^74s}')
                        # Update state update failure flag
                        is_su_fail = torch.tensor(True, device=self._device)
                        # Leave Newton-Raphson iterative loop (failed solution)
                        break
                    else:
                        # Increment iteration counter
                        nr_iter = nr_iter + 1
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Solve return-mapping linearized equation
                    d_iter = torch.linalg.solve(jacobian, -residual)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute convergence norm of iterative solution vector
                    conv_diter = d_iter.detach().clone()
                    if torch.linalg.norm(e_trial_strain) > small:
                        conv_diter[:len(comp_order_sym)] = \
                            conv_diter[:len(comp_order_sym)] \
                                /torch.linalg.norm(e_trial_strain)
                    if torch.abs(acc_p_strain_old) > small:
                        conv_diter[len(comp_order_sym)] = \
                            conv_diter[len(comp_order_sym)]/acc_p_strain_old
                        conv_diter[len(comp_order_sym)+1] = \
                            conv_diter[len(comp_order_sym)+1]/acc_p_strain_old
                    conv_diter_norm = torch.linalg.norm(conv_diter)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Extract iterative solution
                    e_strain_iter = vget_tensor_from_mf(
                        d_iter[:len(comp_order_sym)], n_dim, comp_order_sym,
                        is_kelvin_notation=True, device=self._device)
                    acc_p_strain_iter = d_iter[len(comp_order_sym)]
                    inc_p_mult_iter = d_iter[len(comp_order_sym) + 1]
                    # Update iterative unknowns
                    e_strain = e_strain + e_strain_iter
                    acc_p_strain = acc_p_strain + acc_p_strain_iter
                    inc_p_mult = inc_p_mult + inc_p_mult_iter
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update elastic strain
            e_strain_mf = vget_tensor_mf(e_strain, n_dim, comp_order_sym,
                                         is_kelvin_notation=True,
                                         device=self._device)
            # Update stress
            stress_mf = torch.matmul(e_consistent_tangent_mf, e_strain_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set state variables to NaN if state update fails
            if is_su_fail:
                # Set elastic strain to NaN if state update fails
                e_strain_mf = torch.full(e_strain_mf.shape, torch.nan,
                                         device=self._device)
                # Set stress to NaN if state update fails
                stress_mf = torch.full(stress_mf.shape, torch.nan,
                                       device=self._device)
                # Set accumulated plastic strain to NaN if state update fails
                acc_p_strain = torch.tensor(torch.nan, device=self._device)
                # Set incremental plastic multiplier to NaN if state update
                # fails
                inc_p_mult = torch.tensor(torch.nan, device=self._device)
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
    def get_stress_invariants(self, stress):
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
    def get_stress_invariants_and_derivatives(self, n_dim, stress):
        """Compute stress invariants and derivatives w.r.t. stress.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
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
        di1_dstress : torch.Tensor(1d)
            First-order derivative of first invariant of stress tensor
            w.r.t. stress.
        dj2_dstress : torch.Tensor(1d)
            First-order derivative of second invariant of deviatoric stress
            tensor w.r.t. stress.
        dj3_dstress : torch.Tensor(1d)
            First-order derivative of third invariant of deviatoric stress
            tensor w.r.t. stress.
        d2j2_dstress2 : torch.Tensor(1d)
            Second-order derivative of second invariant of deviatoric stress
            tensor w.r.t. stress.
        d2j3_dstress2 : torch.Tensor(1d)
            Second-order derivative of third invariant of deviatoric stress
            tensor w.r.t. stress.
        """
        # Set required fourth-order tensors
        soid, _, _, _, _, _, fodevprojsym = \
            get_id_operators(n_dim, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute deviatoric stress tensor
        dev_stress = ddot42_1(fodevprojsym, stress)
        # Compute determinant of deviatoric stress tensor
        dev_stress_det = torch.det(dev_stress)
        # Compute inverse of deviatoric stress tensor
        dev_stress_inv = torch.inverse(dev_stress)
        # Compute derivative of inverse of deviatoric strss tensor w.r.t itself
        ddsinv_ddsinv = fo_dinv_sym(dev_stress_inv)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliary term
        w6 = ddot24_1(dev_stress_inv, fodevprojsym)
        # Compute auxiliary term derivative
        dw6_dstress = ddot44_1(ddot44_1(fodevprojsym, ddsinv_ddsinv),
                               fodevprojsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants
        i1, i2, i3, j1, j2, j3 = self.get_stress_invariants(stress)
        # Compute derivatives w.r.t. stress
        di1_dstress = soid
        dj2_dstress = dev_stress
        dj3_dstress = dev_stress_det*w6
        # Compute second-order derivatives w.r.t. stress
        d2j2_dstress2 = fodevprojsym
        d2j3_dstress2 = dyad22_1(w6, dj3_dstress) + dev_stress_det*dw6_dstress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return i1, i2, i3, j1, j2, j3, di1_dstress, dj2_dstress, dj3_dstress, \
            d2j2_dstress2, d2j3_dstress2
    # -------------------------------------------------------------------------
    def get_effective_stress(self, stress, yield_a, yield_b, yield_c, yield_d):
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
        effective_stress : torch.Tensor(0d)
            Effective stress.
        """
        # Compute stress invariants
        i1, _, _, _, j2, j3 = self.get_stress_invariants(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliary terms
        w1 = yield_b*i1
        w2 = yield_c*(j3**2)
        w3 = yield_d*j3
        w4 = j2**3 - w2
        w5 = w4**(1/2) - w3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute effective stress
        effective_stress = yield_a*(w1 + (w5**(1/3)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return effective_stress
    # -------------------------------------------------------------------------
    def get_residual_and_jacobian(self, n_dim, comp_order_sym, e_strain,
                                  e_trial_strain, acc_p_strain,
                                  acc_p_strain_old, inc_p_mult,
                                  e_consistent_tangent, init_yield_stress,
                                  hardening_law, hardening_parameters,
                                  a_hardening_law, a_hardening_parameters,
                                  b_hardening_law, b_hardening_parameters,
                                  c_hardening_law, c_hardening_parameters,
                                  d_hardening_law, d_hardening_parameters,
                                  is_associative_hardening=False):
        """Compute state update residuals and Jacobian matrix.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comp_order_sym : list
            Strain/Stress components symmetric order.
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
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        init_yield_stress : torch.Tensor(0d)
            Initial yield stress.
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
        jacobian : torch.Tensor(2d)
            Jacobian matrix.
        """
        # Set associative hardening factor
        associative_hardening_factor = torch.tensor(1.0, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, fosym, _, _, _ = \
            get_id_operators(n_dim, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants and derivatives w.r.t. stress
        i1, _, _, _, j2, j3, di1_dstress, dj2_dstress, dj3_dstress, \
            d2j2_dstress2, d2j3_dstress2 = \
                self.get_stress_invariants_and_derivatives(n_dim, stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute auxiliary terms
        w1 = yield_b*i1
        w2 = yield_c*(j3**2)
        w3 = yield_d*j3
        w4 = j2**3 - w2
        w5 = w4**(1/2) - w3
        # Compute auxiliary terms derivatives w.r.t. stress
        dw1_dstress = yield_b*di1_dstress
        dw2_dstress = 2*yield_c*j3*dj3_dstress
        dw3_dstress = yield_d*dj3_dstress
        dw4_dstress = 3*(j2**2)*dj2_dstress - dw2_dstress
        dw5_dstress = (1/2)*(w4**(-1/2))*dw4_dstress - dw3_dstress
        # Compute auxiliary terms second-order derivatives w.r.t. stress
        d2w2_dstress2 = 2*yield_c*(dyad22_1(dj3_dstress, dj3_dstress)
                                   + j3*d2j3_dstress2)
        d2w3_dstress2 = yield_d*d2j3_dstress2
        d2w4_dstress2 = (6*j2*dyad22_1(dj2_dstress, dj2_dstress)
                         + 3*(j2**2)*d2j2_dstress2 - d2w2_dstress2)
        d2w5_dstress2 = (-(1/4)*(w4**(-3/2))*dyad22_1(dw4_dstress, dw4_dstress)
                         + (1/2)*(w4**(-1/2))*d2w4_dstress2 - d2w3_dstress2)
        # Compute auxiliary terms derivatives w.r.t. accumulated plastic strain
        dw1_daccpstr = i1*b_hard_slope
        dw2_daccpstr = (j3**2)*c_hard_slope
        dw3_daccpstr = j3*d_hard_slope
        dw4_daccpstr = -dw2_daccpstr
        dw5_daccpstr = (1/2)*(w4**(-1/2))*dw4_daccpstr - dw3_daccpstr
        # Compute auxiliary terms cross derivatives w.r.t. stress and
        # accumulated plastic strain
        d2w1_daccpstrdstress = b_hard_slope*soid
        d2w2_daccpstrdstress = 2*j3*c_hard_slope*dj3_dstress
        d2w3_daccpstrdstress = d_hard_slope*dj3_dstress
        d2w4_daccpstrdstress = -d2w2_daccpstrdstress
        d2w5_daccpstrdstress = (-(1/4)*(w4**(-3/2))*dw4_daccpstr*dw4_dstress
                                + (1/2)*(w4**(-1/2))*d2w4_daccpstrdstress
                                - d2w3_daccpstrdstress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute effective stress
        effective_stress = yield_a*(w1 + (w5**(1/3)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute flow vector
        flow_vector = yield_a*(dw1_dstress + (1/3)*(w5**(-2/3))*dw5_dstress)
        # Compute flow vector norm
        norm_flow_vector = torch.linalg.norm(flow_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute first residual
        residual_1 = e_strain - e_trial_strain + inc_p_mult*flow_vector
        # Compute second residual
        if is_associative_hardening:
            residual_2 = (acc_p_strain - acc_p_strain_old
                          - associative_hardening_factor*inc_p_mult)
        else:
            residual_2 = (acc_p_strain - acc_p_strain_old
                          - inc_p_mult*(math.sqrt(2/3))*norm_flow_vector)
        # Compute third residual
        residual_3 = (effective_stress - yield_stress)/init_yield_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of flow vector w.r.t. stress
        dflow_dstress = (1/3)*yield_a*(
            -(2/3)*(w5**(-5/3))*dyad22_1(dw5_dstress, dw5_dstress)
            + (w5**(-2/3))*d2w5_dstress2)
        # Compute derivative of flow vector w.r.t. elastic strain
        dflow_destrain = ddot44_1(dflow_dstress, e_consistent_tangent)
        # Compute derivative of flow vector w.r.t. accumulated plastic strain
        dflow_daccpstr = (
            a_hard_slope*(dw1_dstress + (1/3)*(w5**(-2/3))*dw5_dstress)
            + yield_a*(d2w1_daccpstrdstress
                       - (2/9)*(w5**(-5/3))*dw5_daccpstr*dw5_dstress
                       + (1/3)*(w5**(-2/3))*d2w5_daccpstrdstress))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of effective stress w.r.t. elastic strain
        deff_destrain = ddot24_1(flow_vector, e_consistent_tangent)
        # Compute derivative of effective stress w.r.t. accumulated plastic
        # strain
        deff_daccpstr = (
            a_hard_slope*(w1 + w5**(1/3))
            + yield_a*(dw1_daccpstr + (1/3)*(w5**(-2/3))*dw5_daccpstr))
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
            dr2_destrain = torch.zeros_like(flow_vector, device=self._device)
            # Compute derivative of second residual w.r.t. to accumulated
            # plastic strain
            dr2_daccpstr = torch.tensor(1.0, device=self._device)
            # Compute derivative of second residual w.r.t. to incremental
            # plastic multiplier
            dr2_dincpm = (torch.tensor(-1.0, device=self._device)
                          *associative_hardening_factor)
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
        dr3_destrain = (1.0/init_yield_stress)*deff_destrain
        # Compute derivative of third residual w.r.t. to accumulated plastic
        # strain
        dr3_daccpstr = (1.0/init_yield_stress)*(deff_daccpstr - hard_slope)
        # Compute derivative of third residual w.r.t. to incremental plastic
        # multiplier
        dr3_dincpm = torch.tensor(0.0, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build first residual derivatives matrices
        j11 = vget_tensor_mf(dr1_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True, device=self._device)
        j12 = vget_tensor_mf(dr1_daccpstr, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=self._device).reshape(-1, 1)
        j13 = vget_tensor_mf(dr1_dincpm, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=self._device).reshape(-1, 1)
        # Build second residual derivatives matrices
        j21 = vget_tensor_mf(dr2_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=self._device).reshape(1, -1)
        j22 = dr2_daccpstr.reshape(1, 1)
        j23 = dr2_dincpm.reshape(1, 1)
        # Build third residual derivatives matrices
        j31 = vget_tensor_mf(dr3_destrain, n_dim, comp_order_sym,
                             is_kelvin_notation=True,
                             device=self._device).reshape(1, -1)
        j32 = dr3_daccpstr.reshape(1, 1)
        j33 = dr3_dincpm.reshape(1, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build Jacobian matrix
        jacobian = torch.cat((torch.cat((j11, j12, j13), dim=1),
                              torch.cat((j21, j22, j23), dim=1),
                              torch.cat((j31, j32, j33), dim=1)), dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return residual_1, residual_2, residual_3, jacobian
    # -------------------------------------------------------------------------
    def get_numerical_jacobian(self, n_dim, comp_order_sym, e_strain,
                               e_trial_strain, acc_p_strain,
                               acc_p_strain_old, inc_p_mult,
                               e_consistent_tangent, init_yield_stress,
                               hardening_law, hardening_parameters,
                               a_hardening_law, a_hardening_parameters,
                               b_hardening_law, b_hardening_parameters,
                               c_hardening_law, c_hardening_parameters,
                               d_hardening_law, d_hardening_parameters,
                               is_associative_hardening=False,
                               is_verbose=False):
        """Compute state update Jacobian matrix with finite differences.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comp_order_sym : list
            Strain/Stress components symmetric order.
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
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        init_yield_stress : torch.Tensor(0d)
            Initial yield stress.
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
        is_associative_hardening : bool, default=False
            If True, then adopt associative hardening rule.
        is_verbose : bool, default=False
            If True, enable verbose output.

        Returns
        -------
        num_jacobian : torch.Tensor(2d)
            Jacobian matrix.
        """
        # Get number of components
        n_comp = len(comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set base elastic strain (matricial form)
        base_e_strain_mf = \
            vget_tensor_mf(e_strain, n_dim, comp_order_sym,
                           is_kelvin_notation=True, device=self._device)
        # Set base accumulated plastic strain
        base_acc_p_strain = acc_p_strain
        # Set base incremental plastic multiplier
        base_inc_p_mult = inc_p_mult
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute base residuals
        base_residual_1, base_residual_2, base_residual_3, jacobian = \
            self.get_residual_and_jacobian(
                n_dim, comp_order_sym, e_strain, e_trial_strain,
                acc_p_strain, acc_p_strain_old, inc_p_mult,
                e_consistent_tangent, init_yield_stress,
                hardening_law, hardening_parameters,
                a_hardening_law, a_hardening_parameters,
                b_hardening_law, b_hardening_parameters,
                c_hardening_law, c_hardening_parameters,
                d_hardening_law, d_hardening_parameters,
                is_associative_hardening=is_associative_hardening)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base residuals (matricial form)
        base_r1_mf = vget_tensor_mf(base_residual_1, n_dim, comp_order_sym,
                                    is_kelvin_notation=True,
                                    device=self._device)
        base_r2 = base_residual_2.reshape(-1)
        base_r3 = base_residual_3.reshape(-1)
        # Build base residual vector (matricial form)
        base_residual_mf = torch.cat((base_r1_mf, base_r2, base_r3), dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set perturbation
        delta = torch.tensor(1e-6)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize numerical Jacobian
        num_jacobian = torch.zeros_like(jacobian)
        # Loop over components
        for i in range(n_comp + 2):
            # Initialize perturbed elastic strain (matricial form)
            pert_e_strain_mf = base_e_strain_mf.clone()
            # Initialize perturbed accumulated plastic strain
            pert_acc_p_strain = base_acc_p_strain.clone()
            # Initialize perturbed incremental plastic multiplier
            pert_inc_p_mult = base_inc_p_mult.clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add perturbation
            if i == n_comp + 1:
                # Set pertubation
                pert = delta*torch.max(torch.abs(pert_inc_p_mult),
                                                 torch.tensor(1e-6))
                # Set perturbed incremental plastic multiplier
                pert_inc_p_mult += pert
            elif i == n_comp:
                # Set pertubation
                pert = delta*torch.max(torch.abs(pert_acc_p_strain),
                                                 torch.tensor(1e-6))
                # Set perturbed accumulated plastic strain
                pert_acc_p_strain += pert
            else:
                # Set pertubation
                pert = delta*torch.max(torch.abs(pert_e_strain_mf[i]),
                                                 torch.tensor(1e-6))
                # Set perturbed elastic strain (matricial form)
                pert_e_strain_mf[i] += pert
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get perturbed elastic strain
            pert_e_strain = vget_tensor_from_mf(
                pert_e_strain_mf, n_dim, comp_order_sym,
                is_kelvin_notation=True, device=self._device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute perturbed residuals
            pert_residual_1, pert_residual_2, pert_residual_3, _ = \
                self.get_residual_and_jacobian(
                    n_dim, comp_order_sym, pert_e_strain, e_trial_strain,
                    pert_acc_p_strain, acc_p_strain_old, pert_inc_p_mult,
                    e_consistent_tangent, init_yield_stress,
                    hardening_law, hardening_parameters,
                    a_hardening_law, a_hardening_parameters,
                    b_hardening_law, b_hardening_parameters,
                    c_hardening_law, c_hardening_parameters,
                    d_hardening_law, d_hardening_parameters,
                    is_associative_hardening=is_associative_hardening)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build perturbed residuals (matricial form)
            pert_r1_mf = vget_tensor_mf(pert_residual_1, n_dim, comp_order_sym,
                                        is_kelvin_notation=True,
                                        device=self._device)
            pert_r2 = pert_residual_2.reshape(-1)
            pert_r3 = pert_residual_3.reshape(-1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build perturbed residual vector (matricial form)
            pert_residual_mf = torch.cat((pert_r1_mf, pert_r2, pert_r3), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute and assemble numerical derivative
            num_jacobian[:, i] = (pert_residual_mf - base_residual_mf)/pert
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display Jacobian comparison
        if is_verbose:
            torch.set_printoptions(linewidth=1000)
            print('\nJacobian comparison:')
            print('\nAnalytical:')
            print(jacobian)
            print('\nNumerical:')
            print(num_jacobian)
            print()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display numerical derivatives
        if is_verbose:
            self.compute_num_derivatives(n_dim, comp_order_sym, e_strain,
                                         e_consistent_tangent)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return num_jacobian
    # -------------------------------------------------------------------------
    def compute_num_derivatives(self, n_dim, comp_order_sym, e_strain,
                                e_consistent_tangent):
        """Compute numerical derivatives by finite differences.
        
        Parameters
        ----------
        n_dim : int
            Problem number of spatial dimensions.
        comp_order_sym : list
            Strain/Stress components symmetric order.
        e_strain : torch.Tensor(2d)
            Elastic strain.
        e_consistent_tangent : torch.Tensor(4d)
            Elastic consistent tangent modulus.
        """
        # Get number of components
        n_comp = len(comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute base stress
        base_stress = ddot42_1(e_consistent_tangent, e_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set base stress (matricial form)
        base_stress_mf = \
            vget_tensor_mf(base_stress, n_dim, comp_order_sym,
                           is_kelvin_notation=True, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set numerical derivative option
        option = 'sod_stress_invariants'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set display options
        torch.set_printoptions(linewidth=1000)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute numerical derivatives
        if option == 'fod_stress_invariants':
            # Compute base stress invariants
            base_i1, _, _, _, base_j2, base_j3 = \
                self.get_stress_invariants(base_stress)
                
            base_i1, _, _, _, base_j2, base_j3, di1_dstress, dj2_dstress, \
                dj3_dstress, _, _ = self.get_stress_invariants_and_derivatives(
                    n_dim, base_stress)
            # Build base vector
            base_vector = torch.cat(
                (base_i1.view(-1), base_j2.view(-1), base_j3.view(-1)), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set perturbation
            delta = torch.tensor(1e-6)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize numerical derivatives
            num_derivatives = torch.zeros(3, n_comp)
            # Loop over components
            for i in range(n_comp):
                # Initialize perturbed stress (matricial form)
                pert_stress_mf = base_stress_mf.clone()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set pertubation
                pert = delta*torch.max(torch.abs(pert_stress_mf[i]),
                                                 torch.tensor(1e-6))
                # Set perturbed stress (matricial form)
                pert_stress_mf[i] += \
                    delta*torch.max(torch.abs(pert_stress_mf[i]),
                                    torch.tensor(1e-6)) 
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get perturbed stress
                pert_stress = vget_tensor_from_mf(
                    pert_stress_mf, n_dim, comp_order_sym,
                    is_kelvin_notation=True, device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute perturbed stress invariants
                pert_i1, _, _, _, pert_j2, pert_j3, _, _, _, _, _ = \
                    self.get_stress_invariants_and_derivatives(
                        n_dim, pert_stress)
                # Build perturbed vector
                pert_vector = torch.cat(
                    (pert_i1.view(-1), pert_j2.view(-1), pert_j3.view(-1)),
                    dim=0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute and assemble numerical derivatives
                num_derivatives[:, i] = (pert_vector - base_vector)/pert
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build analytical first-order derivatives (matricial form)
            di1_dstress_mf = \
                vget_tensor_mf(di1_dstress, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            dj2_dstress_mf = \
                vget_tensor_mf(dj2_dstress, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            dj3_dstress_mf = \
                vget_tensor_mf(dj3_dstress, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            # Concatenate analytical first-order derivatives
            djx_dstress_mf = torch.stack(
                (di1_dstress_mf, dj2_dstress_mf, dj3_dstress_mf), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display numerical derivatives
            print('\nStress invariants numerical first-order derivatives '
                  '(i1, j2, j3) comparison:')
            print('\nAnalytical:')
            print(djx_dstress_mf)
            print('\nNumerical:')
            print(num_derivatives)
            print()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif option == 'sod_stress_invariants':
            # Compute base stress invariants derivatives w.r.t. stress
            _, _, _, _, _, _, _, base_dj2_dstress, base_dj3_dstress, \
                d2j2_dstress2, d2j3_dstress2 = \
                    self.get_stress_invariants_and_derivatives(
                        n_dim, base_stress)     
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build base derivatives (matricial form)
            base_dj2_dstress_mf = \
                vget_tensor_mf(base_dj2_dstress, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            base_dj3_dstress_mf = \
                vget_tensor_mf(base_dj3_dstress, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            # Build base vector
            base_vector = torch.cat(
                (base_dj2_dstress_mf, base_dj3_dstress_mf), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set perturbation
            delta = torch.tensor(1e-6)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize numerical derivatives
            num_derivatives = torch.zeros(2*n_comp, n_comp)
            # Loop over components
            for i in range(n_comp):
                # Initialize perturbed stress (matricial form)
                pert_stress_mf = base_stress_mf.clone()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set pertubation
                pert = delta*torch.max(torch.abs(pert_stress_mf[i]),
                                                 torch.tensor(1e-6))
                # Set perturbed stress (matricial form)
                pert_stress_mf[i] += \
                    delta*torch.max(torch.abs(pert_stress_mf[i]),
                                    torch.tensor(1e-6)) 
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get perturbed stress
                pert_stress = vget_tensor_from_mf(
                    pert_stress_mf, n_dim, comp_order_sym,
                    is_kelvin_notation=True, device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute perturbed stress invariants derivatives
                _, _, _, _, _, _, _, pert_dj2_dstress, pert_dj3_dstress, _, \
                    _ = self.get_stress_invariants_and_derivatives(
                        n_dim, pert_stress)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build perturbed derivatives (matricial form)
                pert_dj2_dstress_mf = \
                    vget_tensor_mf(pert_dj2_dstress, n_dim, comp_order_sym,
                                   is_kelvin_notation=True,
                                   device=self._device)
                pert_dj3_dstress_mf = \
                    vget_tensor_mf(pert_dj3_dstress, n_dim, comp_order_sym,
                                   is_kelvin_notation=True,
                                   device=self._device)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build perturbed vector
                pert_vector = torch.cat(
                    (pert_dj2_dstress_mf, pert_dj3_dstress_mf), dim=0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute and assemble numerical derivatives
                num_derivatives[:, i] = (pert_vector - base_vector)/pert
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build analytical second-order derivatives (matricial form)
            d2j2_dstress2_mf = \
                vget_tensor_mf(d2j2_dstress2, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            d2j3_dstress2_mf = \
                vget_tensor_mf(d2j3_dstress2, n_dim, comp_order_sym,
                               is_kelvin_notation=True, device=self._device)
            # Concatenate analytical second-order derivatives
            d2jx_dstress2_mf = torch.cat(
                (d2j2_dstress2_mf, d2j3_dstress2_mf), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display numerical derivatives
            print('\nStress invariants numerical second-order derivatives '
                  '(j2, j3) comparison:')
            print('\nAnalytical:')
            print(d2jx_dstress2_mf)
            print('\nNumerical:')
            print(num_derivatives)
            print()
    # -------------------------------------------------------------------------
    @classmethod
    def convexity_return_mapping(cls, yield_c, yield_d):
        """Perform convexity return-mapping.
        
        For a given set (c, d), the convexity return-mapping works as follows:
        
        (1) If the yield parameters (c, d) lie inside the convexity domain
            (yield surface is convex), then they are kept unchanged;
            
        (2) If the yield parameters (c, d) lie outside the convexity domain
            (yield surface is not convex), then they are updated to the
            convexity domain boundary point along the same angular direction.

        Parameters
        ----------
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.

        Returns
        -------
        is_convex : bool
            If True, then yield surface is convex, False otherwise.
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
        """
        # Check yield surface convexity
        is_convex = cls.check_yield_surface_convexity(yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform convexity return-mapping
        if not is_convex:
            # Compute angular direction
            theta = torch.atan2(yield_d, yield_c)
            # Compute convexity boundary point
            yield_c, yield_d = cls.directional_convex_boundary(theta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_convex, yield_c, yield_d
    # -------------------------------------------------------------------------
    @classmethod
    def compute_convex_boundary(cls, n_theta=360):
        """Compute convexity domain boundary.
        
        Parameters
        ----------
        n_theta : int, default=360
            Number of discrete angular coordinates to discretize the convexity
            boundary domain.

        Returns
        -------
        convex_boundary : torch.Tensor(2d)
            Convexity domain boundary stored as torch.Tensor(2d) of shape
            (n_point, 2), where each point is stored as (yield_c, yield_d).
        """
        # Set discrete angular coordinates
        thetas = torch.linspace(0, 2.0*torch.pi, steps=n_theta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize convexity domain boundary
        convex_boundary = torch.zeros(n_theta, 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete angular coordinates
        for i, theta in enumerate(thetas):
            # Compute directional convexity domain boundary
            yield_c, yield_d = cls.directional_convex_boundary(theta)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store convexity domain boundary point
            convex_boundary[i, :] = torch.tensor((yield_c, yield_d)) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return convex_boundary
    # -------------------------------------------------------------------------
    @classmethod
    def directional_convex_boundary(cls, theta, r_lower=0.0, r_upper=4.0,
                                    search_tol=1e-6):
        """Compute convexity domain boundary along given angular direction.
        
        Parameters
        ----------
        theta : torch.Tensor(0d)
            Angular coordinate in yield parameters domain (radians).
        r_lower : float, default=0.0
            Initial searching radius lower bound.
        r_upper : float, default=4.0
            Initial searching radius upper bound.
        search_tol : float, default = 1e-6
            Searching window tolerance.
        
        Return
        ------
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.
        """
        # Store input angle type
        input_dtype = theta.dtype
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize searching window
        r_window = r_upper - r_lower
        # Initialize mean searching radius
        r_mean = (r_upper + r_lower)/2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convexity boundary searching loop
        while (r_window > search_tol):
            # Compute yield parameters
            yield_c = r_mean*torch.cos(theta)
            yield_d = r_mean*torch.sin(theta)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check yield surface convexity
            is_convex = cls.check_yield_surface_convexity(yield_c, yield_d)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update searching bounds
            if is_convex:
                r_lower = r_mean
            else:
                r_upper = r_mean
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update mean searching radius
            r_mean = (r_upper + r_lower)/2
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update searching window
            r_window = r_upper - r_lower
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Enforce consistent output type
        yield_c = yield_c.to(input_dtype)
        yield_d = yield_d.to(input_dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return yield_c, yield_d
    # =========================================================================
    @classmethod
    def check_yield_surface_convexity(cls, yield_c, yield_d):
        """Check yield surface convexity.
        
        Geometry-Inspired Numerical Convex Analysis (GINCA) method.
        
        Parameters
        ----------
        yield_c : torch.Tensor(0d)
            Yield parameter.
        yield_d : torch.Tensor(0d)
            Yield parameter.

        Returns
        -------
        is_convex : bool
            If True, then yield surface is convex, False otherwise.
        """
        def get_dev_stress(lode_angle):
            """Compute deviatoric stress from Lode angle.
            
            Parameters
            ----------
            lode_angle : torch.Tensor(0d)
                Lode angle (radians).
                
            Returns
            -------
            dev_stress : torch.Tensor(2d)
                Deviatoric stress.
            """
            # Compute principal deviatoric stresses
            s1 = (2/3)*torch.cos(lode_angle)
            s2 = (2/3)*torch.cos((2*torch.pi/3) - lode_angle)
            s3 = (2/3)*torch.cos((4*torch.pi/3) - lode_angle)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build deviatoric stress tensor
            dev_stress = torch.diag(torch.stack([s1, s2, s3]))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return dev_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def convexity_function(dev_stress, yield_c, yield_d):
            """Function to evaluate convexity.
            
            Parameters
            ----------
            dev_stress : torch.Tensor(2d)
                Deviatoric stress.
            yield_c : torch.Tensor(0d)
                Yield parameter.
            yield_d : torch.Tensor(0d)
                Yield parameter.
                
            Returns
            -------
            val : torch.Tensor(0d)
                Convexity function value.
            """
            # Compute second invariant of deviatoric stress tensor
            j2 = 0.5*torch.sum(dev_stress*dev_stress)
            # Compute third invariant of deviatoric stress tensor
            j3 = torch.det(dev_stress)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute convexity function
            val = ((j2**3 - yield_c*j3**2)**(1/2) - yield_d*j3)**(1/3)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return val
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def evaluate_convexity_lode(lode_angle, yield_c, yield_d, d_lode=None):
            """Evaluate convexity function for given Lode angle.
            
            Parameters
            ----------
            lode_angle : torch.Tensor(0d)
                Lode angle (radians).
            yield_c : torch.Tensor(0d)
                Yield parameter.
            yield_d : torch.Tensor(0d)
                Yield parameter.
            d_lode : torch.Tensor(0d), default=None
                Infinitesimal Lode angle (radians).
            
            Returns
            -------
            convex_fun_val : torch.Tensor(0d)
                Convexity function value.
            """
            # Enforce double precision
            lode_angle = lode_angle.double()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set infinitesimal Lode angle
            if d_lode is None:
                lode_small = torch.deg2rad(
                    torch.tensor(0.001, device=lode_angle.device))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute arc point A deviatoric stress
            lode_angle_a = lode_angle
            dev_stress_a = get_dev_stress(lode_angle_a)
            dev_stress_a = \
                dev_stress_a/convexity_function(dev_stress_a, yield_c, yield_d)
            # Compute arc point B deviatoric stress
            lode_angle_b = lode_angle + lode_small
            dev_stress_b = get_dev_stress(lode_angle_b)
            dev_stress_b = \
                dev_stress_b/convexity_function(dev_stress_b, yield_c, yield_d)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute midpoint C deviatoric stress
            dev_stress_c = (dev_stress_a + dev_stress_b)/2
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evaluate convexity function
            convex_fun_val = convexity_function(dev_stress_c, yield_c, yield_d)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return convex_fun_val
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get yield parameters device
        device = yield_c.device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set discrete Lode angles
        lode_angles = torch.deg2rad(
            torch.linspace(0, 360, steps=1000, device=device))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized convexity function computation (batch along Lode
        # angles)
        vmap_evaluate_convexity_lode = \
            torch.vmap(evaluate_convexity_lode,
                    in_dims=(0, None, None), out_dims=(0,))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute convexity function values
        convex_fun_vals = \
            vmap_evaluate_convexity_lode(lode_angles, yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield surface convexity
        is_convex = torch.all(convex_fun_vals <= 1.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_convex
    # =========================================================================
    @classmethod
    def plot_convexity_boundary(cls, convex_boundary, parameters_paths=None,
                                is_path_arrows=True, rect_search_domain=None,
                                is_plot_legend=False, save_dir=None,
                                is_save_fig=False, is_stdout_display=False,
                                is_latex=False):
        """Plot convexity domain boundary.
        
        Parameters
        ----------
        convex_boundary : torch.Tensor(2d)
            Convexity domain boundary stored as torch.Tensor(2d) of shape
            (n_point, 2), where each point is stored as (yield_c, yield_d).
        parameters_paths : dict, default=None
            For each yield parameters path (key, str), store a torch.Tensor(2d)
            (item, torch.Tensor) of shape (n_point, 2), where each point is
            stored as (yield_c, yield_d).
        is_path_arrows : bool, default=True
            If True, then yield parameters paths include directional arrows
            along the path, False otherwise.
        rect_search_domain : tuple, default=None
            Rectangular search domain boundary defined by the corresponding
            limits along each direction as ((x_min, x_max), (y_min, y_max)).
        is_plot_legend : bool, default=False
            If True, then plot legend.
        save_dir : str, default=None
            Directory where data set plots are saved.
        is_save_fig : bool, default=False
            Save figure.
        is_stdout_display : bool, default=False
            True if displaying figure to standard output device, False
            otherwise.
        is_latex : bool, default=False
            If True, then render all strings in LaTeX. If LaTex is not
            available, then this option is silently set to False and all input
            strings are processed to remove $(...)$ enclosure.
        """
        # Set data array
        data_xy = convex_boundary.numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set data labels
        if is_plot_legend:
            data_labels = ['Convexity boundary',]
        else:
            data_labels = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axes labels
        x_label = 'Yield parameter $c$'
        y_label = 'Yield parameter $d$'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot convexity domain boundary
        figure, axes = plot_xy_data(
            data_xy, data_labels=data_labels, x_label=x_label, y_label=y_label,
            x_scale='linear', y_scale='linear', is_latex=is_latex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot yield parameters paths
        if isinstance(parameters_paths, dict):
            # Set line width
            if is_path_arrows:
                lw = 0
            else:
                lw = None
            # Loop over paths
            for path_label, path_points in parameters_paths.items():
                # Convert parameters path
                path_points = path_points.numpy()
                # Plot parameters path points
                (line, ) = axes.plot(path_points[:, 0], path_points[:, 1],
                                     lw=lw, marker='o', ms=3,
                                     markeredgecolor='k', markeredgewidth=0.5,
                                     label=path_label, zorder=10)
                # Plot parameters path directional arrows
                if path_points.shape[0] > 1:
                    axes.quiver(path_points[:-1, 0], path_points[:-1, 1],
                                np.diff(path_points[:, 0]),
                                np.diff(path_points[:, 1]),
                                angles="xy", color=line.get_color(),
                                scale_units="xy", scale=1, width=0.005,
                                zorder=5)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot rectangular search domain boundary
        if isinstance(rect_search_domain, tuple):
            # Get rectangular search domain boundaries
            search_x_min, search_x_max = rect_search_domain[0]
            search_y_min, search_y_max = rect_search_domain[1]
            # Build rectangular search domain boundary
            search_domain = \
                patches.Rectangle((search_x_min, search_y_min),
                                  search_x_max - search_x_min,
                                  search_y_max - search_y_min,
                                  edgecolor='#555555', facecolor='none',
                                  linewidth=1.5, linestyle='--', zorder=2)
            # Plot rectangular search domain boundary
            axes.add_patch(search_domain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot legend
        if is_plot_legend:
            legend = axes.legend(loc='best', frameon=True, fancybox=True,
                                facecolor='inherit', edgecolor='inherit',
                                fontsize=8, framealpha=1.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set filename
        filename = f'lou_yield_convexity_domain'
        # Save figure
        if is_save_fig:
            save_figure(figure, filename, format='pdf', save_dir=save_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_stdout_display:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close('all')
# =============================================================================
"""
        # Compute Jacobian in the particular case of:
        # - Yield parameters c=0 and d=0
        # - Associative hardening rule
        # - Constant yield parameters a and b
        #
        # How to use: Paste at the end of get_jacobian() 
        #
        # Compute Jacobian tensor terms
        j11_tensor = fosym + 0.5*yield_a*inc_p_mult*(
            dyad22_1(dev_stress, -0.5*(j2**(-3/2))*ddot24_1(
                dev_stress, e_consistent_tangent))
            + (j2**(-1/2))*ddot44_1(fodevprojsym, e_consistent_tangent))        
        j21_tensor = torch.zeros_like(flow_vector, device=self._device)
        j31_tensor = (1.0/init_yield_stress)*(
            yield_a*(0.5*(j2**(-1/2))*ddot24_1(dev_stress,
                                               e_consistent_tangent))
            + yield_a*yield_b*ddot24_1(soid, e_consistent_tangent))
        j12_tensor = torch.zeros_like(flow_vector, device=self._device)
        j22_tensor = torch.tensor(1.0, device=self._device)
        j32_tensor = -(1.0/init_yield_stress)*hard_slope
        j13_tensor = flow_vector
        j23_tensor = torch.tensor(-1.0, device=self._device)
        j33_tensor = torch.tensor(0.0, device=self._device)
        # Get Jacobian terms matricial form
        val_j11 = vget_tensor_mf(j11_tensor, n_dim, comp_order_sym,
                                 is_kelvin_notation=True, device=self._device)
        val_j21 = vget_tensor_mf(j21_tensor, n_dim, comp_order_sym,
                                 is_kelvin_notation=True,
                                 device=self._device).reshape(1, -1)
        val_j31 = vget_tensor_mf(j31_tensor, n_dim, comp_order_sym,
                                 is_kelvin_notation=True,
                                 device=self._device).reshape(1, -1)
        val_j12 = vget_tensor_mf(j12_tensor, n_dim, comp_order_sym,
                                 is_kelvin_notation=True,
                                 device=self._device).reshape(-1, 1)
        val_j22 = j22_tensor.reshape(1, 1)
        val_j32 = j32_tensor.reshape(1, 1)
        val_j13 = vget_tensor_mf(j13_tensor, n_dim, comp_order_sym,
                                 is_kelvin_notation=True,
                                 device=self._device).reshape(-1, 1)
        val_j23 = j23_tensor.reshape(1, 1)
        val_j33 = j33_tensor.reshape(1, 1)
        # Assemble Jacobian matrix
        val_jacobian = torch.cat(
            (torch.cat((val_j11, val_j12, val_j13), dim=1),
             torch.cat((val_j21, val_j22, val_j23), dim=1),
             torch.cat((val_j31, val_j32, val_j33), dim=1)), dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Jacobian condition number
        jacobian_cnum = torch.linalg.norm(jacobian)*torch.linalg.norm(
            torch.inverse(jacobian))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display Jacobian comparison
        is_print_jacobian = True
        if is_print_jacobian:
            torch.set_printoptions(linewidth=1000)
            print('\n\nJACOBIAN VALIDATION')
            print('\nGeneral jacobian:')
            print(jacobian)
            print('\nParticular jacobian:')
            print(val_jacobian)
            print('\nRelative error:')
            eps = 1e-6
            abs_diff = torch.abs(jacobian - val_jacobian)
            abs_b = torch.abs(val_jacobian)
            mask = abs_b >= eps
            rerror = abs_diff.clone()
            rerror[mask] = abs_diff[mask]/abs_b[mask]
            print(rerror)
            print(f'\nJacobian condition number = {jacobian_cnum}')
            print()
"""