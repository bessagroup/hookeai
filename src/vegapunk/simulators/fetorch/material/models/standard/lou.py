"""Lou-Zhang-Yoon model with general differentiable yield function.

This module includes the implementation of the Lou-Zhang-Yoon model with
general differentiable yield function and isotropic hardening.

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
import copy
# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt
# Local
from simulators.fetorch.material.models.interface import ConstitutiveModel
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf, vget_state_3Dmf_from_2Dmf, \
    vget_state_2Dmf_from_3Dmf
from simulators.fetorch.math.tensorops import get_id_operators, dyad22_1, \
    ddot42_1, ddot24_1, ddot22_1, ddot44_1, fo_dinv_sym
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
    get_effective_stress(self, stress, yield_a, yield_b, yield_c, yield_d)
        Compute effective stress.
    get_flow_vector(self, stress, yield_a, yield_b, yield_c, yield_d)
        Compute flow vector.
    get_residual(self, e_strain, e_trial_strain, acc_p_strain, \
                 acc_p_strain_old, inc_p_mult, effective_stress, \
                 yield_stress, init_yield_stress, flow_vector, \
                 norm_flow_vector, is_associative_hardening=False)
        Compute state update residuals.
    get_jacobian(self, n_dim, comp_order_sym, stress, inc_p_mult, \
                 flow_vector, norm_flow_vector, init_yield_stress, \
                 hard_slope, yield_a, a_hard_slope, yield_b, b_hard_slope, \
                 yield_c, c_hard_slope, yield_d, d_hard_slope, \
                 e_consistent_tangent, is_associative_hardening=False)
        Compute state update Jacobian matrix.
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
                 device_type='cpu'):
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
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
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
        # Set state update convergence tolerance
        su_conv_tol = 1e-5
        # Set state update maximum number of iterations
        su_max_n_iterations = 20
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental strain tensor matricial form
        inc_strain_mf = vget_tensor_mf(inc_strain, self._n_dim,
                                       self._comp_order_sym,
                                       device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material properties
        E = self._model_parameters['E']
        v = self._model_parameters['v']
        # Get material isotropic strain hardening law
        hardening_law = self._model_parameters['hardening_law']
        hardening_parameters = self._model_parameters['hardening_parameters']
        # Get yield parameters hardening laws
        a_hardening_law = self._model_parameters['a_hardening_law']
        a_hardening_parameters = \
            self._model_parameters['a_hardening_parameters']
        b_hardening_law = self._model_parameters['b_hardening_law']
        b_hardening_parameters = \
            self._model_parameters['b_hardening_parameters']
        c_hardening_law = self._model_parameters['c_hardening_law']
        c_hardening_parameters = \
            self._model_parameters['c_hardening_parameters']
        d_hardening_law = self._model_parameters['d_hardening_law']
        d_hardening_parameters = \
            self._model_parameters['d_hardening_parameters']
        # Get hardening rule associativity
        is_associative_hardening = \
            self._model_parameters['is_associative_hardening']
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
        if self._problem_type == 1:
            e_strain_33_old = state_variables_old['e_strain_33']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state update failure flag
        is_su_fail = False
        # Initialize plastic step flag
        is_plast = False
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
            is_plast = True
            # Get elastic trial strain tensor
            e_trial_strain = vget_tensor_from_mf(e_trial_strain_mf, n_dim,
                                                 comp_order_sym,
                                                 is_kelvin_notation=True,
                                                 device=self._device)
            # Compute initial yield stress
            init_yield_stress, _ = \
                hardening_law(hardening_parameters, acc_p_strain=0.0)
            # Set unknowns initial iterative guess
            e_strain = e_trial_strain
            acc_p_strain = acc_p_strain_old
            inc_p_mult = 0
            # Initialize Newton-Raphson iteration counter
            nr_iter = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize norm of iterative solution vector (convergence check)
            diter_norm = 0.0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_verbose:
                print(f'\n\nPlastic Increment - Newton-Raphson')
                print('----------------------------------')
                print('nr_iter   conv_norm_res_1   conv_norm_res_2   '
                      'conv_norm_res_3   diter_norm')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Start Newton-Raphson iterative loop
            while True:
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
                effective_stress = self.get_effective_stress(
                    stress, yield_a, yield_b, yield_c, yield_d)
                # Compute current flow vector and norm
                flow_vector = self.get_flow_vector(
                    stress, yield_a, yield_b, yield_c, yield_d)
                norm_flow_vector = torch.linalg.norm(flow_vector)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residuals
                residual_1, residual_2, residual_3 = self.get_residual(
                    e_strain, e_trial_strain, acc_p_strain, acc_p_strain_old,
                    inc_p_mult, effective_stress, yield_stress,
                    init_yield_stress, flow_vector, norm_flow_vector,
                    is_associative_hardening=is_associative_hardening)
                # Build residuals matrices
                r1 = vget_tensor_mf(residual_1, n_dim, comp_order_sym,
                                    is_kelvin_notation=True,
                                    device=self._device)
                r2 = residual_2.reshape(-1)
                r3 = residual_3.reshape(-1)
                # Build residual vector
                residual = torch.cat((r1, r2, r3), dim=0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residuals convergence norm
                conv_norm_res_1 = (torch.linalg.norm(residual_1)/
                                   torch.linalg.norm(e_trial_strain))
                if abs(acc_p_strain_old) < 1e-8:
                    conv_norm_res_2 = abs(residual_2)
                else:
                    conv_norm_res_2 = abs(residual_2/acc_p_strain_old)
                conv_norm_res_3 = abs(residual_3)*(init_yield_stress/E)
                # Compute residual vector convergence norm
                conv_norm_residual = torch.mean(
                    torch.tensor((conv_norm_res_1, conv_norm_res_2,
                                  conv_norm_res_3)))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Display iterative residuals
                if is_verbose:
                    print(f'{nr_iter:7d}   {conv_norm_res_1:^15.4e}   '
                          f'{conv_norm_res_2:^15.4e}   '
                          f'{conv_norm_res_3:^15.4e}   {diter_norm:^10.4e}')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check Newton-Raphson iterative procedure convergence
                is_converged = (conv_norm_residual < su_conv_tol
                                and diter_norm < su_conv_tol)
                # Control Newton-Raphson iteration loop flow
                if is_converged:
                    # Display convergence status
                    if is_verbose:
                        print(f'{"Solution converged!":^74s}')
                    # Leave Newton-Raphson iterative loop (converged solution)
                    break
                elif nr_iter == su_max_n_iterations:
                    # Display convergence status
                    if is_verbose:
                        print(f'{"Solution convergence failure!":^74s}')
                    # If the maximum number of Newton-Raphson iterations is
                    # reached without achieving convergence, recover last
                    # converged state variables, set state update failure flag
                    # and return
                    state_variables = copy.deepcopy(state_variables_old)
                    state_variables['is_su_fail'] = True
                    return state_variables, None
                else:
                    # Increment iteration counter
                    nr_iter = nr_iter + 1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute return-mapping Jacobian
                jacobian = self.get_jacobian(
                    n_dim, comp_order_sym, stress, inc_p_mult, flow_vector,
                    norm_flow_vector, init_yield_stress, hard_slope, yield_a,
                    a_hard_slope, yield_b, b_hard_slope, yield_c, c_hard_slope,
                    yield_d, d_hard_slope, e_consistent_tangent,
                    is_associative_hardening=is_associative_hardening)
                # Solve return-mapping linearized equation
                d_iter = torch.linalg.solve(jacobian, -residual)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute norm of iterative solution vector (convergence check)
                diter_norm = torch.linalg.norm(d_iter)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        #
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
        effective_stress : torch.Tensor(2d)
            Effective stress.
        """
        # Compute stress invariants
        i1, _, _, _, j2, j3 = self.get_stress_invariants(stress)
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
    def get_flow_vector(self, stress, yield_a, yield_b, yield_c, yield_d):
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, _, _, _, fodevprojsym = \
            get_id_operators(n_dim, device=self._device)
        # Compute deviatoric stress tensor
        dev_stress = ddot42_1(fodevprojsym, stress)
        # Compute inverse of deviatoric stress tensor
        dev_stress_inv = torch.inverse(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants
        _, _, _, _, j2, j3 = self.get_stress_invariants(stress)
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
    def get_residual(self, e_strain, e_trial_strain, acc_p_strain,
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
    def get_jacobian(self, n_dim, comp_order_sym, stress, inc_p_mult,
                     flow_vector, norm_flow_vector, init_yield_stress,
                     hard_slope, yield_a, a_hard_slope, yield_b, b_hard_slope,
                     yield_c, c_hard_slope, yield_d, d_hard_slope,
                     e_consistent_tangent, is_associative_hardening=False):
        """Compute state update Jacobian matrix.
        
        Parameters
        ----------
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
        is_associative_hardening : bool, default=False
            If True, then adopt associative hardening rule.

        Returns
        -------
        jacobian : torch.Tensor(2d)
            Jacobian matrix.
        """
        # Set number of spatial dimensions
        n_dim = 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        soid, _, _, fosym, _, _, fodevprojsym = \
            get_id_operators(n_dim, device=self._device)
        # Compute deviatoric stress tensor
        dev_stress = ddot42_1(fodevprojsym, stress)
        # Compute inverse of deviatoric stress tensor
        dev_stress_inv = torch.inverse(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stress invariants
        i1, _, _, _, j2, j3 = self.get_stress_invariants(stress)
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
            dr2_destrain = torch.zeros_like(flow_vector, device=self._device)
            # Compute derivative of second residual w.r.t. to accumulated
            # plastic strain
            dr2_daccpstr = torch.tensor(1.0, device=self._device)
            # Compute derivative of second residual w.r.t. to incremental
            # plastic multiplier
            dr2_dincpm = torch.tensor(-1.0, device=self._device)
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
        return jacobian
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
                lode_small = torch.deg2rad(torch.tensor(0.001))
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
        # Set discrete Lode angles
        lode_angles = torch.deg2rad(torch.linspace(0, 360, steps=1000))
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
            # Loop over paths
            for path_label, path_points in parameters_paths.items():
                # Convert parameters path
                path_points = path_points.numpy()
                # Plot parameters path points
                (line, ) = axes.plot(path_points[:, 0], path_points[:, 1],
                                     lw=0, marker='o', ms=3, label=path_label)
                # Plot parameters path directional arrows
                if path_points.shape[0] > 1:
                    axes.quiver(path_points[:-1, 0], path_points[:-1, 1],
                                np.diff(path_points[:, 0]),
                                np.diff(path_points[:, 1]),
                                angles="xy", color=line.get_color(),
                                scale_units="xy", scale=1, width=0.005)
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