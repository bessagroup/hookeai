"""Von Mises elasto-plastic constitutive model with isotropic hardening.

This module includes the implementation of the von Mises constitutive model
with isotropic strain hardening.

Apart from a conversion to a PyTorch framework, the code is taken from the
package cratepy [#]_.

.. [#] Ferreira, B.P., Andrade Pires, F.M., and Bessa, M.A. (2023). CRATE: A
       Python package to perform fast material simulations. The Journal of Open
       Source Software, 8(87)
       (see `here <https://joss.theoj.org/papers/10.21105/joss.05594>`_)

Classes
-------
VonMises
    Von Mises constitutive model with isotropic strain hardening.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import torch
import numpy as np
import copy
# Local
from simulators.fetorch.material.models.interface import ConstitutiveModel
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_tensor_mf, vget_tensor_from_mf, vget_state_3Dmf_from_2Dmf, \
    vget_state_2Dmf_from_3Dmf
from simulators.fetorch.math.tensorops import get_id_operators, dyad22_1
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class VonMises(ConstitutiveModel):
    """Von Mises constitutive model with isotropic strain hardening.

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
        self._name = 'von_mises'
        # Set constitutive model strain formulation
        self._strain_type = 'finite-kinext'
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
        # Check finite strains formulation
        if self._strain_formulation == 'finite' and \
                elastic_symmetry != 'isotropic':
            raise RuntimeError('The von Mises constitutive model is only '
                               'available under finite strains for the '
                               'elastic isotropic case.')
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

        Returns
        -------
        model_parameters_names : tuple[str]
            Material constitutive model parameters names (str).
        """
        # Set material properties names
        model_parameters_names = ('elastic_symmetry', 'elastic_moduli',
                                  'euler_angles', 'hardening_law',
                                  'hardening_parameters')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model_parameters_names
    # -------------------------------------------------------------------------
    def state_init(self):
        """Get initialized material constitutive model state variables.

        Constitutive model state variables:

        * ``e_strain_mf``

            * *Infinitesimal strains*: Elastic infinitesimal strain tensor
              (matricial form).

            * *Finite strains*: Elastic spatial logarithmic strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon^{e}}` /
              :math:`\\boldsymbol{\\varepsilon^{e}}`

        * ``acc_p_strain``

            * Accumulated plastic strain.

            * *Symbol*: :math:`\\bar{\\varepsilon}^{p}`

        * ``strain_mf``

            * *Infinitesimal strains*: Infinitesimal strain tensor
              (matricial form).

            * *Finite strains*: Spatial logarithmic strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon}` /
              :math:`\\boldsymbol{\\varepsilon}`

        * ``stress_mf``

            * *Infinitesimal strains*: Cauchy stress tensor (matricial form).

            * *Finite strains*: Kirchhoff stress tensor (matricial form) within
              :py:meth:`state_update`, first Piola-Kirchhoff stress tensor
              (matricial form) otherwise.

            * *Symbol*: :math:`\\boldsymbol{\\sigma}` /
              (:math:`\\boldsymbol{\\tau}`, :math:`\\boldsymbol{P}`)

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
            torch.zeros((self._n_dim, self._n_dim),
                        dtype=torch.float, device=self._device),
                        self._n_dim, self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            state_variables_init['e_strain_mf'].clone()
        # Initialize stress tensors
        if self._strain_formulation == 'infinitesimal':
            # Cauchy stress tensor (symmetric)
            state_variables_init['stress_mf'] = vget_tensor_mf(
                torch.zeros((self._n_dim, self._n_dim),
                            dtype=torch.float, device=self._device),
                            self._n_dim, self._comp_order_sym)
        else:
            # First Piola-Kirchhoff stress tensor (nonsymmetric)
            state_variables_init['stress_mf'] = vget_tensor_mf(
                torch.zeros((self._n_dim, self._n_dim),
                            dtype=torch.float, device=self._device),
                            self._n_dim, self._comp_order_nsym)
        # Initialize internal variables
        state_variables_init['acc_p_strain'] = \
            torch.tensor(0.0, dtype=torch.float, device=self._device)
        # Initialize state flags
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
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
        # Set state update convergence tolerance
        su_conv_tol = 1e-6
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
        # Compute flow vector
        if torch.allclose(dev_trial_stress_mf,
                          torch.zeros(dev_trial_stress_mf.shape,
                                      dtype=torch.float, device=self._device),
                          atol=1e-10):
            flow_vector_mf = torch.zeros(dev_trial_stress_mf.shape,
                                         dtype=torch.float,
                                         device=self._device)
        else:
            flow_vector_mf = np.sqrt(3.0/2.0)*(
                dev_trial_stress_mf/torch.norm(dev_trial_stress_mf))
        # Compute von Mises equivalent trial stress
        vm_trial_stress = np.sqrt(3.0/2.0)*torch.norm(dev_trial_stress_mf)
        # Compute trial accumulated plastic strain
        acc_p_trial_strain = acc_p_strain_old
        # Compute trial yield stress
        yield_stress, _ = \
            hardening_law(hardening_parameters, acc_p_trial_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield function
        yield_function = vm_trial_stress - yield_stress
        # If the trial stress state lies inside the von Mises yield function,
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
            # Set incremental plastic multiplier initial iterative guess
            inc_p_mult = 0
            # Initialize Newton-Raphson iteration counter
            nr_iter = 0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Start Newton-Raphson iterative loop
            while True:
                # Compute current yield stress and hardening modulus
                yield_stress, H = hardening_law(hardening_parameters,
                                                acc_p_strain_old + inc_p_mult)
                # Compute return-mapping residual (scalar)
                residual = vm_trial_stress - 3.0*G*inc_p_mult - yield_stress
                # Check Newton-Raphson iterative procedure convergence
                error = abs(residual/yield_stress)
                is_converged = error < su_conv_tol
                # Control Newton-Raphson iteration loop flow
                if is_converged:
                    # Leave Newton-Raphson iterative loop (converged solution)
                    break
                elif nr_iter == su_max_n_iterations:
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
                # Compute return-mapping Jacobian (scalar)
                Jacobian = -3.0*G - H
                # Solve return-mapping linearized equation
                d_iter = -residual/Jacobian
                # Update incremental plastic multiplier
                inc_p_mult = inc_p_mult + d_iter
            # Update elastic strain
            e_strain_mf = e_trial_strain_mf - inc_p_mult*flow_vector_mf
            # Update stress
            stress_mf = torch.matmul(e_consistent_tangent_mf, e_strain_mf)
            # Update accumulated plastic strain
            acc_p_strain = acc_p_strain_old + inc_p_mult
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
        if is_plast:
            # Compute elastoplastic consistent tangent modulus
            factor_1 = ((inc_p_mult*6.0*G**2)/vm_trial_stress)
            factor_2 = (6.0*G**2)*((inc_p_mult/vm_trial_stress)
                                   - (1.0/(3.0*G + H)))
            unit_flow_vector = \
                np.sqrt(2.0/3.0)*vget_tensor_from_mf(flow_vector_mf, n_dim,
                                                     comp_order_sym,
                                                     device=self._device)
            consistent_tangent = e_consistent_tangent \
                - factor_1*fodevprojsym + factor_2*dyad22_1(
                    unit_flow_vector, unit_flow_vector)
        else:
            consistent_tangent = e_consistent_tangent
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
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return state_variables, consistent_tangent_mf