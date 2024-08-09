"""Bazant M7 concrete microplane model.

Classes
-------
BazantM7
    Bazant M7 concrete microplane model.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import numpy as np
# Local
from simulators.fetorch.material.models.interface import ConstitutiveModel
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    get_tensor_mf, get_tensor_from_mf
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Weican Li (weican_li@brown.edu)'
__credits__ = ['Weican Li', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

class BazantM7(ConstitutiveModel):
    """Bazant M7 concrete microplane model.

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite
        strain formulation through kinematic extension (infinitesimal
        constitutive formulation and purely finite strain kinematic extension -
        'finite-kinext').
    _model_parameters : dict
        Material constitutive model parameters.
    _ndim : int
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
            Constitutive model material properties (key, str) values
            (item, {int, float, bool}).
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Set material constitutive model name
        self._name = 'bazant_m7'
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
    # -------------------------------------------------------------------------
    @staticmethod
    def get_required_model_parameters():
        """Get required material constitutive model parameters.

        Returns
        -------
        model_parameters_names : tuple[str]
            Material constitutive model parameters names (str).
        """
        # Set material properties names
        model_parameters_names = ()
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

        * ``M7_microstate``

            * A total of 190 inner state variables, related to state on all 37
              microplane level.
              
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
        state_variables_init['e_strain_mf'] = get_tensor_mf(
            torch.zeros((self._n_dim, self._n_dim),
                        dtype=torch.float, device=self._device),
                        self._n_dim, self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            state_variables_init['e_strain_mf'].clone()
        # Initialize stress tensors
        state_variables_init['stress_mf'] = get_tensor_mf(
            torch.zeros((self._n_dim, self._n_dim),
                        dtype=torch.float, device=self._device),
                        self._n_dim, self._comp_order_sym)
        # Initialize internal variables
        state_variables_init['M7_microstate'] = \
            torch.zeros((38, 5), dtype=torch.float, device=self._device)
        # Initialize state flags
        state_variables_init['is_su_fail'] = False
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
        def M7FMATERIAL(strainInc, epsOld, stressOld, stateOld, NTENS):
            """State update internal function."""
            # Material properties
            # global noip, k_1
            vh_ini = stateOld[:189]

            # Extracting values from strainInc and epsOld
            sig_old = stressOld * unit_conv
            eps = epsOld
            deps = strainInc
            
            # Compute material response
            sig = np.zeros(NTENS)
            E_tan_N = np.zeros((6, noip))
            E_tan_L = np.zeros((6, noip))
            E_tan_M = np.zeros((6, noip))

            eps_N = mydot(eps, qn)
            eps_L = mydot(eps, ql)
            eps_M = mydot(eps, qm)
            deps_N = mydot(deps, qn)
            deps_L = mydot(deps, ql)
            deps_M = mydot(deps, qm)

            sv_ini = vh_ini[0]
            phi0 = vh_ini[1]
            sN_ini = vh_ini[2:nvhi:nvhm]
            eps_N0_pos = vh_ini[5:nvhi:nvhm]
            eps_N0_neg = vh_ini[6:nvhi:nvhm]
            zeta0 = vh_ini[188]
            zeta = zeta0
            ev_ini = (eps[0][0]+eps[0][1]+eps[0][2])/3.0
            dev = (deps[0][0]+deps[0][1]+deps[0][2])/3.0

            # More calculations and assignments...
            # Initialize 
            sN_ela = np.zeros(noip)
            E_N = np.zeros(noip)
            E_tan_ela = np.zeros((6, noip))
            svneg = 0.0
            E_tan_vol = np.zeros((6, noip))
            sd_fin = np.zeros(noip)
            sdneg = np.zeros(noip)
            sdpos = np.zeros(noip)
            E_tan_dev = np.zeros((6, noip))
            sN_boundary = np.zeros(noip)
            E_tan_norm = np.zeros((6, noip))
            sL_fin = np.zeros(noip)
            sM_fin = np.zeros(noip)
            

            sN_ela = c_norm_elastic(eps_N, deps_N, sN_ini, eps_N0_pos, eps_N0_neg, sv_ini, sN_ela, E_N, zeta, E_tan_ela)

            ev_fin = ev_ini + dev
            
            svneg, E_tan_vol = c_vol(dev, ev_ini, sv_ini, deps_N, eps_N,svneg,E_tan_vol)
            ded = deps_N - dev
            ed_ini = eps_N - ev_ini     
            ed_fin = ed_ini + ded


            sdneg, sdpos, E_tan_dev = c_dev(ded, ed_ini, dev, ev_ini, sv_ini, sd_fin, sdneg, sdpos, E_tan_dev)

            eN_fin = ev_fin + ed_fin 


            sN_boundary, E_tan_norm = c_norm(eN_fin, sv_ini, sN_boundary, E_tan_norm)

            # Assuming vh_ini, sN_ela, sN_boundary, svneg, sdneg, eN_fin, E_tan_norm, E_tan_vol, E_tan_dev, E_tan_ela, w, zeta0, dev, deps_L, deps_M are defined

            phi0 = vh_ini[1]

            sig_N = np.zeros(noip)

            for i in range(noip):
                if sN_ela[i] > sN_boundary[i]:
                    sig_N[i] = sN_boundary[i]
                    eps_N0_pos[i] = eN_fin[i]
                    E_tan_N[:, i] = E_tan_norm[:, i]
                elif sN_ela[i] < svneg + sdneg[i]:
                    sig_N[i] = svneg + sdneg[i]
                    eps_N0_neg[i] = eN_fin[i]
                    E_tan_N[:, i] = E_tan_vol[:, i] + E_tan_dev[:, i]
                else:
                    sig_N[i] = sN_ela[i]
                    E_tan_N[:, i] = E_tan_ela[:, i]


            sum_sN_fin = mydot(sig_N, w)
            sv_fin = sum_sN_fin / 3.0

            sum_bulk = mydot(E_N, w)
            bulk = sum_bulk / 3.0

            zeta = zeta0
            if sv_ini > 0.0 and sv_fin > 0.0:
                devv = np.abs(dev - ((sv_fin - sv_ini) / bulk))
                zeta += np.abs(devv)

            sL_ini = vh_ini[3:nvhi:nvhm]
            sM_ini = vh_ini[4:nvhi:nvhm]

            sN_fin = sig_N

            eN_fin += deps_N
            deL = deps_L
            deM = deps_M

            E_ela_L = np.zeros((6, noip))
            E_ela_M = np.zeros((6, noip))
            sL_fin, sM_fin, E_tan_L, E_tan_M, E_ela_L, E_ela_M = c_shear2(eps_L, eps_M, sN_fin, deL, deM, sL_ini, sM_ini, E_tan_N, sL_fin, sM_fin, ev_fin, E_tan_L, E_tan_M)

            

            # Final Stress Vector
            sig = mydot(qn, (w* sN_fin)) + mydot(qm, (w* sM_fin)) + mydot(ql, (w* sL_fin))

            E_tan_N[:, :noip] *= w
            E_tan_L[:, :noip] *= w
            E_tan_M[:, :noip] *= w
            jacobian = mydot(qn, E_tan_N.T) + mydot(qm, E_tan_M.T) + mydot(ql, E_tan_L.T)
            sig /= unit_conv
            jacobian /= unit_conv
            # Update microplane normal and shear stresses
            vh_fin = np.zeros(190) #careful, should be 189 if follow strictly with original Hoang
            vh_fin[0] = sv_fin
            vh_fin[2:nvhi:nvhm] = sN_fin
            vh_fin[3:nvhi:nvhm] = sL_fin
            vh_fin[4:nvhi:nvhm] = sM_fin
            vh_fin[5:nvhi:nvhm] = eps_N0_pos
            vh_fin[6:nvhi:nvhm] = eps_N0_neg
            vh_fin[188] = zeta

            stateNew = vh_fin.copy()
            stateNew[189] = k_1
            stressNew = sig.copy()
            DDSDDE = np.zeros((NTENS, NTENS))

            for loop1 in range(6):
                for loop2 in range(6):
                    DDSDDE[loop1, loop2] = jacobian[loop1, loop2]
                    
            E_tan_ela[:, :noip] *= w
            E_ela_L[:, :noip] *= w
            E_ela_M[:, :noip] *= w
                    
            jacobian2 = mydot(qn, E_tan_ela.T) + mydot(qm, E_ela_M.T) + mydot(ql, E_ela_L.T)
            jacobian2 /= unit_conv

            transposed_array = jacobian2
            
            
            # BPF: Ignoring elastic strain (not needed for now)
            #elastic_strain =  np.linalg.solve(jacobian2, stressNew)
            elastic_strain = np.zeros_like(stressNew)

            return stressNew, stateNew, DDSDDE, elastic_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def c_norm_elastic(eN_ini, deN, sN_ini, eps_N0_pos, eps_N0_neg, sv_ini, sN_ela, E_N, zeta, c_tan_ela):
            """State update internal function."""
            # Calculate initial bulk modulus
            E_N_0 = young / (1.0 - 2.0 * poisson)

            for i in range(noip):

                if sN_ini[i] >= 0:
                    weight_factor = 40.0
                    t0 = 1.0
                    t1 = (weight_factor * zeta) ** 2
                    t2 = (weight_factor * zeta) ** 4
                    t3 = (weight_factor * zeta) ** 6
                    t4 = (weight_factor * zeta) ** 8
                    tsum = t0 + t1 + t2
                    fzeta = 1.0 / tsum
                    E_N[i] = E_N_0 * fzeta * np.exp(-c_19 * eps_N0_pos[i])

                    if sN_ini[i] > E_N_0 * (eN_ini[i] + deN[i]) :
                        if sN_ini[i] * deN[i] < 0.0:
                            E_N[i] = E_N_0
                else:
                    E_N[i] = E_N_0 * (np.exp(-c_20 * abs(eps_N0_neg[i]) / (1.0 + c_18 * max(-sv_ini, 0) / E_N_0)) +
                                      c_21 * max(-sv_ini, 0) / E_N_0)

                c_tan_ela[:, i] = E_N[i] * qn[:, i]

            for i in range(noip):
                sN_ela[i] = sN_ini[i] + E_N[i] * deN[i]
            
            return sN_ela
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def c_vol(dev, ev_ini, sv_ini, deps_N, eps_N, svneg, c_tan_vol):
            """State update internal function."""
            
            # Initialize unit vector
            unit_vector = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

            xk0 = k_3 * k_1 * young
            Cv0 = young / (1.0 - 2.0 * poisson)
            prStrainDiff = np.max(eps_N + deps_N) - np.min(eps_N + deps_N)
            xk4 = (k_6 * (prStrainDiff / k_1) ** k_7) / (1.0 + np.min([np.max([-sv_ini, 0.0]), 250.0]) / Cv0) + k_4

            ev_fin = ev_ini + dev
            svneg = -xk0 * np.exp(-ev_fin / (xk4 * k_1))

            for i in range(noip):
                if eps_N[i] + deps_N[i] == np.max(eps_N + deps_N):
                    beta_e = 1.0
                elif eps_N[i] + deps_N[i] == np.min(eps_N + deps_N):
                    beta_e = -1.0
                else:
                    beta_e = 0.0
                c_tan_vol[:, i] = xk0 * np.exp(-ev_fin / (xk4 * k_1)) * \
                                   (1.0 / (3.0 * xk4 * k_1) * unit_vector - ev_fin / k_1 * 1.0 / xk4 ** 2 * k_6 /
                                    (1.0 + np.min([np.max([-sv_ini, 0.0]), 250.0]) / Cv0) * k_7 *
                                    (prStrainDiff / k_1) ** (k_7 - 1.0) * beta_e / k_1 * qn[:, i])

            return svneg, c_tan_vol
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def c_dev(ded, ed_ini, dev, ev_ini, sv_ini, sd_fin, sdneg, sdpos, c_tan_dev):
            """State update internal function."""
            
            unit_vector = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            # Define constants
            c_5_0 = 1.3e-2
            c_5_1 = 4.0
            c_7_0 = 1.2e-2
            c_7_1 = 35.0e2
            c_8_0 = 1.2e-2
            c_8_1 = 20.0
            c_6_0 = 4.0e2
            c_6_1 = 4.0e1
            c_6_2 = 13.0
            c_9_0 = 4.0e2
            c_9_1 = 4.0e1
            c_9_2 = 13.0
            c_5_M4 = 3.0
            c_6_M4 = 1.30
            c_7_M4 = 10.e-1
            c_8_M4 = 8.0
            c_9_M4 = 0.e-1
            f_c0 = 15.08
            E_0 = 20000.0
            c_40 = 1.0

            f_cp = 90.3
            beta_15 = c_5_1 * np.exp(-c_40 * (f_cp / young - f_c0 / E_0))
            beta_16 = c_8_1 * np.exp(-c_40 * (f_cp / young - f_c0 / E_0))
            beta_17 = c_7_1 * np.exp(-c_40 * (f_cp / young - f_c0 / E_0))
            beta_18 = c_6_0 * np.exp(-c_40 * (f_cp / young - f_c0 / E_0))
            beta_19 = c_9_0 * np.exp(-c_40 * (f_cp / young - f_c0 / E_0))

            ev_fin = ev_ini + dev

            c_5 = beta_15 * np.tanh(c_5_0 * max(-ev_fin, 0.0) / k_1) + c_5_M4

            c_8 = beta_16 * np.tanh(c_8_0 * max(-ev_fin, 0.0) / k_1) + c_8_M4

            c_7 = beta_17 * np.tanh(c_7_0 * max(-ev_fin, 0.0) / k_1) + c_7_M4

            if beta_18 * max(-ev_fin / k_1 - c_6_1, 0.0) >= np.log(c_6_2):
                c_6 = c_6_M4 * c_6_2
            else:
                c_6 = c_6_M4 * np.exp(beta_18 * max(-ev_fin / k_1 - c_6_1, 0.0))

            if beta_19 * max(-ev_fin / k_1 - c_9_1, 0.0) >= np.log(c_9_2):
                c_9 = c_9_M4 * c_9_2
                beta_e = 0.0
            else:
                c_9 = c_9_M4 * np.exp(beta_19 * max(-ev_fin / k_1 - c_9_1, 0.0))
                beta_e = c_9_M4 * np.exp(beta_19 * max(-ev_fin / k_1 - c_9_1, 0.0)) * beta_19 * hside(-ev_fin / k_1 - c_9_1) * (-1.0 / (3.0 * k_1))

            ed_fin = ed_ini + ded

            sdneg[:] = -young * k_1 * c_8 / (1.0 + (np.maximum(-ed_fin - c_9 * c_8 * k_1, 0.0) / (c_7 * k_1)) ** 2.0)
            sdpos[:] = young * k_1 * c_5 / (1.0 + (np.maximum(ed_fin - c_6 * c_5 * k_1, 0.0) / (c_7 * k_1 * c_20)) ** 2.0)

            for i in range(noip):
                c_tan_dev[:, i] = young * k_1 * c_8 / (1.0 + (np.maximum(-ed_fin[i] - c_9 * c_8 * k_1, 0.0) / (c_7 * \
                                   k_1)) ** 2.0) ** 2.0 * 2.0 * np.maximum(-ed_fin[i] - c_9 * c_8 * k_1, 0.0) * \
                                   hside(-ed_fin[i] - c_9 * c_8 * k_1) / (c_7 * k_1) ** 2.0 * (-qn[:, i] + \
                                   unit_vector / 3.0 - k_1 * c_9 * beta_16 * (1.0 - (np.tanh(c_8_0 * max(-ev_fin, 0.0) / k_1)) \
                                   ** 2.0) * c_8_0 / k_1 * hside(-ev_fin) * (-unit_vector / 3.0) - k_1 * c_8 * beta_e * \
                                   unit_vector) - young * k_1 / (1.0 + (np.maximum(-ed_fin[i] - c_9 * c_8 * k_1, 0.0) / (c_7 * k_1)) \
                                   ** 2.0) * beta_16 * (1.0 - (np.tanh(c_8_0 * max(-ev_fin, 0.0) / k_1)) ** 2.0) * c_8_0 / k_1 * \
                                   hside(-ev_fin) * (-unit_vector / 3.0) + young * k_1 * c_8 / (1.0 + (np.maximum(-ed_fin[i] - c_9 * c_8 * k_1, 0.0) / (c_7 * k_1)) ** 2.0) ** 2.0 \
                                   * (np.maximum(-ed_fin[i] - c_9 * c_8 * \
                                   k_1, 0.0) / k_1) ** 2.0 * (-2.0 / c_7 ** 3.0) * beta_17 * (1.0 - (np.tanh(c_7_0 * \
                                   max(-ev_fin, 0.0) / k_1)) ** 2.0) * c_7_0 / k_1 * hside(-ev_fin) * (-unit_vector / 3.0)

            return sdneg, sdpos, c_tan_dev
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def c_norm(eN_fin, sv_ini, sN_boundary, c_tan_norm):
            """State update internal function."""

            d_1 = 0.095
            d_2 = 35.0
            d_3 = 1.7
            d_4 = 1.7
            d_5 = 1000.0
            d_6 = 25.0
            V_f = 0.0
            
            E_N_0 = young / (1.0 - 2.0 * poisson)
            c_1 = d_1 * np.tanh(d_2 * V_f - d_3) + d_4 * np.exp(-np.maximum(-sv_ini - d_6, 0.0) / E_N_0 * d_5)

            if sv_ini < 0.0:
                eb_N = c_3 * k_1 - c_4 / E_N_0 * sv_ini
            else:
                eb_N = c_3 * k_1

            fstar = k_1 * young * c_1
            beta_N = c_2 * c_1 * k_1

            sN_boundary = fstar * np.exp(-np.maximum(eN_fin - beta_N, 0.0) / eb_N)

            for i in range(noip):
                c_tan_norm[:, i] = fstar * np.exp(-np.maximum(eN_fin[i] - beta_N, 0.0) / eb_N) * (-hside(eN_fin[i] - beta_N) / eb_N) * qn[:, i]
            
            return sN_boundary, c_tan_norm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def c_shear2(eps_L, eps_M, sN_fin, deL, deM, sL_ini,sM_ini,c_tan_N,sL_fin,sM_fin,ev_fin,c_tan_L,c_tan_M):
            """State update internal function."""

            # Initialize other variables
            unit_vector = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            sT = 0.1 * young * k_1

            Ct = young / (1.0 + poisson) * (1.0 - 4.0 * poisson) / (1.0 - 2.0 * poisson)
            c_6_p = c_10
            c_12_p = c_11
            c_13_p = c_12 * 3.371e-4
            s_0 = k_1 * k_2 * Ct

            sig_o = max(Ct * k_1 * (c_12_p - c_13_p * max(ev_fin, 0.0) / k_1), 0.0)
            
            # Weican original code (raising division by zero warnings)
            #fsp = 0.5 * (1 - np.tanh(sN_fin / sT)) * (
            #        ((c_6_p * np.maximum(-sN_fin + sig_o, 0.0)) ** (-1.0) + s_0 ** (-1.0)) ** (-1.0)) + \
            #      0.5 * (1 + np.tanh(sN_fin / sT)) * (
            #              ((c_6_p * np.maximum(sig_o, 0.0)) ** (-1.0) + s_0 ** (-1.0)) ** (-1.0))
            # Weican refactored code
            fsp = np.zeros(37)
            for i in range(37):
                if (-sN_fin[i] + sig_o > 0):
                    fsp[i] = fsp[i] +  0.5 * (1 - np.tanh(sN_fin[i] / sT)) * (
                        ((c_6_p *(-sN_fin[i] + sig_o)) ** (-1.0) + s_0 ** (-1.0)) ** (-1.0))
                if (sig_o > 0):
                    fsp[i] = fsp[i] +  0.5 * (1 + np.tanh(sN_fin[i] / sT)) * (
                        ((c_6_p * (sig_o) ) ** (-1.0) + s_0 ** (-1.0)) ** (-1.0))


            E_T = Ct

            sL_fin_e = sL_ini + E_T * deL
            sM_fin_e = sM_ini + E_T * deM
            stfe = np.sqrt(sL_fin_e * sL_fin_e + sM_fin_e * sM_fin_e)

            c_tan_L = np.zeros((6, noip))
            c_tan_M = np.zeros((6, noip))

            for i in range(noip):
                if stfe[i] > fsp[i] and stfe[i] != 0.0:
                    if sig_o > sN_fin[i] and sig_o > 0.0: #tested
                        sL_fin[i] = sL_fin_e[i] / stfe[i] * fsp[i]
                        sM_fin[i] = sM_fin_e[i] / stfe[i] * fsp[i]
                        c_tan_L[:, i] = (fsp[i]/stfe[i]*E_T*ql[:,i]-fsp[i]*
                                         sL_fin_e[i]/stfe[i]**2.0*(sL_fin_e[i]/stfe[i]*E_T*ql[:,i] +
                                         sM_fin_e[i]/stfe[i]*E_T*qm[:,i]) + sL_fin_e[i]/stfe[i]*( 
                                         -0.50*(1.0-(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(-sN_fin[i]+
                                         sig_o,0.0))**(-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:,i]/sT +
                                         0.50*(1.0-np.tanh(sN_fin[i]/sT))*(((c_6_p*max(-sN_fin[i]+sig_o,
                                         0.0))**(-1.0)+s_0**(-1.0))**(-2.0)*(c_6_p*max(-sN_fin[i]+
                                         sig_o,0.0))**(-2.0)*c_6_p*hside(-sN_fin[i]+sig_o)*(hside(Ct*
                                         k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*(-Ct*c_13_p*
                                         hside(ev_fin))*unit_vector/3.0-c_tan_N[:,i])) + 0.50*
                                         (1.0+(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(sig_o,0.0))**
                                         (-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:,i]/sT + 0.50*(1.0+
                                         np.tanh(sN_fin[i]/sT))*(((c_6_p*max(sig_o,0.0))**(-1.0)+s_0**
                                         (-1.0))**(-2.0)*(c_6_p*max(sig_o,0.0))**(-2.0)*c_6_p*
                                         hside(sig_o)*(hside(Ct*k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*
                                         (-Ct*c_13_p*hside(ev_fin))*unit_vector/3.0)) ))

                        c_tan_M[:, i] = (fsp[i]/stfe[i]*E_T*qm[:,i]-fsp[i]*
                                         sM_fin_e[i]/stfe[i]**2.0*(sM_fin_e[i]/stfe[i]*E_T*qm[:,i] +
                                         sL_fin_e[i]/stfe[i]*E_T*ql[:,i]) + sM_fin_e[i]/stfe[i]*( 
                                         -0.50*(1.0-(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(-sN_fin[i]+
		                                 sig_o,0.0))**(-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:,i]/sT +
		                                 0.50*(1.0-np.tanh(sN_fin[i]/sT))*(((c_6_p*max(-sN_fin[i]+sig_o,
		                                 0.0))**(-1.0)+s_0**(-1.0))**(-2.0)*(c_6_p*max(-sN_fin[i]+
		                                 sig_o,0.0))**(-2.0)*c_6_p*hside(-sN_fin[i]+sig_o)*(hside(Ct*
		                                 k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*(-Ct*c_13_p*
		                                 hside(ev_fin))*unit_vector/3.0-c_tan_N[:,i])) + 0.50*
		                                 (1.0+(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(sig_o,0.0))**
		                                 (-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:,i]/sT + 0.50*(1.0+
		                                 np.tanh(sN_fin[i]/sT))*(((c_6_p*max(sig_o,0.0))**(-1.0)+s_0**
		                                 (-1.0))**(-2.0)*(c_6_p*max(sig_o,0.0))**(-2.0)*c_6_p*
		                                 hside(sig_o)*(hside(Ct*k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*
		                                 (-Ct*c_13_p*hside(ev_fin))*unit_vector/3.0)) ))

                    elif sig_o > sN_fin[i] and sig_o <= 0.0: #tested
                        sL_fin[i] = sL_fin_e[i] / stfe[i] * fsp[i]
                        sM_fin[i] = sM_fin_e[i] / stfe[i] * fsp[i]

                        c_tan_L[:, i] = (fsp[i]/stfe[i]*E_T*ql[:, i]-fsp[i]*
                        		        sL_fin_e[i]/stfe[i]**2.0*(sL_fin_e[i]/stfe[i]*E_T*ql[:, i] +
                        		        sM_fin_e[i]/stfe[i]*E_T*qm[:, i]) + sL_fin_e[i]/stfe[i]*( 
                        		        -0.5*(1.0-(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(-sN_fin[i]+
                        		        sig_o,0.0))**(-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:, i]/sT +
                        		        0.5*(1.0-np.tanh(sN_fin[i]/sT))*(((c_6_p*max(-sN_fin[i]+sig_o,
                        		        0.0))**(-1.0)+s_0**(-1.0))**(-2.0)*(c_6_p*max(-sN_fin[i]+
                        		        sig_o,0.0))**(-2.0)*c_6_p*hside(-sN_fin[i]+sig_o)*(hside(Ct*
                        		        k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*(-Ct*c_13_p*
                        		        hside(ev_fin))*unit_vector/3.0-c_tan_N[:, i])) ) )

                        c_tan_M[:, i] = (fsp[i]/stfe[i]*E_T*qm[:, i]-fsp[i]*
                        		        sM_fin_e[i]/stfe[i]**2.0*(sM_fin_e[i]/stfe[i]*E_T*qm[:, i] +
                        		        sL_fin_e[i]/stfe[i]*E_T*ql[:, i]) + sM_fin_e[i]/stfe[i]*( 
                        		        -0.50*(1.0-(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(-sN_fin[i]+
                        		        sig_o,0.0))**(-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:, i]/sT +
                        		        0.50*(1.0-np.tanh(sN_fin[i]/sT))*(((c_6_p*max(-sN_fin[i]+sig_o,
                        		        0.0))**(-1.0)+s_0**(-1.0))**(-2.0)*(c_6_p*max(-sN_fin[i]+
                        		        sig_o,0.0))**(-2.0)*c_6_p*hside(-sN_fin[i]+sig_o)*(hside(Ct*
                        		        k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*(-Ct*c_13_p*
                        		        hside(ev_fin))*unit_vector/3.0-c_tan_N[:, i])) ) )
                        		        
                    elif sig_o <= sN_fin[i] and sig_o > 0.0: #tested
                        sL_fin[i] = sL_fin_e[i] / stfe[i] * fsp[i]
                        sM_fin[i] = sM_fin_e[i] / stfe[i] * fsp[i]
                        c_tan_L[:, i] = (fsp[i]/stfe[i]*E_T*ql[:, i]-fsp[i]*
                                         sL_fin_e[i]/stfe[i]**2.0*(sL_fin_e[i]/stfe[i]*E_T*ql[:, i] +
                                         sM_fin_e[i]/stfe[i]*E_T*qm[:, i]) + sL_fin_e[i]/stfe[i]*
                                         ( 0.50*
                                         (1.0+(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(sig_o,0.0))**
                                         (-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:, i]/sT + 0.50*(1.0+
                                         np.tanh(sN_fin[i]/sT))*(((c_6_p*max(sig_o,0.0))**(-1.0)+s_0**
                                         (-1.0))**(-2.0)*(c_6_p*max(sig_o,0.0))**(-2.0)*c_6_p*
                                         hside(sig_o)*(hside(Ct*k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*
                                         (-Ct*c_13_p*hside(ev_fin))*unit_vector/3.0)) ))
                        c_tan_M[:, i] = (fsp[i]/stfe[i]*E_T*qm[:, i]-fsp[i]*
                                         sM_fin_e[i]/stfe[i]**2.0*(sM_fin_e[i]/stfe[i]*E_T*qm[:, i] +
                                         sL_fin_e[i]/stfe[i]*E_T*ql[:, i]) + sM_fin_e[i]/stfe[i]*
                                         ( 0.50*
                                         (1.0+(np.tanh(sN_fin[i]/sT))**2.0)*(((c_6_p*max(sig_o,0.0))**
                                         (-1.0)+s_0**(-1.0))**(-1.0))*c_tan_N[:, i]/sT + 0.50*(1.0+
                                         np.tanh(sN_fin[i]/sT))*(((c_6_p*max(sig_o,0.0))**(-1.0)+s_0**
                                         (-1.0))**(-2.0)*(c_6_p*max(sig_o,0.0))**(-2.0)*c_6_p*
                                         hside(sig_o)*(hside(Ct*k_1*(c_12_p-c_13_p*max(ev_fin,0.0)/k_1))*
                                         (-Ct*c_13_p*hside(ev_fin))*unit_vector/3.0)) ))

                    else: #tested
                        sL_fin[i] = sL_fin_e[i]
                        sM_fin[i] = sM_fin_e[i]
                        c_tan_L[:, i] = E_T * ql[:, i]
                        c_tan_M[:, i] = E_T * qm[:, i]
                        sL_fin[i] = sL_fin_e[i] / stfe[i] * fsp[i]
                        sM_fin[i] = sM_fin_e[i] / stfe[i] * fsp[i]

                        c_tan_L[:, i] = fsp[i] / stfe[i] * E_T * ql[:, i] - \
                        		        fsp[i] * sL_fin_e[i] / stfe[i]**2 * \
                        		        (sL_fin_e[i] / stfe[i] * E_T * ql[:, i] + \
                        		        sM_fin_e[i] / stfe[i] * E_T * qm[:, i])

                        c_tan_M[:, i] = fsp[i] / stfe[i] * E_T * qm[:, i] - \
                        		        fsp[i] * sM_fin_e[i] / stfe[i]**2 * \
                        		        (sM_fin_e[i] / stfe[i] * E_T * qm[:, i] + \
                        		        sL_fin_e[i] / stfe[i] * E_T * ql[:, i])

                else: #tested
                    sL_fin[i] = sL_fin_e[i]
                    sM_fin[i] = sM_fin_e[i]
                    c_tan_L[:, i] = E_T * ql[:, i]
                    c_tan_M[:, i] = E_T * qm[:, i]

            c_ela_L = np.zeros((6, noip))
            c_ela_M = np.zeros((6, noip))
            for i in range(noip):
                c_ela_L[:, i] = E_T * ql[:, i]
                c_ela_M[:, i] = E_T * qm[:, i]
            return sL_fin, sM_fin, c_tan_L, c_tan_M,c_ela_L, c_ela_M
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def hside(x):
            """State update internal function."""
            if x >= 0.0:
                return 1.0
            else:
                return 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
        def mydot(a,b):
            """State update internal function."""
            c = np.dot(a,b)
            c = np.squeeze(c)
            return c
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PI = 3.1415926535897932384626433832795
        noip = 37
        nvhm = 5
        nvhf = 2
        nvhi = noip * nvhm + nvhf
        # Define common arrays
        qn = np.zeros((6, noip))
        ql = np.zeros((6, noip))
        qm = np.zeros((6, noip))
        w = np.zeros(noip)
        te = np.zeros((4, noip))

        # Generate the table to numerically calculate the spherical integral
        ij = np.array([1, 2, 3, 1, 1, 2, 1, 2, 3, 2, 3, 3])
        ij = np.array(ij).reshape(2, 6)
        # Define the elements of the array
        elements = [
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 1.072388573030e-02],
            [0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00, 1.072388573030e-02],
            [1.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.072388573030e-02],
            [0.000000000000e+00, 7.071067811870e-01, 7.071067811870e-01, 2.114160951980e-02],
            [0.000000000000e+00, -7.071067811870e-01, 7.071067811870e-01, 2.114160951980e-02],
            [7.071067811870e-01, 0.000000000000e+00, 7.071067811870e-01, 2.114160951980e-02],
            [-7.071067811870e-01, 0.000000000000e+00, 7.071067811870e-01, 2.114160951980e-02],
            [7.071067811870e-01, 7.071067811870e-01, 0.000000000000e+00, 2.114160951980e-02],
            [-7.071067811870e-01, 7.071067811870e-01, 0.000000000000e+00, 2.114160951980e-02],
            [0.000000000000e+00, 3.089512677750e-01, 9.510778696510e-01, 5.355055908370e-03],
            [0.000000000000e+00, -3.089512677750e-01, 9.510778696510e-01, 5.355055908370e-03],
            [0.000000000000e+00, 9.510778696510e-01, 3.089512677750e-01, 5.355055908370e-03],
            [0.000000000000e+00, -9.510778696510e-01, 3.089512677750e-01, 5.355055908370e-03],
            [3.089512677750e-01, 0.000000000000e+00, 9.510778696510e-01, 5.355055908370e-03],
            [-3.089512677750e-01, 0.000000000000e+00, 9.510778696510e-01, 5.355055908370e-03],
            [9.510778696510e-01, 0.000000000000e+00, 3.089512677750e-01, 5.355055908370e-03],
            [-9.510778696510e-01, 0.000000000000e+00, 3.089512677750e-01, 5.355055908370e-03],
            [3.089512677750e-01, 9.510778696510e-01, 0.000000000000e+00, 5.355055908370e-03],
            [-3.089512677750e-01, 9.510778696510e-01, 0.000000000000e+00, 5.355055908370e-03],
            [9.510778696510e-01, 3.089512677750e-01,  0.000000000000e+00, 5.355055908370e-03],
            [-9.510778696510e-01, 3.089512677750e-01, 0.000000000000e+00, 5.355055908370e-03],
            [8.805355183100e-01, 3.351545919390e-01, 3.351545919390e-01, 1.677709091560e-02],
            [-8.805355183100e-01, 3.351545919390e-01, 3.351545919390e-01, 1.677709091560e-02],
            [8.805355183100e-01, -3.351545919390e-01, 3.351545919390e-01, 1.677709091560e-02],
            [-8.805355183100e-01, -3.351545919390e-01, 3.351545919390e-01, 1.677709091560e-02],
            [3.351545919390e-01, 8.805355183100e-01, 3.351545919390e-01, 1.677709091560e-02],
            [-3.351545919390e-01, 8.805355183100e-01, 3.351545919390e-01, 1.677709091560e-02],
            [3.351545919390e-01, -8.805355183100e-01, 3.351545919390e-01, 1.677709091560e-02],
            [-3.351545919390e-01, -8.805355183100e-01, 3.351545919390e-01, 1.677709091560e-02],
            [3.351545919390e-01, 3.351545919390e-01, 8.805355183100e-01, 1.677709091560e-02],
            [-3.351545919390e-01, 3.351545919390e-01, 8.805355183100e-01, 1.677709091560e-02],
            [3.351545919390e-01, -3.351545919390e-01, 8.805355183100e-01, 1.677709091560e-02],
            [-3.351545919390e-01, -3.351545919390e-01, 8.805355183100e-01, 1.677709091560e-02],
            [5.773502691900e-01, 5.773502691900e-01, 5.773502691900e-01, 1.884823095080e-02],
            [-5.773502691900e-01, 5.773502691900e-01, 5.773502691900e-01, 1.884823095080e-02],
            [5.773502691900e-01, -5.773502691900e-01, 5.773502691900e-01, 1.884823095080e-02],
            [-5.773502691900e-01, -5.773502691900e-01, 5.773502691900e-01, 1.884823095080e-02]
        ]

        # Reshape the array to [4, noip]
        te = np.array(elements).reshape(4, -1)

        xn = np.zeros(3)

        for jp in range(noip):  
            w[jp] = elements[jp][3] * 6.0

            xn[0] = elements[jp][2]
            xn[1] = elements[jp][1]
            xn[2] = elements[jp][0]    
            

            # BPF: Avoid messing with the global numpy random seed
            #np.random.seed(jp+10)  #specific random seed to ensure consistency for qn, qm and ql. Treat qn, qm and ql as material parameters
            #rand_vec = np.random.rand(3)
            def generate_random_vector(size, seed):
                random_state = np.random.RandomState(seed)
                return random_state.rand(size)
            rand_vec = generate_random_vector(3, jp+10)
            

            xm = rand_vec - mydot(xn, rand_vec) * xn
            xm /= np.linalg.norm(xm)
            xl = np.cross(xn, xm)
            xl /= np.linalg.norm(xl)

            for k in range(6):
                i = ij[0, k] - 1
                j = ij[1, k] - 1
                qn[k, jp] = xn[i] * xn[j]
                qm[k, jp] = 0.5 * (xn[i] * xm[j] + xn[j] * xm[i])
                ql[k, jp] = 0.5 * (xn[i] * xl[j] + xn[j] * xl[i])

        # Define common variables
        young = 30173.0
        poisson = 0.18000
#        E = self._material_properties['E']
#        v = self._material_properties['v']
        
        th_del = 0.005e10
        unit_conv = 1.0

        k_1 = 142.00e-6
        #k_1 = 0.001
        k_2 = 110.000
        k_3 = 20.0  # was 12.0
        k_4 = 40.0  # was 38.0
        k_8 = 3.0
        k_9 = 5.0e-1
        k_5 = 1.0e-4  # was 1.0
        k_6 = 1.0e-4
        k_7 = 1.8

        c_2 = 1.76e-1
        c_3 = 4.0
        c_4 = 50.0
        c_10 = 3.3e-1
        c_11 = 5.0e-1
        c_12 = 7.00e3
        
        c_16 = 10.0
        c_17 = 1.00e-2
        c_18 = 4.0e3
        c_19 = 4.5e3
        c_20 = 3.0e2
        c_21 = 6.0e1

        NTENS = 6

        strainInc = np.zeros((1,6))
        
        strainInc[0][0] = inc_strain[0][0]
        strainInc[0][1] = inc_strain[1][1]
        strainInc[0][2] = inc_strain[2][2]
        strainInc[0][3] = 2*inc_strain[0][1]
        strainInc[0][4] = 2*inc_strain[0][2]
        strainInc[0][5] = 2*inc_strain[1][2]

        epsOld = np.zeros((1,6))
        temp_strain = get_tensor_from_mf(state_variables_old['strain_mf'], self._n_dim, self._comp_order_sym) 
        epsOld[0][0] = temp_strain[0][0]
        epsOld[0][1] = temp_strain[1][1]
        epsOld[0][2] = temp_strain[2][2]
        epsOld[0][3] = 2*temp_strain[0][1]
        epsOld[0][4] = 2*temp_strain[0][2]
        epsOld[0][5] = 2*temp_strain[1][2]
        
        stressOld = np.zeros((1,6))
        temp_stress = get_tensor_from_mf(state_variables_old['stress_mf'], self._n_dim, self._comp_order_sym) 
        stressOld[0][0] = temp_stress[0][0]
        stressOld[0][1] = temp_stress[1][1]
        stressOld[0][2] = temp_stress[2][2]
        stressOld[0][3] = temp_stress[0][1]
        stressOld[0][4] = temp_stress[0][2]
        stressOld[0][5] = temp_stress[1][2]
        
        stateOld = np.zeros(190)
        for i in range(38):
            for j in range(5):
                stateOld[i*5+j] = state_variables_old['M7_microstate'][i][j]

        stressNew, stateNew, DDSDDE ,myelasticstrain= M7FMATERIAL(strainInc, epsOld, stressOld, stateOld, NTENS)
        
#        print ('----------DDSDDE----------\n')
#        transposed_array = DDSDDE
#        #transposed_array = np.transpose(elements)
#        for row in transposed_array:
#            formatted_row = ' '.join("{:25.16f}".format(num) for num in row)
#            print(formatted_row)
#        
#        print ('----------strainInc----------\n', strainInc)
#        print ('----------epsOld----------\n', epsOld)
#        print ('----------stressOld----------\n', stressOld)
##        print ('----------stateOld----------\n', stateOld)
##        print ('----------stateNew----------\n', stateNew)
#        print ('----------stressNew----------\n', stressNew)
        
        # Initialize state variables dictionary
        state_variables = self.state_init()

        temp_strain[0][0] = myelasticstrain[0]
        temp_strain[1][1] = myelasticstrain[1]
        temp_strain[2][2] = myelasticstrain[2]
        temp_strain[0][1] = 0.5*myelasticstrain[3]
        temp_strain[0][2] = 0.5*myelasticstrain[4]
        temp_strain[1][2] = 0.5*myelasticstrain[5]
        temp_strain[1][0] = 0.5*myelasticstrain[3]
        temp_strain[2][0] = 0.5*myelasticstrain[4]
        temp_strain[2][1] = 0.5*myelasticstrain[5]

        state_variables['e_strain_mf'] = get_tensor_mf(temp_strain, self._n_dim, self._comp_order_sym)

        temp_strain[0][0] = strainInc[0][0] + epsOld[0][0]
        temp_strain[1][1] = strainInc[0][1] + epsOld[0][1]
        temp_strain[2][2] = strainInc[0][2] + epsOld[0][2]
        temp_strain[0][1] = 0.5*(strainInc[0][3] + epsOld[0][3])
        temp_strain[0][2] = 0.5*(strainInc[0][4] + epsOld[0][4])
        temp_strain[1][2] = 0.5*(strainInc[0][5] + epsOld[0][5])
        temp_strain[1][0] = 0.5*(strainInc[0][3] + epsOld[0][3])
        temp_strain[2][0] = 0.5*(strainInc[0][4] + epsOld[0][4])
        temp_strain[2][1] = 0.5*(strainInc[0][5] + epsOld[0][5])

        state_variables['strain_mf'] = get_tensor_mf(temp_strain, self._n_dim, self._comp_order_sym)

        temp_stress[0][0] = stressNew[0]
        temp_stress[1][1] = stressNew[1]
        temp_stress[2][2] = stressNew[2]
        temp_stress[0][1] = stressNew[3]
        temp_stress[0][2] = stressNew[4]
        temp_stress[1][2] = stressNew[5]
        temp_stress[1][0] = stressNew[3]
        temp_stress[2][0] = stressNew[4]
        temp_stress[2][1] = stressNew[5]
        
        state_variables['stress_mf'] = get_tensor_mf(temp_stress, self._n_dim, self._comp_order_sym)
        
        # BPF: Converted to torch.Tensor
        state_variables['M7_microstate'] = torch.zeros((38, 5), dtype=torch.float, device=self._device)

        # BPF: Handle potential state variables overflow
        if max(stateNew) > np.finfo(np.float32).max:
            state_variables['is_su_fail'] = True
        else:
            for i in range(38):
                for j in range(5):
                    state_variables['M7_microstate'][i][j] = stateNew[i*5+j]

        # BPF: Converted to torch.Tensor
        consistent_tangent_mf = torch.zeros((6, 6), dtype=torch.float, device=self._device)
        temp_order = np.array([0, 1, 2, 3, 5, 4])
        
        # BPF: Ignoring consistent tangent modulus (not needed for now)
        #for i in range(6):
        #    for j in range(6):
        #        if i >= 3:
        #            DDSDDE[i][j] = DDSDDE[i][j]*np.sqrt(2)
        #        if j >= 3:
        #            DDSDDE[i][j] = DDSDDE[i][j]*np.sqrt(2)
        #        consistent_tangent_mf[temp_order[i]][temp_order[j]] = DDSDDE[i][j]
            
#        print ('----------consistent_tangent_mf----------\n')
#        transposed_array = consistent_tangent_mf
#        #transposed_array = np.transpose(elements)
#        for row in transposed_array:
#            formatted_row = ' '.join("{:25.16f}".format(num) for num in row)
#            print(formatted_row)
        
        return state_variables, consistent_tangent_mf
