

"""

            s0 = hardening_parameters['s0']
            a = hardening_parameters['a']
            b = torch.tensor(hardening_parameters['b'], device= self._device)
            ep0 = torch.tensor(hardening_parameters['ep0'], device= self._device)



            in_vals = tuple((E, v, s0, a, b, ep0))
            in_vals = torch.stack(in_vals)
            
            out_vals = inc_p_mult

            inc_p_mult = implicit_function_newton_raphson(in_vals, out_vals, implicit_func, self._problem_type,
                e_trial_strain_mf, acc_p_strain_old, hardening_law,
                su_conv_tol, su_max_n_iterations, self._device)

            H = 1.0




def implicit_func(in_vals, out_vals, problem_type, e_trial_strain_mf,
                  acc_p_strain_old, hardening_law, device):
    # Extract input values
    E, v, s0, a, b, ep0 = in_vals
    # Extract output values
    inc_p_mult = out_vals
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Lamé parameters
    lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
    miu = E/(2.0*(1.0 + v))
    # Compute shear modulus
    G = E/(2.0*(1.0 + v))
    # Build hardening parameters
    hardening_parameters = {'s0': s0, 'a': a, 'b': b, 'ep0': ep0}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set 3D problem parameters
    n_dim, comp_order_sym, _ = get_problem_type_parameters(4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required fourth-order tensors
    _, _, _, fosym, fodiagtrace, _, fodevprojsym = \
        get_id_operators(n_dim, device=device)
    fodevprojsym_mf = vget_tensor_mf(fodevprojsym, n_dim, comp_order_sym,
                                     device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute elastic consistent tangent modulus according to problem type
    # and store it in matricial form
    if problem_type in [1, 4]:
        e_consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    e_consistent_tangent_mf = vget_tensor_mf(e_consistent_tangent,
                                             n_dim, comp_order_sym,
                                             device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute trial stress
    trial_stress_mf = torch.matmul(e_consistent_tangent_mf,
                                    e_trial_strain_mf)
    # Compute deviatoric trial stress
    dev_trial_stress_mf = torch.matmul(fodevprojsym_mf, trial_stress_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute von Mises equivalent trial stress
    vm_trial_stress = math.sqrt(3.0/2.0)*torch.norm(dev_trial_stress_mf)
    # Compute trial accumulated plastic strain
    acc_p_trial_strain = acc_p_strain_old
    # Compute trial yield stress
    yield_stress, _ = \
        hardening_law(hardening_parameters, acc_p_trial_strain)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute return-mapping residual (scalar)
    residual = vm_trial_stress - 3.0*G*inc_p_mult - yield_stress
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return residual



class ImplicitFunctionNewtonRaphson(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_vals, out_vals0, implicit_func, problem_type,
                e_trial_strain_mf, acc_p_strain_old, hardening_law,
                su_conv_tol, su_max_n_iterations, device):
        
        # Set residual implicit function (fixed input values)
        residual_fun = lambda x: implicit_func(
            in_vals, x, problem_type, e_trial_strain_mf, acc_p_strain_old,
            hardening_law, device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract parameters
        E, v, s0, a, b, ep0 = in_vals
        # Compute shear modulus
        G = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build hardening parameters
        hardening_parameters = {'s0': s0, 'a': a, 'b': b, 'ep0': ep0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize output values
        inc_p_mult = out_vals0.detach().clone()
        # Initialize Newton-Raphson iteration counter
        nr_iter = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Start Newton-Raphson iterative loop
        while True:
            # Compute current yield stress and hardening modulus
            yield_stress, H = hardening_law(hardening_parameters,
                                            acc_p_strain_old + inc_p_mult)
            # Compute return-mapping residual (scalar)
            residual = residual_fun(inc_p_mult)
            # Check Newton-Raphson iterative procedure convergence
            error = abs(residual/yield_stress)
            is_converged = error < su_conv_tol
            # Control Newton-Raphson iteration loop flow
            if is_converged:
                # Leave Newton-Raphson iterative loop (converged solution)
                break
            elif nr_iter == su_max_n_iterations:
                # Update state update failure flag
                is_su_fail = torch.tensor(True, device=device)
                # Leave Newton-Raphson iterative loop (failed solution)
                break
            else:
                # Increment iteration counter
                nr_iter = nr_iter + 1
            # Compute return-mapping Jacobian (scalar)
            Jacobian = -3.0*G - H
            # Solve return-mapping linearized equation
            d_iter = -residual/Jacobian
            # Update incremental plastic multiplier
            inc_p_mult = inc_p_mult + d_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store output values
        out_vals = inc_p_mult
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save required data for backward propagation
        ctx.save_for_backward(E, v, s0, a, b, ep0, out_vals)
        # Store implicit function
        ctx.implicit_func = implicit_func
        
        ctx.problem_type = problem_type
        ctx.e_trial_strain_mf = e_trial_strain_mf
        ctx.acc_p_strain_old = acc_p_strain_old
        ctx.hardening_law = hardening_law
        ctx.device = device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return out_vals
    # -------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output):
        # Get required data for backward propagation
        E, v, s0, a, b, ep0, out_vals = ctx.saved_tensors
        # Get implicit function
        implicit_func = ctx.implicit_func
        
        problem_type = ctx.problem_type
        e_trial_strain_mf = ctx.e_trial_strain_mf
        acc_p_strain_old = ctx.acc_p_strain_old
        hardening_law = ctx.hardening_law
        device = ctx.device

        in_vals = tuple((E, v, s0, a, b, ep0))
        in_vals = torch.stack(in_vals)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of implicit function w.r.t. output values
        jac_out = torch.autograd.functional.jacobian(
            lambda x: implicit_func(in_vals, x, problem_type, e_trial_strain_mf,
                  acc_p_strain_old, hardening_law, device), out_vals)
        # Compute derivative of implicit function w.r.t. input values
        jac_in = torch.autograd.functional.jacobian(
            lambda x: implicit_func(x, out_vals, problem_type, e_trial_strain_mf,
                  acc_p_strain_old, hardening_law, device), in_vals)
        
        
        
        jac_out = torch.diag(jac_out.expand(6))
        
        #print(jac_out)
        #print(jac_in)
        #
        #print(grad_output)
        
        dout_din = -torch.linalg.solve(jac_out, jac_in)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative 
        #grad_in_vals = (-torch.linalg.solve(jac_out, jac_in).T @ grad_output)
        
        grad_in_vals = grad_output*dout_din
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return (grad_in_vals, None, None, None, None, None, None, None, None,
                None, None)


def implicit_function_newton_raphson(
    in_vals, out_vals0, implicit_func, problem_type,
                e_trial_strain_mf, acc_p_strain_old, hardening_law,
                su_conv_tol, su_max_n_iterations, device):
    return ImplicitFunctionNewtonRaphson.apply(
        in_vals, out_vals0, implicit_func, problem_type,
                e_trial_strain_mf, acc_p_strain_old, hardening_law,
                su_conv_tol, su_max_n_iterations, device
    )
    
"""