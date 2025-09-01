# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import torch

def complex_max(t, th=1e-6):
	t_abs = torch.abs(t)
	return t*(t_abs > th) + th*torch.exp(1j*torch.angle(t))*(t_abs <= th)

class BICGSTAB:
    def __init__(self, model, max_iter=100, tol=1e-6, pre_step=None, post_step=None):
        super().__init__()
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.M = None
        
        self.pre_step = pre_step if pre_step is not None else lambda x: x
        self.post_step = post_step if post_step is not None else lambda x: x
   
    def setup_M(self, eps, source, Sxf_I, Sxb_I, Syf_I, Syb_I, wl, dL, source_mult):
        # setup eps
        freq = torch.tensor([dL/wl], dtype=torch.float32, device=source.device)
        all_S = 1+1j*(Sxf_I + Sxb_I + Syf_I + Syb_I)/4
        eps_imag = 1/torch.abs(all_S)
        eps_map = torch.stack([eps, eps_imag], dim=-1)
        self.model.setup(eps_map, freq=freq)

        def apply_M(rhs, x):
            bs, sx, sy = rhs.shape 

            x_RI = torch.view_as_real(x)
            rhs_RI = torch.view_as_real(rhs)

            ##### for old model #####
            # r_scale = torch.maximum(torch.mean(torch.abs(rhs_RI), dim=(1,2,3), keepdim=True), torch.tensor(1e-9))
            # x_scale = torch.maximum(torch.mean(torch.abs(x_RI), dim=(1,2,3), keepdim=True), torch.tensor(1e-9))
            # output = r_scale*self.model(x_RI/x_scale, eps, rhs_RI/r_scale, source, Sxf_I, Syf_I, source_mult)

            ##### for new model #####
            output = self.model(rhs_RI, freq=freq)
            return torch.view_as_complex(output)

        self.M = apply_M
        self.x = torch.view_as_complex(1e-3*torch.randn_like(source, device=source.device)) # keep x as class variable for reuse
    
    def setup_Aop(self, op):
        # Aop is the operator that does A(x), here it takes in
        self.Aop = op

    def matvec(self, x):
        raise NotImplementedError
    
    def dot(self, x, y):
        # return torch.sum(x * y)
        prod = torch.sum(torch.conj(x) * y)
        # print('>>> dot product: ', prod)
        return prod
    
    def zeros_like(self, x):
        return torch.zeros_like(x)
    
    def scale(self, x, a):
        return a * x
    
    def axby(self, a, x, b, y):
        return a * x + b * y

    def vecnorm(self, x):
        # return torch.norm(x)    
        _norm = torch.norm(x)
        # print('>>> norm: ', _norm)
        return _norm
    
    def solve(self, b, tol=None, max_iter=None, apply_M_steps=None, verbose=False, return_xr_history=False):
        if max_iter is None:
            max_iter = self.max_iter
        if apply_M_steps is None:
            apply_M_steps = max_iter
        if tol is None:
            tol = self.tol

        with torch.no_grad():
            b = self.pre_step(b)

            assert torch.is_complex(b), "b must be complex"

            r = b.clone()
            r0_hat = r.clone()
            rho = self.dot(r0_hat, r)
            p = r.clone()

            x = self.zeros_like(b)

            beta0 = self.vecnorm(r)
            # print("Initial residual norm: ", beta)
            if verbose:
                # print("Iteration: ", 0, "Residual norm: ", beta, "Relative residual norm: ", 1.0)
                print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (0, beta0, 1.0))
        
            relres_history = [1.0]
            if return_xr_history:
                x_history = [x]
                r_history = [b]

            for j in range(max_iter):
                if j % apply_M_steps == 0:
                    y = self.M(p, x)
                else:
                    y = p
                v = self.Aop(y)

                r0_hat_v = self.dot(r0_hat, v)

                alpha = rho/complex_max(r0_hat_v, 1e-12)

                h = x + alpha*y
                s = r - alpha * v

                if j < apply_M_steps:
                    z = self.M(s, x)
                else:
                    z = s
                t = self.Aop(z)

                w = self.dot(t, s)/self.dot(t, t)

                x = h + w * z

                r = s - w*t

                r0_hat_r = self.dot(r0_hat, r)
                beta = r0_hat_r/complex_max(rho, 1e-12) * alpha/complex_max(w, 1e-12)
                rho = r0_hat_r
                p = r + beta*(p-w*v)

                # restart:
                if torch.mean(torch.abs(r0_hat_r)) < 1e-9:
                    print("restart")
                    r0_hat = r.clone()
                    p = r.clone()
                
                if return_xr_history:
                    x_history.append(x.clone())
                    r_history.append(r.clone())

                residual_norm = self.vecnorm(p)

                relres_history.append(torch.abs(residual_norm)/torch.abs(beta0))

                # Check for convergence
                if verbose:
                    print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (j, residual_norm, torch.abs(residual_norm)/torch.abs(beta0)))
                    
                if torch.abs(residual_norm)/torch.abs(beta0) < tol:
                    break

            if 1 and not verbose:
                print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (max_iter, residual_norm, torch.abs(residual_norm)/torch.abs(beta0)))

            x = self.post_step(x)

        if return_xr_history:
            return x, r, relres_history, x_history, r_history
        else:
            return x, r, relres_history, None, None
