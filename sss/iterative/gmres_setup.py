# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import torch

import gin

def complex_max(t, th=1e-9):
	t_abs = torch.abs(t)
	return t*(t_abs > th) + th*torch.exp(1j*torch.angle(t))*(t_abs <= th)

@gin.configurable
class GMRES_setup:
    def __init__(
        self,
        model,
        max_iter: int = 100,
        tol: float = 1e-6,
        # Aop,
        # pre_step=None,
        # post_step=None
    ):
        self.max_iter = max_iter
        self.tol = tol
        
        self.model = model
        # self.Aop = Aop
        # self.pre_step = pre_step if pre_step is not None else lambda x: x
        # self.post_step = post_step if post_step is not None else lambda x: x
    
    def setup_M(self, eps, source, Sxf_I, Sxb_I, Syf_I, Syb_I, wl, dL, source_mult, init_x=None):
        # setup eps
        freq = torch.tensor([dL/wl], dtype=torch.float32, device=source.device)
        all_S = 1+1j*(Sxf_I + Sxb_I + Syf_I + Syb_I)/4
        eps_imag = 1/torch.abs(all_S)
        eps_map = torch.stack([eps, eps_imag], dim=-1)
        self.model.setup(eps_map, freq=freq)
        self.Sx_map = torch.abs(1+1j*Sxf_I)
        self.Sy_map = torch.abs(1+1j*Syf_I)

        self.S_descale = (self.Sx_map * self.Sy_map) ** 1

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
        if init_x is not None:
            self.x = init_x.to(source.device)
        else:
            self.x = torch.view_as_complex(1e-2*torch.randn_like(source, device=source.device)) # keep x as class variable for reuse
        # self.x = torch.view_as_complex(torch.zeros_like(source, device=source.device))
    
    def clear_x(self):
        self.x = torch.view_as_complex(1e-2*torch.randn_like(self.source, device=self.source.device)) # keep x as class variable for reuse
        # self.x = torch.view_as_complex(torch.zeros_like(self.source, device=self.source.device))
    
    def setup_Aop(self, op):
        # Aop is the operator that does A(x), here it takes in
        self.Aop = op
    
    def dot(self, x, y):
        prod = torch.sum(torch.conj(x) * y, dim=(1,2), keepdim=False)
        return prod
    
    def scale(self, x, a):
        return a * x
    
    def axby(self, a, x, b, y):
        return a * x + b * y

    def vecnorm(self, x):
        _norm = torch.sqrt(self.dot(x, x))
        return _norm

    @torch.no_grad()
    def solve(self, b, verbose=False, max_iter=None, tol=None, return_xr_history=False):
        max_iter = max_iter if max_iter is not None else self.max_iter
        tol = tol if tol is not None else self.tol

        bs = b.shape[0]

        assert torch.is_complex(b), "b must be complex"

        # b = self.pre_step(b)

        # x might be updated from previous DDM iters, so update init x and b:
        b_for_zero_x = b.clone()
        b = self.axby(1, b, -1, self.Aop(self.x))

        beta = self.vecnorm(b)
        if verbose:
            print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (0, torch.abs(beta), 1.0))
        V = []
        Z = []
        V.append(self.scale(b, 1/complex_max(beta[:, None, None])))
        H = torch.zeros((bs, max_iter + 1, max_iter), dtype=torch.complex64, device=b.device)

        # Arnoldi process
        x = self.x.clone()
        relres_history = [1.0]
        if return_xr_history:
            x_history = [x]
            r_history = [b]
        for j in range(max_iter):
            z = self.M(V[j], x)
            Z.append(z)
            w = self.Aop(z)
            for i in range(j + 1):
                H[:, i, j] = self.dot(w, V[i])
                w = self.axby(1, w, -H[:, i, j][:, None, None], V[i])
                assert torch.is_complex(w), "w must be complex"

            H[:, j + 1, j] = self.vecnorm(w)
            V.append(self.scale(w, 1/complex_max(H[:, j + 1, j][:, None, None])))

            num_iter = j + 1
            # Solve the least squares problem
            e1 = torch.zeros((bs, num_iter + 1), dtype=torch.complex64, device=b.device)
            e1[:, 0] = beta
            regularizer = 1e-6*torch.max(torch.abs(H[:, :num_iter + 1, :num_iter]))*torch.eye(num_iter + 1, dtype=torch.complex64, device=b.device)
            y, residual_norm, _, _ = torch.linalg.lstsq(H[:, :num_iter + 1, :num_iter] + regularizer[None, :num_iter + 1, :num_iter], e1, rcond=None)
            residual_norm = torch.sqrt(residual_norm)

            # Compute the approximate solution
            x = self.x.clone()
            for i in range(num_iter):
                x = self.axby(1, x, y[:, i][:, None, None], Z[i])

            # Compute the residual
            r = self.axby(1, b_for_zero_x, -1, self.Aop(x))
            assert torch.is_complex(r), "r must be complex"

            if return_xr_history:
                x_history.append(x)
                r_history.append(r)

            residual_norm = self.vecnorm(r)

            relres_history.append(torch.abs(residual_norm)/torch.abs(beta))

            # Check for convergence
            if verbose:
                print("Iteration: %d, Residual norm: %e, Relative residual norm: %e" % (num_iter, torch.abs(residual_norm), torch.abs(residual_norm)/torch.abs(beta)))
            
            r_ratio = torch.mean(torch.abs(residual_norm))/torch.mean(torch.abs(beta))
            # print("r_ratio: ", r_ratio)
            if r_ratio < tol:
                print("early converge at iteration: ", j)
                break
        
        # self.x = x.clone()
        self.x = x / self.S_descale
        # self.x = torch.where(self.Sx_map + self.Sy_map > 2, 0, x)

        # x = self.post_step(x)

        if return_xr_history:
            return x, r, relres_history, x_history, r_history
        else:
            return x, r, relres_history, None, None
    
    @torch.no_grad()
    def solve_with_restart(self, b, restart, verbose=False, tol=None, max_iter=None):
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter

        # check if b is complex
        assert torch.is_complex(b), "b must be complex"
        
        with torch.no_grad():
            print("Using restart solve with restart: ", restart)
            relres = 1
            norm_b = self.vecnorm(b)
            x = torch.zeros_like(b)
            sum_iters = 0
            relres_history = [1.0]
            res = b
            res_norm = norm_b
            relres_restart_history = []
            while(relres > tol and sum_iters < max_iter):
                e, e_relres_history = super().solve(res, tol/relres, restart, verbose=verbose)
                sum_iters += len(e_relres_history) - 1
                e_relres_history = [val*res_norm for val in e_relres_history]
                relres_history += e_relres_history[1:]
                x = self.axby(1, x, 1, e)
                res = b - self.Aop(x)
                res_norm = self.vecnorm(res)
                relres = torch.abs(res_norm / norm_b)
                relres_restart_history.append(relres)
                print(">>> Relative residual: ", relres)
                if len(e_relres_history) <= 2:
                    break

        print('ITERATION: ', sum_iters, '>>> residual norm: ', self.vecnorm(b - self.Aop(x)))
        return x, relres_history
