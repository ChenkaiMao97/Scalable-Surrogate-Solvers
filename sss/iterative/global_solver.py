# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from sss.iterative.gmres import GMRES
from sss.data.ceviche_solver import CevicheSolver
from sss.iterative.subdomain_solver import SubdomainSolver
from sss.iterative.two_level_schwarz import TwoLevelSchwarz
from sss.iterative.nonuniform_schwarz import NonUniformSchwarz
from sss.iterative.nonuniform_schwarz_roll import NonUniformSchwarzRoll
from sss.utils.solver_utils import SparseDirectSolver

from sss.utils.PML_utils import make_Sx_Sy
from sss.utils.PDE_utils import maxwell_robin_residue
from sss.utils.plot_utils import plot_helper, setup_plot_data
from sss.utils.UI import printc

from ceviche.constants import C_0, EPSILON_0

import gin

def MAE(a, b):
    return torch.mean(torch.abs(a - b)) / torch.mean(torch.abs(b))

def debug_plot_b(b, thickness = 4, name='debug_b'):
    bs, _, four_d_sx = b.shape
    d_sx = four_d_sx // 4
    for i in range(bs):
        res = np.zeros((d_sx, d_sx))
        res[:thickness,:] = np.abs(b.cpu().numpy())[i,0,None,0:d_sx]
        res[-thickness:,:] = np.abs(b.cpu().numpy())[i,0,None,d_sx:2*d_sx]
        res[:,0:thickness] = np.abs(b.cpu().numpy())[i,0,2*d_sx:3*d_sx, None]
        res[:,-thickness:] = np.abs(b.cpu().numpy())[i,0,3*d_sx:4*d_sx, None]
        plt.figure()
        plt.imshow(res, cmap='turbo')
        plt.title(f"{name}_{i}")
        plt.colorbar()
        plt.savefig(f"{name}_{i}.png")
        plt.close()

@gin.configurable
class GlobalSolver:
    """
    GlobalSolver has two main modules:

    1. Global domain solver (Schwarz, two-level Schwarz, FDTI):
        - partition of global data to subdomain shapes 
        - boundary update for subdomain problems
        - solve additional coarse space corrections

    2. Subdomain solver (GMRES, BICGSTAB, etc.):
        - solve a batch of (independent) subdomain problems
    """
    def __init__(
        self,
        output_dir: str,
        Nx: int,
        Ny: int,
        solver_type: str = "two_level_Schwarz",
        DDM_iters: int = 10,
        momentum: float = 0.1,
        bc_change_th: float = None,
        half_periodic: bool = False,
        save_intermediate = False,
        GMRES_iter_for_solve = 1,
        GMRES_iter_for_coarse_space = 1,
        apply_coarse_space_every_N_iter: int = 1,
        stop_error_th = None,
        debug_coarse_space: bool = False,
        spacing = 0,
        plot_N: int = 6,
        seed: int = 0,
        gpu_id: int = 0,
    ):
        self.output_dir = output_dir
        self.Nx = Nx
        self.Ny = Ny
        self.solver_type = solver_type
        self.DDM_iters = DDM_iters
        self.momentum = momentum
        self.bc_change_th = bc_change_th
        self.half_periodic = half_periodic
        self.apply_coarse_space_every_N_iter = apply_coarse_space_every_N_iter
        self.stop_error_th = stop_error_th
        self.debug_coarse_space = debug_coarse_space
        self.save_intermediate = save_intermediate
        self.spacing = spacing
        self.plot_N = plot_N
        self.plot_iters = [round(x) for x in np.linspace(0, self.DDM_iters-1, self.plot_N).astype(int)]
        self.seed = seed
        self.gpu_id = gpu_id
        self.PML_setup_done = False
        self.direct_solver = None
        self.GMRES_iter_for_solve = GMRES_iter_for_solve
        self.GMRES_iter_for_coarse_space = GMRES_iter_for_coarse_space

    def init(self):
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")

        self.subdomain_solver = SubdomainSolver(gpu_id=self.gpu_id)
        self.subdomain_solver.init() # sets up model & GMRES
    
        if self.solver_type in ["two_level_Schwarz", "Schwarz_GMRES"]:
            self.global_DDM = TwoLevelSchwarz((self.Nx-2*self.spacing, self.Ny-2*self.spacing))
        elif self.solver_type == "nonuniform_Schwarz":
            assert self.spacing == 0, "spacing must be 0 for nonuniform Schwarz, meaning you are solving the full problem"
            self.global_DDM = NonUniformSchwarz((self.Nx, self.Ny))
            self.direct_solver = SparseDirectSolver()
        elif self.solver_type == "nonuniform_Schwarz_roll":
            assert self.spacing == 0, "spacing must be 0 for nonuniform Schwarz, meaning you are solving the full problem"
            self.global_DDM = NonUniformSchwarzRoll((self.Nx, self.Ny))
            self.direct_solver = SparseDirectSolver()
        else:
            raise ValueError(f"Solver type {self.solver_type} not supported")

        if self.save_intermediate and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def clear_PML_precompute(self):
        self.PML_setup_done = False
        if self.direct_solver is not None:
            self.direct_solver.clear()
    
    def source_scale(self, wl, dL):
        return 1j*2*np.pi*C_0*dL**2/wl*EPSILON_0
    
    def solve_from_random(self):
        ceviche_solver = CevicheSolver()
        ceviche_solver.init() # sets up model & GMRES

        # generate random problem:
        pattern = ceviche_solver.prepare_grayscale_pattern()
        wl = ceviche_solver.wavelengths[0] * 1e-3
        dL = ceviche_solver.dLs[0] * 1e-3

        ceviche_solve_time = []
        DDM_solve_time = []
        for i in range(pattern.shape[0]):
            # use ceviche to solve 
            self.clear_PML_precompute()
            tic_i = time.time()
            input_eps, Hx_out_forward, Hy_out_forward, Ez_out_forward, source_out = ceviche_solver.generation_one_device(pattern[i], wl, dL)
            toc_i = time.time()
            ceviche_solve_time.append(toc_i - tic_i)

            tic_i = time.time()
            # use subdomain solver to solve
            eps_torch = torch.from_numpy(input_eps).to(torch.float32).to(self.device)
            source_torch = torch.from_numpy(source_out).to(torch.complex64).to(self.device)
            results, end_time = self.DDM_solve(eps_torch, source_torch, wl, dL, gt=Ez_out_forward, npml=ceviche_solver.npml)
            DDM_solve_time.append(end_time-tic_i)

            if not self.global_DDM.periodic_padding:
                gt_ceviche = Ez_out_forward[self.spacing:-self.spacing, self.spacing:-self.spacing]
            else:
                gt_ceviche = Ez_out_forward

            mae = MAE(results.cpu(), torch.from_numpy(gt_ceviche)).numpy()

            os.makedirs(self.output_dir, exist_ok=True)
            vm = np.max(np.abs(gt_ceviche))
            plt.figure(figsize=(12, 4))
            plt.subplot(1,3,1)
            plt.imshow(gt_ceviche.real, cmap='seismic', vmin=-vm, vmax=vm)
            plt.colorbar()
            plt.title("ceviche")
            plt.subplot(1,3,2)
            plt.imshow(results.cpu().numpy().real, cmap='seismic', vmin=-vm, vmax=vm)
            plt.colorbar()
            plt.title("DDM")
            plt.subplot(1,3,3)
            plt.imshow(results.cpu().numpy().real - gt_ceviche.real, cmap='seismic')
            plt.colorbar()
            plt.title("error")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"sample_{i}_ceviche_ddm_wl{wl:.2e}_dL{dL:.2e}.png"))
            plt.close()
            print(f"Sample {i}: ceviche solve time: {ceviche_solve_time[i]:.2e}, DDM solve time: {DDM_solve_time[i]:.2e}, MAE: {mae:.2e}")
    
    def DDM_solve(self, eps, source, wl, dL, npml, gt=None, init_x=None):
        source_scale = self.source_scale(wl, dL)
        source = source * source_scale # after scaling, source and gt match in the residual fn
        
        source_max = torch.max(torch.abs(source)) # this scaling makes sure the input to DDM has a max of 10
        source = source / source_max
        if gt is not None:
            gt = gt / source_max.item() * self.subdomain_solver.source_mult

        if self.solver_type == "two_level_Schwarz":
            res, end_time = self.Schwarz_solve(eps, source, wl, dL, npml, gt=gt, init_x=init_x)
        elif self.solver_type == "Schwarz_GMRES":
            res, end_time = self.Schwarz_GMRES_solve(eps, source, wl, dL, npml, gt=gt, init_x=init_x)
        elif self.solver_type in ["nonuniform_Schwarz", "nonuniform_Schwarz_roll"]:
            res, end_time = self.nonuniform_Schwarz_solve(eps, source, wl, dL, npml, gt=gt, init_x=init_x)
        else:
            raise ValueError(f"Solver type {self.solver_type} not supported")
        
        # scale solution back to original scale:
        res = res * source_max / self.subdomain_solver.source_mult 

        return res, end_time
    
    @torch.no_grad()
    def Schwarz_solve(self, eps, source, wl, dL, npml, gt=None, init_x=None):
        t1 = time.time()
        ####### 1. partition the global data into subdomains #######
        wl = torch.tensor([wl]).to(self.device)
        dL = torch.tensor([dL]).to(self.device)
        omega = 2 * np.pi * C_0 / wl

        Sx_2D_f, Sy_2D_f = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='f')
        Sx_2D_b, Sy_2D_b = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='b')

        Sx_2D_f_I = torch.from_numpy(Sx_2D_f.imag).to(torch.float32).to(self.device)
        Sy_2D_f_I = torch.from_numpy(Sy_2D_f.imag).to(torch.float32).to(self.device)
        Sx_2D_b_I = torch.from_numpy(Sx_2D_b.imag).to(torch.float32).to(self.device)
        Sy_2D_b_I = torch.from_numpy(Sy_2D_b.imag).to(torch.float32).to(self.device)

        self.global_DDM.init_grid()
        model_bs = np.prod(self.global_DDM.grid_shape)
        d_sx, d_sy = self.global_DDM.region_size

        if not self.global_DDM.periodic_padding:
            # need to set global bc:
            if gt is None:
                gt = torch.from_numpy(np.zeros(source.shape, dtype=np.complex64)).to(self.device)
            else:
                gt = torch.from_numpy(gt).to(self.device)
            if self.spacing > 0:
                eps = eps[self.spacing:-self.spacing, self.spacing:-self.spacing]
                source = source[self.spacing:-self.spacing, self.spacing:-self.spacing]
                Sx_2D_f_I = Sx_2D_f_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
                Sy_2D_f_I = Sy_2D_f_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
                Sx_2D_b_I = Sx_2D_b_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
                Sy_2D_b_I = Sy_2D_b_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
                gt = gt[self.spacing:-self.spacing, self.spacing:-self.spacing]
            gt_batch = torch.stack(self.global_DDM.partition(gt), dim=0)

        eps_batch = torch.stack(self.global_DDM.partition(eps), dim=0)
        source_batch = torch.stack(self.global_DDM.partition(source), dim=0)
        Sx_f_I_batch = torch.stack(self.global_DDM.partition(Sx_2D_f_I), dim=0)
        Sy_f_I_batch = torch.stack(self.global_DDM.partition(Sy_2D_f_I), dim=0)
        Sx_b_I_batch = torch.stack(self.global_DDM.partition(Sx_2D_b_I), dim=0)
        Sy_b_I_batch = torch.stack(self.global_DDM.partition(Sy_2D_b_I), dim=0)

        sim_size_in_wl = eps.shape[0]*dL/wl*torch.mean(eps)**.5
        max_wavenumber = 2*np.pi*eps.shape[0]*dL/(wl/torch.max(eps)**.5)
        max_size_in_wl_subdomain = d_sx*dL/wl*torch.max(eps)**.5
        lambda_mean_div_dL = wl/torch.mean(eps)**.5/dL
        printc(f"sim_size_in_wl: {sim_size_in_wl.item():.2e}, max_size_in_wl_subdomain: {max_size_in_wl_subdomain.item():.2e}, max_wavenumber: {max_wavenumber.item():.2e}, lambda_mean_div_dL: {lambda_mean_div_dL.item():.2e}", color="g")

        if not self.global_DDM.periodic_padding:
            self.global_DDM.set_global_bc(gt_batch, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)

        init_x_batch = None
        if init_x is not None:
            init_x_batch = torch.stack(self.global_DDM.partition(init_x), dim=0)

        ####### 2. init zero solution and boundary conditions #######
        x = torch.zeros(model_bs, d_sx, d_sy, dtype=torch.complex64, device=self.device)
        top_bc, bottom_bc, left_bc, right_bc = self.global_DDM.get_bcs(x, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)
        
        t2 = time.time()
        # print(f"Time for setup: {t2 - t1:.2e} seconds")

        # prepare coarse space (solve eigen value problem B u = lambda u)
        if self.global_DDM.use_coarse_space:
            self.subdomain_solver.solver.max_iter = self.GMRES_iter_for_coarse_space
            self.subdomain_solver.setup((torch.zeros_like(torch.view_as_real(source_batch)), Sx_f_I_batch, Sy_f_I_batch, Sx_b_I_batch, Sy_b_I_batch, eps_batch, dL, wl), init_x=init_x_batch)
            self.global_DDM.prepare_coarse_space(self.subdomain_solver, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL, debug=self.debug_coarse_space, output_dir=self.output_dir)

        self.subdomain_solver.solver.max_iter = self.GMRES_iter_for_solve
        self.subdomain_solver.setup((torch.view_as_real(source_batch), Sx_f_I_batch, Sy_f_I_batch, Sx_b_I_batch, Sy_b_I_batch, eps_batch, dL, wl), init_x=init_x_batch)

        ####### 3. DDM iteration with momentum update #######
        global_x_history = []
        global_r_history = []
        pbar = tqdm(range(self.DDM_iters), desc="DDM iteration", leave=False)
        for i in pbar:
            if self.global_DDM.use_coarse_space and i>0 and (i+1) % self.apply_coarse_space_every_N_iter == 0:
                # apply coarse space every N iter, also skip the first iteration since there is no bc_error from zero initialization
                x = solution
                global_bc_error = self.global_DDM.get_bc_errors(x, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL, stack=True, zero_global_bcs=True)
                BQTr = self.global_DDM.apply_BQTr(-global_bc_error)
                x_coarse, BQTBQx = self.global_DDM.solve_x_coarse(BQTr)
                x_coarse = self.global_DDM.reassemble_x_coarse(x_coarse)
                if self.debug_coarse_space:
                    self.global_DDM.coarse_space_debug_plot(i, x_coarse, x, BQTr, BQTBQx, eps_batch, gt_batch, Sx_f_I_batch, Sy_f_I_batch, dL, wl, self.output_dir)
                
                new_x = self.global_DDM.update_x_from_coarse_solve(x_coarse, x)
                # update x in the subdomain solver, and update bcs:
                self.subdomain_solver.solver.x = new_x
                top_bc, bottom_bc, left_bc, right_bc = self.global_DDM.get_bcs(new_x, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)
                
            # (2) local Schwarz update
            t3 = time.time()
            rhs, solution, relres_history, x_history, r_history = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
            t4 = time.time()

            new_top_bc, new_bottom_bc, new_left_bc, new_right_bc = self.global_DDM.get_bcs(solution, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)

            top_bc = self.momentum*top_bc + (1-self.momentum)*new_top_bc
            bottom_bc = self.momentum*bottom_bc + (1-self.momentum)*new_bottom_bc
            left_bc = self.momentum*left_bc + (1-self.momentum)*new_left_bc
            right_bc = self.momentum*right_bc + (1-self.momentum)*new_right_bc
            t5 = time.time()

            if self.save_intermediate and i in self.plot_iters:
                global_x_history.append(self.global_DDM.combine(solution))
                global_r_history.append(self.global_DDM.combine(rhs))
            t6 = time.time()
            # print(f"step {i}: subdomain solve: {t4 - t3:.2e} seconds, boundary update: {t5 - t4:.2e} seconds, saving intermediate: {t6 - t5:.2e} seconds")
            global_MAE = MAE(solution, gt_batch)
            pbar.set_postfix(global_MAE=f"{global_MAE:.2e}")
            if self.stop_error_th is not None and global_MAE < self.stop_error_th:
                printc(f"Stopping DDM iteration at {i} iterations, global MAE: {global_MAE:.2e}", color="r")
                break

        end_time = time.time()

        if self.save_intermediate:
            colored_setup = setup_plot_data(eps.cpu(), source.cpu(), sx_f=Sx_2D_f_I.cpu(), sy_f=Sy_2D_f_I.cpu())
            plot_data = [colored_setup,source.imag,(Sx_2D_f_I+Sy_2D_f_I).cpu(),torch.abs(global_r_history[0])]
            plot_title = ['eps', 'source', 'pml', 'residue_input']
            cmaps = [None, 'seismic', None, 'turbo']
            center_zero = [False, False, False, False]
            row, column = 1, 4
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_residue_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

            plot_data = [*[x.real for x in global_x_history], *[torch.abs(r) for r in global_r_history]]
            plot_title = [*[f"output_{j}" for j in self.plot_iters], *[f"residual_{j}" for j in self.plot_iters]]
            cmaps = [*['seismic' for _ in range(self.plot_N)], *['turbo' for _ in range(self.plot_N)]]
            center_zero = [*[True for _ in range(self.plot_N)], *[False for _ in range(self.plot_N)]]
            row, column = 2, self.plot_N
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_debug_recurrent_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

        return self.global_DDM.combine(solution).to(eps.device), end_time
    
    @torch.no_grad()
    def nonuniform_Schwarz_solve(self, eps, source, wl, dL, npml, gt=None, init_x=None):
        t1 = time.time()
        ####### 1. partition the global data into subdomains #######
        wl = torch.tensor([wl]).to(eps.device)
        dL = torch.tensor([dL]).to(eps.device)
        omega = 2 * np.pi * C_0 / wl

        if not self.PML_setup_done:
            Sx_2D_f, Sy_2D_f = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='f')
            Sx_2D_b, Sy_2D_b = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='b')

            self.Sx_2D_f_I = torch.from_numpy(Sx_2D_f.imag).to(torch.float32)
            self.Sy_2D_f_I = torch.from_numpy(Sy_2D_f.imag).to(torch.float32)
            self.Sx_2D_b_I = torch.from_numpy(Sx_2D_b.imag).to(torch.float32)
            self.Sy_2D_b_I = torch.from_numpy(Sy_2D_b.imag).to(torch.float32)

            self.global_DDM.init_grid(npml)

            device = eps.device
            eps_batch, eps_pml = self.global_DDM.partition(eps.cpu())
            source_batch, source_pml = self.global_DDM.partition(source.cpu())
            eps_batch = eps_batch.to(device)
            source_batch = source_batch.to(device)

            Sx_f_I_batch, Sx_f_I_pml = self.global_DDM.partition(self.Sx_2D_f_I)
            assert torch.sum(torch.abs(Sx_f_I_batch)) == 0, f"sum of Sx_f_I_batch: {torch.sum(torch.abs(Sx_f_I_batch))}"
            Sy_f_I_batch, Sy_f_I_pml = self.global_DDM.partition(self.Sy_2D_f_I)
            assert torch.sum(torch.abs(Sy_f_I_batch)) == 0, f"sum of Sy_f_I_batch: {torch.sum(torch.abs(Sy_f_I_batch))}"
            Sx_b_I_batch, Sx_b_I_pml = self.global_DDM.partition(self.Sx_2D_b_I)
            assert torch.sum(torch.abs(Sx_b_I_batch)) == 0, f"sum of Sx_b_I_batch: {torch.sum(torch.abs(Sx_b_I_batch))}"
            Sy_b_I_batch, Sy_b_I_pml = self.global_DDM.partition(self.Sy_2D_b_I)
            assert torch.sum(torch.abs(Sy_b_I_batch)) == 0, f"sum of Sy_b_I_batch: {torch.sum(torch.abs(Sy_b_I_batch))}"

            self.eps_pml = eps_pml
            self.source_pml = source_pml
            self.Sx_f_I_batch = Sx_f_I_batch.to(eps_batch.device)
            self.Sy_f_I_batch = Sy_f_I_batch.to(eps_batch.device)
            self.Sx_b_I_batch = Sx_b_I_batch.to(eps_batch.device)
            self.Sy_b_I_batch = Sy_b_I_batch.to(eps_batch.device)
            self.Sx_f_I_pml = Sx_f_I_pml
            self.Sy_f_I_pml = Sy_f_I_pml
            self.Sx_b_I_pml = Sx_b_I_pml
            self.Sy_b_I_pml = Sy_b_I_pml

            self.direct_solver.setup(eps_pml, Sx_f_I_pml, Sy_f_I_pml, Sx_b_I_pml, Sy_b_I_pml, dL.cpu(), wl.cpu()) # on cpu
            
            self.PML_setup_done = True
        else:
            # only need to update eps and source:
            assert wl.item() == self.direct_solver.wl and dL.item() == self.direct_solver.dl, "wl or dL is not consistent"
            eps_batch, _ = self.global_DDM.partition(eps)
            source_batch, _ = self.global_DDM.partition(source)

        model_bs = len(self.global_DDM.regular_indices)
        d_sx, d_sy = self.global_DDM.region_size
        if gt is not None:
            gt = torch.from_numpy(gt)
            gt_batch, gt_pml = self.global_DDM.partition(gt)
        
        init_x_batch = None
        if init_x is not None:
            init_x_batch, init_x_pml = self.global_DDM.partition(init_x)
        
        self.subdomain_solver.setup((torch.view_as_real(source_batch), self.Sx_f_I_batch, self.Sy_f_I_batch, self.Sx_b_I_batch, self.Sy_b_I_batch, eps_batch, dL, wl), init_x=init_x_batch)

        ####### 2. init zero solution and boundary conditions #######
        x = torch.zeros(model_bs, d_sx, d_sy, dtype=torch.complex64, device=eps_batch.device)
        x_pml = [torch.zeros_like(source) for source in self.source_pml]
        top_bc_batch, bottom_bc_batch, left_bc_batch, right_bc_batch, top_bc_pml, bottom_bc_pml, left_bc_pml, right_bc_pml = self.global_DDM.get_bcs(x.cpu(), x_pml, eps_batch.cpu(), self.eps_pml, self.Sx_f_I_batch.cpu(), self.Sx_f_I_pml, self.Sy_f_I_batch.cpu(), self.Sy_f_I_pml, wl.cpu(), dL.cpu())
        top_bc_batch, bottom_bc_batch, left_bc_batch, right_bc_batch = top_bc_batch.to(eps_batch.device), bottom_bc_batch.to(eps_batch.device), left_bc_batch.to(eps_batch.device), right_bc_batch.to(eps_batch.device)
        
        t2 = time.time()
        # print(f"Time for setup: {t2 - t1:.2e} seconds")

        ####### 3. DDM iteration with momentum update #######
        global_x_history = []
        global_r_history = []
        zero_rhs_pml = [torch.zeros_like(source) for source in self.source_pml]

        pbar = tqdm(range(self.DDM_iters), desc="DDM iteration", leave=False)
        for i in pbar:
            t3 = time.time()
            rhs_batch, solution_batch, relres_history, x_history, r_history = self.subdomain_solver.solve((torch.view_as_real(top_bc_batch), torch.view_as_real(bottom_bc_batch), torch.view_as_real(left_bc_batch), torch.view_as_real(right_bc_batch)))
            t4 = time.time()
            solution_pml = self.direct_solver.solve_all(self.source_pml, top_bc_pml, bottom_bc_pml, left_bc_pml, right_bc_pml)
            t5 = time.time()

            new_top_bc_batch, new_bottom_bc_batch, new_left_bc_batch, new_right_bc_batch, new_top_bc_pml, new_bottom_bc_pml, new_left_bc_pml, new_right_bc_pml = self.global_DDM.get_bcs(solution_batch.cpu(), solution_pml, eps_batch.cpu(), self.eps_pml, self.Sx_f_I_batch.cpu(), self.Sx_f_I_pml, self.Sy_f_I_batch.cpu(), self.Sy_f_I_pml, wl.cpu(), dL.cpu())
            t6 = time.time()
            new_top_bc_batch, new_bottom_bc_batch, new_left_bc_batch, new_right_bc_batch = new_top_bc_batch.to(eps_batch.device), new_bottom_bc_batch.to(eps_batch.device), new_left_bc_batch.to(eps_batch.device), new_right_bc_batch.to(eps_batch.device)

            bc_percent_change = 1/4* (MAE(new_top_bc_batch, top_bc_batch) + MAE(new_bottom_bc_batch, bottom_bc_batch) + MAE(new_left_bc_batch, left_bc_batch) + MAE(new_right_bc_batch, right_bc_batch))
            pbar.set_postfix(bc_percent_change=f"{bc_percent_change:.2e}")
            if self.bc_change_th is not None and bc_percent_change < self.bc_change_th:
                break

            top_bc_batch = self.momentum*top_bc_batch + (1-self.momentum)*new_top_bc_batch
            bottom_bc_batch = self.momentum*bottom_bc_batch + (1-self.momentum)*new_bottom_bc_batch
            left_bc_batch = self.momentum*left_bc_batch + (1-self.momentum)*new_left_bc_batch
            right_bc_batch = self.momentum*right_bc_batch + (1-self.momentum)*new_right_bc_batch

            for j in range(len(top_bc_pml)):
                top_bc_pml[j] = self.momentum*top_bc_pml[j] + (1-self.momentum)*new_top_bc_pml[j]
                bottom_bc_pml[j] = self.momentum*bottom_bc_pml[j] + (1-self.momentum)*new_bottom_bc_pml[j]
                left_bc_pml[j] = self.momentum*left_bc_pml[j] + (1-self.momentum)*new_left_bc_pml[j]
                right_bc_pml[j] = self.momentum*right_bc_pml[j] + (1-self.momentum)*new_right_bc_pml[j]
            t7 = time.time()

            if self.save_intermediate and i in self.plot_iters:
                global_x_history.append(self.global_DDM.combine(solution_batch.cpu(), solution_pml))
                global_r_history.append(self.global_DDM.combine(rhs_batch.cpu(), zero_rhs_pml))
            # print(f"step {i}: subdomain NN solve: {t4 - t3:.2e} seconds, PML direct solve: {t5 - t4:.2e} seconds, compute bc: {t6 - t5:.2e} seconds, boundary update: {t7 - t6:.2e} seconds")
        
        end_time = time.time()

        if self.save_intermediate:
            colored_setup = setup_plot_data(eps.cpu(), source.cpu(), sx_f=self.Sx_2D_f_I.cpu(), sy_f=self.Sy_2D_f_I.cpu())
            plot_data = [colored_setup,source.imag,(self.Sx_2D_f_I+self.Sy_2D_f_I).cpu(),torch.abs(global_r_history[0])]
            plot_title = ['eps', 'source', 'pml', 'residue_input']
            cmaps = [None, 'seismic', None, 'turbo']
            center_zero = [False, False, False, False]
            row, column = 1, 4
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_residue_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

            plot_data = [*[x.real for x in global_x_history], *[torch.abs(r) for r in global_r_history]]
            plot_title = [*[f"output_{j}" for j in self.plot_iters], *[f"residual_{j}" for j in self.plot_iters]]
            cmaps = [*['seismic' for _ in range(self.plot_N)], *['turbo' for _ in range(self.plot_N)]]
            center_zero = [*[True for _ in range(self.plot_N)], *[False for _ in range(self.plot_N)]]
            row, column = 2, self.plot_N
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_debug_recurrent_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

        return self.global_DDM.combine(solution_batch.cpu(), solution_pml).to(eps.device), end_time
    
    @torch.no_grad()
    def Schwarz_GMRES_solve(self, eps, source, wl, dL, npml, gt=None):
        t1 = time.time()
        ####### 1. partition the global data into subdomains #######
        wl = torch.tensor([wl]).to(self.device)
        dL = torch.tensor([dL]).to(self.device)
        omega = 2 * np.pi * C_0 / wl

        Sx_2D_f, Sy_2D_f = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='f')
        Sx_2D_b, Sy_2D_b = make_Sx_Sy(omega, dL, eps.shape[0], npml[0], eps.shape[1], 0 if self.half_periodic else npml[1], _dir='b')

        Sx_2D_f_I = torch.from_numpy(Sx_2D_f.imag).to(torch.float32).to(self.device)
        Sy_2D_f_I = torch.from_numpy(Sy_2D_f.imag).to(torch.float32).to(self.device)
        Sx_2D_b_I = torch.from_numpy(Sx_2D_b.imag).to(torch.float32).to(self.device)
        Sy_2D_b_I = torch.from_numpy(Sy_2D_b.imag).to(torch.float32).to(self.device)

        self.global_DDM.init_grid()
        model_bs = np.prod(self.global_DDM.grid_shape)
        d_sx, d_sy = self.global_DDM.region_size
        sim_size_in_wl = eps.shape[0]*dL/wl*torch.mean(eps)**.5
        max_size_in_wl_subdomain = d_sx*dL/wl*torch.max(eps)**.5
        # printc(f"sim_size_in_wl: {sim_size_in_wl.item():.2e}, max_size_in_wl_subdomain: {max_size_in_wl_subdomain.item():.2e}", color="g")
        # printc(f"DDM grid shape: {self.global_DDM.grid_shape}, with overlap: x: {self.global_DDM.x_overlaps}, y: {self.global_DDM.y_overlaps}", color="b")

        if not self.global_DDM.periodic_padding:
            # need to set global bc:
            eps = eps[self.spacing:-self.spacing, self.spacing:-self.spacing]
            source = source[self.spacing:-self.spacing, self.spacing:-self.spacing]
            Sx_2D_f_I = Sx_2D_f_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
            Sy_2D_f_I = Sy_2D_f_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
            Sx_2D_b_I = Sx_2D_b_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
            Sy_2D_b_I = Sy_2D_b_I[self.spacing:-self.spacing, self.spacing:-self.spacing]
            gt = torch.from_numpy(gt[self.spacing:-self.spacing, self.spacing:-self.spacing]).to(self.device)
            gt_batch = torch.stack(self.global_DDM.partition(gt), dim=0)

        eps_batch = torch.stack(self.global_DDM.partition(eps), dim=0)
        source_batch = torch.stack(self.global_DDM.partition(source), dim=0)
        Sx_f_I_batch = torch.stack(self.global_DDM.partition(Sx_2D_f_I), dim=0)
        Sy_f_I_batch = torch.stack(self.global_DDM.partition(Sy_2D_f_I), dim=0)
        Sx_b_I_batch = torch.stack(self.global_DDM.partition(Sx_2D_b_I), dim=0)
        Sy_b_I_batch = torch.stack(self.global_DDM.partition(Sy_2D_b_I), dim=0)

        if not self.global_DDM.periodic_padding:
            self.global_DDM.set_global_bc(gt_batch, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)

        ####### 2. prepare global GMRES and initial bc #######
        def combine_bcs(top_bc, bottom_bc, left_bc, right_bc):
            return torch.cat((top_bc.squeeze(), bottom_bc.squeeze(), left_bc.squeeze(), right_bc.squeeze()), dim=1).reshape(-1)[None, None, :] # shape (1, 1, bs*4*d_sx)

        def split_bcs(bc, model_bs=model_bs, d_sx=d_sx):
            bc = bc.reshape(model_bs, 4, d_sx)
            return bc[:, 0, :].unsqueeze(1), bc[:, 1, :].unsqueeze(1), bc[:, 2, :].unsqueeze(2), bc[:, 3, :].unsqueeze(2)

        first_solve_iters = 40
        rest_solve_iters = self.subdomain_solver.max_iter
        self.subdomain_solver.max_iter = first_solve_iters
        self.subdomain_solver.init()
        self.subdomain_solver.setup((torch.view_as_real(source_batch), Sx_f_I_batch, Sy_f_I_batch, Sx_b_I_batch, Sy_b_I_batch, eps_batch, dL, wl))
        x = torch.zeros(model_bs, d_sx, d_sy, dtype=torch.complex64, device=self.device)
        top_bc, bottom_bc, left_bc, right_bc = self.global_DDM.get_bcs(x, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)
        init_bc = combine_bcs(top_bc, bottom_bc, left_bc, right_bc)
        rhs, init_x, _, _, _ = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
        init_r = torch.zeros_like(init_x)

        gt_solution = gt_batch - init_x
        gt_bcs = combine_bcs(*self.global_DDM.get_current_bcs(gt_solution, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL, zero_global_bcs=True))
        # new_top_bc, new_bottom_bc, new_left_bc, new_right_bc = self.global_DDM.get_bcs(solution, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL)
        # plt.imshow(self.global_DDM.combine(init_x).cpu().numpy().real, cmap='seismic')
        # plt.colorbar()
        # plt.savefig("init_x_debug.png")
        # plt.close()
        # after first solve with source, later solves with zero source
        self.subdomain_solver.max_iter = rest_solve_iters
        self.subdomain_solver.init()
        self.subdomain_solver.setup((torch.view_as_real(torch.zeros_like(source_batch)), Sx_f_I_batch, Sy_f_I_batch, Sx_b_I_batch, Sy_b_I_batch, eps_batch, dL, wl))

        global_gmres = GMRES(None, max_iter=self.DDM_iters, tol=1e-3)
        global_gmres.M = lambda rhs, x: rhs # identity

        # global_gmres.x = torch.zeros_like(init_bc)
        global_gmres.x = gt_bcs
        def Aop(bcs):
            top_bc, bottom_bc, left_bc, right_bc = split_bcs(bcs)
            rhs, solution, _, _, _ = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
            return combine_bcs(*self.global_DDM.get_bc_errors(solution, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL, zero_global_bcs=False))
        global_gmres.setup_Aop(Aop)

        t2 = time.time()
        print(f"Time for setup: {t2 - t1:.2e} seconds")

        ####### 3. GMRES accelerated Schwarz iteration #######
        print("global GMRES solving")
        b = -combine_bcs(*self.global_DDM.get_bc_errors(init_x, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL, zero_global_bcs=True))

        # debug plots:
        # debug_plot_b(b, name='debug_b')
        # debug_plot_b(gt_bcs, name='debug_gt_bcs')
        # gt_zero_bcs = combine_bcs(*self.global_DDM.get_bc_errors(gt_batch, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL))
        # debug_plot_b(gt_zero_bcs, name='debug_gt_zero_bcs')
        # direct_gt_bcs = combine_bcs(*self.global_DDM.get_bc_errors(gt_solution, eps_batch, Sx_f_I_batch, Sy_f_I_batch, wl, dL))
        # debug_plot_b(direct_gt_bcs, name='debug_direct_gt_bcs')
        
        x, r, relres_history, x_history, r_history = global_gmres.solve(b, return_xr_history=self.save_intermediate, verbose=True)

        # debug plots:
        debug_plot_b(b, name='debug_b')
        Aop_gt_bcs = Aop(gt_bcs)
        debug_plot_b(Aop_gt_bcs, name='debug_Aop_gt_bcs')
        Aop_final_x = Aop(x)
        debug_plot_b(Aop_final_x, name='debug_Aop_final_x')
        print(f"difference Aop_gt_bcs - b: {torch.mean(torch.abs(Aop_gt_bcs - b))}")
        print(f"difference Aop_final_x - b: {torch.mean(torch.abs(Aop_final_x - b))}")
        # top_bc, bottom_bc, left_bc, right_bc = split_bcs(x)
        # rhs, debug_solution, _, _, _ = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
        # debug_combined = self.global_DDM.combine(debug_solution+init_x)
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1,2,1)
        # plt.imshow(debug_combined.cpu().numpy().real, cmap='seismic')
        # plt.colorbar()
        # plt.title('debug_solution')
        # plt.subplot(1,2,2)
        # plt.imshow(self.global_DDM.combine(gt_batch).cpu().numpy().real, cmap='seismic')
        # plt.colorbar()
        # plt.title('gt_solution')
        # plt.savefig("debug_solution.png")
        # plt.close()

        ####### 4. fill in the solution to the subdomains #######
        print("assembling solution")

        # reset the subdomain solver to have the correct source
        # self.subdomain_solver.setup((torch.view_as_real(source_batch), Sx_f_I_batch, Sy_f_I_batch, Sx_b_I_batch, Sy_b_I_batch, eps_batch, dL, wl))

        global_x_history = []
        global_r_history = []
        for i in range(self.DDM_iters):
            if self.save_intermediate and i in self.plot_iters:
                top_bc, bottom_bc, left_bc, right_bc = split_bcs(x_history[i])
                rhs, solution, _, _, _ = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
                combined = self.global_DDM.combine(solution + init_x)
                global_x_history.append(combined)

                # top_bc, bottom_bc, left_bc, right_bc = split_bcs(r_history[i])
                # rhs, solution_r, _, _, _ = self.subdomain_solver.solve((torch.view_as_real(top_bc), torch.view_as_real(bottom_bc), torch.view_as_real(left_bc), torch.view_as_real(right_bc)))
                # combined_r = self.global_DDM.combine(solution_r + init_r)
                # global_r_history.append(combined_r)
                global_r_history.append(self.global_DDM.combine(init_r))
        
        end_time = time.time()

        if self.save_intermediate:
            colored_setup = setup_plot_data(eps.cpu(), source.cpu(), sx_f=Sx_2D_f_I.cpu(), sy_f=Sy_2D_f_I.cpu())
            plot_data = [colored_setup,source.real,(Sx_2D_f_I+Sy_2D_f_I).cpu(),torch.abs(global_r_history[0])]
            plot_title = ['eps', 'source', 'pml', 'residue_input']
            cmaps = [None, 'seismic', None, 'turbo']
            center_zero = [False, False, False, False]
            row, column = 1, 4
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_residue_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

            plot_data = [*[x.real for x in global_x_history], *[torch.abs(r) for r in global_r_history]]
            plot_title = [*[f"output_{j}" for j in self.plot_iters], *[f"residual_{j}" for j in self.plot_iters]]
            cmaps = [*['seismic' for _ in range(self.plot_N)], *['turbo' for _ in range(self.plot_N)]]
            center_zero = [*[True for _ in range(self.plot_N)], *[False for _ in range(self.plot_N)]]
            row, column = 2, self.plot_N
            plot_helper(plot_data,row,column,plot_title,os.path.join(self.output_dir, f"sample_{i}_debug_recurrent_wl{wl[0].item():.2e}_dL{dL[0].item():.2e}.png"), cmaps=cmaps, center_zero=center_zero)

        return combined, end_time
    
    @torch.no_grad()
    def FETI_solve(self, eps, source, wl, dL, pml_thickness=40, gt=None):
        raise NotImplementedError("FETI solver not implemented")