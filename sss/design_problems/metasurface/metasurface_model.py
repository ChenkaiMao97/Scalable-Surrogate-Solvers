# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

from ceviche import fdfd_ez
from ceviche.constants import C_0, EPSILON_0, MU_0
from ceviche_challenges import primitives

from typing import Tuple, Sequence

from sss.invde.utils.torch_functions import make_torch_epsilon_r
from sss.utils.PDE_utils import Ez_to_Hx, Ez_to_Hy, maxwell_robin_residue
from sss.utils.GPU_worker_utils import solver_worker

import numpy as np
import autograd.numpy as anp
import torch
import multiprocessing as mp
import threading
import concurrent.futures

from functools import cached_property

import matplotlib.pyplot as plt

import gin

nm = 1e-9

def is_multiple(a, b, tol=1e-9):
    if b == 0:
        return False  # avoid division by zero
    quotient = a / b
    return abs(round(quotient) - quotient) < tol

def source_scale(wl, dL):
  return 1j*2*np.pi*C_0*dL**2/wl*EPSILON_0

@gin.configurable
class MetasurfaceModel:
    """
    metasurface physical layout:
     -----> y direction
    |
    |
    V
    x direction
                      focal_shift
               <----------
                          ----->
    ###############################################
    |                                             |
    |                         ⟋\                 | ---
    |         /⟍           ⟋    \                |  ^
    |        /     ⟍    ⟋         \              |  |
    |       /        ⟋ ⟍            \            |  |
    |      /      ⟋         ⟍         \          |  | max_focal_length
    |     /    ⟋                 ⟍      \        |  |
    |    /  ⟋                         ⟍   \      |  |
    |   /⟋                                 ⟍\    |  v
    |   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   | --
    |   xxxxxxxxxxxx(design region)xxxxxxxxxxxx   | || thickness
    |   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   | --
    |                                             | || source_spacing_nm
    |   =======================================   | --> (source)
    |                                             |
    ###############################################

        |<-----------  width_all  ----------->|

    """
    def __init__(
        self,
        grid_shape,
        design_variable_shape_nm: Tuple[int, int],
        focus_positions_nm: Sequence[Tuple[float, float]],
        wavelengths_nm: Sequence[float],
        source_spacing_nm: int,
        opt_flux_monitor_width_nm: int, # the monitor width for computing the optimization objective
        FOM_flux_monitor_width_nm: int, # the monitor width for computing the focus efficiency (larger than opt_flux_monitor_width_nm)
        dL_nm, # resolution
        eps_meta = 6.0,
        eps_bg = 1.0,
        pml_width=10, # in pixels
        params=None,
        _backend='ceviche',
    ): 
        assert is_multiple(design_variable_shape_nm[0], dL_nm), f"design_variable_shape_nm[0] should be divisible by dL_nm"
        assert is_multiple(design_variable_shape_nm[1], dL_nm), f"design_variable_shape_nm[1] should be divisible by dL_nm"
        assert is_multiple(source_spacing_nm, dL_nm), f"source_spacing_nm should be divisible by dL_nm"
        assert is_multiple(opt_flux_monitor_width_nm, dL_nm), f"flux_monitor_width_nm should be divisible by dL_nm"
        assert is_multiple(FOM_flux_monitor_width_nm, dL_nm), f"flux_monitor_width_nm should be divisible by dL_nm"
        max_focal_length = 0
        for focus_position_nm in focus_positions_nm:
            assert is_multiple(focus_position_nm[0], dL_nm), f"focus_position should be divisible by dL_nm"
            assert is_multiple(focus_position_nm[1], dL_nm), f"focus_position should be divisible by dL_nm"
            max_focal_length = max(max_focal_length, int(focus_position_nm[0]/dL_nm))

        self.grid_shape = grid_shape
        self.dL_nm = dL_nm
        self.dL = dL_nm * nm
        self.design_variable_shape = (int(design_variable_shape_nm[0]/dL_nm), int(design_variable_shape_nm[1]/dL_nm))
        self.focus_positions = [(int(f[0]/dL_nm), int(f[1]/dL_nm)) for f in focus_positions_nm]
        print("focus_positions: ", self.focus_positions)
        self.source_spacing = int(source_spacing_nm/dL_nm)
        self.max_focal_length = max_focal_length
        self.wavelengths_nm = wavelengths_nm
        assert len(self.wavelengths_nm) == len(focus_positions_nm), f"number of wavelengths and focus positions should be the same"
        self.opt_flux_monitor_width = int(opt_flux_monitor_width_nm/dL_nm)
        self.FOM_flux_monitor_width = int(FOM_flux_monitor_width_nm/dL_nm)

        self.pml_width = pml_width
        self.eps_meta = eps_meta
        self.eps_bg = eps_bg
        self.plane_wave_fluxes = {}
        self.source_amp = 1e6/self.dL**2
        
        self.thickness_all = self.source_spacing + self.design_variable_shape[0] + self.max_focal_length
        self.width_all = self.design_variable_shape[1]

        assert self.thickness_all + 2 * self.pml_width < self.grid_shape[0], f"vertical space not enough"
        assert self.width_all + 2 * self.pml_width < self.grid_shape[1], f"horizontal space not enough"

        self._backend = _backend
        self.last_forward_E = {}
        self.last_adjoint_E = {}
    
    def init_DDM_workers(self):
        # init solvers on each GPU
        gpu_ids = list(range(torch.cuda.device_count()))
        self.num_gpus = len(gpu_ids)
        self.task_queues = [mp.Queue() for _ in range(self.num_gpus)]
        self.result_queue = mp.Queue()
        self.init_queues = [mp.Queue() for _ in range(self.num_gpus)] # for passing back values after init
        
        self.processes = []

        init_kwargs = {
            'Nx': self.grid_shape[0],
            'Ny': self.grid_shape[1],
            'save_intermediate': False,
            'output_dir': None,
        }

        for device_id in gpu_ids:
            p = mp.Process(target=solver_worker, args=(device_id, init_kwargs, self.task_queues[device_id], self.result_queue, self.init_queues[device_id]))
            p.start()
            self.processes.append(p)
        
        init_outputs = [q.get() for q in self.init_queues]
        # check if all init outputs are the same
        assert len(set(init_outputs)) == 1
        self.source_mult = init_outputs[0]

        self.task_id_counter = 0
        self.results = {}

        self.results_lock = threading.Lock()
        self.results_cond = threading.Condition(self.results_lock)
        self.listener_thread = threading.Thread(target=self._result_listener, daemon=True)
        self.listener_thread.start()
    
    def _result_listener(self):
        while True:
            item = self.result_queue.get()
            if item is None:
                break
            task_id, result = item
            with self.results_cond:
                self.results[task_id] = result
                self.results_cond.notify_all()
    
    def stop_workers(self):
        if self._backend == 'DDM':
            for i in range(self.num_gpus):
                self.task_queues[i].put(None)
            for p in self.processes:
                p.join()

            self.result_queue.put(None)
            self.listener_thread.join()
        print("all process and threads stopped")
    
    @cached_property
    def density_bg(self):
        d = np.zeros(self.grid_shape, dtype=np.float32)
        return d
    
    def density(self, design_variable):
        return primitives.insert_design_variable(
            design_variable,
            self.density_bg,
            (self.design_region_x_start, self.design_region_y_start, self.design_region_x_end, self.design_region_y_end)
        )

    @cached_property
    def epsilon_bg(self):
        density_bg = self.density_bg
        epsilon_bg = density_bg * (self.eps_meta-self.eps_bg) + self.eps_bg
        return epsilon_bg

    def epsilon_r(self, design_variable):
        assert design_variable.shape == self.design_variable_shape, f"design_variable shape should be {self.design_variable_shape}"

        full_density = self.density(design_variable).astype(np.float32)
        full_eps = full_density * (self.eps_meta - self.eps_bg) + self.eps_bg
        return full_eps
    
    def flux_without_meta(self, wavelength_nm):
        flux_x = self.source_x_start - 2

        eps = np.ones(self.grid_shape)
        omega = 2 * np.pi * C_0 / (wavelength_nm * nm)
        k0 = 2 * np.pi / (wavelength_nm * nm)
        F = fdfd_ez(omega, self.dL, eps, (self.pml_width, self.pml_width))

        plane_wave_source = np.zeros(self.grid_shape, dtype=np.complex64)
        plane_wave_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
        plane_wave_source[self.source_x_start+1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)

        hx_forward, hy_forward, ez_forward = F.solve(plane_wave_source)
        flux = np.sum(np.real(np.conj(ez_forward[flux_x, self.source_y_start:self.source_y_end]) * hy_forward[flux_x, self.source_y_start:self.source_y_end]))

        return flux
    
    def get_flux(self, wavelength_nm):
        if wavelength_nm in self.plane_wave_fluxes:
            return self.plane_wave_fluxes[wavelength_nm]
        else:
            flux = self.flux_without_meta(wavelength_nm)
            print(f"plane wave flux for {wavelength_nm} nm: {flux:.2e}")
            self.plane_wave_fluxes[wavelength_nm] = flux
            return flux

    def simulate(
            self,
            design_variable,
            wavelengths_nm
    ):
        for wavelength_nm in wavelengths_nm:
            assert wavelength_nm in self.wavelengths_nm, f"wavelength {wavelength_nm} not in {self.wavelengths_nm}"

        wavelengths_nm = np.asarray(wavelengths_nm)
        flat_wavelengths = list(wavelengths_nm.ravel(order='C'))
        opt_fluxes = [None] * len(wavelengths_nm)
        FOM_fluxes = [None] * len(wavelengths_nm)
        ezs = [None] * len(wavelengths_nm)

        def _simulate(wavelength_nm):
            omega = 2 * np.pi * C_0 / (wavelength_nm * nm)
            k0 = 2 * np.pi * self.eps_bg**.5 / (wavelength_nm * nm)

            epsilon_r = self.epsilon_r(design_variable)

            # forward plane wave source
            plane_wave_source = np.zeros(self.grid_shape, dtype=complex)
            plane_wave_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
            plane_wave_source[self.source_x_start+1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)

            if self._backend == 'ceviche':
                # Set up the FDFD simulation for TE
                F = fdfd_ez(omega, self.dL, epsilon_r, (self.pml_width, self.pml_width))
                hx, hy, ez = F.solve(plane_wave_source)

                # calculate the focusing flux:
                focus_idx = self.wavelengths_nm.index(wavelength_nm)
                window_x = self.design_region_x_start - self.focus_positions[focus_idx][0]
                opt_window_y_start, opt_window_y_end = self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] - self.opt_flux_monitor_width//2, self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] + self.opt_flux_monitor_width//2
                opt_flux = anp.sum(anp.real(anp.conj(ez[window_x, opt_window_y_start:opt_window_y_end]) * hy[window_x, opt_window_y_start:opt_window_y_end]))
                assert opt_flux > 0, f"flux is negative: {opt_flux}"
                opt_trans_eff = opt_flux/ self.get_flux(wavelength_nm)

                FOM_window_y_start, FOM_window_y_end = self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] - self.FOM_flux_monitor_width//2, self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] + self.FOM_flux_monitor_width//2
                FOM_flux = anp.sum(anp.real(anp.conj(ez[window_x, FOM_window_y_start:FOM_window_y_end]) * hy[window_x, FOM_window_y_start:FOM_window_y_end]))
                assert FOM_flux > 0, f"flux is negative: {FOM_flux}"
                FOM_trans_eff = FOM_flux/ self.get_flux(wavelength_nm)

            elif self._backend == 'DDM':
                omega_torch = torch.tensor([omega], dtype=torch.float32)
                wl_torch = (2 * np.pi * C_0) / omega_torch
                dl_torch = torch.tensor([self.dL], dtype=torch.float32)
                epsilon_r_torch = torch.from_numpy(epsilon_r).to(torch.float32)
                source_torch = torch.from_numpy(plane_wave_source).to(torch.complex64)

                # send task to GPU worker queue
                task_id = self.task_id_counter
                self.task_id_counter += 1
                # device_id = task_id % self.num_gpus
                device_id = flat_wavelengths.index(wavelength_nm) # each device handles one wavelength, to reuse the precomputed PML 
                last_E = self.last_forward_E.get(wavelength_nm, None)
                self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.pml_width, None, last_E)))

                # wait and fetch the result
                with self.results_cond:
                    while task_id not in self.results:
                        self.results_cond.wait()
                    solution = self.results.pop(task_id)

                # self.last_forward_E[wavelength_nm] = solution

                ez = solution[None]
                # actually hy in ceviche convention
                hx = Ez_to_Hx(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)
                # actually hx in ceviche convention
                hy = Ez_to_Hy(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)

                focus_idx = self.wavelengths_nm.index(wavelength_nm)
                window_x = self.design_region_x_start - self.focus_positions[focus_idx][0]
                opt_window_y_start, opt_window_y_end = self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] - self.opt_flux_monitor_width//2, self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] + self.opt_flux_monitor_width//2
                opt_flux = torch.sum(torch.real(torch.conj(ez[0,window_x, opt_window_y_start:opt_window_y_end]) * hx[0,window_x, opt_window_y_start:opt_window_y_end]))
                assert opt_flux > 0, f"flux is negative: {opt_flux}"
                opt_trans_eff = opt_flux / self.get_flux(wavelength_nm)

                FOM_window_y_start, FOM_window_y_end = self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] - self.FOM_flux_monitor_width//2, self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] + self.FOM_flux_monitor_width//2
                FOM_flux = torch.sum(torch.real(torch.conj(ez[0,window_x, FOM_window_y_start:FOM_window_y_end]) * hx[0,window_x, FOM_window_y_start:FOM_window_y_end]))
                assert FOM_flux > 0, f"flux is negative: {FOM_flux}"
                FOM_trans_eff = FOM_flux / self.get_flux(wavelength_nm)

                ez = ez[0]
            else:
                raise ValueError(f"backend {self.backend} not supported")
            
            return wavelength_nm, opt_trans_eff, FOM_trans_eff, ez

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wavelengths_nm)) as executor:
            simute_results = list(executor.map(_simulate, wavelengths_nm))
        
        for wavelength_nm, opt_trans_eff, FOM_trans_eff, ez in simute_results:
            wl_idx = flat_wavelengths.index(wavelength_nm)
            ezs[wl_idx] = ez
            opt_fluxes[wl_idx] = opt_trans_eff
            FOM_fluxes[wl_idx] = FOM_trans_eff
        
        opt_fluxes = anp.stack(opt_fluxes, axis=0)
        FOM_fluxes = anp.stack(FOM_fluxes, axis=0)
        ezs = anp.stack(ezs, axis=0)

        return opt_fluxes, FOM_fluxes, ezs

    def simulate_adjoint(
        self,
        design_variable: np.ndarray,
        wavelengths_nm: np.ndarray,
        forward_output_torch: Tuple[torch.Tensor],
        grad_output_torch: Tuple[torch.Tensor],
    ):
        wavelengths_nm = np.asarray(wavelengths_nm)
        flat_wavelengths = list(wavelengths_nm.ravel(order='C'))
        epsilon_r = self.epsilon_r(design_variable)
        epsilon_r_torch = torch.from_numpy(epsilon_r).to(torch.float32)
       
        def _adjoint_simulate(forward_ez, wl_nm, grad_output):
            omega = 2 * np.pi * C_0 / (wl_nm * nm)
            omega_torch = torch.tensor([omega], dtype=torch.float32)
            dl_torch = torch.tensor([self.dL], dtype=torch.float32)

            def comp_graph(ez):
                ez = ez[None] # add batch dimension for H fields computation
                # actually hy in ceviche convention
                hx = Ez_to_Hx(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)
                # actually hx in ceviche convention
                hy = Ez_to_Hy(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)

                focus_idx = self.wavelengths_nm.index(wl_nm)
                window_x = self.design_region_x_start - self.focus_positions[focus_idx][0]
                opt_window_y_start, opt_window_y_end = self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] - self.opt_flux_monitor_width//2, self.grid_shape[1]//2 + self.focus_positions[focus_idx][1] + self.opt_flux_monitor_width//2
                opt_flux = torch.sum(torch.real(torch.conj(ez[0,window_x, opt_window_y_start:opt_window_y_end]) * hx[0,window_x, opt_window_y_start:opt_window_y_end]))
                assert opt_flux > 0, f"flux is negative: {opt_flux}"
                opt_trans_eff = opt_flux / self.get_flux(wl_nm)

                return opt_trans_eff

            forward_ez = forward_ez.detach().requires_grad_(True)
            opt_trans_eff = comp_graph(forward_ez)
            grad_ez = torch.autograd.grad(opt_trans_eff, forward_ez, grad_outputs=torch.conj(grad_output))[0]

            # compute adjoint simulations on gpu
            wl_torch = (2 * np.pi * C_0) / omega_torch
            source_torch = torch.conj(grad_ez).to(torch.complex64).resolve_conj()  # adjoint source

            # send task to GPU worker queue
            task_id = self.task_id_counter
            self.task_id_counter += 1
            # device_id = task_id % self.num_gpus
            device_id = flat_wavelengths.index(wl_nm) # each device handles one omega, to reuse the precomputed PML 
            last_E = self.last_adjoint_E.get(wl_nm, None)
            self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.pml_width, None, last_E)))
            
            # wait and fetch the result
            with self.results_cond:
                while task_id not in self.results:
                    self.results_cond.wait()
                adjoint_solution = self.results.pop(task_id)
            
            # self.last_adjoint_E[wl_nm] = adjoint_solution

            # F(x, y) = b - A(x)* y
            # compute ∂F/∂x
            design_variable_torch = design_variable.clone().detach().requires_grad_(True)
            epsilon_for_residual = make_torch_epsilon_r(design_variable_torch, self.density_bg, self.eps_bg, self.eps_meta, (self.design_region_x_start, self.design_region_y_start, self.design_region_x_end, self.design_region_y_end))
            sx, sy = adjoint_solution.shape # bs == 1
            bs = 1 # batch dimension for residual computation
            top_bc, bottom_bc, left_bc, right_bc = torch.zeros(bs, 1, sy, 2), torch.zeros(bs, 1, sy, 2), torch.zeros(bs, sx, 1, 2), torch.zeros(bs, sx, 1, 2)
            Sx_f, Sx_b = torch.zeros(bs, sx, sy), torch.zeros(bs, sx, sy)
            Sy_f, Sy_b = torch.zeros(bs, sy, sx), torch.zeros(bs, sy, sx)
            forward_ez = forward_ez.detach()

            # forward plane wave source
            k0 = 2 * np.pi * self.eps_bg**.5 / (wl_nm * nm)
            forward_source = np.zeros(self.grid_shape, dtype=complex)
            forward_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
            forward_source[self.source_x_start+1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)
            forward_source_torch = torch.from_numpy(forward_source).to(torch.complex64)

            # rescale_factor = self.source_mult / source_scale(wl_torch, dl_torch)
            mult_in_res_fn = (dl_torch/wl_torch)**0.5 * MU_0/EPSILON_0*(wl_torch/(2*np.pi*dl_torch))**2
            rescale_factor = 1 / (mult_in_res_fn*source_scale(wl_torch, dl_torch))

            residual = rescale_factor * torch.view_as_complex(maxwell_robin_residue(
                                                torch.view_as_real(forward_ez[None]), 
                                                epsilon_for_residual[None], 
                                                top_bc, bottom_bc, left_bc, right_bc, 
                                                torch.view_as_real(source_scale(wl_torch, dl_torch) * forward_source_torch[None]), 
                                                (Sx_f, Sx_b), (Sy_f, Sy_b), 
                                                dl_torch, wl_torch, bc_mult=1)) # bc not in grad calculation so set to 1

            # for debugging purpose:
            # residual_adjoint = rescale_factor * torch.view_as_complex(maxwell_robin_residue(
            #                                   torch.view_as_real(adjoint_solution[None]), 
            #                                   epsilon_for_residual[None], 
            #                                   top_bc, bottom_bc, left_bc, right_bc, 
            #                                   torch.view_as_real(source_scale(wl_torch, dl_torch) * source_torch[None]), 
            #                                   (Sx_f, Sx_b), (Sy_f, Sy_b), 
            #                                   dl_torch, wl_torch, bc_mult=1))

            input_grad = torch.autograd.grad(residual[0], design_variable_torch, grad_outputs=torch.conj(adjoint_solution))[0]
            return input_grad
        
        input_grads = []
        forward_ezs = forward_output_torch[2] # shape (num_wavelengths, num_excite_ports, height, width)
        grad_output_opt_trans_eff = grad_output_torch[0] # shape (num_wavelengths, num_excite_ports, num_ports)
        
        # map jobs to all available GPUs
        def worker(wl_nm):
            wl_idx = flat_wavelengths.index(wl_nm)
            return _adjoint_simulate(
                forward_ezs[wl_idx],
                wl_nm,
                grad_output_opt_trans_eff[wl_idx],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            input_grads = list(executor.map(worker, flat_wavelengths))
        
        return sum(input_grads), None

    def get_sources(self):
        return np.array([1+1j])

    @cached_property
    def design_region_x_start(self):
        return int(self.grid_shape[0]/2 + self.thickness_all/2) - self.source_spacing - self.design_variable_shape[0]
    @cached_property
    def design_region_x_end(self):
        return self.design_region_x_start + self.design_variable_shape[0]
    @cached_property
    def design_region_y_start(self):
        return int(self.grid_shape[1]/2 - self.width_all/2)
    @cached_property
    def design_region_y_end(self):
        return self.design_region_y_start + self.design_variable_shape[1]
    @cached_property
    def source_x_start(self):
        return self.design_region_x_end + self.source_spacing
    @cached_property
    def source_y_start(self):
        return self.design_region_y_start
    @cached_property
    def source_y_end(self):
        return self.design_region_y_end
    @cached_property
    def monitor_x(self):
        return int((self.wg_x_start + self.wg_x_end) / 2)
    @cached_property
    def monitor_y(self):
        return int(self.grid_shape[1]/2 + self.width_all/2)
    
    @cached_property
    def port(self):
        direction = defs.Direction.Y_NEG
        return modes.WaveguidePort(
                    x=self.monitor_x,
                    y=self.monitor_y,
                    width=self.monitor_size,
                    order=1,
                    dir=direction,
                    offset=5
                )
        
