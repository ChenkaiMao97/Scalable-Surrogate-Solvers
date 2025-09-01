# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

from ceviche import fdfd_ez
from ceviche.constants import C_0, EPSILON_0, MU_0
from ceviche_challenges import modes, ops, defs, primitives
from ceviche_challenges.scattering import calculate_amplitudes as calculate_amplitudes_cc

from typing import Tuple

from sss.invde.utils.torch_functions import calculate_amplitudes as calculate_amplitudes_torch
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

mm = 1e-3

def is_multiple(a, b, tol=1e-9):
    if b == 0:
        return False  # avoid division by zero
    quotient = a / b
    return abs(round(quotient) - quotient) < tol

def source_scale(wl, dL):
  return 1j*2*np.pi*C_0*dL**2/wl*EPSILON_0

@gin.configurable
class GratingCouplerModel:
    """
    grating coupler physical layout:
     -----> y direction
    |
    |
    V
    x direction
    ###############################################
    |                                             |
    |                                             |
    |   ===================   (source)            |-----------   -----
    |                                             |src_spacing     ^
    |   xxxxxxxxxxxxxxxxxxx                       |-----------     |
    |   xx(design region)xx           (monitor)   |           thickness_all
    |   xxxxxxxxxxxxxxxxxxx               |       |                |
    |   0000000000000000000000000000000000|0000000|                V
    |   0000000000(wave guide)000000000000|0000000|              -----
    |                                     |       |
    |                                             |
    ###############################################

                          |<-monitor_dis->|
        |<---------   width_all  -------->|

    """
    def __init__(
        self,
        grid_shape,
        design_variable_shape_mm: Tuple[int, int],
        source_spacing_mm: int,
        wg_thickness_mm: int,
        monitor_distance_mm: int,
        monitor_size_mm: int,
        dL_mm, # resolution
        eps_wg = 6.0,
        eps_bg = 1.0,
        eps_design_max = None,
        eps_design_min = None,
        pml_width=10, # in pixels
        params=None,
        _backend='ceviche',
    ):
        mod_mult = 10 # 
        assert is_multiple(design_variable_shape_mm[0], dL_mm), f"design_variable_shape_mm[0] should be divisible by dL_mm"
        assert is_multiple(design_variable_shape_mm[1], dL_mm), f"design_variable_shape_mm[1] should be divisible by dL_mm"
        assert is_multiple(source_spacing_mm, dL_mm), f"source_spacing_mm should be divisible by dL_mm"
        assert is_multiple(wg_thickness_mm, dL_mm), f"wg_thickness_mm should be divisible by dL_mm"
        assert is_multiple(monitor_distance_mm, dL_mm), f"monitor_distance_mm should be divisible by dL_mm"
        assert is_multiple(monitor_size_mm, dL_mm), f"monitor_size_mm should be divisible by dL_mm"
        self.grid_shape = grid_shape
        self.dL_mm = dL_mm
        self.dL = dL_mm * mm
        self.design_variable_shape = (int(design_variable_shape_mm[0]/dL_mm), int(design_variable_shape_mm[1]/dL_mm))
        self.source_spacing = int(source_spacing_mm/dL_mm)
        self.wg_thickness = int(wg_thickness_mm/dL_mm)
        self.monitor_distance = int(monitor_distance_mm/dL_mm)
        self.monitor_size = int(monitor_size_mm/dL_mm)
        
        self.pml_width = pml_width
        self.eps_wg = eps_wg
        self.eps_bg = eps_bg
        self.eps_design_max = eps_wg if eps_design_max is None else eps_design_max
        self.eps_design_min = eps_bg if eps_design_min is None else eps_design_min
        self.plane_wave_amplitudes = {}
        self.mode_normalizations = {}
        self.source_amp = 1e6/self.dL**2
        
        self.thickness_all = self.source_spacing + self.wg_thickness + self.design_variable_shape[0]
        self.width_all = self.design_variable_shape[1] + self.monitor_distance

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
        d[self.wg_x_start:self.wg_x_end, self.wg_y_start:self.wg_y_end] = 1
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
        epsilon_bg = density_bg * (self.eps_wg-self.eps_bg) + self.eps_bg
        return epsilon_bg

    def epsilon_r(self, design_variable):
        # design_variable is from 0 to 1, we want to transform it to be from a to b,
        # where a * (self.eps_wg - self.eps_bg) + self.eps_bg = eps_design_min
        # and   b * (self.eps_wg - self.eps_bg) + self.eps_bg = eps_design_max
        assert design_variable.shape == self.design_variable_shape, f"design_variable shape should be {self.design_variable_shape}"

        a = (self.eps_design_min - self.eps_bg) / (self.eps_wg - self.eps_bg)
        b = (self.eps_design_max - self.eps_bg) / (self.eps_wg - self.eps_bg)
        scaled_design_variable = design_variable * (b-a) + a

        full_density = self.density(scaled_design_variable).astype(np.float32)
        full_eps = full_density * (self.eps_wg - self.eps_bg) + self.eps_bg
        return full_eps
    
    def flux_without_wg(self, wavelength_mm):
        flux_x = self.design_region_x_start - 2

        eps = np.ones(self.grid_shape)
        omega = 2 * np.pi * C_0 / (wavelength_mm * mm)
        k0 = 2 * np.pi / (wavelength_mm * mm)
        F = fdfd_ez(omega, self.dL, eps, (self.pml_width, self.pml_width))

        plane_wave_source = np.zeros(self.grid_shape, dtype=np.complex64)
        plane_wave_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
        plane_wave_source[self.source_x_start-1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)

        hx_forward, hy_forward, ez_forward = F.solve(plane_wave_source)

        flux = -np.sum(np.real(np.conj(ez_forward[flux_x, self.source_y_start:self.source_y_end]) * hy_forward[flux_x, self.source_y_start:self.source_y_end]))

        return flux
    
    def get_plane_wave_amplitude(self, wavelength_mm):
        if wavelength_mm in self.plane_wave_amplitudes:
            return self.plane_wave_amplitudes[wavelength_mm]
        else:
            amp = self.flux_without_wg(wavelength_mm)**.5
            print(f"plane wave amplitude for {wavelength_mm} mm: {amp:.2e}")
            self.plane_wave_amplitudes[wavelength_mm] = amp
            return amp
    
    # def get_mode_normalization(self, wavelength_mm):
    #     if wavelength_mm in self.mode_normalizations:
    #         return self.mode_normalizations[wavelength_mm]
    #     else:
    #         omega = 2 * np.pi * C_0 / (wavelength_mm * mm)
    #         coords = self.port.coords()
    #         et_m, ht_m, _ = self.port.field_profiles(self.epsilon_bg[coords], omega, self.dL)
    #         em = (0., 0., et_m)
    #         hm = (0., ht_m, 0.) if self.port.dir.is_along_x else (-ht_m, 0., 0.)
    #         normalization = np.sqrt(2*ops.overlap(em, hm, self.port.dir))
    #         self.mode_normalizations[wavelength_mm] = normalization
    #         print(f"mode_normalizations for {wavelength_mm} mm: {normalization:.2e}")
    #         return normalization

    def simulate(
            self,
            design_variable,
            wavelengths_mm):
        wavelengths_mm = np.asarray(wavelengths_mm)
        flat_wavelengths = list(wavelengths_mm.ravel(order='C'))
        amps = [None] * len(wavelengths_mm)
        ezs = [None] * len(wavelengths_mm)

        def _simulate(wavelength_mm):
            omega = 2 * np.pi * C_0 / (wavelength_mm * mm)
            k0 = 2 * np.pi * self.eps_bg**.5 / (wavelength_mm * mm)

            epsilon_r = self.epsilon_r(design_variable)

            # forward plane wave source
            plane_wave_source = np.zeros(self.grid_shape, dtype=complex)
            plane_wave_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
            plane_wave_source[self.source_x_start-1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)

            if self._backend == 'ceviche':
                # Set up the FDFD simulation for TE
                F = fdfd_ez(omega, self.dL, epsilon_r, (self.pml_width, self.pml_width))
                hx, hy, ez = F.solve(plane_wave_source)

                ######### (1) use ceviche amplitudes #########
                a, b = calculate_amplitudes_cc(
                    omega,
                    self.dL,
                    self.port,
                    ez,
                    hy,
                    hx,
                    self.epsilon_bg,
                )
                couple_amp = 2**.5 * b / self.get_plane_wave_amplitude(wavelength_mm)

                ######### (2) use flux #########
                # output_flux = anp.sum(anp.real(anp.conj(ez[self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]) * \
                #                                         hx[self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]))
                # assert output_flux > 0, "output_flux is negative something is wrong"
                # couple_amp = output_flux**.5 / self.get_plane_wave_amplitude(wavelength_mm)

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
                device_id = flat_wavelengths.index(wavelength_mm) # each device handles one wavelength, to reuse the precomputed PML 
                last_E = self.last_forward_E.get(wavelength_mm, None)
                self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.pml_width, None, last_E)))

                # wait and fetch the result
                with self.results_cond:
                    while task_id not in self.results:
                        self.results_cond.wait()
                    solution = self.results.pop(task_id)

                # self.last_forward_E[wavelength_mm] = solution

                ez = solution[None]
                # actually hy in ceviche convention
                hx = Ez_to_Hx(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)
                # actually hx in ceviche convention
                hy = Ez_to_Hy(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)

                
                # output_flux = torch.sum(torch.real(torch.conj(ez[0,self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]) * \
                #                                               hy[0,self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]))
                # output_flux = output_flux.numpy()
                # assert output_flux > 0, f"output_flux is negative: {output_flux} something is wrong"
                # couple_amp = output_flux**.5 / self.get_plane_wave_amplitude(wavelength_mm)
                a, b = calculate_amplitudes_torch(
                    omega,
                    self.dL,
                    self.port,
                    ez[0],
                    hx[0], # hy in ceviche convention
                    hy[0], # hx in ceviche convention
                    self.epsilon_bg,
                )
                couple_amp = 2**.5 * b / self.get_plane_wave_amplitude(wavelength_mm)

                ez = ez[0]
            else:
                raise ValueError(f"backend {self.backend} not supported")
            
            return wavelength_mm, couple_amp, ez

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wavelengths_mm)) as executor:
            simute_results = list(executor.map(_simulate, wavelengths_mm))
        
        for wavelength_mm, amp, ez in simute_results:
            wl_idx = flat_wavelengths.index(wavelength_mm)
            amps[wl_idx] = amp
            ezs[wl_idx] = ez
        
        amps = anp.stack(amps, axis=0)
        ezs = anp.stack(ezs, axis=0)

        return amps, ezs

    def simulate_adjoint(
        self,
        design_variable: np.ndarray,
        wavelengths_mm: np.ndarray,
        forward_output_torch: Tuple[torch.Tensor],
        grad_output_torch: Tuple[torch.Tensor],
    ):
        wavelengths_mm = np.asarray(wavelengths_mm)
        flat_wavelengths = list(wavelengths_mm.ravel(order='C'))
        epsilon_r = self.epsilon_r(design_variable)
        epsilon_r_torch = torch.from_numpy(epsilon_r).to(torch.float32)
       
        def _adjoint_simulate(forward_ez, wl_mm, grad_output):
            omega = 2 * np.pi * C_0 / (wl_mm * mm)
            omega_torch = torch.tensor([omega], dtype=torch.float32)
            dl_torch = torch.tensor([self.dL], dtype=torch.float32)

            def comp_graph(ez):
                ez = ez[None] # add batch dimension for H fields computation
                # actually hy in ceviche convention
                hx = Ez_to_Hx(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)
                # actually hx in ceviche convention
                hy = Ez_to_Hy(ez, dl_torch, omega_torch, torch.ones_like(ez), torch.ones_like(ez), EPSILON_0, periodic=True)

                # output_flux = torch.sum(torch.real(torch.conj(ez[0,self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]) * \
                #                                               hy[0,self.monitor_x-int(self.monitor_size/2):self.monitor_x+int(self.monitor_size/2), self.monitor_y]))
                # assert output_flux > 0, f"output_flux is negative: {output_flux} something is wrong"
                # couple_amp = output_flux**.5 / self.get_plane_wave_amplitude(wl_mm)
                a, b = calculate_amplitudes_torch(
                    omega,
                    self.dL,
                    self.port,
                    ez[0],
                    hx[0],
                    hy[0],
                    self.epsilon_bg,
                )
                couple_amp = 2**.5 * b / self.get_plane_wave_amplitude(wl_mm)

                return couple_amp

            forward_ez = forward_ez.detach().requires_grad_(True)
            amp = comp_graph(forward_ez)
            grad_ez = torch.autograd.grad(amp, forward_ez, grad_outputs=torch.conj(grad_output))[0]

            # compute adjoint simulations on gpu
            wl_torch = (2 * np.pi * C_0) / omega_torch
            source_torch = torch.conj(grad_ez).to(torch.complex64).resolve_conj()  # adjoint source

            # send task to GPU worker queue
            task_id = self.task_id_counter
            self.task_id_counter += 1
            # device_id = task_id % self.num_gpus
            device_id = flat_wavelengths.index(wl_mm) # each device handles one omega, to reuse the precomputed PML 
            last_E = self.last_adjoint_E.get(wl_mm, None)
            self.task_queues[device_id].put((task_id, (epsilon_r_torch, source_torch, wl_torch, dl_torch, self.pml_width, None, last_E)))
            
            # wait and fetch the result
            with self.results_cond:
                while task_id not in self.results:
                    self.results_cond.wait()
                adjoint_solution = self.results.pop(task_id)
            
            # self.last_adjoint_E[wl_mm] = adjoint_solution

            # F(x, y) = b - A(x)* y
            # compute ∂F/∂x
            design_variable_torch = design_variable.clone().detach().requires_grad_(True)
            epsilon_for_residual = make_torch_epsilon_r(design_variable_torch, self.density_bg, self.eps_design_min, self.eps_design_max, (self.design_region_x_start, self.design_region_y_start, self.design_region_x_end, self.design_region_y_end))
            sx, sy = adjoint_solution.shape # bs == 1
            bs = 1 # batch dimension for residual computation
            top_bc, bottom_bc, left_bc, right_bc = torch.zeros(bs, 1, sy, 2), torch.zeros(bs, 1, sy, 2), torch.zeros(bs, sx, 1, 2), torch.zeros(bs, sx, 1, 2)
            Sx_f, Sx_b = torch.zeros(bs, sx, sy), torch.zeros(bs, sx, sy)
            Sy_f, Sy_b = torch.zeros(bs, sy, sx), torch.zeros(bs, sy, sx)
            forward_ez = forward_ez.detach()

            # forward plane wave source
            k0 = 2 * np.pi * self.eps_bg**.5 / (wl_mm * mm)
            forward_source = np.zeros(self.grid_shape, dtype=complex)
            forward_source[self.source_x_start, self.source_y_start:self.source_y_end] = self.source_amp
            forward_source[self.source_x_start-1, self.source_y_start:self.source_y_end] = -self.source_amp * np.exp(-1j*k0*self.dL)
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
        forward_ezs = forward_output_torch[1] # shape (num_wavelengths, num_excite_ports, height, width)
        grad_output_amp = grad_output_torch[0] # shape (num_wavelengths, num_excite_ports, num_ports)
        
        # map jobs to all available GPUs
        def worker(wl_mm):
            wl_idx = flat_wavelengths.index(wl_mm)
            return _adjoint_simulate(
                forward_ezs[wl_idx],
                wl_mm,
                grad_output_amp[wl_idx],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            input_grads = list(executor.map(worker, flat_wavelengths))

        return sum(input_grads), None

    def get_sources(self):
        return np.array([1+1j])

    @cached_property
    def wg_x_start(self):
        return int(self.grid_shape[0]/2 + self.thickness_all/2 - self.wg_thickness)
    @cached_property
    def wg_x_end(self):
        return int(self.grid_shape[0]/2 + self.thickness_all/2)
    @cached_property
    def wg_y_start(self):
        return self.design_region_y_start
    @cached_property
    def wg_y_end(self):
        return self.grid_shape[1]
    @cached_property
    def design_region_x_start(self):
        return self.wg_x_start - self.design_variable_shape[0]
    @cached_property
    def design_region_x_end(self):
        return self.wg_x_start
    @cached_property
    def design_region_y_start(self):
        return int(self.grid_shape[1]/2 - self.width_all/2)
    @cached_property
    def design_region_y_end(self):
        return self.design_region_y_start + self.design_variable_shape[1]
    @cached_property
    def source_x_start(self):
        return self.design_region_x_start - self.source_spacing
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
        
