# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import functools
import jax.tree_util
import jax.numpy as jnp
import numpy as onp
from typing import Sequence, Union, Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass

from sss.invde.ceviche_jax import CevicheJaxComponent, CevicheJaxChallenge, DensityInitializer
from sss.invde.utils.jax_utils import _get_default_initializer
from sss.design_problems.grating_coupler.grating_coupler_model import GratingCouplerModel

import sss.invde.utils.jax_autograd_wrapper as autograd_wrapper
import sss.invde.utils.jax_torch_wrapper as torch_wrapper

import gin

_DENSITY_LABEL = "density"
_FIELDS_EZ_LABEL = "fields_ez"

@dataclass
class ResponseArray:
    array: jnp.ndarray

jax.tree_util.register_pytree_node(
    nodetype=ResponseArray,
    flatten_func=lambda s: ( (s.array,), (None,) ),
    unflatten_func=lambda aux, array: ResponseArray(*array),
)

@gin.configurable
class GratingCouplerComponent(CevicheJaxComponent):
    """grating coupler component with arbitrary number of ports and locations"""
    def __init__(
        self,
        design_resolution_mm: int,
        sim_resolution_mm: int,
        wavelengths_mm: Union[onp.ndarray, Sequence[float]],
        density_initializer_getter: Callable = _get_default_initializer,
        backend: str = 'ceviche',
    ) -> None:
        super().__init__(
            design_resolution_nm=design_resolution_mm,
            sim_resolution_nm=sim_resolution_mm,
            wavelengths_nm=wavelengths_mm,
            model_constructor=functools.partial(GratingCouplerModel, _backend=backend),
            density_initializer=density_initializer_getter(),
            _backend=backend
        )
    
    def construct_jax_sim_fn(self,
            design_resolution_mm: int,
            sim_resolution_mm: int,
            model: Callable,
        ):
        """Constructs the jax-compatible simulation function for the model."""
        # Wrap the model simulation function for use with jax. It has signature:
        # Here, the design must be at the simulation grid resolution.
        if self._backend == 'ceviche':
            _jax_wrapped_sim_fn = autograd_wrapper.jax_wrap_autograd(
                model.simulate, argnums=0, outputnums=0
            )
        elif self._backend == 'DDM':
            _jax_wrapped_sim_fn = torch_wrapper.jax_wrap_torch(
                model.simulate, model.simulate_adjoint, argnums=0
            )
        return _jax_wrapped_sim_fn
    
    def response(
        self,
        params,
        wavelengths_mm: Optional[Sequence[float]] = None,
    ):
        density = params[_DENSITY_LABEL].density

        if wavelengths_mm is None:
            wavelengths_mm = tuple(self._wavelength)
        coupling_eff, ez = self._jax_sim_fn(
            density, wavelengths_mm
        )
        coupling_eff = ResponseArray(array = coupling_eff)
        return coupling_eff, {_FIELDS_EZ_LABEL: ez}



@gin.configurable
class GratingCouplerChallenge(CevicheJaxChallenge):
    """General Block design challenge."""
    def __init__(
        self,
        design_resolution_mm,
        sim_resolution_mm,
        wavelengths_mm: Sequence[float],
        density_initializer_getter: Callable = _get_default_initializer,
    ) -> None:
        super().__init__(
            component=GratingCouplerComponent(
                design_resolution_mm=design_resolution_mm,
                sim_resolution_mm=sim_resolution_mm,
                wavelengths_mm=wavelengths_mm,
                density_initializer_getter=density_initializer_getter,
            ),
            min_transmission=jnp.asarray([0.0] * len(wavelengths_mm)),
            max_transmission=jnp.asarray([1.0] * len(wavelengths_mm)),
        )
    
    def loss(self, response) -> float:
        return jnp.sum((jnp.abs(response.array) ** 2 - self._max_transmission)**2)

# # -*- coding: utf-8 -*-

# # import all the necessities
# import sys,os
# import numpy as np
# import random
# from scipy.ndimage import gaussian_filter

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
# import ceviche
# from ceviche import fdfd_hz
# from ceviche.constants import C_0
# import time
# from tqdm import tqdm

# sys.path.append("../util")
# from modes import get_modes

# dL = np.array([0.4, 0.4])

# Nx = int(sys.argv[1])
# Ny = int(sys.argv[2])
# grid_shape = Nx, Ny

# pml_x = 40
# pml_y = pml_x
# npml = [pml_x, pml_y] # Periodic in x direction

# source_x = int(2*Nx/5)
# source_length_y = int(Ny/6)

# wg_thickness = 15
# design_region_x = 4*wg_thickness
# design_region_y = int(1*Ny/4)
# wg_center = (int(3*Nx/4), int(Ny/2))

# design_center = (int(3*Nx/4)-int((design_region_x-wg_thickness)/2), int(Ny/2))

# wg_x_start = wg_center[0] - int(wg_thickness/2)
# wg_x_end = wg_center[0] - int(wg_thickness/2) + wg_thickness

# x_start = design_center[0] - int(design_region_x/2)
# x_end = design_center[0] - int(design_region_x/2)+design_region_x
# y_start = design_center[1] - int(design_region_y/2)
# y_end = design_center[1] - int(design_region_y/2)+design_region_y

# flux_x = x_start - 2
# probe_y = Ny-2*pml_y
# probe_x_start = wg_x_start-15
# probe_x_end = wg_x_end+15

# assert design_region_x<2000
# assert design_region_y<2000

# eps_max = 4.6
# eps_wg = 4.6
# eps_air = 1

# def greyscale_paint_1(pattern, max_region=None,seed=0):
#     # use several different strategy to paint the binary patterns into greyscale patterns
#     # Strat (1):
#     # use random Fourier waves in 2d to generate continuous map
#     np.random.seed(seed)

#     from voronoi import generate

#     for idx,p in enumerate(pattern):
#         print("p.shape: ", p.shape)
#         min_region = max(1,int(p.shape[1]/500)**2)
#         max_region = max(2,int(p.shape[1]/30)**2) if not max_region else max_region
        
#         N_regions = np.random.randint(min_region,max_region)
#         voronoi_map = generate(
#                 width = p.shape[1],
#                 height = p.shape[2],
#                 regions = N_regions,
#                 colors = np.random.rand(N_regions)*(eps_max-eps_air)
#             )
#         pattern[idx] = voronoi_map*p[0]+eps_air

#     return pattern

# def greyscale_paint_2(pattern, alpha, seed=0, eps_max=eps_max):
#     np.random.seed(seed)
#     from gaussian_random_fields import gaussian_random_field
#     # use several different strategy to paint the binary patterns into greyscale patterns
#     # Strat (2):
#     # use random Gaussian fields

#     for idx,p in enumerate(pattern):
#         GRM = gaussian_random_field(alpha=alpha, size=p.shape[1])
#         GRM = GRM-np.min(GRM)
#         GRM = GRM/np.max(GRM)
#         GRM = GRM*(eps_max-eps_air)

#         pattern[idx] = GRM*p[0]+eps_air

#     return pattern

# def setup_plot_data(data, src, pml_th, flux_x, probe_y):
#     colored_yee = np.zeros((data.shape[0], data.shape[1],3))
#     # Si_color = np.array([245,112,108], dtype=np.uint8)
#     air_color = np.array([249,232,215], dtype=np.uint8)
#     pml_color = np.array([255, 185, 0], dtype=np.uint8)
#     src_color = np.array([52, 181, 168], dtype=np.uint8)

#     top_mat_color = np.array([30, 30, 30], dtype=np.uint8)

#     data = np.asarray(data)

#     colored_yee = air_color+((data[:,:,None]-1)/(eps_max-1)*(top_mat_color.astype(np.float32)-air_color.astype(np.float32))).astype(np.uint8)

#     pml = np.zeros((data.shape[0], data.shape[1]))
#     pml[:pml_th, :] = 1
#     pml[-pml_th:, :] = 1
#     pml[:,:pml_th] = 1
#     pml[:,-pml_th:] = 1
#     colored_yee = (pml[:,:,None]>0.5)*pml_color + (pml[:,:,None]<0.5)*colored_yee 

#     thickened_src = np.abs(src[:,:,None])>1e-3
#     thickened_src = thickened_src + np.roll(thickened_src, 1, axis=0) + np.roll(thickened_src, -1, axis=0) + \
#                                     np.roll(thickened_src, 2, axis=0) + np.roll(thickened_src, -2, axis=0) + \
#                                     np.roll(thickened_src, 3, axis=0) + np.roll(thickened_src, -3, axis=0) + \
#                                     np.roll(thickened_src, 1, axis=1) + np.roll(thickened_src, -1, axis=1) + \
#                                     np.roll(thickened_src, 2, axis=1) + np.roll(thickened_src, -2, axis=1) + \
#                                     np.roll(thickened_src, 3, axis=1) + np.roll(thickened_src, -3, axis=1)

#     colored_yee = (thickened_src>1e-5)*src_color + (np.abs(src[:,:,None])<1e-5)*colored_yee 

#     colored_yee[flux_x, y_start:y_end] += 30
#     colored_yee[probe_x_start:probe_x_end, probe_y] += 60

#     return colored_yee

# def adam_optimizer(grad, theta, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=0):
#     """
#     Adam optimizer for a single step.

#     :param grad: Gradient of the objective function with respect to parameters
#     :param theta: Current parameters
#     :param lr: Learning rate
#     :param beta1: Exponential decay rate for the first moment estimates
#     :param beta2: Exponential decay rate for the second moment estimates
#     :param epsilon: Small value to avoid division by zero
#     :param m: First moment vector (initialized as zeros)
#     :param v: Second moment vector (initialized as zeros)
#     :param t: Time step
#     :return: Updated parameters, updated m, updated v, updated t
#     """
#     if m is None:
#         m = [0] * len(theta)
#     if v is None:
#         v = [0] * len(theta)
    
#     t += 1  # Increment time step
    
#     # Update biased first moment estimate
#     m = [beta1 * m_i + (1 - beta1) * grad_i for m_i, grad_i in zip(m, grad)]
#     # Update biased second raw moment estimate
#     v = [beta2 * v_i + (1 - beta2) * (grad_i ** 2) for v_i, grad_i in zip(v, grad)]
    
#     # Compute bias-corrected first moment estimate
#     m_hat = [m_i / (1 - beta1 ** t) for m_i in m]
#     # Compute bias-corrected second raw moment estimate
#     v_hat = [v_i / (1 - beta2 ** t) for v_i in v]
    
#     # Update parameters
#     theta = [theta_i - lr * m_hat_i / (v_hat_i ** 0.5 + epsilon) for theta_i, m_hat_i, v_hat_i in zip(theta, m_hat, v_hat)]
    
#     return theta, m, v, t


# def FOM(field, mode_Hz):
#     f = field[wg_x_start:wg_x_end, probe_y]
#     mode = mode_Hz[wg_x_start:wg_x_end]

#     return np.real(f.dot(np.conj(mode)))/(mode.dot(mode.conj()))

# def wg_mode_source(wl, dL, eps_line, m=1, pol="TM"):
#     N = eps_line.shape[0]
#     omega = 2 * np.pi * C_0 / wl        # angular frequency (rad/s)
#     Lx = N * dL                         # length in horizontal direction (m)
#     vals, vecs = get_modes(eps_line, omega, dL, npml=pml_x, m=m, pol=pol, filter_PML_mode=True)

#     return vals, vecs

# def flux_from_sim_without_wg(wavelength, flux_x):
#     eps_r = np.ones(grid_shape)
#     omega = 2 * np.pi * C_0 / wavelength
#     k0 = 2 * np.pi / wavelength
#     F = fdfd_hz(omega, dL[0]*1e-3, eps_r, npml)

#     source_amp = 1e6/dL[0]/dL[1]
#     random_source = np.zeros(grid_shape, dtype=complex)
#     random_source[source_x, int(Ny/2 - source_length_y/2):int(Ny/2 - source_length_y/2)+source_length_y] = source_amp

#     Ex_forward, Ey_forward, Hz_forward = F.solve(random_source)

#     flux = np.sum(np.real(np.conj(Ey_forward[flux_x, y_start:y_end]) * Hz_forward[flux_x, y_start:y_end]))

#     return flux


# def simulate(idx, full_pattern, wavelength, direction="f", vecs=None):
#     omega = 2 * np.pi * C_0 / wavelength
#     k0 = 2 * np.pi / wavelength

#     eps_r = np.ones(grid_shape)
#     eps_r[wg_x_start:wg_x_end, :] = eps_wg
#     eps_r[x_start:x_end, y_start:y_end] = full_pattern

#     # Set up the FDFD simulation for TM
#     F = fdfd_hz(omega, dL[0]*1e-3, eps_r, npml)

#     # Source
#     source_amp = 1e6/dL[0]/dL[1]
#     random_source = np.zeros(grid_shape, dtype=complex)
#     if direction == 'f':
#         random_source[source_x, int(Ny/2 - source_length_y/2):int(Ny/2 - source_length_y/2)+source_length_y] = source_amp
#     elif direction == 'a':
#         random_source[:, probe_y] = source_amp*vecs

#     Ex_forward, Ey_forward, Hz_forward = F.solve(random_source)

#     if direction == 'f':
#         return eps_r, Hz_forward, Ex_forward, Ey_forward, random_source
#     elif direction == 'a':
#         return eps_r, Hz_forward, Ex_forward, Ey_forward, random_source, vecs


# def main(seed):
#     print(f"start optimization loop with seed {seed}")
#     np.random.seed(seed)
#     random.seed(seed)
#     ####################

#     tic = time.time()

#     wl = 30 # unit: mm
#     wavelength = wl*1e-3
#     init_max_eps = eps_max
#     opt_steps = 50

#     lr = 3
#     # initialize device as random gaussian field
#     max_size = max(design_region_x, design_region_y)
#     # device = greyscale_paint_1(np.ones((1,1,max_size,max_size)), max_region=20, seed=seed)[0,0,:design_region_x, :design_region_y]
#     device = greyscale_paint_2(np.ones((1,max_size,max_size)), 4.0, seed =seed, eps_max=init_max_eps)[0,:design_region_x, :design_region_y]
    
#     # noise = np.random.rand(design_region_x, design_region_y)
#     # device = eps_air+noise*(eps_max-eps_air)
#     # device = gaussian_filter(device, 3)


#     noise_blur_step = 100
#     plot_step = 1

#     # add blur and noise every now and then to jump out of local minima
#     noise_mult = 0.1
#     sigma = 1

#     m, v, t = None, None, 0 # for adam optimizer
#     scheduler = 0.7 # lr exponential decay

#     input_flux = flux_from_sim_without_wg(wavelength, flux_x)
#     # compute adjoint source:
#     eps_r = np.ones(grid_shape)
#     eps_r[wg_x_start:wg_x_end, :] = eps_wg
#     val, vecs = wg_mode_source(wavelength, dL[0]*1e-3, eps_r[:,probe_y], m=50, pol="TM")
#     # print(f"found {vecs.shape[1]} modes")
#     if vecs.shape[1]>1:
#         assert np.max(val) == val[0], "fundamental mode is not the first one?!"
#     vecs = vecs[:,0] # fundamental mode
#     angle_correction = np.angle(vecs[wg_center[0]])
#     vecs = vecs*np.exp(-1j*angle_correction)

#     # optimization loop:
#     FOMs = []
#     best_fom=0

#     opt_steps = 10
#     pbar = tqdm(range(opt_steps), desc='Starting')
#     for i in range(opt_steps):
#         input_eps, Hz_f, Ex_f, Ey_f, source_f = simulate(i, device, wavelength, direction='f')
#         input_eps, Hz_a, Ex_a, Ey_a, source_a, mode_Hz = simulate(i, device, wavelength, direction='a', vecs=vecs)
#         # f_angle = np.conj(np.angle(Hz_f[wg_center[0], probe_y]))
#         # conj_phase = np.exp(1j*f_angle)
#         conj_phase = 1
        
#         # grad = np.real(Hz_f[x_start:x_end, y_start:y_end]*Ey_a[x_start:x_end, y_start:y_end]*conj_phase)
#         grad = -np.real(Ex_f[x_start:x_end, y_start:y_end]*Ex_a[x_start:x_end, y_start:y_end]*conj_phase) + \
#                -np.real(Ey_f[x_start:x_end, y_start:y_end]*Ey_a[x_start:x_end, y_start:y_end]*conj_phase)
#         grad = grad*1e-3
#         # print("grad stats: ", np.mean(np.abs(grad)), np.max(grad), np.min(grad))

#         # only consider region where 1<eps<eps_max
#         # grad[(np.abs(device-eps_air)<1e-3)*(grad<0)] = 0
#         # grad[(np.abs(device-eps_max)<1e-3)*(grad>0)] = 0

#         # grad_abs = np.abs(grad)
#         # grad[grad_abs<0.2*np.max(grad_abs)] = 0

#         output_flux = np.sum(np.real(np.conj(-Ex_f[probe_x_start:probe_x_end, probe_y]) * Hz_f[probe_x_start:probe_x_end, probe_y]))

#         # fom = FOM(Hz_f, mode_Hz*conj_phase)
#         fom = output_flux/input_flux

#         pbar.set_description(f"step: {i} FOM: {fom:.4f}, lr: {lr:.4f}")
#         pbar.update()
#         FOMs.append(fom)
#         # update using gradient:
#         # device += lr*grad
#         device, m, v, t = adam_optimizer(grad.flatten(), device.flatten(), m=m, v=v, t=t, lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
#         device = np.array(device).reshape((design_region_x, design_region_y))
#         device = np.maximum(np.minimum(device, eps_max), eps_air)

#         if (i+1) % noise_blur_step == 0:
#             noise = noise_mult * np.random.rand(device.shape[0], device.shape[1])
#             device = 0.9*device + 0.1 * noise
#             device = gaussian_filter(device, sigma*(1-i/opt_steps))

#         if fom > best_fom:
#             best_fom = fom
#             plt.figure()
#             plt.subplot(5,1,1)
#             colored_setup = setup_plot_data(input_eps, source_f+source_a, 40, flux_x, probe_y)
#             plt.imshow(colored_setup)
#             plt.colorbar()
#             plt.title(f"fom: {best_fom}")
#             plt.subplot(5,1,2)
#             plt.imshow(Hz_f.real)
#             plt.colorbar()
#             plt.subplot(5,1,3)
#             plt.imshow(Hz_f.imag)
#             plt.colorbar()
#             plt.subplot(5,1,4)
#             plt.imshow(Hz_a.real)
#             plt.colorbar()
#             plt.subplot(5,1,5)
#             plt.imshow(Hz_a.imag)
#             plt.colorbar()
#             plt.savefig(f"./seed_{seed}_best.png", dpi=300)
#             plt.close()

#         if i%5 == 0:
#             lr = lr*scheduler

#     toc = time.time()
#     print(f"The total time of the optimization is {toc - tic}s")

#     plt.plot(FOMs)
#     plt.savefig(f"FOM_seed{seed}.png", dpi=200)



# if __name__ == '__main__':
#     for seed in range(1):
#         main(seed)
    