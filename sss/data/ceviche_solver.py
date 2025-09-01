# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import numpy as np
from ceviche import fdfd_ez
from ceviche.constants import C_0
import time
import sys,os
from tqdm import tqdm
import matplotlib.pyplot as plt

import gin

from sss.utils.data_utils import random_2d_gaussian, generate_voronoi_map, random_line_src
from sss.utils.plot_utils import setup_plot_data
sys.path.append("../../../util")

@gin.configurable
class CevicheSolver:
    def __init__(
        self,
        output_dir: str,
        Nx: int,
        Ny: int,
        wavelengths: list[float], # in mm
        dLs: list[float], # in mm
        eps_zooms: list[int],
        eps_sigmas: list[float],
        pml_x: int = 40,
        pml_y = 40,
        eps_max = 8, 
        eps_air = 1,
        N_src = 10,
        num_shapes: int = 10,
        binary_shape_mask = None, # optional binary shape mask to mask the device
        save_plots = False,
        seed: int = 0,
    ):
        self.output_dir = output_dir
        self.Nx = Nx
        self.Ny = Ny
        self.wavelengths = wavelengths
        self.dLs = dLs
        self.npml = (pml_x, pml_y)
        self.eps_zooms = eps_zooms
        self.eps_sigmas = eps_sigmas
        self.eps_max = eps_max
        self.eps_air = eps_air
        self.N_src = N_src
        self.num_shapes = num_shapes

        self.binary_shape_mask = binary_shape_mask
        self.save_plots = save_plots
        self.seed = seed

        self.init()
    
    def init(self):
        np.random.seed(self.seed)
        if self.output_dir and (not os.path.isdir(self.output_dir)):
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir+'/plots', exist_ok=True)

    def generation_one_device(self, full_pattern, wavelength, dL):
        N_src = self.N_src
        grid_shape = (self.Nx, self.Ny)
        omega = 2 * np.pi * C_0 / wavelength
        k0 = 2 * np.pi / wavelength
        
        # Set up the FDFD simulation for TM
        F = fdfd_ez(omega, dL, full_pattern, self.npml)
        
        random_source = np.zeros(grid_shape, dtype=complex)
        for _ in range(N_src//2):
            random_source += random_line_src(grid_shape, wavelength, dL, n_angles = 6, pml_x=self.npml[0], pml_y=self.npml[1], direction='h', source_PML_spacing=5)
            random_source += random_line_src(grid_shape, wavelength, dL, n_angles = 6, pml_x=self.npml[0], pml_y=self.npml[1], direction='v', source_PML_spacing=5)

        # Solve the FDFD simulation for the fields, offset the phase such that Ex has 0 phase at the center of the bottom row of the window
        Hx_forward, Hy_forward, Ez_forward = F.solve(random_source)
        
        return full_pattern, Hx_forward, Hy_forward, Ez_forward, random_source

    def prepare_grayscale_pattern(self):
        if self.binary_shape_mask is not None:
            full_pattern = np.load(self.binary_shape_mask)
            assert len(full_pattern.shape) == 3
            assert full_pattern.shape[1] >= self.Nx
            assert full_pattern.shape[2] >= self.Ny
        else:
            full_pattern = np.ones((self.num_shapes, self.Nx, self.Ny))

        grayscale_pattern = np.zeros((full_pattern.shape[0], self.Nx, self.Ny))
        for i in range(full_pattern.shape[0]):
            r = np.random.rand()
            zoom_factor = np.random.choice(self.eps_zooms)
            sigma = np.random.choice(self.eps_sigmas)
            if r < 0.5:
                pattern = random_2d_gaussian((self.Nx, self.Ny), zoom_factor, sigma, clip=3, norm_min=self.eps_air, norm_max=self.eps_max)
                grayscale_pattern[i] = np.where(full_pattern[i] > 0.5, pattern, self.eps_air)
            else:
                num_points = max(10, int(self.Nx*self.Ny/zoom_factor**2/4))
                pattern = generate_voronoi_map((self.Nx, self.Ny), num_points, norm_min=self.eps_air, norm_max=self.eps_max)
                grayscale_pattern[i] = np.where(full_pattern[i] > 0.5, pattern, self.eps_air)
        return grayscale_pattern

        # grayscale_pattern = np.ones((1, self.Nx, self.Ny))
        # side_space = 100
        # for i in range(side_space,self.Nx-side_space):
        #     grayscale_pattern[0,i,int(side_space):int(self.Ny-side_space)] = 4
        # return grayscale_pattern

    def generate_dataset(self):
        grayscale_pattern = self.prepare_grayscale_pattern()

        gray_N = grayscale_pattern.shape[0]

        total_N = gray_N * len(self.wavelengths) * len(self.dLs)

        tic = time.time()

        each_device_time =[]
        # Initialize output fields
        input_eps = np.empty([total_N, self.Nx, self.Ny], dtype = np.float32)
        Hx_out_forward = np.empty([total_N, self.Nx, self.Ny], dtype = np.complex64)
        Hy_out_forward = np.empty([total_N, self.Nx, self.Ny], dtype = np.complex64)
        Ez_out_forward = np.empty([total_N, self.Nx, self.Ny], dtype = np.complex64)
        source_out = np.empty([total_N, self.Nx, self.Ny], dtype = np.complex64)
        wls_out = np.empty([total_N], dtype = np.float32)
        dLs_out = np.empty([total_N], dtype = np.float32)

        for wl_idx, wl in tqdm(enumerate(self.wavelengths)):
            wavelength = wl*1e-3
            for dL_idx, dL in tqdm(enumerate(self.dLs)):
                dL_m = dL*1e-3
                for i in tqdm(range(gray_N)):
                    data_idx = wl_idx*len(self.dLs)*gray_N + dL_idx*gray_N + i
                    tic_i = time.time()
                    input_eps[data_idx], Hx_out_forward[data_idx], Hy_out_forward[data_idx], Ez_out_forward[data_idx], source_out[data_idx] = self.generation_one_device(grayscale_pattern[i], wavelength, dL_m)
                    wls_out[data_idx] = wl
                    dLs_out[data_idx] = dL
                    toc_i = time.time()
                    each_device_time.append(toc_i - tic_i)

                    if self.save_plots:
                        plt.figure()
                        plt.subplot(1,2,1)
                        colored_setup = setup_plot_data(input_eps[data_idx], source_out[data_idx], 40)
                        plt.imshow(colored_setup)
                        plt.colorbar()
                        plt.subplot(1,2,2)
                        plt.imshow(Ez_out_forward[data_idx].real)
                        plt.colorbar()
                        plt.savefig(self.output_dir+'/plots'+f"/wl{wavelength}_dL{dL}_idx{i}.png", dpi=200)

        Hx_out_forward_RI = np.stack((np.real(Hx_out_forward), np.imag(Hx_out_forward)), axis = -1)
        Hy_out_forward_RI = np.stack((np.real(Hy_out_forward), np.imag(Hy_out_forward)), axis = -1)
        Ez_out_forward_RI = np.stack((np.real(Ez_out_forward), np.imag(Ez_out_forward)), axis = -1)
        source_RI = np.stack((np.real(source_out), np.imag(source_out)), axis = -1)

        toc = time.time()
        print(f"Device finished: {total_N}, The total time of the data generation is {toc - tic}s")

        each_device_time = np.array(each_device_time)
        print(f"for simulation domain of size {self.Nx} by {self.Ny}, average time is {np.mean(each_device_time)}, variance is {np.var(each_device_time)}")

        np.save(self.output_dir+"/"+f"input_eps.npy", input_eps)
        np.save(self.output_dir+"/"+f"Hx_out_forward_RI.npy", Hx_out_forward_RI)
        np.save(self.output_dir+"/"+f"Hy_out_forward_RI.npy", Hy_out_forward_RI)
        np.save(self.output_dir+"/"+f"Ez_out_forward_RI.npy", Ez_out_forward_RI)
        np.save(self.output_dir+"/"+f"source_RI.npy", source_RI)
        np.save(self.output_dir+"/"+f"wls.npy", wls_out)
        np.save(self.output_dir+"/"+f"dLs.npy", dLs_out)
