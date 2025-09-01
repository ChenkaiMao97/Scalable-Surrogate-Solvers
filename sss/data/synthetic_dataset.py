# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os
import  os.path
import  numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree

from sss.utils.PML_utils import create_sfactor_f, create_sfactor_b
from sss.utils.UI import printc

from ceviche.constants import C_0

import gin


@gin.configurable
class SyntheticDataset(Dataset):
    # dataset with PML bondary on all sides and synthetic parameter maps
    # used for unsupervised learning of physics residue only (no ground truth field labels)
    def __init__(
        self,
        shape,
        pml_sizes,
        eps_min,
        eps_max,
        dataset_size, 
        zoom_eps_list,
        zoom_src_list,
        sigma_eps_list,
        sigma_src_list,
        generate_bcs=False,
        wl_range=None, # range of wavelengths, in nm. e.g. [380,750]
        lambda_in_pixel_range=None, # range of wavelengths, in # of pixels. e.g. [30,60]
    ):
        # shape: (sx, sy, sz)
        self.shape = shape
        self.pml_sizes = pml_sizes
        self.dataset_size = dataset_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.zoom_eps_list = zoom_eps_list
        self.zoom_src_list = zoom_src_list
        self.sigma_eps_list = sigma_eps_list
        self.sigma_src_list = sigma_src_list
        self.gamma = 1.0 # make sure it matches full_residual_util_for_training.py

        self.wl_range = wl_range
        self.lambda_in_pixel_range = lambda_in_pixel_range
        self.generate_bcs = generate_bcs
    
    def __len__(self):
        return self.dataset_size
    
    def build_pml_feature(self, eps, sx_f, sy_f):
        # mask select eps in the PML region
        eps_x = torch.where(torch.abs(sx_f) > 1, eps, 0.)
        eps_y = torch.where(torch.abs(sy_f) > 1, eps, 0.)

        # compute the average phase along the tangential direction for each PML direction
        eps_mean_x = torch.mean(eps_x, dim=0)
        phase_x = torch.cumsum(eps_mean_x, dim=0)
        phase_x = (phase_x - torch.mean(phase_x)) / phase_x.shape[0]
        pml_feature_x = phase_x[None,:] * sx_f.imag

        eps_mean_y = torch.mean(eps_y, dim=1)
        phase_y = torch.cumsum(eps_mean_y, dim=0)
        phase_y = (phase_y - torch.mean(phase_y)) / phase_y.shape[0]
        pml_feature_y = phase_y[:,None] * sy_f.imag

        return pml_feature_x, pml_feature_y
    
    def build_PML_and_eps(self, eps, wl, dL):
        r = np.random.rand()
        if r < 0.5 or self.pml_sizes == (0,0):
            # no PML:
            sx_f = torch.ones(self.shape, dtype=torch.complex64)
            sx_b = torch.ones(self.shape, dtype=torch.complex64)
            sy_f = torch.ones(self.shape, dtype=torch.complex64)
            sy_b = torch.ones(self.shape, dtype=torch.complex64)
            eps = torch.stack([eps, torch.ones_like(eps)], dim=-1)
        else:
            # PML:
            omega = 2*np.pi * C_0 / wl
            sx_f_vec = create_sfactor_f(omega, dL, self.shape[0]+2*self.pml_sizes[0], self.pml_sizes[0])
            sx_b_vec = create_sfactor_b(omega, dL, self.shape[0]+2*self.pml_sizes[0], self.pml_sizes[0])
            sy_f_vec = create_sfactor_f(omega, dL, self.shape[1]+2*self.pml_sizes[1], self.pml_sizes[1])
            sy_b_vec = create_sfactor_b(omega, dL, self.shape[1]+2*self.pml_sizes[1], self.pml_sizes[1])

            shift_x = np.random.randint(round(-2*self.pml_sizes[0]), round(4*self.pml_sizes[0]))
            shift_y = np.random.randint(round(-2*self.pml_sizes[1]), round(4*self.pml_sizes[1]))
            sx_f_vec_shifted = np.roll(sx_f_vec, -shift_x, axis=0)[:self.shape[0]]
            sx_b_vec_shifted = np.roll(sx_b_vec, -shift_x, axis=0)[:self.shape[0]]
            sy_f_vec_shifted = np.roll(sy_f_vec, -shift_y, axis=0)[:self.shape[1]]
            sy_b_vec_shifted = np.roll(sy_b_vec, -shift_y, axis=0)[:self.shape[1]]

            sx_f = torch.from_numpy(sx_f_vec_shifted[:,None].repeat(self.shape[1],axis=1)).to(torch.complex64)
            sx_b = torch.from_numpy(sx_b_vec_shifted[:,None].repeat(self.shape[1],axis=1)).to(torch.complex64)
            sy_f = torch.from_numpy(sy_f_vec_shifted[None,:].repeat(self.shape[0],axis=0)).to(torch.complex64)
            sy_b = torch.from_numpy(sy_b_vec_shifted[None,:].repeat(self.shape[0],axis=0)).to(torch.complex64)

            eps_imag = 4/torch.abs(sx_f + sx_b + sy_f + sy_b)

            # pml_feature_x, pml_feature_y = self.build_pml_feature(eps, sx_f, sy_f)
            # eps = torch.stack([eps, eps_imag, pml_feature_x, pml_feature_y], dim=-1)

            eps = torch.stack([eps, eps_imag], dim=-1)

        return eps, sx_f, sx_b, sy_f, sy_b

    def random_2d_gaussian(self, shape, zoom_factor, sigma, clip=3, norm_min=0, norm_max=1):
        # generate a random 3d gaussian with the given shape and sigma
        # zoom is the zoom factor
        # return a 3d gaussian with the given shape and sigma
        small_shape = tuple(int(s / zoom_factor)+1 for s in shape)
        x = np.random.randn(*small_shape).astype(np.float32)
        x = zoom(x, zoom_factor, order=0)[:shape[0], :shape[1]]

        if sigma > 0:
            x = gaussian_filter(x, sigma, order=0)

        x = np.clip(x, -clip, clip)

        # convert to (0, 1)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x * (norm_max - norm_min) + norm_min
        return x
    
    def generate_voronoi_map(self, shape, num_points, norm_min=0.0, norm_max=1.0, seed=None):
        width, height = shape
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random control points
        points = np.random.rand(num_points, 2) * [width, height]
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Generate image coordinate grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack((xx, yy), axis=-1).reshape(-1, 2)

        # Find nearest Voronoi point for each pixel
        tree = cKDTree(points)
        _, regions = tree.query(coords)
        
        # Assign random value per region
        values = np.random.uniform(norm_min, norm_max, size=num_points).astype(np.float32)
        voronoi_map = values[regions].reshape((height, width))
        
        return voronoi_map
    
    def reduce_source_within_pml(self, source, sx, sy):
        source = torch.view_as_complex(source)
        source = torch.where(torch.abs((sx + sy).imag) < 1e-6, source, 0.)
        source = torch.view_as_real(source)
        return source
    
    def build_bcs(self, shape, zoom_bcs, sigma_bcs, all_S, norm_min=-1, norm_max=1):
        bcs = torch.stack([torch.from_numpy(self.random_2d_gaussian(shape, zoom_bcs, sigma_bcs, norm_min=norm_min, norm_max=norm_max)) for _ in range(2)], dim=-1)
        # reduce the bcs within the PML:
        bcs = torch.view_as_complex(bcs)
        bcs = bcs * (4/torch.abs(all_S))**2
        bcs = torch.view_as_real(bcs)

        top_bc = bcs[:1,:,:]
        bottom_bc = bcs[-1:,:,:]
        left_bc = bcs[:,:1,:]
        right_bc = bcs[:,-1:,:]
        return top_bc, bottom_bc, left_bc, right_bc

    def __getitem__(self, idx):
        # generate random parameter maps
        np.random.seed(idx)
        zoom_eps = np.random.choice(self.zoom_eps_list)
        zoom_src = np.random.choice(self.zoom_src_list)
        sigma_eps = np.random.choice(self.sigma_eps_list)
        sigma_src = np.random.choice(self.sigma_src_list)

        zoom_bcs = np.random.choice(self.zoom_eps_list)
        sigma_bcs = np.random.choice(self.sigma_eps_list)

        wl = np.random.randint(self.wl_range[0], self.wl_range[1]+1) * 1e-9
        lambda_in_pixel = np.random.randint(self.lambda_in_pixel_range[0], self.lambda_in_pixel_range[1]+1)
        dL = wl / lambda_in_pixel

        r = np.random.rand()
        if r < 0.5:
            num_points = max(round(np.prod(self.shape)/zoom_eps/zoom_eps/8), 5)
            eps = torch.from_numpy(self.generate_voronoi_map(self.shape, num_points, norm_min=self.eps_min, norm_max=self.eps_max))
        else:
            eps = torch.from_numpy(self.random_2d_gaussian(self.shape, zoom_eps, sigma_eps, norm_min=self.eps_min, norm_max=self.eps_max))
        eps, sx_f, sx_b, sy_f, sy_b = self.build_PML_and_eps(eps, wl, dL) # complex eps with absorptions as well

        source = torch.stack([torch.from_numpy(self.random_2d_gaussian(self.shape, zoom_src, sigma_src, norm_min=-3, norm_max=3)) for _ in range(2)], dim=-1)
        source = self.reduce_source_within_pml(source, sx_f + sy_f, sx_b + sy_b)
        
        sample = {'eps': eps, 'source': source, 'dL': dL, 'wl': wl,
                  'sx_f': sx_f.imag, 'sx_b': sx_b.imag, 'sy_f': sy_f.imag, 'sy_b': sy_b.imag}
        
        if self.generate_bcs:
            all_S = sx_f + sx_b + sy_f + sy_b
            top_bc, bottom_bc, left_bc, right_bc = self.build_bcs(self.shape, zoom_bcs, sigma_bcs, all_S, norm_min=-3, norm_max=3)
            sample['top_bc'] = top_bc  
            sample['bottom_bc'] = bottom_bc
            sample['left_bc'] = left_bc
            sample['right_bc'] = right_bc
        
        return sample

@gin.configurable
class SyntheticDataset_same_wl_dL(SyntheticDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return idx
    
    def collate_fn_same_wl_dL(self, batch_indices):
        batch_size = len(batch_indices)
        np.random.seed(batch_indices[0])
        wl = np.random.randint(self.wl_range[0], self.wl_range[1]+1) * 1e-9
        lambda_in_pixel = np.random.randint(self.lambda_in_pixel_range[0], self.lambda_in_pixel_range[1]+1)
        dL = wl / lambda_in_pixel

        batch = []
        for idx in batch_indices:
            np.random.seed(idx)
            zoom_eps = np.random.choice(self.zoom_eps_list)
            zoom_src = np.random.choice(self.zoom_src_list)
            sigma_eps = np.random.choice(self.sigma_eps_list)
            sigma_src = np.random.choice(self.sigma_src_list)

            zoom_bcs = np.random.choice(self.zoom_eps_list)
            sigma_bcs = np.random.choice(self.sigma_eps_list)

            r = np.random.rand()
            if r < 0.5:
                num_points = max(round(np.prod(self.shape)/zoom_eps/zoom_eps/8), 5)
                eps = torch.from_numpy(self.generate_voronoi_map(self.shape, num_points, norm_min=self.eps_min, norm_max=self.eps_max))
            else:
                eps = torch.from_numpy(self.random_2d_gaussian(self.shape, zoom_eps, sigma_eps, norm_min=self.eps_min, norm_max=self.eps_max))
            # eps = self.build_complex_eps(eps) # complex eps with absorptions as well
            eps, sx_f, sx_b, sy_f, sy_b = self.build_PML_and_eps(eps, wl, dL) # complex eps with absorptions as well

            source = torch.stack([torch.from_numpy(self.random_2d_gaussian(self.shape, zoom_src, sigma_src, norm_min=-3, norm_max=3)) for _ in range(2)], dim=-1)
            source = self.reduce_source_within_pml(source, sx_f + sy_f, sx_b + sy_b)
            
            if self.generate_bcs:
                all_S = sx_f + sx_b + sy_f + sy_b
                top_bc, bottom_bc, left_bc, right_bc = self.build_bcs(self.shape, zoom_bcs, sigma_bcs, all_S, norm_min=-3, norm_max=3)
                batch.append([eps, source, dL, wl, sx_f, sx_b, sy_f, sy_b, top_bc, bottom_bc, left_bc, right_bc])
            else:
                batch.append([eps, source, dL, wl, sx_f, sx_b, sy_f, sy_b])
            
        if self.generate_bcs:
            eps, source, dL, wl, sx_f, sx_b, sy_f, sy_b, top_bc, bottom_bc, left_bc, right_bc = zip(*batch)
            return {
                'eps': torch.stack(eps),
                'source': torch.stack(source),
                'dL': torch.tensor(dL),
                'wl': torch.tensor(wl),
                'sx_f': torch.stack(sx_f).imag,
                'sx_b': torch.stack(sx_b).imag,
                'sy_f': torch.stack(sy_f).imag,
                'sy_b': torch.stack(sy_b).imag,
                'top_bc': torch.stack(top_bc),
                'bottom_bc': torch.stack(bottom_bc),
                'left_bc': torch.stack(left_bc),
                'right_bc': torch.stack(right_bc)
            }
        else:
            eps, source, dL, wl, sx_f, sx_b, sy_f, sy_b = zip(*batch)
            return {
                'eps': torch.stack(eps),
                'source': torch.stack(source),
                'dL': torch.tensor(dL),
                'wl': torch.tensor(wl),
                'sx_f': torch.stack(sx_f).imag,
                'sx_b': torch.stack(sx_b).imag,
                'sy_f': torch.stack(sy_f).imag,
                'sy_b': torch.stack(sy_b).imag
            }
        
        