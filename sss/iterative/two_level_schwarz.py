# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os
import math
import numpy as np
import torch
from tqdm import tqdm

from sss.utils.PDE_utils import plane_wave_bc, random_bc, random_fourier_bc
from sss.utils.plot_utils import setup_plot_data
from sss.utils.UI import printc

import cupy as cp
import cupyx.scipy.sparse as cpx
import cupyx.scipy.sparse.linalg as cpxl

import matplotlib.pyplot as plt

import gin

def robin(eps, bc, d_w, wl, dL):
    # bc: the boundary field to be transform
    # d_w: the derivative of fields in w
    g = 1j*2*np.pi*torch.sqrt(eps)/wl*dL*bc+d_w
    return g

def r2c(x):
    return torch.view_as_complex(x)

def c2r(x):
    return torch.view_as_real(x)

def bcs_to_vector(top_bc, bottom_bc, left_bc, right_bc):
	bs = top_bc.shape[0]
	return torch.cat((top_bc.view(bs, -1), bottom_bc.view(bs, -1), left_bc.view(bs, -1), right_bc.view(bs, -1)), dim=1)

def vector_to_bcs(vector):
	bs = vector.shape[0]
	total_length = vector.shape[1]
	assert total_length % 4 == 0
	length = total_length // 4
	return c2r(vector[:, :length].view(bs, 1, length)), \
		   c2r(vector[:, length:2*length].view(bs, 1, length)), \
		   c2r(vector[:, 2*length:3*length].view(bs, length, 1)), \
		   c2r(vector[:, 3*length:4*length].view(bs, length, 1))

def gram_schmidt(Y, xi):
	# assume Y is a list of orthonormal vectors (Y[i] of shape (bs, length))
	bs = xi.shape[0]
	yi = xi.clone()
	for i in range(len(Y)):
		yi = yi - torch.sum(torch.conj(Y[i]) * yi, dim=1, keepdim=True) * Y[i]
	yi = yi / torch.sum(torch.abs(yi)**2, dim=1, keepdim=True)**.5
	return yi

@gin.configurable
class TwoLevelSchwarz:
    def __init__(
        self,
        unpadded_shape,
        min_overlap,
        region_size,
        coarse_space_momentum=1.0,
        eval_k_per_subdomain=4,
        keep_k_per_subdomain=4,
        use_coarse_space=False,
        sparse_BQTBQ=False,
        plot_basis_func=False,
        plot_basis_func_num=10,
        variable_overlap=False,
        periodic_padding=False,
        sommerfeld=False,
    ):
        self.unpadded_shape = unpadded_shape
        self.min_overlap = min_overlap
        self.region_size = region_size
        self.coarse_space_momentum = coarse_space_momentum
        print("coarse_space_momentum: ", coarse_space_momentum)
        self.use_coarse_space = use_coarse_space
        self.sparse_BQTBQ = sparse_BQTBQ
        self.plot_basis_func = plot_basis_func
        self.plot_basis_func_num = plot_basis_func_num
        self.eval_k_per_subdomain = eval_k_per_subdomain
        self.keep_k_per_subdomain = keep_k_per_subdomain
        self.variable_overlap = variable_overlap
        self.periodic_padding = periodic_padding
        self.global_bcs = None
        self.x_overlaps, self.y_overlaps, self.grid_shape, self.padded_shape = self.init_grid()
        print("x_overlaps: ",self.x_overlaps, "y_overlaps: ", self.y_overlaps, "grid_shape: ", self.grid_shape, "padded_shape: ", self.padded_shape)
        self.regions, self.POU_maps = self.init_regions()  # Stores slices for each region
        self.sommerfeld = sommerfeld

    def init_grid(self):
        if self.variable_overlap:
            if self.periodic_padding:
                rows = math.ceil(self.unpadded_shape[0]/(self.region_size[0]-self.min_overlap))
                cols = math.ceil(self.unpadded_shape[1]/(self.region_size[1]-self.min_overlap))
            else:
                rows = math.ceil((self.unpadded_shape[0]-self.min_overlap)/(self.region_size[0]-self.min_overlap))
                cols = math.ceil((self.unpadded_shape[1]-self.min_overlap)/(self.region_size[1]-self.min_overlap))
            grid_shape = (rows, cols)
            total_gap_x = rows * (self.region_size[0]-self.min_overlap) - self.unpadded_shape[0] + (not self.periodic_padding) * self.min_overlap
            total_gap_y = cols * (self.region_size[1]-self.min_overlap) - self.unpadded_shape[1] + (not self.periodic_padding) * self.min_overlap

            num_overlaps_x = rows if self.periodic_padding else rows - 1
            num_overlaps_y = cols if self.periodic_padding else cols - 1
            x_overlaps = [self.min_overlap + round((i+1)*total_gap_x/num_overlaps_x) - round(i*total_gap_x/num_overlaps_x) for i in range(num_overlaps_x)] + [self.min_overlap] * (not self.periodic_padding)
            y_overlaps = [self.min_overlap + round((i+1)*total_gap_y/num_overlaps_y) - round(i*total_gap_y/num_overlaps_y) for i in range(num_overlaps_y)] + [self.min_overlap] * (not self.periodic_padding)
        else:
            assert self.min_overlap < self.region_size[0] and self.min_overlap < self.region_size[1]
            assert self.unpadded_shape[0] % (self.region_size[0]-self.min_overlap) == (not self.periodic_padding) * self.min_overlap
            assert self.unpadded_shape[1] % (self.region_size[1]-self.min_overlap) == (not self.periodic_padding) * self.min_overlap
            grid_shape = (self.unpadded_shape[0]//(self.region_size[0]-self.min_overlap), self.unpadded_shape[1]//(self.region_size[1]-self.min_overlap))
            x_overlaps = [self.min_overlap] * grid_shape[0]
            y_overlaps = [self.min_overlap] * grid_shape[1]

        padded_shape = (self.unpadded_shape[0] + x_overlaps[-1], self.unpadded_shape[1] + y_overlaps[-1]) if self.periodic_padding else self.unpadded_shape
        
        return x_overlaps, y_overlaps, grid_shape, padded_shape

    def init_regions(self):
        num_rows, num_cols = self.grid_shape
        regions = []
        POU_maps = []
        count_map = torch.zeros(self.padded_shape)

        # (1) set as all ones
        POU_map = torch.ones(self.region_size[0], self.region_size[1])
        
        # (2) set as smoothly decaying at edge
        # temp_overlap = max(self.x_overlaps[0], self.y_overlaps[0])
        # POU_map = torch.zeros(self.region_size[0], self.region_size[1])
        # for k in range(temp_overlap):
        #     POU_map[k:self.region_size[0]-k, k:self.region_size[1]-k] = (k+1)**1/temp_overlap**1

        for i in range(num_rows):
            for j in range(num_cols):
                # Calculate slices for the current region
                row_start = i * self.region_size[0] - sum(self.x_overlaps[:i])
                row_end = row_start + self.region_size[0]
                col_start = j * self.region_size[1] - sum(self.y_overlaps[:j])
                col_end = col_start + self.region_size[1]

                count_map[row_start:row_end, col_start:col_end] += POU_map

                regions.append((slice(row_start, row_end), slice(col_start, col_end)))
                POU_maps.append(POU_map)
        
        # re-normalize the POU_maps
        for i in range(len(regions)):
            POU_maps[i] = POU_maps[i] / count_map[regions[i]]

        return regions, torch.stack(POU_maps, dim=0)
    
    def partition(self, A):
        if self.periodic_padding:
            A_shape = A.shape
            padded_A = torch.nn.functional.pad(A.reshape((-1,A.shape[-2],A.shape[-1])), (0, self.y_overlaps[-1], 0, self.x_overlaps[-1]), mode='circular').reshape((*A_shape[:-2], A_shape[-2]+self.x_overlaps[-1], A_shape[-1]+self.y_overlaps[-1]))
        else:
            padded_A = A
        return [padded_A[region] for region in self.regions]

    def combine(self, partitions_x, scale_with_POU=True):
        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size
        POU_maps = self.POU_maps.to(partitions_x.device)

        reconstructed = torch.zeros(self.padded_shape, dtype = torch.complex64).to(partitions_x.device)

        if scale_with_POU:
            for i, r in enumerate(self.regions):
                reconstructed[r] += partitions_x[i] * POU_maps[i]
        else:
            for i, r in enumerate(self.regions):
                reconstructed[r] += partitions_x[i]

        return reconstructed[:self.unpadded_shape[0], :self.unpadded_shape[1]]

    def get_bcs(self, x, eps, Sx_f, Sy_f, wl, dL, bloch_phases=None, zero_global_bcs=False):
        assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)

        device = x.device
        if not torch.is_complex(Sx_f):
            Sx_f = 1 + 1j*Sx_f
        if not torch.is_complex(Sy_f):
            Sy_f = 1 + 1j*Sy_f
        
        if self.sommerfeld:
            assert self.global_bcs is not None
            Sx_f = torch.ones_like(Sx_f)
            Sy_f = torch.ones_like(Sy_f)

        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size

        patched_x = x.view(x_patches, y_patches, dsx, dsy)
        patched_eps = eps.view(x_patches, y_patches, dsx, dsy)

        patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.view(x_patches, y_patches, dsx, dsy)
        patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.view(x_patches, y_patches, dsx, dsy)

        top_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64, device=device)
        bottom_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64, device=device)
        left_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64, device=device)
        right_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64, device=device)

        # top bc
        roll = torch.roll(patched_x, 1, dims=0)
        for i in range(x_patches):
            eps = patched_eps_Sx[i:i+1,:,:1,:]
            ol = self.x_overlaps[i-1]
            
            bc = 1/2 * (roll[i:i+1,:,dsx-ol:dsx-ol+1,:] + roll[i:i+1,:,dsx-ol+1:dsx-ol+2,:])
            d_w = (roll[i:i+1,:,dsx-ol:dsx-ol+1,:] - roll[i:i+1,:,dsx-ol+1:dsx-ol+2,:])
            if i==0 and bloch_phases is not None and bloch_phases[0] != 0:
                bc[0] = bc[0]*np.exp(-1j*bloch_phases[0])
                d_w[0] = d_w[0]*np.exp(-1j*bloch_phases[0])
            top_bc[i:i+1,:,:,:] = robin(eps, bc, d_w=d_w, wl=wl, dL=dL)
        
        if not self.periodic_padding:
            top_bc[0:1,:,:,:] = 0 if zero_global_bcs else self.global_bcs[0]

        # bottom bc
        roll = torch.roll(patched_x, -1, dims=0)
        for i in range(x_patches):
            eps = patched_eps_Sx[i:i+1,:,-1:,:]
            ol = self.x_overlaps[i]
            
            bc = 1/2 * (roll[i:i+1,:,ol-1:ol,:] + roll[i:i+1,:,ol-2:ol-1,:])
            d_w = (roll[i:i+1,:,ol-1:ol,:] - roll[i:i+1,:,ol-2:ol-1,:])
            if i==x_patches-1 and bloch_phases is not None and bloch_phases[0] != 0:
                bc[-1] = bc[-1]*np.exp(1j*bloch_phases[0])
                d_w[-1] = d_w[-1]*np.exp(1j*bloch_phases[0])
            bottom_bc[i:i+1,:,:,:] = robin(eps, bc, d_w=d_w, wl=wl, dL=dL)
        
        if not self.periodic_padding:
            bottom_bc[-1:,:,:,:] = 0 if zero_global_bcs else self.global_bcs[1]

        # left bc
        roll = torch.roll(patched_x, 1, dims=1)
        for j in range(y_patches):
            eps = patched_eps_Sy[:,j:j+1,:,:1]
            ol = self.y_overlaps[j-1]
            
            bc = 1/2 * (roll[:,j:j+1,:,dsy-ol:dsy-ol+1] + roll[:,j:j+1,:,dsy-ol+1:dsy-ol+2])
            d_w = (roll[:,j:j+1,:,dsy-ol:dsy-ol+1] - roll[:,j:j+1,:,dsy-ol+1:dsy-ol+2])
            if j==0 and bloch_phases is not None and bloch_phases[1] != 0:
                bc[:,0] = bc[:,0]*np.exp(-1j*bloch_phases[1])
                d_w[:,0] = d_w[:,0]*np.exp(-1j*bloch_phases[1])
            left_bc[:,j:j+1,:,:] = robin(eps, bc, d_w=d_w, wl=wl, dL=dL)
        
        if not self.periodic_padding:
            left_bc[:,0:1,:,:] = 0 if zero_global_bcs else self.global_bcs[2]

        # right bc
        roll = torch.roll(patched_x, -1, dims=1)
        for j in range(y_patches):
            eps = patched_eps_Sy[:,j:j+1,:,-1:]
            ol = self.y_overlaps[j]
            
            bc = 1/2 * (roll[:,j:j+1,:,ol-1:ol] + roll[:,j:j+1,:,ol-2:ol-1])
            d_w = (roll[:,j:j+1,:,ol-1:ol] - roll[:,j:j+1,:,ol-2:ol-1])
            if j==y_patches-1 and bloch_phases is not None and bloch_phases[1] != 0:
                bc[:,-1] = bc[:,-1]*np.exp(1j*bloch_phases[1])
                d_w[:,-1] = d_w[:,-1]*np.exp(1j*bloch_phases[1])
            right_bc[:,j:j+1,:,:] = robin(eps, bc, d_w=d_w, wl=wl, dL=dL)
        
        if not self.periodic_padding:
            right_bc[:,-1:,:,:] = 0 if zero_global_bcs else self.global_bcs[3]

        return top_bc.reshape((x_patches*y_patches, 1, dsy)), \
               bottom_bc.reshape((x_patches*y_patches, 1, dsy)), \
               left_bc.reshape((x_patches*y_patches, dsx, 1)), \
               right_bc.reshape((x_patches*y_patches, dsx, 1))

    def get_current_bcs(self, x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=False):
        assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)

        if not torch.is_complex(Sx_f):
            Sx_f = 1 + 1j*Sx_f
        if not torch.is_complex(Sy_f):
            Sy_f = 1 + 1j*Sy_f

        if self.sommerfeld:
            assert self.global_bcs is not None
            Sx_f = torch.ones_like(Sx_f)
            Sy_f = torch.ones_like(Sy_f)

        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size

        patched_x = x.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)

        patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)
        # patched_eps_Sx = patched_eps
        # patched_eps_Sy = patched_eps

        top_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(x.device)
        bottom_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(x.device)
        left_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(x.device)
        right_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(x.device)

        # top bc
        top_bc = robin(patched_eps_Sx[:,:,:1,:], 1/2 * (patched_x[:,:,0:1,:] + patched_x[:,:,1:2,:]), d_w=(patched_x[:,:,0:1,:] - patched_x[:,:,1:2,:]), wl=wl, dL=dL)
        bottom_bc = robin(patched_eps_Sx[:,:,-1:,:], 1/2 * (patched_x[:,:,-1:,:] + patched_x[:,:,-2:-1,:]), d_w=(patched_x[:,:,-1:,:] - patched_x[:,:,-2:-1,:]), wl=wl, dL=dL)
        left_bc = robin(patched_eps_Sy[:,:,:,:1], 1/2 * (patched_x[:,:,:,0:1] + patched_x[:,:,:,1:2]), d_w=(patched_x[:,:,:,0:1] - patched_x[:,:,:,1:2]), wl=wl, dL=dL)
        right_bc = robin(patched_eps_Sy[:,:,:,-1:], 1/2 * (patched_x[:,:,:,-1:] + patched_x[:,:,:,-2:-1]), d_w=(patched_x[:,:,:,-1:] - patched_x[:,:,:,-2:-1]), wl=wl, dL=dL)

        if zero_global_bcs:
            top_bc[0:1,:,:,:] = 0
            bottom_bc[-1:,:,:,:] = 0
            left_bc[:,0:1,:,:] = 0
            right_bc[:,-1:,:,:] = 0

        return top_bc.reshape((x_patches*y_patches, 1, dsy)), \
               bottom_bc.reshape((x_patches*y_patches, 1, dsy)), \
               left_bc.reshape((x_patches*y_patches, dsx, 1)), \
               right_bc.reshape((x_patches*y_patches, dsx, 1))

    def get_bc_errors(self, x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=False, stack=False):
        assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)

        # if not periodic padding, meaning we have global bc, set global bc error to zeros
        a = self.get_bcs(x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=not self.periodic_padding)
        b = self.get_current_bcs(x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=zero_global_bcs)
        bc_errors = [i-j for i, j in zip(a, b)]
        if stack:
            return torch.stack([(i-j).squeeze() for i, j in zip(a, b)], dim=1)
        else:
            return [i-j for i, j in zip(a, b)]

    def set_global_bc(self, gt, eps, Sx_f, Sy_f, wl, dL):
        # if we don't assume periodic padding, we need to construct the global bc using groundtruth data
        assert not self.periodic_padding

        if not torch.is_complex(Sx_f):
            Sx_f = 1 + 1j*Sx_f
        if not torch.is_complex(Sy_f):
            Sy_f = 1 + 1j*Sy_f
        
        if self.sommerfeld:
            print("set zero global bc")
            Sx_f = torch.ones_like(Sx_f)
            Sy_f = torch.ones_like(Sy_f)
            gt = torch.zeros_like(gt)

        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size
        patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)
        patched_gt = gt.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)

        self.global_bcs = []

        # top bc
        eps = patched_eps_Sx[0:1,:,:1,:]
        bc = 1/2 * (patched_gt[0:1,:,0:1,:] + patched_gt[0:1,:,1:2,:])
        d_w = (patched_gt[0:1,:,0:1,:] - patched_gt[0:1,:,1:2,:])
        self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

        # bottom bc
        eps = patched_eps_Sx[-1:,:,-1:,:]
        bc = 1/2 * (patched_gt[-1:,:,-1:,:] + patched_gt[-1:,:,-2:-1,:])
        d_w = (patched_gt[-1:,:,-1:,:] - patched_gt[-1:,:,-2:-1,:])
        self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

        # left bc
        eps = patched_eps_Sy[:,0:1,:,:1]
        bc = 1/2 * (patched_gt[:,0:1,:,0:1] + patched_gt[:,0:1,:,1:2])
        d_w = (patched_gt[:,0:1,:,0:1] - patched_gt[:,0:1,:,1:2])
        self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

        # right bc
        eps = patched_eps_Sy[:,-1:,:,-1:]
        bc = 1/2 * (patched_gt[:,-1:,:,-1:] + patched_gt[:,-1:,:,-2:-1])
        d_w = (patched_gt[:,-1:,:,-1:] - patched_gt[:,-1:,:,-2:-1])
        self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))
    
    def get_bi_from_result(self, field, eps, dL, wl):
        bs, sx , sy = field.shape
        assert self.grid_shape[0] * self.grid_shape[1] == bs

        field = field.view(self.grid_shape[0], self.grid_shape[1], sx, sy)
        eps = eps.view(self.grid_shape[0], self.grid_shape[1], sx, sy)

        top_bc = torch.zeros(self.grid_shape[0], self.grid_shape[1], 1, sx, dtype=torch.complex64).cuda()
        bottom_bc = torch.zeros(self.grid_shape[0], self.grid_shape[1], 1, sx, dtype=torch.complex64).cuda()
        left_bc = torch.zeros(self.grid_shape[0], self.grid_shape[1], sy, 1, dtype=torch.complex64).cuda()
        right_bc = torch.zeros(self.grid_shape[0], self.grid_shape[1], sy, 1, dtype=torch.complex64).cuda()

        for i in range(1, self.grid_shape[0]):
            ol = self.x_overlaps[i-1]
            top_bc[i, :] = 1j*2*torch.pi*torch.sqrt(eps[i, :, ol-1:ol, :])*dL/wl*1/2*(field[i, :, ol-1:ol, :]+field[i, :, ol-2:ol-1, :]) + field[i, :, ol-1:ol, :]-field[i, :, ol-2:ol-1, :]
        for i in range(self.grid_shape[0]-1):
            ol = self.x_overlaps[i]
            bottom_bc[i, :] = 1j*2*torch.pi*torch.sqrt(eps[i, :, sx-ol:sx-ol+1, :])*dL/wl*1/2*(field[i, :, sx-ol:sx-ol+1, :]+field[i, :, sx-ol+1:sx-ol+2, :]) + field[i, :, sx-ol:sx-ol+1, :]-field[i, :, sx-ol+1:sx-ol+2, :]
        for i in range(1, self.grid_shape[1]):
            ol = self.y_overlaps[i-1]
            left_bc[:, i] = 1j*2*torch.pi*torch.sqrt(eps[:, i, :, ol-1:ol])*dL/wl*1/2*(field[:, i, :, ol-1:ol]+field[:, i, :, ol-2:ol-1]) + field[:, i, :, ol-1:ol]-field[:, i, :, ol-2:ol-1]
        for i in range(self.grid_shape[1]-1):
            ol = self.y_overlaps[i]
            right_bc[:, i] = 1j*2*torch.pi*torch.sqrt(eps[:, i, :, sy-ol:sy-ol+1])*dL/wl*1/2*(field[:, i, :, sy-ol:sy-ol+1]+field[:, i, :, sy-ol+1:sy-ol+2]) + field[:, i, :, sy-ol:sy-ol+1]-field[:, i, :, sy-ol+1:sy-ol+2]

        bi = torch.cat((top_bc.view(bs, -1), bottom_bc.view(bs, -1), left_bc.view(bs, -1), right_bc.view(bs, -1)), dim=1)
        return bi

    def prepare_coarse_space(self, solver, eps_batch, Sx_f_I, Sy_f_I, wl, dL, debug=False, output_dir=None):
        printc("Preparing coarse space...", "g")
        # (1) solve eigen value problem for the subdoamins, to get the basis vectors and eigen values
        printc("Solving eigen value problem for RtR Map of the subdomains...", "y")
        self.coarse_basis,self.eigen_values = self.prepare_RtR_arnoldi_bases(solver, eps_batch, wl, dL, debug=self.plot_basis_func, output_dir=output_dir)
        # coarse_basis also notated as Q

        # (2) prepare the bc_array, which contains the boundary mismatch info for the basis
        printc("Preparing bc_array...", "y")
        self.bc_array = self.Q_to_bc_array(self.coarse_basis, eps_batch, Sx_f_I, Sy_f_I, dL, wl)

        # (3) assemble the BQTBQ matrix:
        printc("Assembling BQTBQ matrix...", "y")
        if self.sparse_BQTBQ:
            self.BQTBQ, self.non_zero_basis_indices, self.BQTBQ_factor = self.assemble_BQTBQ_sparse(self.bc_array)
        else:
            self.BQTBQ, self.non_zero_basis_indices = self.assemble_BQTBQ(self.bc_array)

    def prepare_RtR_arnoldi_bases(self, solver, eps_batch, wl, dL, debug=False, output_dir=None):
        plane_wave_directions = [(np.cos(2*np.pi*i/self.eval_k_per_subdomain), np.sin(2*np.pi*i/self.eval_k_per_subdomain)) for i in range(self.eval_k_per_subdomain)]
        field_shape = (*eps_batch.shape,2)

        source_RI = torch.zeros(field_shape).cuda() # no need of source for the coarse space
        Sx_f_I = torch.zeros(eps_batch.shape).cuda() # use zero PML for now
        Sx_b_I = torch.zeros(eps_batch.shape).cuda()
        Sy_f_I = torch.zeros(eps_batch.shape).cuda()
        Sy_b_I = torch.zeros(eps_batch.shape).cuda()

        X = []
        Y = []
        B = []
        basis_fields = []

        for idx, plane_wave_direction in tqdm(enumerate(plane_wave_directions), desc="Preparing RtR bases", leave=False):
            top_bc_RI, bottom_bc_RI, left_bc_RI, right_bc_RI = plane_wave_bc(dL, wl, eps_batch, Sx_f_I, Sy_f_I, plane_wave_direction=plane_wave_direction, x_patches=self.grid_shape[0], y_patches=self.grid_shape[1], periodic=False)
            # top_bc_RI, bottom_bc_RI, left_bc_RI, right_bc_RI = random_fourier_bc(eps_batch, x_patches=self.grid_shape[0], y_patches=self.grid_shape[1])
            # top_bc_RI, bottom_bc_RI, left_bc_RI, right_bc_RI = random_bc(eps_batch, x_patches=self.grid_shape[0], y_patches=self.grid_shape[1])
        
            xi = bcs_to_vector(r2c(top_bc_RI), r2c(bottom_bc_RI), r2c(left_bc_RI), r2c(right_bc_RI))
            yi = gram_schmidt(Y, xi)
            # yi = xi
            X.append(xi)
            Y.append(yi)

            top_bc_RI, bottom_bc_RI, left_bc_RI, right_bc_RI = vector_to_bcs(yi)
            rhs, solution, relres_history, x_history, r_history = solver.solve((top_bc_RI, bottom_bc_RI, left_bc_RI, right_bc_RI))

            bi = self.get_bi_from_result(solution, eps_batch, dL, wl)
            B.append(bi)
            basis_fields.append(solution)

        Y = torch.stack(Y, dim=0).permute(1,2,0) # shape: bs, 4*sx, k
        B = torch.stack(B, dim=0).permute(1,2,0) # shape: bs, 4*sx, k

        H = torch.conj(Y.transpose(1,2)) @ B

        ############### (1) compute eigen values and vectors #################
        eigen_values, eigen_vectors = torch.linalg.eig(H)
        sorted_indices = torch.argsort(torch.abs(eigen_values), axis=1, descending=True)[:,:self.keep_k_per_subdomain]
        eigen_values = torch.gather(eigen_values,1,sorted_indices) # shape: bs, k
        eigen_vectors = torch.gather(eigen_vectors,2,sorted_indices.unsqueeze(1).repeat(1, eigen_vectors.shape[1], 1)) # shape: bs, k, k
        ############### (2) apply SVD: ###############
        # U, S, V = torch.linalg.svd(H)
        # eigen_values = S[:,:self.keep_k_per_subdomain]
        # eigen_vectors = V[:,:,:self.keep_k_per_subdomain]
        ##############################################
        
        basis_fields = torch.stack(basis_fields, dim=1) # shape (bs, k, sx, sy)
        coarse_basis = torch.zeros(eps_batch.shape[0], eigen_values.shape[1], eps_batch.shape[1], eps_batch.shape[2], 2).cuda()
        for i in range(eigen_values.shape[1]):
            field_RI = c2r(torch.sum(basis_fields*eigen_vectors[:,:,i,None,None], dim=1))
            field_RI = field_RI/(torch.mean(torch.abs(field_RI), dim=(1,2,3), keepdim=True)+1e-10)
            coarse_basis[:,i,:,:,:] = field_RI
            # coarse_basis[:,i,:,:,:] = field_RI*self.POU_maps[...,None]
        
        if debug:
            # pass
            assert output_dir is not None, "output_dir is required for logging debug plots"
            num_to_plot = 2
            plt.figure(figsize=((self.plot_basis_func_num+1)*2.8, 4*num_to_plot))
            for i in range(num_to_plot):
                for j in range(self.plot_basis_func_num+1):
                    plt.subplot(num_to_plot,self.plot_basis_func_num+1,i*(self.plot_basis_func_num+1)+j+1)
                    if j == 0:
                        colored_setup = setup_plot_data(eps_batch[i,:,:].cpu().numpy(),
                                                        np.zeros_like(eps_batch[i,:,:].cpu().numpy(), dtype=np.float32),
                                                        pml_th=0)
                        plt.imshow(colored_setup)
                        plt.xticks([])
                        plt.yticks([])
                    else:
                        vm = np.max(np.abs(torch.view_as_complex(coarse_basis[i,j-1,:,:,:]).real.cpu().numpy()))
                        plt.imshow(torch.view_as_complex(coarse_basis[i,j-1,:,:,:]).real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
                        plt.xticks([])
                        plt.yticks([])
                        if eigen_values is not None:
                            egv = eigen_values[i,j-1].cpu().numpy()
                            plt.title(f"eigenvalue:\n{egv.real:.1e}+{egv.imag:.1e}j")
                    # plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "Q_debug.png"), dpi=200)
            plt.close()
        
        return coarse_basis, eigen_values

    def Q_to_bc_array(self, Q, eps, sx_f, sy_f, dL, wl, periodic=False):
        if periodic:
            raise NotImplementedError("Periodic boundary conditions are not implemented yet")
        
        # Q shape: (model_bs, keep_k_per_subdomain, sx, sy, 2)
        model_bs, _, sx, sy, _ = Q.shape
        pd = 0 # optional padding
        assert self.region_size == (sx, sy)
        assert self.grid_shape[0]*self.grid_shape[1] == model_bs
        assert sx == sy

        if not torch.is_complex(sx_f):
            sx_f = 1+1j*sx_f
            sy_f = 1+1j*sy_f
        
        eps_with_sx = eps.to(torch.complex64) * sx_f
        eps_with_sy = eps.to(torch.complex64) * sy_f

        Ez = Q[:,:,:,:,0] + 1j*Q[:,:,:,:,1]

        bc_array = torch.zeros((self.grid_shape[0], self.grid_shape[1], self.keep_k_per_subdomain, 8, sx), dtype=torch.complex64).to(Q.device)
        Ez = Ez.reshape(self.grid_shape[0], self.grid_shape[1], self.keep_k_per_subdomain, sx, sy)
        eps_with_sx = eps_with_sx.reshape(self.grid_shape[0], self.grid_shape[1], sx, sy)
        eps_with_sy = eps_with_sy.reshape(self.grid_shape[0], self.grid_shape[1], sx, sy)

        # first compute all the bcs and store in an array:
        bc_array[:,:,:,0,:] -= (Ez[:,:,:, 0+pd,:]-Ez[:,:,:, 1+pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[:,:,None, 0+pd,:])/wl*dL*1/2*(Ez[:,:,:, 0+pd,:]+Ez[:,:,:, 1+pd,:])
        bc_array[:,:,:,1,:] -= (Ez[:,:,:,-1-pd,:]-Ez[:,:,:,-2-pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[:,:,None,-1-pd,:])/wl*dL*1/2*(Ez[:,:,:,-1-pd,:]+Ez[:,:,:,-2-pd,:])
        bc_array[:,:,:,2,:] -= (Ez[:,:,:,:, 0+pd]-Ez[:,:,:,:, 1+pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,:,None,:, 0+pd])/wl*dL*1/2*(Ez[:,:,:,:, 0+pd]+Ez[:,:,:,:, 1+pd])
        bc_array[:,:,:,3,:] -= (Ez[:,:,:,:,-1-pd]-Ez[:,:,:,:,-2-pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,:,None,:,-1-pd])/wl*dL*1/2*(Ez[:,:,:,:,-1-pd]+Ez[:,:,:,:,-2-pd])

        for i in range(1,self.grid_shape[0]):
            ol = self.x_overlaps[i-1]
            bc_array[i,:,:,4,:] += (Ez[i,:,:,pd+ol-1,:]-Ez[i,:,:,pd+ol-2,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i,:,None,pd+ol-1,:])/wl*dL*1/2*(Ez[i,:,:,pd+ol-1,:]+Ez[i,:,:,pd+ol-2,:])
        for i in range(self.grid_shape[0]-1):
            ol = self.x_overlaps[i]
            bc_array[i,:,:,5,:] += (Ez[i,:,:,sx-pd-ol,:]-Ez[i,:,:,sx-pd-ol+1,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i,:,None,sx-pd-ol,:])/wl*dL*1/2*(Ez[i,:,:,sx-pd-ol,:]+Ez[i,:,:,sx-pd-ol+1,:])
        for i in range(1,self.grid_shape[1]):
            ol = self.y_overlaps[i-1]
            bc_array[:,i,:,6,:] += (Ez[:,i,:,:,pd+ol-1]-Ez[:,i,:,:,pd+ol-2])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i,None,:,pd+ol-1])/wl*dL*1/2*(Ez[:,i,:,:,pd+ol-1]+Ez[:,i,:,:,pd+ol-2])
        for i in range(self.grid_shape[1]-1):
            ol = self.y_overlaps[i]
            bc_array[:,i,:,7,:] += (Ez[:,i,:,:,sy-pd-ol]-Ez[:,i,:,:,sy-pd-ol+1])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i,None,:,sy-pd-ol])/wl*dL*1/2*(Ez[:,i,:,:,sy-pd-ol]+Ez[:,i,:,:,sy-pd-ol+1])

        return bc_array.reshape(model_bs, self.keep_k_per_subdomain, 8, sx)

    def assemble_BQTBQ(self, bc_array):
        # bc_array shape: N, k, 8, sx
        # BQTBQ shape: N*k, N*k
        N, k, _, sx = bc_array.shape

        BQTBQ = torch.zeros((N,k, N,k), dtype=torch.complex64).to(bc_array.device)

        for i in range(N):
            r1, c1 = i // self.grid_shape[1], i % self.grid_shape[1]
            for j in range(N):
                r2, c2 = j // self.grid_shape[1], j % self.grid_shape[1]
                if r1 == r2 and c1 == c2:
                    BQTBQ[i,:,j,:] = torch.einsum('abc,dbc->ad', torch.conj(bc_array[i,:,:,:]), bc_array[j,:,:,:]) # (k, 8, sx), (k, 8, sx) -> (k, k)
                elif r2 == r1 and c2 == c1+1:
                    BQTBQ[i,:,j,:] = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,7,:]), bc_array[j,:,2,:]) + \
                                    torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,3,:]), bc_array[j,:,6,:]) # (k, sx), (k, sx) -> (k, k)
                elif r2 == r1 and c2 == c1-1:
                    BQTBQ[i,:,j,:] = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,6,:]), bc_array[j,:,3,:]) + \
                                    torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,2,:]), bc_array[j,:,7,:]) # (k, sx), (k, sx) -> (k, k)
                elif r2 == r1+1 and c2 == c1:
                    BQTBQ[i,:,j,:] = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,5,:]), bc_array[j,:,0,:]) + \
                                    torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,1,:]), bc_array[j,:,4,:]) # (k, sx), (k, sx) -> (k, k)
                elif r2 == r1-1 and c2 == c1:
                    BQTBQ[i,:,j,:] = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,4,:]), bc_array[j,:,1,:]) + \
                                    torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,0,:]), bc_array[j,:,5,:]) # (k, sx), (k, sx) -> (k, k)

        non_zero_basis_indices = torch.sum(torch.abs(bc_array.reshape(-1,8,sx)), dim=(1,2)) > 1e-9
        BQTBQ = BQTBQ.reshape(N*k, N*k)[non_zero_basis_indices, :][:, non_zero_basis_indices]
        return BQTBQ, non_zero_basis_indices
    
    def assemble_BQTBQ_sparse(self, bc_array):
        N, k, _, sx = bc_array.shape

        # BQTBQ = torch.zeros((N,k, N,k), dtype=torch.complex64).to(bc_array.device)
        data = []
        row_indices = []
        col_indices = []

        for i in range(N):
            r1, c1 = i // self.grid_shape[1], i % self.grid_shape[1]
            for j in range(N):
                r2, c2 = j // self.grid_shape[1], j % self.grid_shape[1]
                if r1 == r2 and c1 == c2:
                    val = torch.einsum('abc,dbc->ad', torch.conj(bc_array[i,:,:,:]), bc_array[j,:,:,:])
                elif r2 == r1 and c2 == c1+1:
                    val = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,7,:]), bc_array[j,:,2,:]) + \
                        torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,3,:]), bc_array[j,:,6,:])
                elif r2 == r1 and c2 == c1-1:
                    val = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,6,:]), bc_array[j,:,3,:]) + \
                        torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,2,:]), bc_array[j,:,7,:])
                elif r2 == r1+1 and c2 == c1:
                    val = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,5,:]), bc_array[j,:,0,:]) + \
                        torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,1,:]), bc_array[j,:,4,:])
                elif r2 == r1-1 and c2 == c1:
                    val = torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,4,:]), bc_array[j,:,1,:]) + \
                        torch.einsum('ab,cb->ac', torch.conj(bc_array[i,:,0,:]), bc_array[j,:,5,:])
                else:
                    continue  # Skip zero elements

                # Flatten and store values
                data.append(cp.asarray(val.flatten()))
                col_idx, row_idx = cp.meshgrid(cp.arange(j*k,(j+1)*k), cp.arange(i*k,(i+1)*k))
                row_indices.append(row_idx.flatten())
                col_indices.append(col_idx.flatten())

        data = cp.concatenate(data)
        row_indices = cp.concatenate(row_indices)
        col_indices = cp.concatenate(col_indices)
        BQTBQ_sparse = cpx.csr_matrix((data, (row_indices, col_indices)), shape=(N * bc_array.shape[1], N * bc_array.shape[1]), dtype=np.complex64)

        non_zero_basis_indices = torch.sum(torch.abs(bc_array.reshape(-1,8,sx)), dim=(1,2)) > 1e-9
        indices_cp = cp.asarray(non_zero_basis_indices.cpu())
        BQTBQ_sparse = BQTBQ_sparse[indices_cp, :][:, indices_cp]

        BQTBQ_factor = cpxl.splu(BQTBQ_sparse)

        return BQTBQ_sparse, non_zero_basis_indices, BQTBQ_factor
    
    
    def apply_BQTr(self, r):
        # self.bc_array shape: N, k, 8, sx
        # r shape: N, 4, sx
        N, k, _, sx = self.bc_array.shape
        assert self.grid_shape[0]*self.grid_shape[1] == N

        BQTr = torch.zeros((N, k), dtype=torch.complex64).to(self.bc_array.device)

        if self.periodic_padding:
            raise NotImplementedError("Periodic boundary condition is not implemented yet")
        else:
            for i in range(N):
                r1, c1 = i // self.grid_shape[1], i % self.grid_shape[1]
                BQTr[i,:] += torch.sum(torch.conj(self.bc_array[i,:,:4,:])*r[i,None, :,:], dim=(1,2))
                if c1 != self.grid_shape[1] - 1:
                    BQTr[i,:] += torch.sum(torch.conj(self.bc_array[i,:,7,:])*r[i+1,None, 2,:], dim=-1)
                if c1 != 0:
                    BQTr[i,:] += torch.sum(torch.conj(self.bc_array[i,:,6,:])*r[i-1,None, 3,:], dim=-1)
                if r1 != self.grid_shape[0] - 1:
                    BQTr[i,:] += torch.sum(torch.conj(self.bc_array[i,:,5,:])*r[i+self.grid_shape[1],None, 0,:], dim=-1)
                if r1 != 0:
                    BQTr[i,:] += torch.sum(torch.conj(self.bc_array[i,:,4,:])*r[i-self.grid_shape[1],None, 1,:], dim=-1)
        BQTr = BQTr.reshape(N * k)[self.non_zero_basis_indices]
        return BQTr
    
    def solve_x_coarse(self, BQTr):
        if self.sparse_BQTBQ:
            BQTr = cp.asarray(BQTr, dtype=cp.complex64)
            x_coarse = self.BQTBQ_factor.solve(BQTr)
            BQTBQx = self.BQTBQ @ x_coarse
            x_coarse = torch.as_tensor(x_coarse.get(), device="cuda")
        else:
            x_coarse = torch.linalg.solve(self.BQTBQ, BQTr)
            BQTBQx = self.BQTBQ @ x_coarse
        return x_coarse, BQTBQx
        
    def reassemble_x_coarse(self, x_coarse):
        # x_coarse shape: l, 1
        # non_zero_basis_indices: binary array of shape: bs*k
        # l < bs*k, since there are zero basis functions.
        result = torch.zeros_like(self.non_zero_basis_indices, dtype=x_coarse.dtype).to(x_coarse.device)
        result[self.non_zero_basis_indices] = x_coarse
        result = result.reshape(-1, self.coarse_basis.shape[1])
        return result
    
    def update_x_from_coarse_solve(self, x_coarse, x):
        x_fine = x_coarse[:,:,None,None] * torch.view_as_complex(self.coarse_basis)
        x_fine = torch.sum(x_fine, dim=1, keepdim=False)
        new_x = x + self.coarse_space_momentum*x_fine
        return new_x

    def update_bc_from_coarse_solve(self, x_coarse, top_bc, bottom_bc, left_bc, right_bc):
        # x_coarse shape: N, k
        # bc_array shape: N, k, 8, sx
        N, k, _, sx = self.bc_array.shape

        summed_bc_error = torch.sum(x_coarse[:,:,None,None] * self.bc_array, dim=1) # shape (N, 8, sx)
        summed_bc_error = summed_bc_error.reshape(self.grid_shape[0], self.grid_shape[1], 8, sx)
        
        d_top_bc = torch.roll(summed_bc_error, shifts=(1,0), dims=(0,1))[:,:,5,:]
        d_bottom_bc = torch.roll(summed_bc_error, shifts=(-1,0), dims=(0,1))[:,:,4,:]
        d_left_bc = torch.roll(summed_bc_error, shifts=(0,1), dims=(0,1))[:,:,7,:]
        d_right_bc = torch.roll(summed_bc_error, shifts=(0,-1), dims=(0,1))[:,:,6,:]

        top_bc = top_bc + self.coarse_space_momentum*d_top_bc.reshape(N, 1, sx)
        bottom_bc = bottom_bc + self.coarse_space_momentum*d_bottom_bc.reshape(N, 1, sx)
        left_bc = left_bc + self.coarse_space_momentum*d_left_bc.reshape(N, sx, 1)
        right_bc = right_bc + self.coarse_space_momentum*d_right_bc.reshape(N, sx, 1)

        if not self.periodic_padding:
            top_bc = top_bc.reshape((self.grid_shape[0], self.grid_shape[1], 1, sx))
            bottom_bc = bottom_bc.reshape((self.grid_shape[0], self.grid_shape[1], 1, sx))
            left_bc = left_bc.reshape((self.grid_shape[0], self.grid_shape[1], sx, 1))
            right_bc = right_bc.reshape((self.grid_shape[0], self.grid_shape[1], sx, 1))

            top_bc[0:1,:,:,:] = self.global_bcs[0]
            bottom_bc[-1:,:,:,:] = self.global_bcs[1]
            left_bc[:,0:1,:,:] = self.global_bcs[2]
            right_bc[:,-1:,:,:] = self.global_bcs[3]

            top_bc = top_bc.reshape((self.grid_shape[0]*self.grid_shape[1], 1, sx))
            bottom_bc = bottom_bc.reshape((self.grid_shape[0]*self.grid_shape[1], 1, sx))
            left_bc = left_bc.reshape((self.grid_shape[0]*self.grid_shape[1], sx, 1))
            right_bc = right_bc.reshape((self.grid_shape[0]*self.grid_shape[1], sx, 1))

        return top_bc, bottom_bc, left_bc, right_bc
    
    def bc_error_vector_to_map(self, bc_error_vector, shape):
        bc_error_map = torch.zeros(shape, dtype=torch.complex64, device=bc_error_vector.device)
        bc_error_map[:,0,:] = bc_error_vector[:,0,:]
        bc_error_map[:,-1,:] = bc_error_vector[:,1,:]
        bc_error_map[:,:,0] = bc_error_vector[:,2,:]
        bc_error_map[:,:,-1] = bc_error_vector[:,3,:]
        return self.combine(bc_error_map, scale_with_POU=False)
    
    def sum_BQ_xcoarse(self, x_coarse):
        # bc_array shape: N, k, 8, sx
        # x_coarse shape: N, k

        N, k, _, sx = self.bc_array.shape
        assert x_coarse.shape == (N, k)
        bc_error = torch.zeros((N, 4, sx), dtype=torch.complex64).to(self.bc_array.device)

        for i in range(N):
            r1, c1 = i // self.grid_shape[1], i % self.grid_shape[1]
            bc_error[i,:,:] += torch.sum(self.bc_array[i,:,:4,:]*x_coarse[i,:,None,None], dim=0)
            if r1 != 0:
                bc_error[i,0,:] += torch.sum(self.bc_array[i-self.grid_shape[1],:,5,:]*x_coarse[i-self.grid_shape[1],:,None], dim=0)
            if r1 != self.grid_shape[0] - 1:
                bc_error[i,1,:] += torch.sum(self.bc_array[i+self.grid_shape[1],:,4,:]*x_coarse[i+self.grid_shape[1],:,None], dim=0)
            if c1 != 0:
                bc_error[i,2,:] += torch.sum(self.bc_array[i-1,:,7,:]*x_coarse[i-1,:,None], dim=0)
            if c1 != self.grid_shape[1] - 1:
                bc_error[i,3,:] += torch.sum(self.bc_array[i+1,:,6,:]*x_coarse[i+1,:,None], dim=0)
        return bc_error
    
    def coarse_space_debug_plot(self, i, x_coarse, x, BQTr, BQTBQx, eps_batch, y_batch, Sx_f_batch_I, Sy_f_batch_I, dL, wl, output_dir):
        try:
            import matplotlib.font_manager as font_manager
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            font_dirs = ['/home/chenkaim/fonts/Microsoft_Aptos_Fonts']
            font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
        except:
            pass

        plt.rcParams.update({
            'font.family': 'Aptos'
        })

        x_fine = x_coarse[:,:,None,None] * torch.view_as_complex(self.coarse_basis)
        x_fine = torch.sum(x_fine, dim=1, keepdim=False)
        new_x = x + x_fine

        combined = self.combine(x)
        coarse_space_correction = self.combine(x_fine)
        new_combined = self.combine(new_x)

        x_bc_error = self.get_bc_errors(x, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        x_bc_error_map = self.bc_error_vector_to_map(x_bc_error, x.shape)

        x_fine_bc_error = self.get_bc_errors(x_fine, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        x_fine_bc_error_map = self.bc_error_vector_to_map(x_fine_bc_error, x.shape)

        error_batch = y_batch.to(x.device) - x
        error_bc_error = self.get_bc_errors(error_batch, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        error_bc_error_map = self.bc_error_vector_to_map(error_bc_error, x.shape)

        new_x_bc_error = self.get_bc_errors(new_x, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        new_x_bc_error_map = self.bc_error_vector_to_map(new_x_bc_error, x.shape)

        def math_bold(s: str) -> str:
            return r"$\bf{" + s.replace(" ", r"\ ") + "}$"
                        
        gt = self.combine(y_batch).real.cpu().numpy()
        vm = np.max(np.abs(gt))
        
        plt.figure(figsize=(28, 4))
        plt.subplot(1,7,1)
        total_eps = self.combine(eps_batch).real.cpu().numpy()
        colored_setup = setup_plot_data(total_eps,
                                        np.zeros_like(total_eps, dtype=np.float32),
                                        pml_th=0)

        plt.imshow(colored_setup)
        plt.xticks([])
        plt.yticks([])
        plt.title(r"$\bf{Epsilon,\ step\ }$" + f"{math_bold(str(i))}")
        
        ax = plt.subplot(1,7,2)
        im = ax.imshow(combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{One\ level\ solve}$" + f"\nBoundary error: {torch.mean(torch.abs(x_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = plt.subplot(1,7,3)
        vm = np.max(np.abs(gt-combined.real.cpu().numpy()))
        im = ax.imshow(gt-combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{One\ level\ field\ error}$")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = plt.subplot(1,7,4)
        vm = np.max(np.abs(coarse_space_correction.cpu().numpy()))
        im = ax.imshow(coarse_space_correction.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{Coarse\ correction}$" + f"\nBoundary error: {torch.mean(torch.abs(x_fine_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = plt.subplot(1,7,5)
        vm = np.max(np.abs(gt))
        im = ax.imshow(new_combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{Two\ level\ solve}$" + f"\nBoundary error: {torch.mean(torch.abs(new_x_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = plt.subplot(1,7,6)
        vm = np.max(np.abs(gt-new_combined.real.cpu().numpy()))
        im = ax.imshow(gt - new_combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{Two\ level\ field\ error}$")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        ax = plt.subplot(1,7,7)
        vm = np.max(np.abs(gt))
        im = ax.imshow(gt, cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(r"$\bf{Ground\ truth}$")
        plt.xticks([])
        plt.yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_fine_debug_{i}.png"), dpi=300, pad_inches=0.1)
        plt.close()

        BQT_r_fine = self.apply_BQTr(x_fine_bc_error)
        BQT_r_fine = BQT_r_fine.reshape(y_batch.shape[0], self.keep_k_per_subdomain)

        Qx_summed_bc_error = self.sum_BQ_xcoarse(x_coarse)
        BQT_Qx_summed_r_fine = self.apply_BQTr(Qx_summed_bc_error)
        BQT_Qx_summed_r_fine = BQT_Qx_summed_r_fine.reshape(y_batch.shape[0], self.keep_k_per_subdomain)

        # print("rel error of BQx and r: ", torch.mean(torch.abs(BQx - (-global_bc_error).reshape(-1)))/torch.mean(torch.abs(-global_bc_error)))
        print("rel error 12: ", torch.mean(torch.abs(BQTBQx - BQTr))/torch.mean(torch.abs(BQTr)))
        print("rel error 34: ", torch.mean(torch.abs(BQT_r_fine - BQT_Qx_summed_r_fine))/torch.mean(torch.abs(BQT_Qx_summed_r_fine)))
        print("rel error 23: ", torch.mean(torch.abs(BQTBQx.reshape(y_batch.shape[0], self.keep_k_per_subdomain) - BQT_r_fine))/torch.mean(torch.abs(BQT_r_fine)))

        plt.figure(figsize=(15,3))
        plt.subplot(1,5,1)
        plt.imshow(BQTr.real.cpu().numpy().reshape(y_batch.shape[0], self.keep_k_per_subdomain))
        plt.title("BQTr")
        plt.colorbar()
        plt.subplot(1,5,2)
        plt.imshow(x_coarse.real.cpu().numpy())
        plt.title("x_coarse")
        plt.colorbar()
        plt.subplot(1,5,3)
        plt.imshow(BQTBQx.reshape(y_batch.shape[0], self.keep_k_per_subdomain).real.cpu().numpy())
        plt.title("BQTBQx")
        plt.colorbar()
        plt.subplot(1,5,4)
        plt.imshow(BQT_r_fine.real.cpu().numpy())
        plt.title("BQT_r_fine")
        plt.colorbar()
        plt.subplot(1,5,5)
        plt.imshow(BQT_Qx_summed_r_fine.real.cpu().numpy())
        plt.title("BQT_Qx_summed_r_fine")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_coarse_debug_{i}.png"), dpi=200)
        plt.close()


    def coarse_space_debug_plot1(self, i, x_coarse, x, BQTr, BQTBQx, eps_batch, y_batch, Sx_f_batch_I, Sy_f_batch_I, dL, wl, output_dir):
        x_fine = x_coarse[:,:,None,None] * torch.view_as_complex(self.coarse_basis)
        x_fine = torch.sum(x_fine, dim=1, keepdim=False)
        new_x = x + x_fine

        combined = self.combine(x)
        coarse_space_correction = self.combine(x_fine)
        new_combined = self.combine(new_x)

        x_bc_error = self.get_bc_errors(x, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        x_bc_error_map = self.bc_error_vector_to_map(x_bc_error, x.shape)

        x_fine_bc_error = self.get_bc_errors(x_fine, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        x_fine_bc_error_map = self.bc_error_vector_to_map(x_fine_bc_error, x.shape)

        error_batch = y_batch.to(x.device) - x
        error_bc_error = self.get_bc_errors(error_batch, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        error_bc_error_map = self.bc_error_vector_to_map(error_bc_error, x.shape)

        new_x_bc_error = self.get_bc_errors(new_x, eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True, zero_global_bcs=True)
        new_x_bc_error_map = self.bc_error_vector_to_map(new_x_bc_error, x.shape)

        # gt_bc_error = self.get_bc_errors(y_batch.to(x.device), eps_batch, Sx_f_batch_I, Sy_f_batch_I, wl, dL, stack=True)
        # gt_bc_error_map = self.bc_error_vector_to_map(gt_bc_error, x.shape)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(gt_bc_error_map.real.cpu().numpy(), cmap='seismic')
        # plt.colorbar()
        # plt.savefig(os.path.join(output_dir, f"gt_bc_error_map_{i}.png"), dpi=400)
        # plt.close()
                        
        gt = self.combine(y_batch).real.cpu().numpy()
        vm = np.max(np.abs(gt))
        
        plt.figure(figsize=(18, 6))
        plt.subplot(2,5,1)
        total_eps = self.combine(eps_batch).real.cpu().numpy()
        colored_setup = setup_plot_data(total_eps,
                                        np.zeros_like(total_eps, dtype=np.float32),
                                        pml_th=0)

        plt.imshow(colored_setup)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("eps")
        plt.subplot(2,5,2)
        plt.imshow(combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title("x\nlocal_solve")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.subplot(2,5,3)
        plt.imshow(gt-combined.real.cpu().numpy(), cmap='seismic')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("error\n(gt-x)")
        plt.subplot(2,5,4)
        vm = np.max(np.abs(coarse_space_correction.cpu().numpy()))
        plt.imshow(coarse_space_correction.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("coarse space\ncorrection")
        plt.subplot(2,5,5)
        vm = np.max(np.abs(gt))
        plt.imshow(new_combined.real.cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("final solution\n(x + coarse_space_correction)")
        plt.subplot(2,5,6)
        vm = np.max(np.abs(gt))
        plt.imshow(gt, cmap='seismic', vmax=vm, vmin=-vm)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("gt")
        plt.subplot(2,5,7)
        vm = np.max(np.abs(x_bc_error_map.cpu().numpy()))
        plt.imshow(torch.real(x_bc_error_map).cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(f"x_bc_error_map\nmean residual:\n{torch.mean(torch.abs(x_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.subplot(2,5,8)
        plt.imshow(torch.real(error_bc_error_map).cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(f"error_bc_error_map\nmean residual:\n{torch.mean(torch.abs(error_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.subplot(2,5,9)
        plt.imshow(torch.real(x_fine_bc_error_map).cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(f"x_fine_bc_error_map\nmean residual:\n{torch.mean(torch.abs(x_fine_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.subplot(2,5,10)
        plt.imshow(torch.real(new_x_bc_error_map).cpu().numpy(), cmap='seismic', vmax=vm, vmin=-vm)
        plt.title(f"new_x_bc_error_map\nmean residual:\n{torch.mean(torch.abs(new_x_bc_error_map)):.2e}")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_fine_debug_{i}.png"), dpi=200)
        plt.close()


        BQT_r_fine = self.apply_BQTr(x_fine_bc_error)
        BQT_r_fine = BQT_r_fine.reshape(y_batch.shape[0], self.keep_k_per_subdomain)

        Qx_summed_bc_error = self.sum_BQ_xcoarse(x_coarse)
        BQT_Qx_summed_r_fine = self.apply_BQTr(Qx_summed_bc_error)
        BQT_Qx_summed_r_fine = BQT_Qx_summed_r_fine.reshape(y_batch.shape[0], self.keep_k_per_subdomain)

        # print("rel error of BQx and r: ", torch.mean(torch.abs(BQx - (-global_bc_error).reshape(-1)))/torch.mean(torch.abs(-global_bc_error)))
        print("rel error 12: ", torch.mean(torch.abs(BQTBQx - BQTr))/torch.mean(torch.abs(BQTr)))
        print("rel error 34: ", torch.mean(torch.abs(BQT_r_fine - BQT_Qx_summed_r_fine))/torch.mean(torch.abs(BQT_Qx_summed_r_fine)))
        print("rel error 23: ", torch.mean(torch.abs(BQTBQx.reshape(y_batch.shape[0], self.keep_k_per_subdomain) - BQT_r_fine))/torch.mean(torch.abs(BQT_r_fine)))

        plt.figure(figsize=(15,3))
        plt.subplot(1,5,1)
        plt.imshow(BQTr.real.cpu().numpy().reshape(y_batch.shape[0], self.keep_k_per_subdomain))
        plt.title("BQTr")
        plt.colorbar()
        plt.subplot(1,5,2)
        plt.imshow(x_coarse.real.cpu().numpy())
        plt.title("x_coarse")
        plt.colorbar()
        plt.subplot(1,5,3)
        plt.imshow(BQTBQx.reshape(y_batch.shape[0], self.keep_k_per_subdomain).real.cpu().numpy())
        plt.title("BQTBQx")
        plt.colorbar()
        plt.subplot(1,5,4)
        plt.imshow(BQT_r_fine.real.cpu().numpy())
        plt.title("BQT_r_fine")
        plt.colorbar()
        plt.subplot(1,5,5)
        plt.imshow(BQT_Qx_summed_r_fine.real.cpu().numpy())
        plt.title("BQT_Qx_summed_r_fine")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_coarse_debug_{i}.png"), dpi=200)
        plt.close()
