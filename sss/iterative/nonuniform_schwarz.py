# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import math
import numpy as np
import torch

import gin

def robin(eps, bc, d_w, wl, dL):
    # bc: the boundary field to be transform
    # d_w: the derivative of fields in w
    g = 1j*2*np.pi*torch.sqrt(eps)/wl*dL*bc+d_w
    return g

@gin.configurable
class NonUniformSchwarz:
    """
    non-uniform partitioner for the schwarz method
    primarily designed for dealing with PML region, e.g. apply direct solve for PML regions
    -----------------------------------------------------
    |PML|        PML          |         PML         |PML|
    |---------------------------------------------------|
    |   |                     |                     |   |
    |   |                     |                     |   |
    |PML|      regular        |       regular       |PML|
    |   |      domain         |       domain        |   |
    |   |                     |                     |   |
    |   |                     |                     |   |
    |---------------------------------------------------|
    |   |                     |                     |   |
    |   |                     |                     |   |
    |PML|      regular        |       regular       |PML|
    |   |      domain         |       domain        |   |
    |   |                     |                     |   |
    |   |                     |                     |   |
    |---------------------------------------------------|
    |PML|         PML         |         PML         |PML|
    -----------------------------------------------------
    """

    def __init__(
        self,
        unpadded_shape,
        min_overlap,
        region_size,
        extra_pml_pixels = 1
    ):
        self.unpadded_shape = unpadded_shape
        self.min_overlap = min_overlap
        self.region_size = region_size
        self.extra_pml_pixels = extra_pml_pixels
        self.periodic_padding = True # compatibility with other solvers
        
    def init_grid(self, pml_thickness):
        # first process x direction:
        pml_x = pml_thickness[0]
        if pml_x > 0:
            center_length = self.unpadded_shape[0] - 2*pml_x - 2*self.extra_pml_pixels # extra pixels to completely remove PML in regular regions
            center_rows = math.ceil((center_length-self.min_overlap)/(self.region_size[0]-self.min_overlap))
            total_gap_x = center_rows * (self.region_size[0]-self.min_overlap) + self.min_overlap - center_length
            num_overlaps_x = center_rows-1
            x_overlaps = [self.min_overlap + round((i+1)*total_gap_x/num_overlaps_x) - round(i*total_gap_x/num_overlaps_x) for i in range(num_overlaps_x)]
            # first min_overlap for PML, last two min_overlaps for last subdomain in the center, and PML in the end
            x_overlaps = [self.min_overlap] + x_overlaps + [self.min_overlap, self.min_overlap]
            rows = center_rows + 2
        else:
            rows = math.ceil(self.unpadded_shape[0]/(self.region_size[0]-self.min_overlap))
            total_gap_x = rows * (self.region_size[0]-self.min_overlap) - self.unpadded_shape[0]
            num_overlaps_x = rows
            x_overlaps = [self.min_overlap + round((i+1)*total_gap_x/num_overlaps_x) - round(i*total_gap_x/num_overlaps_x) for i in range(num_overlaps_x)]
        
        # then process y direction:
        pml_y = pml_thickness[1]
        if pml_y > 0:
            center_length = self.unpadded_shape[1] - 2*pml_y - 2*self.extra_pml_pixels # extra pixels to completely remove PML in regular regions
            center_cols = math.ceil((center_length-self.min_overlap)/(self.region_size[1]-self.min_overlap))
            total_gap_y = center_cols * (self.region_size[1]-self.min_overlap) + self.min_overlap - center_length
            num_overlaps_y = center_cols-1
            y_overlaps = [self.min_overlap + round((i+1)*total_gap_y/num_overlaps_y) - round(i*total_gap_y/num_overlaps_y) for i in range(num_overlaps_y)]
            y_overlaps = [self.min_overlap] + y_overlaps + [self.min_overlap, self.min_overlap]
            cols = center_cols + 2
        else:
            cols = math.ceil(self.unpadded_shape[1]/(self.region_size[1]-self.min_overlap))
            total_gap_y = cols * (self.region_size[1]-self.min_overlap) - self.unpadded_shape[1]
            num_overlaps_y = cols
            y_overlaps = [self.min_overlap + round((i+1)*total_gap_y/num_overlaps_y) - round(i*total_gap_y/num_overlaps_y) for i in range(num_overlaps_y)]

        self.pml_thickness = pml_thickness
        self.x_overlaps = x_overlaps
        self.y_overlaps = y_overlaps
        self.grid_shape = (rows, cols)
        self.padded_shape = (self.unpadded_shape[0] + x_overlaps[-1], self.unpadded_shape[1] + y_overlaps[-1])
        print(f"overlaps: {self.x_overlaps}, {self.y_overlaps}, grid_shape: {self.grid_shape}, padded_shape: {self.padded_shape}")
        self.init_regions()

    def init_regions(self):
        num_rows, num_cols = self.grid_shape
        regions = []
        POU_maps = []
        count_map = torch.zeros(self.padded_shape)

        PML_indices = []
        regular_indices = []

        index = 0
        row_start = 0
        for i in range(num_rows):
            col_start = 0
            if self.pml_thickness[0] > 0 and i == 0:
                sx = self.pml_thickness[0] + self.extra_pml_pixels + self.x_overlaps[0]
            elif self.pml_thickness[0] > 0 and i == num_rows-1:
                # attention: last pml is larger than first pml since it wraps around
                sx = self.pml_thickness[0] + self.extra_pml_pixels + self.x_overlaps[-2] + self.x_overlaps[-1]
            else:
                sx = self.region_size[0]
            for j in range(num_cols):
                # Calculate slices for the current region
                if self.pml_thickness[1] > 0 and j == 0:
                    sy = self.pml_thickness[1] + self.extra_pml_pixels + self.y_overlaps[0]
                elif self.pml_thickness[1] > 0 and j == num_cols-1:
                    # attention: last pml is larger than first pml since it wraps around
                    sy = self.pml_thickness[1] + self.extra_pml_pixels + self.y_overlaps[-2] + self.y_overlaps[-1]
                else:
                    sy = self.region_size[1]
                
                POU_map = torch.ones(sx, sy)

                count_map[row_start:row_start+sx, col_start:col_start+sy] += POU_map

                if self.pml_thickness[0] > 0 and i in [0, num_rows-1] or self.pml_thickness[1] > 0 and j in [0, num_cols-1]:
                    PML_indices.append(index)
                else:
                    regular_indices.append(index)

                regions.append((slice(row_start, row_start+sx), slice(col_start, col_start+sy)))
                POU_maps.append(POU_map)
                col_start += sy-self.y_overlaps[j]
                index += 1
            row_start += sx-self.x_overlaps[i]
        
        # re-normalize the POU_maps
        for i in range(len(regions)):
            POU_maps[i] = POU_maps[i] / count_map[regions[i]]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(count_map.cpu().numpy())
        plt.colorbar()
        plt.savefig("count_map.png")
        plt.close()
        self.regions = regions
        self.POU_maps = POU_maps
        self.PML_indices = PML_indices
        self.regular_indices = regular_indices
        print(f"Regions initialized.")
        self.print_regions()
    
    def print_regions(self):
        rx, ry = self.grid_shape
        for i in range(rx):
            for j in range(ry):
                idx = i*ry+j
                if idx in self.PML_indices:
                    print("PML\t", end="")
                else:
                    print("REG\t", end="")
            print()

    def split(self, list_data):
        return torch.stack([list_data[i] for i in self.regular_indices], dim=0), [list_data[i] for i in self.PML_indices]
    
    def assemble(self, list_regular, list_pml):
        total_list = []
        regular_idx = 0
        pml_idx = 0
        for i in range(len(self.regular_indices) + len(self.PML_indices)):
            if i in self.PML_indices:
                total_list.append(list_pml[pml_idx])
                pml_idx += 1
            else:
                total_list.append(list_regular[regular_idx])
                regular_idx += 1
        return total_list
    
    def partition(self, A):
        A_shape = A.shape
        padded_A = torch.nn.functional.pad(A.reshape((-1,A.shape[-2],A.shape[-1])), (0, self.y_overlaps[-1], 0, self.x_overlaps[-1]), mode='circular').reshape((*A_shape[:-2], A_shape[-2]+self.x_overlaps[-1], A_shape[-1]+self.y_overlaps[-1]))
        all_regions = [padded_A[region] for region in self.regions]
        return self.split(all_regions)

    def combine(self, x, x_pml, scale_with_POU=True):
        total_x = self.assemble(x, x_pml)
        POU_maps = self.POU_maps

        reconstructed = torch.zeros(self.padded_shape, dtype = torch.complex64)

        if scale_with_POU:
            for i, r in enumerate(self.regions):
                reconstructed[r] += total_x[i] * POU_maps[i]
        else:
            for i, r in enumerate(self.regions):
                reconstructed[r] += total_x[i]

        return reconstructed[:self.unpadded_shape[0], :self.unpadded_shape[1]]

    def get_bcs(self, x, x_pml, eps, eps_pml, Sx_f_I, Sx_f_I_pml, Sy_f_I, Sy_f_I_pml, wl, dL, PML_descale=False):
        device = x.device
        total_x = self.assemble(x, x_pml)
        total_eps = self.assemble(eps, eps_pml)
        total_Sx_f_I = self.assemble(Sx_f_I, Sx_f_I_pml)
        total_Sy_f_I = self.assemble(Sy_f_I, Sy_f_I_pml)

        x_patches, y_patches = self.grid_shape

        total_top_bcs = [None] * x_patches * y_patches
        total_bottom_bcs = [None] * x_patches * y_patches
        total_left_bcs = [None] * x_patches * y_patches
        total_right_bcs = [None] * x_patches * y_patches

        def idx(i, j):
            return i * y_patches + j

        for i in range(x_patches):
            for j in range(y_patches):
                p = idx(i, j)
                
                x = total_x[p]
                eps = total_eps[p]
                sx_f_I = 1+1j*total_Sx_f_I[p]
                sy_f_I = 1+1j*total_Sy_f_I[p]

                if PML_descale:
                    descale_map = torch.abs(sx_f_I) * torch.abs(sy_f_I)
                nx, ny = x.shape

                # top bc, bottom for neighbor
                ol = self.x_overlaps[(i-1) % x_patches]
                eps_slice = (eps.to(torch.complex64) * sx_f_I)[ol-1:ol,:]
                bc = 1/2 * (x[ol-1:ol,:] + x[ol-2:ol-1,:])
                d_w = (x[ol-1:ol,:] - x[ol-2:ol-1,:])
                total_bottom_bcs[idx((i-1) % x_patches, j)] = robin(eps_slice, bc, d_w=d_w, wl=wl, dL=dL)
                if PML_descale:
                    total_bottom_bcs[idx((i-1) % x_patches, j)] /= descale_map[ol-1:ol,:]

                # bottom bc, top for neighbor
                ol = self.x_overlaps[i]
                eps_slice = (eps.to(torch.complex64) * sx_f_I)[nx-ol:nx-ol+1,:]
                bc = 1/2 * (x[nx-ol:nx-ol+1,:] + x[nx-ol+1:nx-ol+2,:])
                d_w = (x[nx-ol:nx-ol+1,:] - x[nx-ol+1:nx-ol+2,:])
                total_top_bcs[idx((i+1) % x_patches, j)] = robin(eps_slice, bc, d_w=d_w, wl=wl, dL=dL)
                if PML_descale:
                    total_top_bcs[idx((i+1) % x_patches, j)] /= descale_map[nx-ol:nx-ol+1,:]

                # left bc, right for neighbor
                ol = self.y_overlaps[(j-1) % y_patches]
                eps_slice = (eps.to(torch.complex64) * sy_f_I)[:,ol-1:ol]
                bc = 1/2 * (x[:,ol-1:ol] + x[:,ol-2:ol-1])
                d_w = (x[:,ol-1:ol] - x[:,ol-2:ol-1])
                total_right_bcs[idx(i, (j-1) % y_patches)] = robin(eps_slice, bc, d_w=d_w, wl=wl, dL=dL)
                if PML_descale:
                    total_right_bcs[idx(i, (j-1) % y_patches)] /= descale_map[:,ol-1:ol]

                # right bc, left for neighbor
                ol = self.y_overlaps[j]
                eps_slice = (eps.to(torch.complex64) * sy_f_I)[:,ny-ol:ny-ol+1]
                bc = 1/2 * (x[:,ny-ol:ny-ol+1] + x[:,ny-ol+1:ny-ol+2])
                d_w = (x[:,ny-ol:ny-ol+1] - x[:,ny-ol+1:ny-ol+2])
                total_left_bcs[idx(i, (j+1) % y_patches)] = robin(eps_slice, bc, d_w=d_w, wl=wl, dL=dL)
                if PML_descale:
                    total_left_bcs[idx(i, (j+1) % y_patches)] /= descale_map[:,ny-ol:ny-ol+1]

        top_bc, top_bc_pml = self.split(total_top_bcs)
        bottom_bc, bottom_bc_pml = self.split(total_bottom_bcs)
        left_bc, left_bc_pml = self.split(total_left_bcs)
        right_bc, right_bc_pml = self.split(total_right_bcs)

        return top_bc, bottom_bc, left_bc, right_bc, top_bc_pml, bottom_bc_pml, left_bc_pml, right_bc_pml

    # def get_current_bcs(self, x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=False):
    #     assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)

    #     x_patches, y_patches = self.grid_shape
    #     dsx, dsy = self.region_size

    #     patched_x = x.reshape(x_patches, y_patches, dsx, dsy)
    #     patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)

    #     # patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
    #     # patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)
    #     patched_eps_Sx = patched_eps
    #     patched_eps_Sy = patched_eps

    #     top_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(x.device)
    #     bottom_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(x.device)
    #     left_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(x.device)
    #     right_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(x.device)

    #     # top bc
    #     top_bc = robin(patched_eps_Sx[:,:,:1,:], 1/2 * (patched_x[:,:,0:1,:] + patched_x[:,:,1:2,:]), d_w=(patched_x[:,:,0:1,:] - patched_x[:,:,1:2,:]), wl=wl, dL=dL)
    #     bottom_bc = robin(patched_eps_Sx[:,:,-1:,:], 1/2 * (patched_x[:,:,-1:,:] + patched_x[:,:,-2:-1,:]), d_w=(patched_x[:,:,-1:,:] - patched_x[:,:,-2:-1,:]), wl=wl, dL=dL)
    #     left_bc = robin(patched_eps_Sy[:,:,:,:1], 1/2 * (patched_x[:,:,:,0:1] + patched_x[:,:,:,1:2]), d_w=(patched_x[:,:,:,0:1] - patched_x[:,:,:,1:2]), wl=wl, dL=dL)
    #     right_bc = robin(patched_eps_Sy[:,:,:,-1:], 1/2 * (patched_x[:,:,:,-1:] + patched_x[:,:,:,-2:-1]), d_w=(patched_x[:,:,:,-1:] - patched_x[:,:,:,-2:-1]), wl=wl, dL=dL)

    #     if zero_global_bcs:
    #         top_bc[0:1,:,:,:] = 0
    #         bottom_bc[-1:,:,:,:] = 0
    #         left_bc[:,0:1,:,:] = 0
    #         right_bc[:,-1:,:,:] = 0

    #     return top_bc.reshape((x_patches*y_patches, 1, dsy)), \
    #            bottom_bc.reshape((x_patches*y_patches, 1, dsy)), \
    #            left_bc.reshape((x_patches*y_patches, dsx, 1)), \
    #            right_bc.reshape((x_patches*y_patches, dsx, 1))

    # def get_bc_errors(self, x, eps, Sx_f, Sy_f, wl, dL):
    #     assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)

    #     # if not periodic padding, meaning we have global bc, set global bc error to zeros
    #     a = self.get_bcs(x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=not self.periodic_padding)
    #     b = self.get_current_bcs(x, eps, Sx_f, Sy_f, wl, dL, zero_global_bcs=not self.periodic_padding)
    #     return [i-j for i, j in zip(a, b)]

    # def set_global_bc(self, gt, eps, Sx_f, Sy_f, device, wl, dL):
    #     # if we don't assume periodic padding, we need to construct the global bc using groundtruth data
    #     assert not self.periodic_padding

    #     if not torch.is_complex(Sx_f):
    #         Sx_f = 1 + 1j*Sx_f
    #     if not torch.is_complex(Sy_f):
    #         Sy_f = 1 + 1j*Sy_f

    #     x_patches, y_patches = self.grid_shape
    #     dsx, dsy = self.region_size
    #     patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)
    #     patched_gt = gt.reshape(x_patches, y_patches, dsx, dsy)
    #     patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
    #     patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)

    #     self.global_bcs = []

    #     # top bc
    #     eps = patched_eps_Sx[0:1,:,:1,:]
    #     bc = 1/2 * (patched_gt[0:1,:,0:1,:] + patched_gt[0:1,:,1:2,:])
    #     d_w = (patched_gt[0:1,:,0:1,:] - patched_gt[0:1,:,1:2,:])
    #     self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

    #     # bottom bc
    #     eps = patched_eps_Sx[-1:,:,-1:,:]
    #     bc = 1/2 * (patched_gt[-1:,:,-1:,:] + patched_gt[-1:,:,-2:-1,:])
    #     d_w = (patched_gt[-1:,:,-1:,:] - patched_gt[-1:,:,-2:-1,:])
    #     self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

    #     # left bc
    #     eps = patched_eps_Sy[:,0:1,:,:1]
    #     bc = 1/2 * (patched_gt[:,0:1,:,0:1] + patched_gt[:,0:1,:,1:2])
    #     d_w = (patched_gt[:,0:1,:,0:1] - patched_gt[:,0:1,:,1:2])
    #     self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))

    #     # right bc
    #     eps = patched_eps_Sy[:,-1:,:,-1:]
    #     bc = 1/2 * (patched_gt[:,-1:,:,-1:] + patched_gt[:,-1:,:,-2:-1])
    #     d_w = (patched_gt[:,-1:,:,-1:] - patched_gt[:,-1:,:,-2:-1])
    #     self.global_bcs.append(robin(eps, bc, d_w=d_w, wl=wl, dL=dL))
