# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import numpy as np
import math
import torch
import copy

from matplotlib import pyplot as plt

import gin

def robin(eps, bc, d_v=None, d_w=None, wl=1050e-9, dL=6.25e-9):
    # bc: the boundary field to be transform
    # d_v: the derivative of fields in v
    # d_w: the derivative of fields in w
    g = 1j*2*np.pi*torch.sqrt(eps)/wl*dL*bc+d_w
    return g

@gin.configurable
class Partitioner:
    def __init__(
        self, 
        unpadded_shape, 
        min_overlap, 
        region_size, 
        variable_overlap=False, 
        periodic_padding=True
    ):
        """
        Initializes the partitioner with support for non-uniform grids.
        """
        self.periodic_padding = periodic_padding
        self.unpadded_shape = unpadded_shape
        self.min_overlap = min_overlap
        self.variable_overlap = variable_overlap
        self.region_size = region_size  # Size of each region (height, width)
        self.global_bcs = None

        self.x_overlaps, self.y_overlaps, self.grid_shape, self.padded_shape = self.init_grid()
        self.regions, self.POU_maps = self.init_regions()  # Stores slices for each region
    
    def init_grid(self):
        if self.variable_overlap:
            rows = math.ceil(self.unpadded_shape[0]/(self.region_size[0]-self.min_overlap))
            cols = math.ceil(self.unpadded_shape[1]/(self.region_size[1]-self.min_overlap))
            grid_shape = (rows, cols)
            total_gap_x = rows * (self.region_size[0]-self.min_overlap) - self.unpadded_shape[0] + (not self.periodic_padding) * self.min_overlap
            total_gap_y = cols * (self.region_size[1]-self.min_overlap) - self.unpadded_shape[1] + (not self.periodic_padding) * self.min_overlap

            num_overlaps_x = cols if self.periodic_padding else cols - 1
            num_overlaps_y = rows if self.periodic_padding else rows - 1
            x_overlaps = [self.min_overlap + round((i+1)*total_gap_x/num_overlaps_x) - round(i*total_gap_x/num_overlaps_x) for i in range(num_overlaps_x)] + [self.min_overlap] * (not self.periodic_padding)
            y_overlaps = [self.min_overlap + round((i+1)*total_gap_y/num_overlaps_y) - round(i*total_gap_y/num_overlaps_y) for i in range(num_overlaps_y)] + [self.min_overlap] * (not self.periodic_padding)
        else:
            assert self.min_overlap < self.region_size[0] and self.min_overlap < self.region_size[1]
            assert self.unpadded_shape[0] % (self.region_size[0]-self.min_overlap) == (not self.periodic_padding) * self.min_overlap
            assert self.unpadded_shape[1] % (self.region_size[1]-self.min_overlap) == (not self.periodic_padding) * self.min_overlap
            grid_shape = (self.unpadded_shape[0]//(self.region_size[0]-self.min_overlap), self.unpadded_shape[1]//(self.region_size[1]-self.min_overlap))
            x_overlaps = [self.min_overlap] * self.grid_shape[0]
            y_overlaps = [self.min_overlap] * self.grid_shape[1]

        padded_shape = (self.unpadded_shape[0] + x_overlaps[-1], self.unpadded_shape[1] + y_overlaps[-1]) if self.periodic_padding else self.unpadded_shape
        
        return x_overlaps, y_overlaps, grid_shape, padded_shape

    def init_regions(self):
        num_rows, num_cols = self.grid_shape
        regions = []
        POU_maps = []
        count_map = torch.zeros(self.padded_shape)

        # (1) set as all ones
        # POU_map = torch.ones(self.region_size[0], self.region_size[1])
        
        # (2) set as smoothly decaying at edge
        temp_overlap = max(self.x_overlaps[0], self.y_overlaps[0])
        POU_map = torch.zeros(self.region_size[0], self.region_size[1])
        for k in range(temp_overlap):
            POU_map[k:self.region_size[0]-k, k:self.region_size[1]-k] = (k+1)**1/temp_overlap**1

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

        # debugging:
        # summed_map = torch.zeros(self.padded_shape)
        # for i in range(len(regions)):
        #     summed_map[regions[i]] += POU_maps[i]
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(count_map)
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow(summed_map)
        # plt.colorbar()
        # plt.savefig('POU_maps_debug.png', dpi=500)
        # plt.close()
        # a = input("check the POU_maps_debug.png")

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

    
    def get_bcs(self, x, eps, Sx_f, Sy_f, device, bloch_phases=None, wl=1050e-9, dL=6.25e-9):
        assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)
        
        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size

        patched_x = x.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)
            
        patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)
        
        top_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(device)
        bottom_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(device)
        left_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(device)
        right_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(device)

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
            top_bc[0:1,:,:,:] = self.global_bcs[0]

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
            bottom_bc[-1:,:,:,:] = self.global_bcs[1]

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
            left_bc[:,0:1,:,:] = self.global_bcs[2]

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
            right_bc[:,-1:,:,:] = self.global_bcs[3]

        return top_bc.reshape((x_patches*y_patches, 1, dsy)), \
               bottom_bc.reshape((x_patches*y_patches, 1, dsy)), \
               left_bc.reshape((x_patches*y_patches, dsx, 1)), \
               right_bc.reshape((x_patches*y_patches, dsx, 1))

    def get_current_bcs(self, x, eps, Sx_f, Sy_f, device, wl=1050e-9, dL=6.25e-9):
        assert self.periodic_padding or (not self.periodic_padding and self.global_bcs is not None)
        
        x_patches, y_patches = self.grid_shape
        dsx, dsy = self.region_size

        patched_x = x.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps = eps.reshape(x_patches, y_patches, dsx, dsy)
            
        patched_eps_Sx = patched_eps.to(torch.complex64)*Sx_f.reshape(x_patches, y_patches, dsx, dsy)
        patched_eps_Sy = patched_eps.to(torch.complex64)*Sy_f.reshape(x_patches, y_patches, dsx, dsy)
        
        top_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(device)
        bottom_bc = torch.zeros((x_patches, y_patches, 1, dsy), dtype=torch.complex64).to(device)
        left_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(device)
        right_bc = torch.zeros((x_patches, y_patches, dsx, 1), dtype=torch.complex64).to(device)

        # top bc
        top_bc = robin(patched_eps_Sx[:,:,:1,:], 1/2 * (patched_x[:,:,0:1,:] + patched_x[:,:,1:2,:]), d_w=1/2 * (patched_x[:,:,0:1,:] - patched_x[:,:,1:2,:]), wl=wl, dL=dL)
        bottom_bc = robin(patched_eps_Sx[:,:,-1:,:], 1/2 * (patched_x[:,:,-1:,:] + patched_x[:,:,-2:-1,:]), d_w=1/2 * (patched_x[:,:,-1:,:] - patched_x[:,:,-2:-1,:]), wl=wl, dL=dL)
        left_bc = robin(patched_eps_Sy[:,:,:,:1], 1/2 * (patched_x[:,:,:,0:1] + patched_x[:,:,:,1:2]), d_w=1/2 * (patched_x[:,:,:,0:1] - patched_x[:,:,:,1:2]), wl=wl, dL=dL)
        right_bc = robin(patched_eps_Sy[:,:,:,-1:], 1/2 * (patched_x[:,:,:,-1:] + patched_x[:,:,:,-2:-1]), d_w=1/2 * (patched_x[:,:,:,-1:] - patched_x[:,:,:,-2:-1]), wl=wl, dL=dL)

        return top_bc.reshape((x_patches*y_patches, 1, dsy)), \
               bottom_bc.reshape((x_patches*y_patches, 1, dsy)), \
               left_bc.reshape((x_patches*y_patches, dsx, 1)), \
               right_bc.reshape((x_patches*y_patches, dsx, 1))

    def set_global_bc(self, gt, eps, Sx_f, Sy_f, device, wl=1050e-9, dL=6.25e-9):
        # if we don't assume periodic padding, we need to construct the global bc using groundtruth data
        assert not self.periodic_padding

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
    
    def new_extended_partitioner(self, pad_each_side=1):
        # pad_each_side: pixels to extend the overlapping region on each side (if pad_each_side=1, then new overlapping would be old_ol + 2)
        new_partitioner = copy.deepcopy(self)
        new_partitioner.region_size = (new_partitioner.region_size[0]+2*pad_each_side, new_partitioner.region_size[1]+2*pad_each_side)
        if new_partitioner.periodic_padding:
            # TO DO: fix this, since the boundary has some issues.
            new_partitioner.x_overlaps = [o + pad_each_side for o in self.x_overlaps]
            new_partitioner.y_overlaps = [o + pad_each_side for o in self.y_overlaps]
            new_partitioner.padded_shape = (new_partitioner.unpadded_shape[0] + new_partitioner.x_overlaps[-1], new_partitioner.unpadded_shape[1] + new_partitioner.y_overlaps[-1])
        else:
            new_partitioner.x_overlaps = [self.x_overlaps[0] + 4*pad_each_side] + [o+2*pad_each_side for o in self.x_overlaps[1:-1]] + [self.x_overlaps[-1] + 4*pad_each_side]
            new_partitioner.y_overlaps = [self.y_overlaps[0] + 4*pad_each_side] + [o+2*pad_each_side for o in self.y_overlaps[1:-1]] + [self.y_overlaps[-1] + 4*pad_each_side]
        
        new_partitioner.regions, new_partitioner.POU_maps = new_partitioner.init_regions()
        return new_partitioner
    
    def update(self, regions, update_func):
        # Get the number of regions in grid
        num_rows, num_cols = self.grid_shape
        
        updated_array = A.copy()
        
        # Update using vectorized operations for efficiency
        for idx, region_slices in enumerate(self.regions):
            i, j = divmod(idx, num_cols)
            
            # Get the current region
            region = partitions[idx]
            
            # Collect neighbors efficiently
            neighbors = {}
            if i > 0:  # Top neighbor
                neighbors['top'] = updated_array[self.regions[idx - num_cols]][-self.y_overlap[i]:, :]
            if i < num_rows - 1:  # Bottom neighbor
                neighbors['bottom'] = updated_array[self.regions[idx + num_cols]][:self.y_overlap[i + 1], :]
            if j > 0:  # Left neighbor
                neighbors['left'] = updated_array[self.regions[idx - 1]][:, -self.x_overlap[j]:]
            if j < num_cols - 1:  # Right neighbor
                neighbors['right'] = updated_array[self.regions[idx + 1]][:, :self.x_overlap[j + 1]]
            
            # Update region using neighbors
            updated_region = update_func(region, neighbors)
            
            # Update the corresponding part of the array
            updated_array[region_slices] = updated_region
        
        return updated_array
    
    def visualize_regions(self, A):
        A_padded = np.pad(A, ((0, self.x_overlaps[-1]), (0, self.y_overlaps[-1])), mode='wrap')
        fig, ax = plt.subplots()
        ax.imshow(A_padded, cmap='gray')
        for region in self.regions:
            ax.add_patch(plt.Rectangle((region[1].start, region[0].start), region[1].stop - region[1].start, region[0].stop - region[0].start, fill=False, edgecolor='red'))
        plt.savefig('regions.png')
    
    def get_batch_size(self):
        return self.grid_shape[0]*self.grid_shape[1]
    
    def Q_to_BQTBQ(self, Q, eps, sx_f, sy_f, dL, wl, pad_each_side=0, periodic=False):
        if periodic:
            raise NotImplementedError("Periodic boundary conditions are not implemented yet")
        # Q shape: (model_bs, k_directions, sx, sy, 2)
        model_bs, k_directions, sx, sy, _ = Q.shape
        pd = pad_each_side
        assert self.region_size == (sx, sy)
        assert self.grid_shape[0]*self.grid_shape[1] == model_bs
        assert sx == sy

        if not torch.is_complex(sx_f):
            sx_f = 1+1j*sx_f
            sy_f = 1+1j*sy_f
        
        eps_with_sx = eps.to(torch.complex64) * sx_f
        eps_with_sy = eps.to(torch.complex64) * sy_f

        Ez = Q[:,:,:,:,0] + 1j*Q[:,:,:,:,1]

        bc_array = torch.zeros((self.grid_shape[0], self.grid_shape[1], k_directions, 8, sx), dtype=torch.complex64).to(Q.device)
        Ez = Ez.reshape(self.grid_shape[0], self.grid_shape[1], k_directions, sx, sy)
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

        bc_array = bc_array.reshape(model_bs, k_directions, 8, sx)

        BQTBQ = torch.zeros((model_bs, k_directions, model_bs, k_directions), dtype=torch.complex64).to(Q.device)

        for i in range(model_bs):
            r1, c1 = i//self.grid_shape[1], i%self.grid_shape[1]
            for j in range(model_bs):
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

        # filter out zero entries in bc_array, as well as the corresponding rows and columns in BQTBQ
        non_zero_basis_indices = torch.sum(torch.abs(bc_array.reshape(-1,8,sx)), dim=(2,3), keepdim=True) > 1e-9
        BQTBQ = BQTBQ.reshape(model_bs * k_directions, model_bs * k_directions)[non_zero_basis_indices, :][:, non_zero_basis_indices]

        if self.periodic_padding:
            raise NotImplementedError("Periodic boundary conditions are not implemented yet")

        return BQTBQ, bc_array, non_zero_basis_indices
    
    def Q_to_bc_array(self, Q, eps, sx_f, sy_f, dL, wl, pad_each_side=0, periodic=False):
        if periodic:
            raise NotImplementedError("Periodic boundary conditions are not implemented yet")
        # Q shape: (model_bs, k_directions, sx, sy, 2)
        model_bs, k_directions, sx, sy, _ = Q.shape
        pd = pad_each_side
        assert self.region_size == (sx, sy)
        assert self.grid_shape[0]*self.grid_shape[1] == model_bs
        assert sx == sy

        if not torch.is_complex(sx_f):
            sx_f = 1+1j*sx_f
            sy_f = 1+1j*sy_f
        
        eps_with_sx = eps.to(torch.complex64) * sx_f
        eps_with_sy = eps.to(torch.complex64) * sy_f

        Ez = Q[:,:,:,:,0] + 1j*Q[:,:,:,:,1]

        bc_array = torch.zeros((self.grid_shape[0], self.grid_shape[1], k_directions, 8, sx), dtype=torch.complex64).to(Q.device)
        Ez = Ez.reshape(self.grid_shape[0], self.grid_shape[1], k_directions, sx, sy)
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

        bc_array = bc_array.reshape(model_bs, k_directions, 8, sx)
        return bc_array

    def get_bc_error(self, Ez, eps, sx_f, sy_f, dL, wl, pad_each_side=0, compute_global_error=True):
        model_bs, sx, sy = Ez.shape
        pd = pad_each_side
        assert self.grid_shape[0]*self.grid_shape[1] == model_bs

        if not torch.is_complex(sx_f):
            sx_f = 1+1j*sx_f
            sy_f = 1+1j*sy_f
        eps_with_sx = (eps.to(torch.complex64) * sx_f).reshape(self.grid_shape[0], self.grid_shape[1], sx, sy)
        eps_with_sy = (eps.to(torch.complex64) * sy_f).reshape(self.grid_shape[0], self.grid_shape[1], sx, sy)

        Ez = Ez.reshape(self.grid_shape[0], self.grid_shape[1], sx, sy)
        bc_error = torch.zeros((self.grid_shape[0], self.grid_shape[1], 4, sx), dtype=torch.complex64, device=Ez.device)

        if self.periodic_padding:
            raise NotImplementedError
        else:
            for i in range(1,self.grid_shape[0]):
                ol = self.x_overlaps[i-1]
                bc_neighbor = (Ez[i-1,:,sx-pd-ol,:]-Ez[i-1,:,sx-pd-ol+1,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i-1,:,sx-pd-ol,:])/wl*dL*1/2*(Ez[i-1,:,sx-pd-ol,:]+Ez[i-1,:,sx-pd-ol+1,:])
                bc_this = (Ez[i,:,0+pd,:]-Ez[i,:,1+pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i,:,0+pd,:])/wl*dL*1/2*(Ez[i,:,0+pd,:]+Ez[i,:,1+pd,:])
                bc_error[i,:,0,:] = bc_neighbor - bc_this
            for i in range(self.grid_shape[0]-1):
                ol = self.x_overlaps[i]
                bc_neighbor = (Ez[i+1,:,pd+ol-1,:]-Ez[i+1,:,pd+ol-2,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i+1,:,pd+ol-1,:])/wl*dL*1/2*(Ez[i+1,:,pd+ol-1,:]+Ez[i+1,:,pd+ol-2,:])
                bc_this = (Ez[i,:,sx-1-pd,:]-Ez[i,:,sx-2-pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[i,:,sx-1-pd,:])/wl*dL*1/2*(Ez[i,:,sx-1-pd,:]+Ez[i,:,sx-2-pd,:])
                bc_error[i,:,1,:] = bc_neighbor - bc_this
            for i in range(1,self.grid_shape[1]):
                ol = self.y_overlaps[i-1]
                bc_neighbor = (Ez[:,i-1,:,sy-pd-ol]-Ez[:,i-1,:,sy-pd-ol+1])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i-1,:,sy-pd-ol])/wl*dL*1/2*(Ez[:,i-1,:,sy-pd-ol]+Ez[:,i-1,:,sy-pd-ol+1])
                bc_this = (Ez[:,i,:,0+pd]-Ez[:,i,:,1+pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i,:,0+pd])/wl*dL*1/2*(Ez[:,i,:,0+pd]+Ez[:,i,:,1+pd])
                bc_error[:,i,2,:] = bc_neighbor - bc_this
            for i in range(self.grid_shape[1]-1):
                ol = self.y_overlaps[i]
                bc_neighbor = (Ez[:,i+1,:,pd+ol-1]-Ez[:,i+1,:,pd+ol-2])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i+1,:,pd+ol-1])/wl*dL*1/2*(Ez[:,i+1,:,pd+ol-1]+Ez[:,i+1,:,pd+ol-2])
                bc_this = (Ez[:,i,:,sy-1-pd]-Ez[:,i,:,sy-2-pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,i,:,sy-1-pd])/wl*dL*1/2*(Ez[:,i,:,sy-1-pd]+Ez[:,i,:,sy-2-pd])
                bc_error[:,i,3,:] = bc_neighbor - bc_this
            
            if compute_global_error:
                # top:
                bc_neighbor = torch.zeros((self.grid_shape[1], sx), dtype=torch.complex64, device=Ez.device)
                bc_this = (Ez[0,:,0+pd,:]-Ez[0,:,1+pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[0,:,0+pd,:])/wl*dL*1/2*(Ez[0,:,0+pd,:]+Ez[0,:,1+pd,:])
                bc_error[0,:,0,:] = bc_neighbor - bc_this
                # bottom:
                bc_neighbor = torch.zeros((self.grid_shape[1], sx), dtype=torch.complex64, device=Ez.device)
                bc_this = (Ez[-1,:,sx-1-pd,:]-Ez[-1,:,sx-2-pd,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[-1,:,sx-1-pd,:])/wl*dL*1/2*(Ez[-1,:,sx-1-pd,:]+Ez[-1,:,sx-2-pd,:])
                bc_error[-1,:,1,:] = bc_neighbor - bc_this
                # left:
                bc_neighbor = torch.zeros((self.grid_shape[0], sy), dtype=torch.complex64, device=Ez.device)
                bc_this = (Ez[:,0,:,0+pd]-Ez[:,0,:,1+pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,0,:,0+pd])/wl*dL*1/2*(Ez[:,0,:,0+pd]+Ez[:,0,:,1+pd])
                bc_error[:,0,2,:] = bc_neighbor - bc_this
                # right:
                bc_neighbor = torch.zeros((self.grid_shape[0], sy), dtype=torch.complex64, device=Ez.device)
                bc_this = (Ez[:,-1,:,sy-1-pd]-Ez[:,-1,:,sy-2-pd])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,-1,:,sy-1-pd])/wl*dL*1/2*(Ez[:,-1,:,sy-1-pd]+Ez[:,-1,:,sy-2-pd])
                bc_error[:,-1,3,:] = bc_neighbor - bc_this

        return bc_error.reshape(model_bs, 4, sx)

    def bc_error_vector_to_map(self, bc_error_vector, shape):
        bc_error_map = torch.zeros(shape, dtype=torch.complex64, device=bc_error_vector.device)
        bc_error_map[:,0,:] = bc_error_vector[:,0,:]
        bc_error_map[:,-1,:] = bc_error_vector[:,1,:]
        bc_error_map[:,:,0] = bc_error_vector[:,2,:]
        bc_error_map[:,:,-1] = bc_error_vector[:,3,:]
        return self.combine(bc_error_map, scale_with_POU=False)



class OverlappingQuadTree:
    def __init__(self, coords, wl, dL, eps, threshold=None, leaf_overlap=None, leaf_size=None):
        self.coords = coords
        self.wl = wl
        self.dL = dL
        self.threshold = threshold

        self.subdivide(eps, leaf_overlap, leaf_size)
    
    def subdivide(self, eps, leaf_overlap, leaf_size):
        x, y, w, h = self.coords
        eps_slice = eps[x:x+w, y:y+h]
    
        if eps_slice.size == 0:
            return # skip empty slice
        if w * self.dL * np.sqrt(np.mean(eps_slice)) / self.wl > self.threshold or eps_slice.shape[0] < w or eps_slice.shape[1] < h:
            mid_w, mid_h = int(w // 2), int(h // 2) # it's okay to have uneven split
            self.children = [
                OverlappingQuadTree((x, y, mid_w, mid_h), self.wl, self.dL, eps, self.threshold, leaf_overlap, leaf_size),
                OverlappingQuadTree((x, y + mid_h, mid_w, h - mid_h), self.wl, self.dL, eps, self.threshold, leaf_overlap, leaf_size),
                OverlappingQuadTree((x + mid_w, y, w - mid_w, mid_h), self.wl, self.dL, eps, self.threshold, leaf_overlap, leaf_size),
                OverlappingQuadTree((x + mid_w, y + mid_h, w - mid_w, h - mid_h), self.wl, self.dL, eps, self.threshold, leaf_overlap, leaf_size)
            ]
        else:
            self.is_leaf = True
            self.overlap = int(leaf_overlap * w/leaf_size)
    
    def visualize(self, eps_data=None, src_data=None, field_data=None, ax=None, path='./quadtree.png'):
        fig, ax = plt.subplots(figsize=(10, 10))
        if eps_data is not None:
            color_yee_data = setup_plot_data(eps_data, src_data)
            ax.imshow(color_yee_data)

        self._draw_node(ax)

        ax.set_xlim(0, self.coords[3])
        ax.set_ylim(self.coords[2], 0)
        ax.set_aspect('equal')
        ax.set_title('QuadTree Visualization')
        plt.tight_layout()
        plt.savefig(path, transparent=True, dpi=300)
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 10))
        if field_data is not None:
            ax.imshow(field_data.real, cmap='seismic')

        self._draw_node(ax)

        ax.set_xlim(0, self.coords[3])
        ax.set_ylim(self.coords[2], 0)
        ax.set_aspect('equal')
        ax.set_title('QuadTree Visualization')
        plt.tight_layout()
        plt.savefig('./quadtree_field.png', transparent=True, dpi=300)
        plt.close()



    def _draw_node(self, ax):
        y, x, h, w = self.coords
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=3)
        ax.add_patch(rect)
        rect = patches.Rectangle((x-self.overlap, y-self.overlap), w+2*self.overlap, h+2*self.overlap, fill=False, edgecolor='g', linewidth=3)
        ax.add_patch(rect)

        if not self.is_leaf:
            for child in self.children:
                child._draw_node(ax)


class QuadTreePartitioner:
    def __init__(self, coords, wl, dL, eps, threshold=None, leaf_overlap=None, leaf_size=None, periodic_padding=True, pad_pixels=4):
        self.periodic_padding = periodic_padding
        # self.coords = coords
        # self.wl = wl
        # self.dL = dL
        # self.eps = eps
        # self.threshold = threshold
        # self.leaf_overlap = leaf_overlap
        # self.leaf_size = leaf_size
        self.pad_pixels = pad_pixels

        self.quadtree = OverlappingQuadTree(coords, wl, dL, eps, threshold, leaf_overlap, leaf_size)
        self.regions = None
        self.grid_shape = None
    


if __name__ == "__main__":
    A = np.zeros((100, 110))
    partitioner = Partitioner(A.shape, 4, (16, 16), variable_overlap=True)
    print(partitioner.x_overlaps, partitioner.y_overlaps)
    print(partitioner.grid_shape, partitioner.padded_shape)
    partitioner.visualize_regions(A)
    # print(partitioner.regions)