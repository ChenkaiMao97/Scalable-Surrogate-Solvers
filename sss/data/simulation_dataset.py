# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import gin

@gin.configurable
class SimulationDataset(Dataset):
    def __init__(
        self, 
        data_folder, 
        total_sample_number = None, 
        transform = None, 
        data_mult = 1, 
        pml_dominate=False
    ):

        self.eps = np.load(os.path.join(data_folder, 'cropped_eps.npy'), mmap_mode='r')
        print("eps.shape: ", self.eps.shape, self.eps.dtype)

        self.Ez_forward = np.load(os.path.join(data_folder, 'cropped_Ezs.npy'), mmap_mode='r')
        self.Ez_forward = data_mult*np.stack((np.real(self.Ez_forward),np.imag(self.Ez_forward)),axis=3).astype(np.float32)
        print("Ez_forward.shape: ", self.Ez_forward.shape, self.Ez_forward.dtype)

        self.sources = np.load(os.path.join(data_folder, 'cropped_sources.npy'), mmap_mode='r')
        self.sources = data_mult*np.stack((np.real(self.sources),np.imag(self.sources)),axis=3).astype(np.float32)
        print("sources.shape: ", self.sources.shape, self.sources.dtype)

        self.Sx_f = np.load(os.path.join(data_folder, 'cropped_Sx_f.npy'), mmap_mode='r').astype(np.float32, copy=False)
        self.Sx_b = np.load(os.path.join(data_folder, 'cropped_Sx_b.npy'), mmap_mode='r').astype(np.float32, copy=False)
        print("Sx_f.shape: ", self.Sx_f.shape, self.Sx_f.dtype)

        self.Sy_f = np.load(os.path.join(data_folder, 'cropped_Sy_f.npy'), mmap_mode='r').astype(np.float32, copy=False)
        self.Sy_b = np.load(os.path.join(data_folder, 'cropped_Sy_b.npy'), mmap_mode='r').astype(np.float32, copy=False)

        self.top_bc = np.load(os.path.join(data_folder, 'cropped_top_bc.npy'), mmap_mode='r')
        self.top_bc = data_mult*np.stack((np.real(self.top_bc),np.imag(self.top_bc)),axis=3).astype(np.float32)
        print("top_bc.shape: ", self.top_bc.shape, self.top_bc.dtype)
        
        self.bottom_bc = np.load(os.path.join(data_folder, 'cropped_bottom_bc.npy'), mmap_mode='r')
        self.bottom_bc = data_mult*np.stack((np.real(self.bottom_bc),np.imag(self.bottom_bc)),axis=3).astype(np.float32)
        print("bottom_bc.shape: ", self.bottom_bc.shape, self.bottom_bc.dtype)

        self.left_bc = np.load(os.path.join(data_folder, 'cropped_left_bc.npy'), mmap_mode='r')
        self.left_bc = data_mult*np.stack((np.real(self.left_bc),np.imag(self.left_bc)),axis=3).astype(np.float32)
        print("left_bc.shape: ", self.left_bc.shape, self.left_bc.dtype)

        self.right_bc = np.load(os.path.join(data_folder, 'cropped_right_bc.npy'), mmap_mode='r')
        self.right_bc = data_mult*np.stack((np.real(self.right_bc),np.imag(self.right_bc)),axis=3).astype(np.float32)
        print("right_bc.shape: ", self.right_bc.shape, self.right_bc.dtype)

        self.wls = np.load(os.path.join(data_folder, 'cropped_wls.npy'), mmap_mode='r').astype(np.float32)
        self.dLs = np.load(os.path.join(data_folder, 'cropped_dLs.npy'), mmap_mode='r').astype(np.float32)
        print("wls.shape: ", self.wls.shape, self.wls.dtype)
        print("dLs.shape: ", self.dLs.shape, self.dLs.dtype)

        self.fields = self.Ez_forward
        
        if total_sample_number:
            np.random.seed(1234)
            indices = np.array(np.random.choice(self.Ez_forward.shape[0], total_sample_number, replace=False))
            
            self.eps = np.take(self.eps, indices, axis=0)
            
            self.fields = np.take(self.fields, indices, axis=0)
            self.sources = np.take(self.sources, indices, axis=0)
            self.Sx_f = np.take(self.Sx_f, indices, axis=0)
            self.Sx_b = np.take(self.Sx_b, indices, axis=0)
            self.Sy_f = np.take(self.Sy_f, indices, axis=0)
            self.Sy_b = np.take(self.Sy_b, indices, axis=0)

            self.top_bc = np.take(self.top_bc, indices, axis=0)
            self.bottom_bc = np.take(self.bottom_bc, indices, axis=0)
            self.left_bc = np.take(self.left_bc, indices, axis=0)
            self.right_bc = np.take(self.right_bc, indices, axis=0)
            
            self.wls = np.take(self.wls, indices, axis=0)
            self.dLs = np.take(self.dLs, indices, axis=0)
            print("finished indexing")

        self.transform = transform

        if pml_dominate:
            self.filter_pml_dominate_data()
        # else:
        #     self.filter_valid_data()

    def filter_valid_data(self):
        # idea: the effective wavelength in the domain shouldn't be more than 3. 
        # eps_mean**.5 * dL * 64 / wl < 3
        # wl: in meter, dL: in meter

        keep_indices = []
        types = {}
        for i in tqdm(range(self.fields.shape[0])):
            dL = self.dLs[i]
            wl = self.wls[i]
            eps_max = np.max(self.eps[i])
            if eps_max**.5 * dL * 64 / wl < 3:
                keep_indices.append(i)
                if (wl, dL) in types:
                    types[(wl, dL)] += 1
                else:
                    types[(wl, dL)] = 1

        print(f"finished filtering, total keep indices: {len(keep_indices)}")
        print("with types: ", types)

        print("pruning data ...")
        self.eps = np.take(self.eps, keep_indices, axis=0)  
        self.fields = np.take(self.fields, keep_indices, axis=0)
        self.sources = np.take(self.sources, keep_indices, axis=0)
        self.Sx_f = np.take(self.Sx_f, keep_indices, axis=0)
        self.Sy_f = np.take(self.Sy_f, keep_indices, axis=0)
        self.Sx_b = np.take(self.Sx_b, keep_indices, axis=0)
        self.Sy_b = np.take(self.Sy_b, keep_indices, axis=0)
        self.top_bc = np.take(self.top_bc, keep_indices, axis=0)
        self.bottom_bc = np.take(self.bottom_bc, keep_indices, axis=0)
        self.left_bc = np.take(self.left_bc, keep_indices, axis=0)
        self.right_bc = np.take(self.right_bc, keep_indices, axis=0)
        self.wls = np.take(self.wls, keep_indices, axis=0)
        self.dLs = np.take(self.dLs, keep_indices, axis=0)
        print("finished")

    def filter_pml_dominate_data(self):
        keep_indices = []
        PML_data_count = 0
        other_data_count = 0
        for i in tqdm(range(self.fields.shape[0])):
            sx_f = self.Sx_f[i]
            sy_f = self.Sy_f[i]
            if np.max(np.abs(sx_f+sy_f)) > 2+1e-6:
                keep_indices.append(i)
                PML_data_count += 1
            else:
                if other_data_count < PML_data_count:
                    keep_indices.append(i)
                    other_data_count += 1

        print(f"finished filtering, total keep indices: {len(keep_indices)}")
        print(f"PML data count: {PML_data_count}, other data count: {other_data_count}")

        print("pruning data ...")
        self.eps = np.take(self.eps, keep_indices, axis=0)  
        self.fields = np.take(self.fields, keep_indices, axis=0)
        self.sources = np.take(self.sources, keep_indices, axis=0)
        self.Sx_f = np.take(self.Sx_f, keep_indices, axis=0)
        self.Sy_f = np.take(self.Sy_f, keep_indices, axis=0)
        self.Sx_b = np.take(self.Sx_b, keep_indices, axis=0)
        self.Sy_b = np.take(self.Sy_b, keep_indices, axis=0)
        self.top_bc = np.take(self.top_bc, keep_indices, axis=0)
        self.bottom_bc = np.take(self.bottom_bc, keep_indices, axis=0)
        self.left_bc = np.take(self.left_bc, keep_indices, axis=0)
        self.right_bc = np.take(self.right_bc, keep_indices, axis=0)
        self.wls = np.take(self.wls, keep_indices, axis=0)
        self.dLs = np.take(self.dLs, keep_indices, axis=0)
        print("finished")

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        field = self.fields[idx, :, :, :]
        source = self.sources[idx, :, :, :]
        Sx_f = self.Sx_f[idx, :, :]
        Sy_f = self.Sy_f[idx, :, :]
        Sx_b = self.Sx_b[idx, :, :]
        Sy_b = self.Sy_b[idx, :, :]

        top_bc = self.top_bc[idx, :, :, :]
        bottom_bc = self.bottom_bc[idx, :, :, :]
        left_bc = self.left_bc[idx, :, :, :]
        right_bc = self.right_bc[idx, :, :, :]
        eps = self.eps[idx, :, :]
        # yeex = self.yeex[idx, :, :]
        # yeey = self.yeey[idx, :, :]
        wl = self.wls[idx]
        dL = self.dLs[idx]

        #means = [self.Ez_meanR, self.Ez_meanI, self.Ex_meanR, self.Ex_meanI, self.Ez_meanR, self.Ez_meanI]

        sample = {'field': field, 'source': source,
                  'Sx_f': Sx_f, 'Sy_f':Sy_f, 'Sx_b': Sx_b, 'Sy_b':Sy_b,
                  'top_bc': top_bc, 'bottom_bc': bottom_bc,
                  'left_bc': left_bc, 'right_bc': right_bc,
                  # 'yeex': yeex, 'yeey': yeey,
                  'eps': eps,
                  'wl': wl, 'dL': dL}#, 'means': means}

        if self.transform:
            sample = self.transform(sample)

        return sample