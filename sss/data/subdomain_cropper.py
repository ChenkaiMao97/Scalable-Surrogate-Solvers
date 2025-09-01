# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import  os.path
import  numpy as np
from datetime import datetime
import sys
from tqdm import tqdm

from ceviche.constants import C_0, EPSILON_0
from sss.utils.PML_utils import make_Sx_Sy

import gin

@gin.configurable
class SubdomainCropper(object):
	def __init__(
		self,
		output_dir,
		input_data_folder,
		crops_per_idx,
		crop_dx=64,
		crop_dy=64,
		x_pad=10,
		y_pad=10,
		pml_thickness=40,
		img_file_name="input_eps.npy",
		field_file_name="Ez_out_forward_RI.npy",
		sources_file_name="source_RI.npy",
		wls_file_name="wls.npy",
		dLs_file_name="dLs.npy",
		exclude_pml = False,
		seed = 0
	):
		################### loading data ################
		self.seed = seed
		np.random.seed(self.seed)
		self.output_dir = output_dir

		self.exclude_pml = exclude_pml

		os.makedirs(self.output_dir, exist_ok=True)
		
		self.input_imgs = np.load(input_data_folder+'/'+img_file_name, mmap_mode='r')[:,:,:] 
		self.input_imgs = self.input_imgs[:, :, :].astype(np.float32, copy=False)
		self.original_size = self.input_imgs.shape[1]
		print("self.original_size is: ", self.original_size)
		print("input_imgs.shape: ", self.input_imgs.shape, self.input_imgs.dtype)
		
		self.Ez_forward = np.load(input_data_folder+'/'+field_file_name, mmap_mode='r').astype(np.float32, copy=False)
		print("Ez_forward.shape: ", self.Ez_forward.shape, self.Ez_forward.dtype)
		self.fields = self.Ez_forward

		self.wls = np.load(input_data_folder+'/'+wls_file_name, mmap_mode='r')
		print("wls.shape: ", self.wls.shape, self.wls.dtype)

		self.dLs = np.load(input_data_folder+'/'+dLs_file_name, mmap_mode='r')
		print("dLs.shape: ", self.dLs.shape, self.dLs.dtype)

		self.sources = np.load(input_data_folder+'/'+sources_file_name, mmap_mode='r').astype(np.float32, copy=False)
		print("Sources.shape: ", self.sources.shape, self.sources.dtype)

		############## make PML and pad #################
		self.pml_thickness = pml_thickness

		self.Sx_2D_f, self.Sy_2D_f = {}, {}
		self.Sx_2D_b, self.Sy_2D_b = {}, {}

		self.x_pad = x_pad
		self.y_pad = y_pad

		print("generating pml maps ...")
		self.Sx_2D_f = {}
		self.Sy_2D_f = {}
		for wl in self.wls:
			for dL in self.dLs:
				if (wl, dL) in self.Sx_2D_f:
					continue
				print(f"add PML for wl {wl} and dL {dL}")

				omega = 2 * np.pi * C_0 / (wl*1e-3)
				Sx_f, Sy_f = make_Sx_Sy(omega, dL*1e-3, self.input_imgs.shape[1], self.pml_thickness, self.input_imgs.shape[2], self.pml_thickness, _dir='f')
				self.Sx_2D_f[(wl, dL)] = np.pad(Sx_f, ((0,self.x_pad),(0,self.y_pad)), mode="wrap").astype(np.csingle, copy=False)
				self.Sy_2D_f[(wl, dL)] = np.pad(Sy_f, ((0,self.x_pad),(0,self.y_pad)), mode="wrap").astype(np.csingle, copy=False)

				Sx_b, Sy_b = make_Sx_Sy(omega, dL*1e-3, self.input_imgs.shape[1], self.pml_thickness, self.input_imgs.shape[2], self.pml_thickness, _dir='b')
				self.Sx_2D_b[(wl, dL)] = np.pad(Sx_b, ((0,self.x_pad),(0,self.y_pad)), mode="wrap").astype(np.csingle, copy=False)
				self.Sy_2D_b[(wl, dL)] = np.pad(Sy_b, ((0,self.x_pad),(0,self.y_pad)), mode="wrap").astype(np.csingle, copy=False)

		any_item = list(self.Sx_2D_f.values())[0]
		print("Sx_2D_f", len(self.Sx_2D_f), any_item.shape, any_item.dtype)

		######### pad remaining large scale data #########
		self.input_imgs = np.pad(self.input_imgs, ((0,0),(0,self.x_pad),(0,self.y_pad)), mode="wrap")
		self.fields = np.pad(self.fields, ((0,0),(0,self.x_pad),(0,self.y_pad),(0,0)), mode="wrap")
		self.sources = np.pad(self.sources, ((0,0),(0,self.x_pad),(0,self.y_pad),(0,0)), mode="wrap")
		print("data padded")
		##################################################

		############## create cropped data ###############
		self.crops_per_idx = crops_per_idx
		self.crop_dx = crop_dx
		self.crop_dy = crop_dy
		self.cropped_Ezs = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.csingle)
		self.cropped_sources = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.csingle)
		self.cropped_Sx_f = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.float32)
		self.cropped_Sy_f = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.float32)
		self.cropped_Sx_b = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.float32)
		self.cropped_Sy_b = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.float32)

		self.cropped_eps = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype=np.float32)
		self.cropped_top_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, 1, self.crop_dy), dtype =np.csingle)
		self.cropped_bottom_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, 1, self.crop_dy), dtype =np.csingle)
		self.cropped_left_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, 1), dtype =np.csingle)
		self.cropped_right_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, 1), dtype =np.csingle)
		self.cropped_wls = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx), dtype=np.float32)
		self.cropped_dLs = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx), dtype=np.float32)
		print("init cropped data")
		###################################################

	def crop_index(self, index):
		this_image = self.input_imgs[index,:,:]

		this_field_RI = self.fields[index,:,:,:]
		this_field = this_field_RI[:,:,0] + 1j*this_field_RI[:,:,1]
		this_wl = self.wls[index]*1e-3
		this_dL = self.dLs[index]*1e-3

		this_Sx_f = self.Sx_2D_f[(self.wls[index], self.dLs[index])]
		this_Sy_f = self.Sy_2D_f[(self.wls[index], self.dLs[index])]
		this_Sx_b = self.Sx_2D_b[(self.wls[index], self.dLs[index])]
		this_Sy_b = self.Sy_2D_b[(self.wls[index], self.dLs[index])]

		this_source_RI = self.sources[index,:,:,:]
		this_source = 1j*2*np.pi*C_0*this_dL**2/this_wl*EPSILON_0*(this_source_RI[:,:,0] + 1j*this_source_RI[:,:,1])

		for i in range(self.crops_per_idx):
			if self.exclude_pml:
				x = np.random.randint(self.pml_thickness, self.input_imgs.shape[1]-self.crop_dx-self.pml_thickness)
				y = np.random.randint(self.pml_thickness, self.input_imgs.shape[2]-self.crop_dy-self.pml_thickness)
			else:
				x = np.random.randint(0, self.input_imgs.shape[1]-self.crop_dx)
				y = np.random.randint(0, self.input_imgs.shape[2]-self.crop_dy)
				
			this_eps = this_image[x:x+self.crop_dx, y:y+self.crop_dy]
			
			Sx_f0 = this_Sx_f[x:x+self.crop_dx, y:y+self.crop_dy]
			Sy_f0 = this_Sy_f[x:x+self.crop_dx, y:y+self.crop_dy]
			Sx_b0 = this_Sx_b[x:x+self.crop_dx, y:y+self.crop_dy]
			Sy_b0 = this_Sy_b[x:x+self.crop_dx, y:y+self.crop_dy]

			this_eps_with_Sx = this_eps.astype(np.csingle) * Sx_f0
			this_eps_with_Sy = this_eps.astype(np.csingle) * Sy_f0

			field_rot0 = this_field[x:x+self.crop_dx, y:y+self.crop_dy]
			source_rot0 = this_source[x:x+self.crop_dx, y:y+self.crop_dy]

			top_bc0 =    1j*2*np.pi*np.sqrt(this_eps_with_Sx[:1, :])*this_dL/this_wl*1/2*(field_rot0[0:1, :]+field_rot0[1:2, :]) + field_rot0[0:1, :]-field_rot0[1:2, :]
			bottom_bc0 = 1j*2*np.pi*np.sqrt(this_eps_with_Sx[-1:, :])*this_dL/this_wl*1/2*(field_rot0[-1:, :]+field_rot0[-2:-1, :]) + field_rot0[-1:, :]-field_rot0[-2:-1, :]
			left_bc0 =   1j*2*np.pi*np.sqrt(this_eps_with_Sy[:, :1])*this_dL/this_wl*1/2*(field_rot0[:, 0:1]+field_rot0[:, 1:2]) + field_rot0[:, 0:1]-field_rot0[:, 1:2]
			right_bc0 =  1j*2*np.pi*np.sqrt(this_eps_with_Sy[:, -1:])*this_dL/this_wl*1/2*(field_rot0[:, -1:]+field_rot0[:, -2:-1]) + field_rot0[:, -1:]-field_rot0[:, -2:-1]

			self.cropped_Ezs[(index*self.crops_per_idx+i)] = field_rot0
			self.cropped_eps[(index*self.crops_per_idx+i)] = this_eps
			self.cropped_top_bc[(index*self.crops_per_idx+i)] = top_bc0
			self.cropped_bottom_bc[(index*self.crops_per_idx+i)] = bottom_bc0
			self.cropped_left_bc[(index*self.crops_per_idx+i)] = left_bc0
			self.cropped_right_bc[(index*self.crops_per_idx+i)] = right_bc0
			self.cropped_sources[(index*self.crops_per_idx+i)] = source_rot0

			self.cropped_wls[(index*self.crops_per_idx+i)] = this_wl
			self.cropped_dLs[(index*self.crops_per_idx+i)] = this_dL
			
			self.cropped_Sx_f[(index*self.crops_per_idx+i)] = Sx_f0.imag
			self.cropped_Sy_f[(index*self.crops_per_idx+i)] = Sy_f0.imag
			self.cropped_Sx_b[(index*self.crops_per_idx+i)] = Sx_b0.imag
			self.cropped_Sy_b[(index*self.crops_per_idx+i)] = Sy_b0.imag

	def crop_all(self):
		for i in tqdm(range(self.input_imgs.shape[0])):
			self.crop_index(i)

		np.save(os.path.join(self.output_dir, "cropped_Ezs.npy"), self.cropped_Ezs)
		np.save(os.path.join(self.output_dir, "cropped_sources.npy"), self.cropped_sources)
		np.save(os.path.join(self.output_dir, "cropped_eps.npy"), self.cropped_eps)
		np.save(os.path.join(self.output_dir, "cropped_wls.npy"), self.cropped_wls)
		np.save(os.path.join(self.output_dir, "cropped_dLs.npy"), self.cropped_dLs)

		np.save(os.path.join(self.output_dir, "cropped_top_bc.npy"), self.cropped_top_bc)
		np.save(os.path.join(self.output_dir, "cropped_bottom_bc.npy"), self.cropped_bottom_bc)
		np.save(os.path.join(self.output_dir, "cropped_left_bc.npy"), self.cropped_left_bc)
		np.save(os.path.join(self.output_dir, "cropped_right_bc.npy"), self.cropped_right_bc)

		np.save(os.path.join(self.output_dir, "cropped_Sx_f.npy"), self.cropped_Sx_f)
		np.save(os.path.join(self.output_dir, "cropped_Sy_f.npy"), self.cropped_Sy_f)
		np.save(os.path.join(self.output_dir, "cropped_Sx_b.npy"), self.cropped_Sx_b)
		np.save(os.path.join(self.output_dir, "cropped_Sy_b.npy"), self.cropped_Sy_b)
		print("cropped data saved")
