# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import os, sys

import torch
import numpy as np

from ceviche.constants import EPSILON_0, MU_0, C_0

def E_to_E(Ez_R, Ez_I, dL, wl, omega, eps, source, Sxs, Sys, EPSILON_0 = EPSILON_0, MU_0 = MU_0, separate_source=False):
	sx_f, sx_b = Sxs
	sy_f, sy_b = Sys
		
	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sx_b = 1+1j*sx_b
		sy_f = 1+1j*sy_f
		sy_b = 1+1j*sy_b

	Ez = Ez_R + 1j*Ez_I
	# Sx = torch.ones(Sx.shape, dtype=torch.complex64, device=Sx.device) + 1j*Sx
	# Sy = torch.ones(Sy.shape, dtype=torch.complex64, device=Sx.device) + 1j*Sy
	FD_Hx = Ez_to_Hx(Ez, dL, omega, sx_b, sy_b, EPSILON_0)
	FD_Hy = Ez_to_Hy(Ez, dL, omega, sx_b, sy_b, EPSILON_0)
	FD_E = H_to_Ez(FD_Hy, FD_Hx, eps, sx_f, sy_f, dL, omega, MU_0)

	FD_E = torch.stack((torch.real(FD_E), torch.imag(FD_E)), axis=-1)

	source_vector = -1/eps[:,1:-1,1:-1,None]*MU_0/EPSILON_0*(wl[:,None,None,None]/(2*np.pi*dL[:,None,None,None]))**2*source[:,1:-1,1:-1, :]

	if separate_source:
		return FD_E, source_vector
	else:
		return FD_E + source_vector

def Ez_to_Hx(Ez, dL, omega, Sx, Sy, EPSILON_0 = EPSILON_0, periodic=False):
	if periodic:
		Hx = -1j * (Ez - torch.roll(Ez, 1, dims=1))/dL[:,None,None]/omega[:,None,None]/MU_0/torch.roll(Sx, -1, dims=1)
	else:
		Hx = -1j * (Ez[:, 1:,1:-1] - Ez[:, :-1,1:-1])/dL[:,None,None]/omega[:,None,None]/MU_0/Sx[:, 1:, 1:-1]
	return Hx

def Ez_to_Hy(Ez, dL, omega, Sx, Sy, EPSILON_0 = EPSILON_0, periodic=False):
	if periodic:
		Hy = 1j * (Ez - torch.roll(Ez, 1, dims=2))/dL[:,None,None]/omega[:,None,None]/MU_0/torch.roll(Sy, -1, dims=2)
	else:
		Hy = 1j * (Ez[:,1:-1, 1:] - Ez[:,1:-1, :-1])/dL[:,None,None]/omega[:,None,None]/MU_0/Sy[:, 1:-1, 1:]
	return Hy

def H_to_Ez(Hy, Hx, eps, Sx, Sy, dL, omega, MU_0 = MU_0):
	Sx_eps = eps.to(torch.complex64)[:,1:-1,1:-1] * Sx[:, 1:-1, 1:-1]
	Sy_eps = eps.to(torch.complex64)[:,1:-1,1:-1] * Sy[:, 1:-1, 1:-1]
	Ez = 1j*((Hy[:,:,1:] - Hy[:,:,:-1])/Sy_eps-(Hx[:, 1:, :] - Hx[:, :-1, :])/Sx_eps)/dL[:,None,None]/omega[:,None,None]/EPSILON_0
	return Ez

def E_to_bc_full_corner(Ez_R, Ez_I, dL, wl, eps, sx_f, sy_f, EPSILON_0 = EPSILON_0):
	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sy_f = 1+1j*sy_f
	eps_with_sx = eps.to(torch.complex64) * sx_f
	eps_with_sy = eps.to(torch.complex64) * sy_f

	Ez = Ez_R + 1j*Ez_I
	# x = 1 / 2 * (eps_grid[:, 1:, :] + eps_grid[:, 0:-1, :]) # Material averaging
	top_bc = (Ez[:,0,:]-Ez[:,1,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[:,0, :])/wl[:,None]*dL[:,None]*1/2*(Ez[:,0,:]+Ez[:,1,:])
	bottom_bc = (Ez[:,-1,:]-Ez[:,-2,:])+1j*2*np.pi*torch.sqrt(eps_with_sx[:,-1, :])/wl[:,None]*dL[:,None]*1/2*(Ez[:,-1,:]+Ez[:,-2,:])
	left_bc = (Ez[:,:,0]-Ez[:,:,1])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,: ,0])/wl[:,None]*dL[:,None]*1/2*(Ez[:,:,0]+Ez[:,:,1])
	right_bc = (Ez[:,:,-1]-Ez[:,:,-2])+1j*2*np.pi*torch.sqrt(eps_with_sy[:,: ,-1])/wl[:,None]*dL[:,None]*1/2*(Ez[:,:,-1]+Ez[:,:,-2])
	return torch.stack((torch.real(top_bc), torch.imag(top_bc), torch.real(bottom_bc), torch.imag(bottom_bc),\
						torch.real(left_bc), torch.imag(left_bc), torch.real(right_bc), torch.imag(right_bc)), axis = -1)

def maxwell_robin_residue(x, eps, top_bc, bottom_bc, left_bc, right_bc, source, Sxs, Sys, dL, wl, bc_mult=1, clamp=None):
	omega = 2 * np.pi * C_0 / wl
	sx_f, sx_b = Sxs
	sy_f, sy_b = Sys
		
	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sx_b = 1+1j*sx_b
		sy_f = 1+1j*sy_f
		sy_b = 1+1j*sy_b

	inner_residue = eps[:,1:-1,1:-1,None]*(E_to_E(x[:,:,:,0], x[:,:,:,1], dL, wl, omega, eps, source, (sx_f, sx_b), (sy_f, sy_b))-x[:,1:-1,1:-1,:])
	boundary_residue = bc_mult*(E_to_bc_full_corner(x[:,:,:,0], x[:,:,:,1], dL, wl, eps, sx_f, sy_f) - torch.cat([top_bc[:,0,:,:],bottom_bc[:,0,:,:],left_bc[:,:,0,:],right_bc[:,:,0,:]], axis=-1))

	residue = torch.zeros_like(x)

	if clamp is not None:
		residue[:,1:-1,1:-1,:] = torch.clamp(inner_residue, min=-clamp, max=clamp)
	else:
		residue[:,1:-1,1:-1,:] = inner_residue

	residue[:,0,:,:] += boundary_residue[:,:,0:2]
	residue[:,-1,:,:] += boundary_residue[:,:,2:4]
	residue[:,:,0,:] += boundary_residue[:,:,4:6]
	residue[:,:,-1,:] += boundary_residue[:,:,6:8]

	residue *= (dL[:,None,None,None]/wl[:,None,None,None])**.5

	return -residue

def maxwell_robin_Aop(x, eps, source, Sxs, Sys, dL, wl, bc_mult):
	omega = 2 * np.pi * C_0 / wl
	sx_f, sx_b = Sxs
	sy_f, sy_b = Sys
		
	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sx_b = 1+1j*sx_b
		sy_f = 1+1j*sy_f
		sy_b = 1+1j*sy_b
	
	inner = eps[:,1:-1,1:-1,None]*(E_to_E(x[:,:,:,0], x[:,:,:,1], dL, wl, omega, eps, source, (sx_f, sx_b), (sy_f, sy_b), separate_source=True)[0]-x[:,1:-1,1:-1,:])
	boundary = bc_mult*(E_to_bc_full_corner(x[:,:,:,0], x[:,:,:,1], dL, wl, eps, sx_f, sy_f))

	Ax = torch.zeros_like(x)

	# Ax[:,1:-1,1:-1,:] = torch.clamp(inner, min=-clamp, max=clamp)
	Ax[:,1:-1,1:-1,:] = inner

	Ax[:,0,:,:] += boundary[:,:,0:2]
	Ax[:,-1,:,:] += boundary[:,:,2:4]
	Ax[:,:,0,:] += boundary[:,:,4:6]
	Ax[:,:,-1,:] += boundary[:,:,6:8]

	return Ax * (dL[:,None,None,None]/wl[:,None,None,None])**.5

def maxwell_robin_damping_residue(x, eps, top_bc, bottom_bc, left_bc, right_bc, source, Sxs, Sys, dL, wl, bc_mult=1, clamp=None, gamma=1.5):
	omega = 2 * np.pi * C_0 / wl
	sx_f, sx_b = Sxs
	sy_f, sy_b = Sys

	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sx_b = 1+1j*sx_b
		sy_f = 1+1j*sy_f
		sy_b = 1+1j*sy_b

	sx_f_no_pml = torch.ones_like(sx_f, dtype=torch.complex64)
	sx_b_no_pml = torch.ones_like(sx_b, dtype=torch.complex64)
	sy_f_no_pml = torch.ones_like(sy_f, dtype=torch.complex64)
	sy_b_no_pml = torch.ones_like(sy_b, dtype=torch.complex64)

	inner_residue = eps[:,1:-1,1:-1,None]*(E_to_E(x[:,:,:,0], x[:,:,:,1], dL, wl, omega, eps, source, (sx_f_no_pml, sx_b_no_pml), (sy_f_no_pml, sy_b_no_pml))-x[:,1:-1,1:-1,:])
	boundary_residue = bc_mult*(E_to_bc_full_corner(x[:,:,:,0], x[:,:,:,1], dL, wl, eps, sx_f, sy_f) - torch.cat([top_bc[:,0,:,:],bottom_bc[:,0,:,:],left_bc[:,:,0,:],right_bc[:,:,0,:]], axis=-1))

	# add a damping loss:
	damping_loss = 1j*gamma*(sx_f.imag[:,1:-1,1:-1] + sy_f.imag[:,1:-1,1:-1]) * torch.view_as_complex(x[:,1:-1,1:-1]) 
	inner_residue += torch.view_as_real(damping_loss)

	residue = torch.zeros_like(x)

	if clamp is not None:
		residue[:,1:-1,1:-1,:] = torch.clamp(inner_residue, min=-clamp, max=clamp)
	else:
		residue[:,1:-1,1:-1,:] = inner_residue

	residue[:,0,:,:] += boundary_residue[:,:,0:2]
	residue[:,-1,:,:] += boundary_residue[:,:,2:4]
	residue[:,:,0,:] += boundary_residue[:,:,4:6]
	residue[:,:,-1,:] += boundary_residue[:,:,6:8]

	residue *= (dL[:,None,None,None]/wl[:,None,None,None])**.5

	return -residue

def maxwell_robin_damping_Aop(x, eps, source, Sxs, Sys, dL, wl, bc_mult=1, gamma=1.5):
	omega = 2 * np.pi * C_0 / wl
	sx_f, sx_b = Sxs
	sy_f, sy_b = Sys

	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sx_b = 1+1j*sx_b
		sy_f = 1+1j*sy_f
		sy_b = 1+1j*sy_b

	sx_f_no_pml = torch.ones_like(sx_f, dtype=torch.complex64)
	sx_b_no_pml = torch.ones_like(sx_b, dtype=torch.complex64)
	sy_f_no_pml = torch.ones_like(sy_f, dtype=torch.complex64)
	sy_b_no_pml = torch.ones_like(sy_b, dtype=torch.complex64)

	inner = eps[:,1:-1,1:-1,None]*(E_to_E(x[:,:,:,0], x[:,:,:,1], dL, wl, omega, eps, source, (sx_f_no_pml, sx_b_no_pml), (sy_f_no_pml, sy_b_no_pml), separate_source=True)[0]-x[:,1:-1,1:-1,:])
	boundary = bc_mult*(E_to_bc_full_corner(x[:,:,:,0], x[:,:,:,1], dL, wl, eps, sx_f, sy_f))

	# add a damping loss:
	damping_loss = 1j*gamma*(sx_f.imag[:,1:-1,1:-1] + sy_f.imag[:,1:-1,1:-1]) * torch.view_as_complex(x[:,1:-1,1:-1]) 
	inner += torch.view_as_real(damping_loss)

	Ax = torch.zeros_like(x)
	Ax[:,1:-1,1:-1,:] = inner

	Ax[:,0,:,:] += boundary[:,:,0:2]
	Ax[:,-1,:,:] += boundary[:,:,2:4]
	Ax[:,:,0,:] += boundary[:,:,4:6]
	Ax[:,:,-1,:] += boundary[:,:,6:8]

	Ax *= (dL[:,None,None,None]/wl[:,None,None,None])**.5

	return Ax

def plane_wave_bc(dL, wl, eps, sx_f, sy_f, plane_wave_direction=(1,0), periodic=False, x_patches=None, y_patches=None):
	if periodic:
		raise NotImplementedError
	
	plane_wave_direction = np.array(plane_wave_direction)

	if not torch.is_complex(sx_f):
		sx_f = 1+1j*sx_f
		sy_f = 1+1j*sy_f
	eps_with_sx = eps.to(torch.complex64) * sx_f
	eps_with_sy = eps.to(torch.complex64) * sy_f

	top_k_prod = np.array([-1,0]).dot(plane_wave_direction)
	bottom_k_prod = np.array([1,0]).dot(plane_wave_direction)
	left_k_prod = np.array([0,-1]).dot(plane_wave_direction)
	right_k_prod = np.array([0,1]).dot(plane_wave_direction)

	########## idea 2, 
	boundary_average_pixels = 3

	top_phase = torch.cumsum(-torch.mean(torch.sqrt(eps_with_sx[:,:boundary_average_pixels, :]), dim=1), dim=-1) * 2*np.pi * np.array([0,1]).dot(plane_wave_direction) * dL[:,None]/wl[:,None]
	left_phase = torch.cumsum(-torch.mean(torch.sqrt(eps_with_sy[:,: ,:boundary_average_pixels]), dim=2), dim=-1) * 2*np.pi * np.array([1,0]).dot(plane_wave_direction) * dL[:,None]/wl[:,None]
	bottom_phase = left_phase[:,-1:] + torch.cumsum(-torch.mean(torch.sqrt(eps_with_sx[:,-boundary_average_pixels:, :]), dim=1), dim=-1) * 2*np.pi * np.array([0,1]).dot(plane_wave_direction) * dL[:,None]/wl[:,None]
	right_phase = top_phase[:,-1:] + torch.cumsum(-torch.mean(torch.sqrt(eps_with_sy[:,: ,-boundary_average_pixels:]), dim=2), dim=-1) * 2*np.pi * np.array([1,0]).dot(plane_wave_direction) * dL[:,None]/wl[:,None]

	phase_mismatch = bottom_phase[:,-1:] - right_phase[:,-1:]
	linear_gradient = torch.linspace(0, 1, eps_with_sx.shape[1]).to(eps.device)

	top_phase = top_phase + linear_gradient[None,:] * phase_mismatch/4
	right_phase = right_phase + (1+linear_gradient[None,:]) * phase_mismatch/4
	left_phase = left_phase - linear_gradient[None,:] * phase_mismatch/4
	bottom_phase = bottom_phase - (1+linear_gradient[None,:]) * phase_mismatch/4
	assert torch.sum(torch.abs(bottom_phase[:,-1] - right_phase[:,-1])) < 1e-3, f"{torch.sum(torch.abs(bottom_phase[:,-1] - right_phase[:,-1]))}, {bottom_phase[:,-1]}, {right_phase[:,-1]}"

	top_bc = torch.exp(1j*top_phase) * 1j*2*np.pi*torch.sqrt(eps_with_sx[:,0, :])/wl[:,None]*dL[:,None] * (-top_k_prod+1)
	bottom_bc = torch.exp(1j*bottom_phase) * 1j*2*np.pi*torch.sqrt(eps_with_sx[:,-1, :])/wl[:,None]*dL[:,None] * (-bottom_k_prod+1)
	left_bc = torch.exp(1j*left_phase) * 1j*2*np.pi*torch.sqrt(eps_with_sy[:,: ,0])/wl[:,None]*dL[:,None] * (-left_k_prod+1)
	right_bc = torch.exp(1j*right_phase) * 1j*2*np.pi*torch.sqrt(eps_with_sy[:,: ,-1])/wl[:,None]*dL[:,None] * (-right_k_prod+1)

	if not periodic and x_patches is not None:
		top_bc = top_bc.reshape((x_patches, y_patches, -1))
		bottom_bc = bottom_bc.reshape((x_patches, y_patches, -1))
		left_bc = left_bc.reshape((x_patches, y_patches, -1))
		right_bc = right_bc.reshape((x_patches, y_patches, -1))

		top_bc[0:1,:] = 0
		bottom_bc[-1:,:] = 0
		left_bc[:,0:1] = 0
		right_bc[:,-1:] = 0

		top_bc = top_bc.reshape((x_patches*y_patches, -1))	
		bottom_bc = bottom_bc.reshape((x_patches*y_patches, -1))
		left_bc = left_bc.reshape((x_patches*y_patches, -1))
		right_bc = right_bc.reshape((x_patches*y_patches, -1))

	return torch.view_as_real(top_bc)[:,None,:,:], torch.view_as_real(bottom_bc)[:,None,:,:], torch.view_as_real(left_bc)[:,:,None,:], torch.view_as_real(right_bc)[:,:,None,:]

def random_bc(eps, x_patches=None, y_patches=None):
	bs, sx, sy = eps.shape

	top_bc = torch.randn(bs, 1, sy, 2).to(eps.device)
	bottom_bc = torch.randn(bs, 1, sy, 2).to(eps.device)
	left_bc = torch.randn(bs, sx, 1, 2).to(eps.device)
	right_bc = torch.randn(bs, sx, 1, 2).to(eps.device)

	return top_bc, bottom_bc, left_bc, right_bc

def lowfreq_random_vector(bs, n, cutoff, device="cpu"):
    # random complex spectrum
    spectrum = torch.zeros(bs, n, dtype=torch.complex64, device=device)

    # fill first cutoff frequencies
    real_part = torch.randn(bs, cutoff, device=device)
    imag_part = torch.randn(bs, cutoff, device=device)
    spectrum[:, :cutoff] = real_part + 1j * imag_part

    # inverse FFT to get time/space domain vector
    signal = torch.view_as_real(torch.fft.ifft(spectrum))
    return signal

def random_fourier_bc(eps, x_patches=None, y_patches=None):
	bs, sx, sy = eps.shape

	top_bc = lowfreq_random_vector(bs, sy, 30).to(eps.device)[:,None,:,:]
	bottom_bc = lowfreq_random_vector(bs, sy, 30).to(eps.device)[:,None,:,:]
	left_bc = lowfreq_random_vector(bs, sx, 30).to(eps.device)[:,:,None,:]
	right_bc = lowfreq_random_vector(bs, sx, 30).to(eps.device)[:,:,None,:]

	return top_bc, bottom_bc, left_bc, right_bc