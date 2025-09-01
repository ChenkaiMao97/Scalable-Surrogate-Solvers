# Copyright (c) 2025 Chenkai Mao <chenkaim@stanford.edu>
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1 (BUSL-1.1).
# See the LICENSE file in the project root for full license information.
# Date: 08/31/2025

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from einops.layers.torch import Rearrange
import gin

class ResizeChannel2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        return self.conv(x)

# class ZeroConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=False):
#         super(ZeroConv2d, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             bias=bias
#         )
#         # Initialize weights and bias to zero
#         nn.init.constant_(self.conv.weight, 0.0)
#         if bias:
#             nn.init.constant_(self.conv.bias, 0.0)

#     def forward(self, x):
#         return self.conv(x)

class ZeroResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, ref_freq=None):
        super(ZeroResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.freq_mlp = nn.Linear(1, 2*out_channels, bias=bias)
        self.ref_freq = ref_freq
        nn.init.constant_(self.conv1.weight, 0.0)
        nn.init.constant_(self.conv2.weight, 0.0)
        nn.init.constant_(self.shortcut.weight, 0.0)
        nn.init.constant_(self.freq_mlp.weight, 0.0)
        if bias:
            nn.init.constant_(self.conv1.bias, 0.0)
            nn.init.constant_(self.conv2.bias, 0.0)
            nn.init.constant_(self.freq_mlp.bias, 0.0)
    
    def forward(self, x, freq):
        output = self.conv2(F.leaky_relu(self.conv1(x), negative_slope=0.1))

        freq_ratio = torch.tensor([self.ref_freq / freq[0]], dtype=torch.float32, device=x.device)[None,:]
        scale, shift = self.freq_mlp(freq_ratio).chunk(2, dim=1)
        output = output * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
        return output + self.shortcut(x)   
        

class Conv2D_Periodic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ALPHA, stride=1, use_shortcut=True, periodic=(0,0,0), linear_mapping=False, ref_freq=None):
        # periodic: whether periodic in x, y and z
        # if linear_mapping is True, no bias, no activation
        super(Conv2D_Periodic, self).__init__()
        self.ALPHA = ALPHA
        self.linear_mapping = linear_mapping
        self.ref_freq = ref_freq

        self.padding = nn.Sequential(
                        nn.CircularPad2d((0,0,1,1)) if periodic[0] else nn.ZeroPad2d((0,0,1,1)),
                        nn.CircularPad2d((1,1,0,0)) if periodic[1] else nn.ZeroPad2d((1,1,0,0)),
                       ) # pay attention to order!

        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=0, bias=not self.linear_mapping)
        
        self.freq_mlp = nn.Linear(1, 2*planes, bias=False)
        
        self.use_shortcut = use_shortcut
        if self.use_shortcut:
            self.shortcut = nn.Sequential(nn.Identity())
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x, freq, mult_tensor=None):
        # if provided, multiply with the coefficient tensor
        if mult_tensor is not None:
            x = x * mult_tensor
            
        if self.linear_mapping:
            out = self.conv(self.padding(x))
        else:
            out = F.leaky_relu(self.conv(self.padding(x)), negative_slope=self.ALPHA)
        
        assert torch.unique(freq).numel() == 1

        freq_ratio = torch.tensor([self.ref_freq / freq[0]], dtype=torch.float32, device=x.device)[None,:]
        scale, shift = self.freq_mlp(freq_ratio).chunk(2, dim=1)
        out = out * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
        
        if self.use_shortcut:
            out += self.shortcut(x)
        return out

def Upsample(dim, dim_out, periodic=(False, False, False), linear_mapping=False):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear'),
        nn.Conv2d(dim, dim_out, 1, bias = not linear_mapping)
    )

def Downsample(dim, dim_out, linear_mapping=False):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1, bias=not linear_mapping) # if linear_mapping is True, no bias
    )

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_freq, ALPHA, modes1, modes2, ref_freq=None):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_freq = hidden_freq
        self.ALPHA = ALPHA
        self.ref_freq = ref_freq
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / hidden_freq / self.in_channels / self.out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.hidden_freq, dtype=torch.float32))
        self.mlp = nn.Linear(self.hidden_freq, self.modes1*self.modes2*2, bias=False) # +1 for the frequency ratio
        self.freq_mlp1 = nn.Linear(1, 128)
        self.freq_mlp2 = nn.Linear(128, self.out_channels)
        

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, freq, mult_tensor=None):
        # if provided, multiply with the coefficient tensor
        if mult_tensor is not None:
            x = x * mult_tensor

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])

        #################### (1) interp with scaled frequency ####################
        assert torch.unique(freq).numel() == 1
        freq = torch.tensor([self.ref_freq / freq[0]-1], dtype=torch.float32, device=x.device)
        freq = F.leaky_relu(self.freq_mlp1(freq[None,:]), negative_slope=self.ALPHA)
        freq = self.freq_mlp2(freq).reshape(-1, self.out_channels)

        selection_weights = F.softmax(freq, dim=-1)
        scaled_weights = self.weights * selection_weights[...,None]
        # scaled_weights = self.weights

        weights = self.mlp(scaled_weights).reshape((self.in_channels, self.out_channels, self.modes1, self.modes2, 2))
        weights_r = F.interpolate(weights[...,0], size=(x.size(-2), x.size(-1)//2+1), mode='bilinear')
        weights_i = F.interpolate(weights[...,1], size=(x.size(-2), x.size(-1)//2+1), mode='bilinear')
        weights = torch.view_as_complex(torch.stack((weights_r, weights_i), dim=-1))
        
        out_ft = self.compl_mul2d(x_ft, weights)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class residual_blocks(nn.Module):
    """
    One level of UNet, which consists a number of residual blocks.
    Each residual block uses either convolution or Fourier neural operator, with or without activation
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        use_fourier,
        modes,
        num_blocks,
        ALPHA,
        periodic,
        hidden_freq=1,
        without_last_relu=False,
        linear_mapping=False,
        ref_freq=None
    ):
        super(residual_blocks, self).__init__()
        self.num_blocks = num_blocks
        self.use_fourier = use_fourier
        self.ALPHA=ALPHA
        self.linear_mapping = linear_mapping
        self.ref_freq = ref_freq
        self.without_last_relu = without_last_relu

        self.convs = []
        self.identity = []
        for i in range(self.num_blocks):
            if i == 0:
                in_c = in_channels
            else:
                in_c = out_channels
            if self.use_fourier:
                self.convs.append(SpectralConv2d(in_c, out_channels, hidden_freq, self.ALPHA, modes[0], modes[1], ref_freq=self.ref_freq))
            else:
                self.convs.append(Conv2D_Periodic(in_c, out_channels, self.ALPHA, use_shortcut=True, periodic=periodic, linear_mapping=self.linear_mapping, ref_freq=self.ref_freq))
            self.identity.append(
                nn.Identity() if in_c == out_channels else nn.ResizeChannel2d(in_c, out_channels, bias=False)
            )
        self.convs = nn.ModuleList(self.convs)
        self.identity = nn.ModuleList(self.identity)

    def forward(self, x, freq, mult_tensor=None):
        for i in range(self.num_blocks):
            x1 = self.convs[i](x, freq, mult_tensor=mult_tensor)
            if self.linear_mapping or (i == self.num_blocks-1 and self.without_last_relu):
                x = x1 + self.identity[i](x)
            else:
                x = F.leaky_relu(x1, negative_slope=self.ALPHA) + self.identity[i](x)
        return x

class DownSizedNetwork(nn.Module):
    """
    Used both as the setup network for the coefficients and the downsample part of the network for the RHS
    """
    def __init__(
        self,
        dim_mults, # e.g. [1,2,2,2]
        conv_blocks, # e.g. [1,2,2,2]
        use_fourier, # e.g. [True, True, True, True]
        f_modes, # e.g. (32, 32, 32)
        modes_descale, # e.g. 2
        HIDDEN_DIM,
        hidden_freq,
        ALPHA,
        periodic, # e.g. (False, False, False)
        linear_mapping=False,
        ref_freq=None
    ):
        super(DownSizedNetwork, self).__init__()
        assert len(dim_mults) == len(conv_blocks) and len(dim_mults) == len(use_fourier)
        self.dim_mults = dim_mults
        self.conv_blocks = conv_blocks
        self.use_fourier = use_fourier
        self.modes_descale = modes_descale
        self.width = HIDDEN_DIM
        self.hidden_freq = hidden_freq
        self.ALPHA = ALPHA
        self.periodic = periodic
        self.linear_mapping = linear_mapping
        self.modes1, self.modes2 = f_modes
        self.ref_freq = ref_freq

        self.downs = nn.ModuleList([])
        res_blocks1 = partial(residual_blocks, ALPHA=self.ALPHA, periodic=self.periodic, hidden_freq=self.hidden_freq)

        dims = [*map(lambda m: self.width * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))
        for ind, ((dim_in, dim_out), num_block) in enumerate(zip(in_out, conv_blocks[:-1])):
            m1, m2 = int(self.modes1 / self.modes_descale**ind), int(self.modes2 / self.modes_descale**ind)
            self.downs.append(nn.ModuleList([
                res_blocks1(dim_in, dim_in, use_fourier[ind], (m1, m2), num_block, linear_mapping=self.linear_mapping, ref_freq=self.ref_freq),
                Downsample(dim_in, dim_out, linear_mapping=self.linear_mapping)
            ]))
        
        m1, m2 = int(self.modes1 / self.modes_descale**(len(dim_mults)-1)), int(self.modes2 / self.modes_descale**(len(dim_mults)-1))
        mid_dim = dims[-1]
        self.mid_block = res_blocks1(mid_dim, mid_dim, use_fourier[-1], (m1, m2), conv_blocks[-1], linear_mapping=self.linear_mapping, ref_freq=self.ref_freq)

    def forward(self, x, freq, coeff_tensors=None):
        assert coeff_tensors is None or len(coeff_tensors) == len(self.downs) + 1
        batch_size = x.shape[0] # shape: [bs, sx, sy, C]

        h = []
        for level, (block, downsample) in enumerate(self.downs):
            coeff_tensor = None if coeff_tensors is None else coeff_tensors[level]
            x = block(x, freq, mult_tensor=coeff_tensor)
            h.append(x)
            x = downsample(x)

        coeff_tensor = None if coeff_tensors is None else coeff_tensors[-1]
        x = self.mid_block(x, freq, mult_tensor=coeff_tensor)
        h.append(x)

        return h

class UpSizedNetwork(nn.Module):
    """
    The upsampling part of the network for solve_net
    """
    def __init__(
        self,
        dim_mults, # e.g. [1,2,2,2]
        conv_blocks, # e.g. [1,2,2,2]
        use_fourier, # e.g. [True, True, True, True]
        f_modes, # e.g. (32, 32, 32)
        modes_descale, # e.g. 2
        HIDDEN_DIM,
        hidden_freq,
        ALPHA,
        periodic, # e.g. (False, False, False)
        linear_mapping=False,
        ref_freq=None
    ):
        super(UpSizedNetwork, self).__init__()

        self.dim_mults = dim_mults
        self.conv_blocks = conv_blocks
        self.use_fourier = use_fourier
        self.modes_descale = modes_descale
        self.width = HIDDEN_DIM
        self.hidden_freq = hidden_freq
        self.ALPHA = ALPHA
        self.periodic = periodic
        self.linear_mapping = linear_mapping
        self.modes1, self.modes2 = f_modes
        self.ref_freq = ref_freq

        self.ups = nn.ModuleList([])
        res_blocks1 = partial(residual_blocks, ALPHA=self.ALPHA, periodic=self.periodic, hidden_freq=self.hidden_freq)
        dims = [*map(lambda m: self.width * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        for ind, ((dim_in, dim_out), num_block) in enumerate(zip(*map(reversed, (in_out, conv_blocks[:-1])))):
            m1, m2 = int(self.modes1 / self.modes_descale**(len(in_out)-1-ind)), int(self.modes2 / self.modes_descale**(len(in_out)-1-ind))
            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, periodic=self.periodic, linear_mapping=self.linear_mapping),
                res_blocks1(dim_in, dim_in, use_fourier[-2-ind], (m1, m2), num_block, linear_mapping=self.linear_mapping, ref_freq=self.ref_freq)
            ]))

    def forward(self, x, h, freq, coeff_tensors=None):
        for level, (upsample, block) in enumerate(self.ups):
            x = upsample(x)
            x = x + h.pop()
            coeff_tensor = None if coeff_tensors is None else coeff_tensors[-2-level]
            x = block(x, freq, mult_tensor=coeff_tensor)
        return x

@gin.configurable
class MGUFO2d_broadband(nn.Module):
    """
    Multi-Grid UNet - Fourier Neural Operator with frequency scaling

    Idea: network takes in parameter map and source (RHS) and outputs field.

    (1) The parameter map goes through a nonlinear setup network, which resembles the encoder of a UNet, the outputs are weights on different levels.

    (2) The source is passed through a linear network, which is a UNet architecture plus FNO, without nonlinear activations.
    The key point is at each level of the UNet, the tensors are multiplied with the outputs from the setup network.
    """
    def __init__(
        self,
        dim_mults, # e.g. [1,2,2,2]
        conv_blocks, # e.g. [1,2,2,2]
        use_fourier_setup, # e.g. [True, True, True, True]
        use_fourier_solve, # e.g. [True, True, True, True]
        setup_HIDDEN_DIM,
        setup_HIDDEN_DIM_freq,
        solve_HIDDEN_DIM,
        solve_HIDDEN_DIM_freq,
        f_modes,
        modes_descale,
        ref_freq=None, 
        domain_sizes = None, # provided by trainer
        paddings = None, # provided by trainer
        input_channel_eps = 1, # eps (1 channel)
        input_channel_rhs = 2, # RHS (residual, 2 channels)
        outc = 2, # Ez, complex
        periodic = (False, False, False),
        ALPHA = 0.1,
        field_mult = 1.0,
    ):
        """
        ref_freq : the reference frequency for the learned weights. 
            - suggested to be the middle of the frequency range in the data
            - for example, if the wavelength range in the data is [380,750], dL range is [10,20], 
            - then max frequency is 20/380 = 1/19, and min frequency is 10/750 = 1/75
            - then ref_freq can be the geometric mean: sqrt(1/19 * 1/75) = sqrt(1/1425) = 1/38

            - another strategy is to prioritize learning the high frequency problem, in which case the ref_freq should be the max frequency (1/19 in the example)
        """
        super().__init__()

        self.field_mult = torch.tensor(field_mult, requires_grad=False)
        # check the fourier modes doesn't exceed half the domain size, for each level
        self.ref_freq = ref_freq
        self.sizex, self.sizey = domain_sizes
        self.modes1, self.modes2= f_modes
        self.padding_x, self.padding_y = paddings
        assert modes_descale<=2
        assert int(self.modes1/modes_descale**(len(dim_mults)-1)) <= 1/2*(self.sizex+self.padding_x)/2**(len(dim_mults)-1)
        assert int(self.modes2/modes_descale**(len(dim_mults)-1)) <= 1/2*(self.sizey+self.padding_y)/2**(len(dim_mults)-1)

        assert setup_HIDDEN_DIM % 2 == 0
        self.eps_in_net = ResizeChannel2d(input_channel_eps, setup_HIDDEN_DIM, bias=True)
        self.rhs_in_net = ResizeChannel2d(input_channel_rhs, solve_HIDDEN_DIM, bias=False)
        self.out_net = ResizeChannel2d(solve_HIDDEN_DIM, outc, bias=False)

        self.setup_net = DownSizedNetwork(
            dim_mults,
            conv_blocks,
            use_fourier_setup,
            f_modes,
            modes_descale,
            setup_HIDDEN_DIM,
            setup_HIDDEN_DIM_freq,
            ALPHA,
            periodic,
            linear_mapping=False,
            ref_freq=self.ref_freq
        )

        self.solve_down_net = DownSizedNetwork(
            dim_mults,
            conv_blocks,
            use_fourier_solve,
            f_modes,
            modes_descale,
            solve_HIDDEN_DIM,
            solve_HIDDEN_DIM_freq,
            ALPHA,
            periodic,
            linear_mapping=True,
            ref_freq=self.ref_freq
        )

        self.solve_up_net = UpSizedNetwork(
            dim_mults,
            conv_blocks,
            use_fourier_solve,
            f_modes,
            modes_descale,
            solve_HIDDEN_DIM,
            solve_HIDDEN_DIM_freq,
            ALPHA,
            periodic,
            linear_mapping=True,
            ref_freq=self.ref_freq
        )

        # zero-conv to initialize the coefficients as zeros
        self.zero_convs = nn.ModuleList([ZeroResidualBlock(setup_HIDDEN_DIM * m, solve_HIDDEN_DIM * m, bias=False, ref_freq=self.ref_freq) for m in dim_mults])
        self.coeff_tensors = None

    def setup(self, eps, freq):
        # eps: [bs, sx, sy, 2]
        eps = self.eps_in_net(eps.permute(0,3,1,2)) # shape: [bs, H, sx, sy]
        eps = F.pad(eps, [self.padding_y//2, self.padding_y//2, self.padding_x//2, self.padding_x//2])
        
        coeff_tensors = self.setup_net(eps, freq)
        # apply a zero-conv to initialize the coefficients as zeros when starting to train
        assert len(coeff_tensors) == len(self.zero_convs)
        coeff_tensors = [self.zero_convs[i](coeff_tensors[i], freq) for i in range(len(coeff_tensors))]

        self.coeff_tensors = coeff_tensors

    def forward(self, rhs, freq):
        # rhs: [bs, sx, sy, 2]
        rhs = self.rhs_in_net(rhs.permute(0,3,1,2)) # shape: [bs, H, sx, sy]
        rhs = F.pad(rhs, [self.padding_y//2, self.padding_y//2, self.padding_x//2, self.padding_x//2])

        h = self.solve_down_net(rhs, freq, coeff_tensors=self.coeff_tensors)
        output = self.solve_up_net(h[-1], h[:-1], freq, coeff_tensors=self.coeff_tensors)

        _,_,n,m = output.shape
        output = output[..., self.padding_x//2:n-self.padding_x//2, self.padding_y//2:m-self.padding_y//2]
        output = self.out_net(output)

        return self.field_mult*output.permute(0, 2, 3, 1).contiguous()
